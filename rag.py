import os
from typing import Optional

import requests
from duckduckgo_search import DDGS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from db import collection

# -----------------------------
# Configuration
# -----------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "gemma4")
FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "mistral")

# Optional AirLLM config
AIRLLM_MODEL = os.getenv("AIRLLM_MODEL", MODEL)
AIRLLM_FALLBACK_MODEL = os.getenv("AIRLLM_FALLBACK_MODEL", FALLBACK_MODEL)
AIRLLM_DEVICE = os.getenv("AIRLLM_DEVICE", "auto")  # "auto" | "cpu" | "cuda"

# requests timeout=(connect_timeout, read_timeout)
OLLAMA_CONNECT_TIMEOUT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "10"))
OLLAMA_READ_TIMEOUT = float(os.getenv("OLLAMA_READ_TIMEOUT", "1000"))

# Lower = closer in many Chroma distance modes; adjust if needed
MAX_ACCEPTED_DISTANCE = 1.2
MAX_CONTEXT_CHARS = 6000

# Lazy AirLLM model cache (loaded on first use)
_AIRLLM_PRIMARY = None
_AIRLLM_FALLBACK = None


# -----------------------------
# HTTP session
# -----------------------------
def _build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=1,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


HTTP = _build_http_session()


# -----------------------------
# Retrieval helpers
# -----------------------------
def web_search(query: str, max_results: int = 5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "body": r.get("body", ""),
                }
            )
    return results


def add_knowledge(text: str, metadata=None):
    import uuid

    collection.add(
        documents=[text],
        metadatas=[metadata or {}],
        ids=[str(uuid.uuid4())],
    )


def query_knowledge(query: str, n_results: int = 5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    packed = []
    for i, doc in enumerate(docs):
        packed.append(
            {
                "text": doc,
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            }
        )
    return packed


def _format_local_context(items):
    lines = []
    for i, item in enumerate(items, start=1):
        meta = item.get("metadata") or {}
        src = meta.get("source", "local_note")
        topic = meta.get("topic", "general")
        dist = item.get("distance")
        lines.append(
            f"[LOCAL {i}] topic={topic} source={src} distance={dist}\n{item['text']}"
        )
    return "\n\n".join(lines)


def _format_web_context(items):
    lines = []
    for i, item in enumerate(items, start=1):
        lines.append(
            f"[WEB {i}] {item.get('title', '')}\nURL: {item.get('url', '')}\n{item.get('body', '')}"
        )
    return "\n\n".join(lines)


# -----------------------------
# Provider implementations
# -----------------------------
def _call_ollama(model: str, prompt: str) -> str:
    response = HTTP.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=(OLLAMA_CONNECT_TIMEOUT, OLLAMA_READ_TIMEOUT),
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "").strip()


def _normalize_airllm_output(result) -> str:
    """
    AirLLM outputs can vary depending on version/backends.
    Handle common shapes defensively.
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, list):
        # Could be list[str], list[dict], or token ids
        if not result:
            return ""
        if all(isinstance(x, str) for x in result):
            return "\n".join(result).strip()
        if isinstance(result[0], dict):
            # Common chat-like output patterns
            if "generated_text" in result[0]:
                return str(result[0].get("generated_text", "")).strip()
            if "text" in result[0]:
                return str(result[0].get("text", "")).strip()
        return str(result).strip()

    if isinstance(result, dict):
        for key in ("generated_text", "text", "response", "output"):
            if key in result:
                return str(result[key]).strip()

    return str(result).strip()


def _load_airllm_model(model_name: str):
    """
    Lazy import and model load to avoid hard dependency unless provider=airllm.
    """
    try:
        from airllm import AutoModel  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "AirLLM import failed. Install AirLLM and compatible runtime dependencies."
        ) from e

    kwargs = {}
    # Best-effort device hinting, only if your AirLLM version supports these args.
    if AIRLLM_DEVICE in ("cpu", "cuda"):
        kwargs["device"] = AIRLLM_DEVICE

    try:
        return AutoModel.from_pretrained(model_name, **kwargs)
    except TypeError:
        # Backward/forward compatibility if signature differs
        return AutoModel.from_pretrained(model_name)


def _ensure_airllm_loaded(primary: bool = True):
    global _AIRLLM_PRIMARY, _AIRLLM_FALLBACK
    if primary:
        if _AIRLLM_PRIMARY is None:
            _AIRLLM_PRIMARY = _load_airllm_model(AIRLLM_MODEL)
        return _AIRLLM_PRIMARY

    if _AIRLLM_FALLBACK is None:
        _AIRLLM_FALLBACK = _load_airllm_model(AIRLLM_FALLBACK_MODEL)
    return _AIRLLM_FALLBACK


def _call_airllm(model_name: str, prompt: str) -> str:
    model = _ensure_airllm_loaded(primary=(model_name == AIRLLM_MODEL))

    # Try several common invocation patterns for compatibility.
    # 1) model.generate(prompt)
    if hasattr(model, "generate"):
        try:
            out = model.generate(prompt)
            return _normalize_airllm_output(out)
        except TypeError:
            # Some variants expect list[str]
            out = model.generate([prompt])
            return _normalize_airllm_output(out)

    # 2) callable model(prompt)
    if callable(model):
        out = model(prompt)
        return _normalize_airllm_output(out)

    raise RuntimeError("AirLLM model object does not support generation methods.")


def _call_llm(prompt: str) -> str:
    provider = LLM_PROVIDER

    if provider == "airllm":
        answer = _call_airllm(AIRLLM_MODEL, prompt)
        if (
            not answer
            and AIRLLM_FALLBACK_MODEL
            and AIRLLM_FALLBACK_MODEL != AIRLLM_MODEL
        ):
            answer = _call_airllm(AIRLLM_FALLBACK_MODEL, prompt)
        return answer

    # Default: ollama
    answer = _call_ollama(MODEL, prompt)
    if not answer and FALLBACK_MODEL and FALLBACK_MODEL != MODEL:
        answer = _call_ollama(FALLBACK_MODEL, prompt)
    return answer


# -----------------------------
# Main answer generation
# -----------------------------
def generate_answer(question: str, use_web: bool = True, n_results: int = 5):
    retrieved = query_knowledge(question, n_results=n_results)

    relevant_local = [
        x
        for x in retrieved
        if x.get("distance") is None or x["distance"] <= MAX_ACCEPTED_DISTANCE
    ]

    local_context = _format_local_context(relevant_local) if relevant_local else ""

    source = "local"
    context = local_context

    if (not local_context or len(local_context) < 120) and use_web:
        web_results = web_search(question, max_results=5)
        web_context = _format_web_context(web_results)
        source = "web+local" if local_context else "web"
        context = (local_context + "\n\n" + web_context).strip()

    context = context[:MAX_CONTEXT_CHARS]

    prompt = f"""
You are a homelab assistant.

Behavior rules:
1) Prioritize LOCAL facts when present. They represent the user's environment.
2) If local facts are missing for a detail, use WEB facts if available.
3) Never claim you can directly inspect the user's machine.
4) Do not invent specs or values.
5) If uncertain, state exactly what is missing.
6) Keep answers practical and concise.

Return format:
- Short answers
- Why / key facts used
- Optional recommendation (if relevant)

Context:
{context}

Question:
{question}
"""

    try:
        answer = _call_llm(prompt)
        if not answer:
            answer = "I am unsure based on the available context."

    except requests.Timeout:
        # Applies to Ollama HTTP path
        return {
            "answer": (
                f"Ollama timed out after connect={OLLAMA_CONNECT_TIMEOUT}s, "
                f"read={OLLAMA_READ_TIMEOUT}s. "
                f"Try a smaller/faster model or increase OLLAMA_READ_TIMEOUT."
            ),
            "source": source,
            "context": context,
            "provider": LLM_PROVIDER,
        }
    except requests.RequestException as e:
        # Applies to Ollama HTTP path
        return {
            "answer": f"Ollama request failed: {e}",
            "source": source,
            "context": context,
            "provider": LLM_PROVIDER,
        }
    except ValueError:
        # Applies to Ollama JSON parsing path
        return {
            "answer": "Ollama returned a non-JSON response.",
            "source": source,
            "context": context,
            "provider": LLM_PROVIDER,
        }
    except Exception as e:
        # Includes AirLLM loading/runtime issues
        return {
            "answer": f"LLM provider '{LLM_PROVIDER}' failed: {e}",
            "source": source,
            "context": context,
            "provider": LLM_PROVIDER,
        }

    return {
        "answer": answer,
        "source": source,
        "context": context,
        "provider": LLM_PROVIDER,
    }
