# Homelab Agent API

A lightweight FastAPI + RAG service for homelab Q&A.

It stores your local knowledge in ChromaDB, optionally augments with web search, and generates answers through a configurable LLM provider.

---

## Features

- FastAPI REST API
- Local knowledge storage with ChromaDB
- Retrieval-augmented generation (RAG)
- Optional web fallback via DuckDuckGo search
- Pluggable LLM provider:
  - `ollama` (default)
  - `airllm` (optional)

---

## Project Structure

```homelab-agent-api/README.md#L1-20
homelab-agent-api/
├── main.py
├── rag.py
├── db.py
├── models.py
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+ recommended
- `pip`
- For Ollama provider:
  - Ollama running locally or remotely
- For AirLLM provider:
  - A compatible model setup and sufficient system resources (RAM/VRAM)
  - Optional GPU/CUDA stack if you want acceleration

---

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```homelab-agent-api/README.md#L1-3
pip install -r requirements.txt
```

---

## Running the API

```homelab-agent-api/README.md#L1-3
uvicorn main:app --reload
```

Default API URL: `http://127.0.0.1:8000`  
Swagger docs: `http://127.0.0.1:8000/docs`

---

## API Endpoints

- `GET /`  
  Health check.

- `POST /knowledge`  
  Add a knowledge document.
  - body:
    - `text: str`
    - `metadata: object | null`

- `GET /knowledge`  
  List stored knowledge entries.

- `DELETE /knowledge`  
  Delete all knowledge entries.

- `DELETE /knowledge/search?text=...`  
  Delete documents that contain a text fragment.

- `POST /ask`  
  Ask a question.
  - body:
    - `question: str`
    - `use_web: bool = true`
    - `n_results: int = 5`

---

## Provider Configuration

Set environment variables (for example in a `.env` file) to choose your LLM backend.

### Common

- `LLM_PROVIDER`  
  Values: `ollama` or `airllm`  
  Default: `ollama`

### Ollama Provider

- `OLLAMA_URL`  
  Default: `http://localhost:11434/api/generate`
- `OLLAMA_MODEL`  
  Default: `gemma4`
- `OLLAMA_FALLBACK_MODEL`  
  Default: `mistral`
- `OLLAMA_CONNECT_TIMEOUT`  
  Default: `10`
- `OLLAMA_READ_TIMEOUT`  
  Default: `1000`

### AirLLM Provider

Suggested variables (names may vary depending on your implementation in `rag.py`):

- `AIRLLM_MODEL`  
  Hugging Face model id or local model path
- `AIRLLM_MAX_NEW_TOKENS`  
  Generation cap per response
- `AIRLLM_TEMPERATURE`  
  Sampling temperature
- `AIRLLM_DEVICE`  
  Example: `cpu`, `cuda`, or provider-specific behavior
- `AIRLLM_TOP_P`  
  Optional sampling config

Example:

```homelab-agent-api/README.md#L1-12
LLM_PROVIDER=airllm
AIRLLM_MODEL=meta-llama/Llama-2-7b-chat-hf
AIRLLM_MAX_NEW_TOKENS=256
AIRLLM_TEMPERATURE=0.2
AIRLLM_TOP_P=0.9
AIRLLM_DEVICE=cuda
```

---

## AirLLM Setup Notes

1. Install AirLLM-related dependencies (exact packages depend on your implementation and hardware setup).
2. Ensure model access rights if using gated Hugging Face models.
3. Expect longer startup time for first model load (cold start).
4. Keep model initialization as a singleton (load once per process).
5. Be mindful of FastAPI worker count: each worker may load its own model instance.

### Security note for model hubs

If you use private/gated model repositories, provide your token through environment variables or secure runtime secrets. Do **not** hardcode secrets in source files.

---

## Example `.env`

```homelab-agent-api/README.md#L1-16
# Provider switch
LLM_PROVIDER=ollama

# Ollama
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=gemma4
OLLAMA_FALLBACK_MODEL=mistral
OLLAMA_CONNECT_TIMEOUT=10
OLLAMA_READ_TIMEOUT=1000

# AirLLM (used when LLM_PROVIDER=airllm)
AIRLLM_MODEL=meta-llama/Llama-2-7b-chat-hf
AIRLLM_MAX_NEW_TOKENS=256
AIRLLM_TEMPERATURE=0.2
AIRLLM_TOP_P=0.9
AIRLLM_DEVICE=cpu
```

---

## Quick Usage

### Add knowledge

```homelab-agent-api/README.md#L1-8
curl -X POST "http://127.0.0.1:8000/knowledge" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"My NAS uses ZFS and snapshots nightly.\",\"metadata\":{\"source\":\"ops-notes\",\"topic\":\"storage\"}}"
```

### Ask question

```homelab-agent-api/README.md#L1-8
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"How should I protect against accidental deletion?\",\"use_web\":true,\"n_results\":5}"
```

---

## Troubleshooting

- **Timeouts with Ollama**
  - Increase `OLLAMA_READ_TIMEOUT`
  - Use a smaller/faster model

- **AirLLM memory issues**
  - Use a smaller model
  - Reduce max tokens
  - Lower worker count
  - Prefer GPU with enough VRAM (or CPU with sufficient RAM)

- **No useful answer**
  - Add more local knowledge via `/knowledge`
  - Increase `n_results`
  - Keep metadata informative (`source`, `topic`)

---

## Notes

- Local knowledge is prioritized when available.
- Web results are only used as fallback when local context is insufficient.
- This API does not directly inspect your machines; it only uses provided context and optional web retrieval.