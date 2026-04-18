import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from db import collection
from models import KnowledgeCreate, Question
from rag import add_knowledge, generate_answer

app = FastAPI(title="Homelab AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/knowledge")
def create_knowledge(item: KnowledgeCreate):
    add_knowledge(item.text, item.metadata)
    return {"status": "stored"}


@app.post("/ask")
def ask_question(q: Question):
    return generate_answer(q.question, use_web=q.use_web, n_results=q.n_results)


@app.post("/ask/stream")
def ask_question_stream(q: Question):
    """
    SSE endpoint for Ollama streaming.

    Event format:
      event: chunk
      data: {"text":"partial token/text"}

      event: done
      data: {"status":"completed"}

      event: error
      data: {"message":"..."}
    """
    try:
        from rag import stream_answer_ollama
    except Exception as e:
        err_msg = f"Streaming unavailable: {str(e)}"
    
        async def import_error_stream():
            payload = {"message": err_msg}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(import_error_stream(), media_type="text/event-stream")

    async def event_generator():
        try:
            for chunk in stream_answer_ollama(
                q.question,
                use_web=q.use_web,
                n_results=q.n_results,
            ):
                payload = {"text": chunk}
                yield f"event: chunk\ndata: {json.dumps(payload)}\n\n"

            yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"

        except Exception as e:
            payload = {"message": str(e)}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


@app.get("/knowledge")
def list_knowledge():
    data = collection.get(include=["documents", "metadatas"])
    return {
        "count": len(data.get("documents", [])),
        "data": [
            {"text": d, "metadata": m}
            for d, m in zip(data.get("documents", []), data.get("metadatas", []))
        ],
    }


@app.delete("/knowledge")
def delete_all():
    collection.delete(where={})
    return {"status": "all deleted"}


@app.delete("/knowledge/search")
def delete_by_text(text: str):
    results = collection.get(include=["documents"])
    ids_to_delete = []

    for i, doc in enumerate(results.get("documents", [])):
        if text.lower() in doc.lower():
            ids_to_delete.append(results["ids"][i])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    return {"deleted_ids": ids_to_delete}
