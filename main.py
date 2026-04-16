from db import collection
from fastapi import FastAPI
from models import KnowledgeCreate, Question
from rag import add_knowledge, generate_answer
from fastapi.middleware.cors import CORSMiddleware


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
