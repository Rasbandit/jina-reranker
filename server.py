import torch
from transformers import AutoModel
from fastapi import FastAPI
from pydantic import BaseModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(
    "jinaai/jina-reranker-v3",
    dtype="auto",
    trust_remote_code=True,
)
model.to(device)
model.eval()

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


class RerankRequest(BaseModel):
    query: str
    texts: list[str]
    top_n: int = 5


@app.post("/rerank")
def rerank(req: RerankRequest):
    results = model.rerank(req.query, req.texts, top_n=req.top_n)
    return [{"score": float(r["relevance_score"]), "text": r["document"]} for r in results]
