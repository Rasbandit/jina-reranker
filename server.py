"""jina-reranker-v3 behind POST /rerank.

The model loads on the first request and unloads after IDLE_SECONDS without
one, so its VRAM is free while nothing is reranking.
"""

import gc
import os
import threading
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel

IDLE_SECONDS = float(os.environ.get("IDLE_SECONDS", "300"))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Held for a whole rerank, so an idle unload can never pull the model out from
# under a request. ponytail: serializes requests; one GPU runs them one at a time anyway.
lock = threading.Lock()
_model = None
_last_used = 0.0
_timer = None


def _load():
    """The loaded model. Caller holds `lock`."""
    global _model
    if _model is None:
        _model = AutoModel.from_pretrained(
            "jinaai/jina-reranker-v3",
            dtype="auto",
            trust_remote_code=True,
        )
        _model.to(device)
        _model.eval()
    return _model


def _unload_if_idle():
    global _model
    with lock:
        # A timer that fired during a request waits on the lock, so re-check.
        if _model is None or time.monotonic() - _last_used < IDLE_SECONDS:
            return
        _model = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def _arm_unload():
    """Restart the idle countdown. Caller holds `lock`."""
    global _timer
    if _timer is not None:
        _timer.cancel()
    _timer = threading.Timer(IDLE_SECONDS, _unload_if_idle)
    _timer.daemon = True
    _timer.start()


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "loaded": _model is not None}


class RerankRequest(BaseModel):
    query: str
    texts: list[str]
    top_n: int = 5


@app.post("/rerank")
def rerank(req: RerankRequest):
    global _last_used
    with lock:
        try:
            results = _load().rerank(req.query, req.texts, top_n=req.top_n)
        finally:
            _last_used = time.monotonic()
            _arm_unload()
    return [{"score": float(r["relevance_score"]), "text": r["document"]} for r in results]
