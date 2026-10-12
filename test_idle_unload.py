"""Checks the load-on-request / unload-when-idle lifecycle without torch or the model.

    python3 test_idle_unload.py
"""

import os
import sys
import time
import types
from pathlib import Path

loads = []


class FakeModel:
    def to(self, _):
        return self

    def eval(self):
        return self

    def rerank(self, query, texts, top_n):
        return [{"relevance_score": 1.0, "document": t} for t in texts[:top_n]]


def from_pretrained(*_, **__):
    loads.append(time.monotonic())
    return FakeModel()


sys.modules["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
sys.modules["transformers"] = types.SimpleNamespace(AutoModel=types.SimpleNamespace(from_pretrained=from_pretrained))

os.environ["IDLE_SECONDS"] = "0.3"
sys.path.insert(0, str(Path(__file__).parent))
import server  # noqa: E402  (stubs must be installed first)

req = server.RerankRequest(query="q", texts=["a", "b"], top_n=1)

assert server._model is None, "loaded at import"
assert server.rerank(req) == [{"score": 1.0, "text": "a"}]
assert len(loads) == 1, "request did not load"

server.rerank(req)
assert len(loads) == 1, "second request reloaded a loaded model"

time.sleep(0.15)
server.rerank(req)
time.sleep(0.2)
assert server._model is not None, "unloaded before IDLE_SECONDS since the last use"

time.sleep(0.3)
assert server._model is None, "did not unload after IDLE_SECONDS"
assert server.health()["loaded"] is False

server.rerank(req)
assert len(loads) == 2, "request after unload did not reload"

print("ok")
