FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    "transformers<5.0.0" \
    sentence-transformers \
    fastapi \
    uvicorn \
    einops

COPY server.py .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082"]
