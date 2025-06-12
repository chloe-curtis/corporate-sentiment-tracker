
# FROM python:3.12-slim
FROM huggingface/transformers-pytorch-cpu:latest

WORKDIR /app

COPY requirements_backend.txt requirements_backend.txt

RUN pip install --no-cache-dir -r requirements_backend.txt

COPY backend backend

EXPOSE 8080

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
