FROM python:3.12-slim

WORKDIR /app

COPY requirements_backend.txt requirements_backend.txt

RUN pip install --no-cache-dir -r requirements_backend.txt

COPY backend backend

RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('ProsusAI/finbert'); AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"

EXPOSE 8080

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}
