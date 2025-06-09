# backend/Dockerfile

FROM python:3.12-slim

WORKDIR /app

#copy backend folder
COPY . .

RUN pip install --no-cache-dir -r requirements_backend.txt

# EXPOSE 8000

# CMD uvicorn main:app --host 0.0.0.0 --port $PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
