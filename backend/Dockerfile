# backend/Dockerfile
# try bookworm image ? maybe smaller
FROM python:3.12-slim

WORKDIR /app

#copy backend folder
COPY . .

#--no-cache-dir
RUN pip install  -r requirements_backend.txt

# EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
