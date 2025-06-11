# # backend/Dockerfile
# # try bookworm image ? maybe smaller
# FROM python:3.12-slim

# WORKDIR /app

# #copy backend folder
# COPY . .

# #--no-cache-dir
# RUN pip install  -r requirements_backend.txt

# # EXPOSE 8000

# CMD uvicorn main:app --host 0.0.0.0 --port $PORT
# # CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]


# backend/Dockerfile (Optimize Edilmiş Versiyon)

# 1. Adım: Temel imajı belirle
FROM python:3.12-slim

# 2. Adım: Konteyner içinde çalışılacak klasörü ayarla
WORKDIR /app

# 3. Adım: ÖNCE sadece gereksinimler dosyasını kopyala
# Bu sayede bu dosya değişmediği sürece, bir sonraki adım önbellekten kullanılır.
COPY requirements_backend.txt .

# 4. Adım: Kütüphaneleri yükle (En yavaş adım)
# Bu adım sadece requirements_backend.txt değiştiğinde tekrar çalışır.
RUN pip install -r requirements_backend.txt

# 5. Adım: SONRA projedeki diğer tüm dosyaları kopyala
# main.py'de değişiklik yaptığınızda sadece bu hızlı adım tekrar çalışır.
COPY . .

# 6. Adım: Konteyner çalıştığında hangi komutun çalışacağını belirt
# Ortam değişkeni ($PORT) kullanmak yerine doğrudan portu belirtmek daha güvenilirdir.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
