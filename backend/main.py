
# main.py

from fastapi import FastAPI
from pydantic import BaseModel
# Hem fonksiyonu hem de test metnini model.py'dan import ediyoruz
from model import get_sentiment_stats_from_text, test_mda

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Backend is working!"}

# --- İSTEĞE BAĞLI METİN ANALİZİ İÇİN MEVCUT ENDPOINT (Bu kalabilir) ---
@app.post("/analyze/")
def analyze_sentiment(mda_text):
    sentiment_results = get_sentiment_stats_from_text(mda_text) #finbert
    return {"sentiment_results": sentiment_results}

# --- YENİ ENDPOINT: Sadece model.py'daki varsayılan metni analiz eder ---
@app.get("/analyze_default_mda/")
def analyze_default_mda():
    """
    Analyzes the hardcoded TEST_MDA string from model.py and returns the results.
    """
    # Doğrudan import edilen TEST_MDA değişkenini kullanıyoruz
    sentiment_results = get_sentiment_stats_from_text(test_mda)

    # Sonuçları JSON olarak geri döndür
    return {"source": "Default TEST_MDA from model.py", "sentiment_results": sentiment_results}
    return {"sentiment_results": sentiment_results}
