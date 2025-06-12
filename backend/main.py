
# main.py

from fastapi import FastAPI
from pydantic import BaseModel
# Hem fonksiyonu hem de test metnini model.py'dan import ediyoruz
from model import get_sentiment_stats_from_text, test_mda
from model import make_prediction, get_prediction_from_mda
import pandas as pd

app = FastAPI()

class Mda_Payload(BaseModel):
    mda_from_front: str


@app.get("/")
def read_root():
    return {"message": "Backend is working!"}

# --- İSTEĞE BAĞLI METİN ANALİZİ İÇİN MEVCUT ENDPOINT (Bu kalabilir) ---
@app.post("/analyze_mda/")
def analyze_sentiment(mda_text):
    sentiment_results = get_sentiment_stats_from_text(mda_text) #finbert
    return {"sentiment_results": sentiment_results}

# --- YENİ ENDPOINT: Sadece model.py'daki varsayılan metni analiz eder ---
# @app.get("/analyze_default_mda/")
# def analyze_default_mda(params):
#     """
#     Analyzes the hardcoded TEST_MDA string from model.py and returns the results.
#     """
#     # Doğrudan import edilen TEST_MDA değişkenini kullanıyoruz
#     sentiment_results = model_prediction(params)

#     # Sonuçları JSON olarak geri döndür
#     return {"source": "Default TEST_MDA from model.py", "sentiment_results": sentiment_results}
#     return {"sentiment_results": sentiment_results}

@app.get("/get_prediction_from_sentiment_processed")
def get_prediction_from_sentiment_stats_proc(net_sentiment, industry, q_num, neutral_dominance):

    # make df out of vars passed in
    print("recevied variables:", net_sentiment, industry, q_num, neutral_dominance)

    X_new = pd.DataFrame([{
                'net_sentiment': net_sentiment,
                'industry': industry,
                'q_num': q_num,
                'neutral_dominance': neutral_dominance
                }])

    X_new = X_new.astype({
        'q_num': 'object',
        'neutral_dominance': 'object'
    })

    prediction = make_prediction(X_new)
    print(prediction)
    return {
        "prediction" : int(prediction)
    }

#try async?
@app.post("/get_prediction_from_mda")
def get_prediction_from_sentiment_stats_proc(mda_payload : Mda_Payload):
    # mda_from_front = mda_payload['mda_from_front']
    print("mda_payload", mda_payload)
    print("mda_from_front", mda_payload.mda_from_front)
    mda_from_front = mda_payload.mda_from_front
    print("backend endpoint received mda text:", mda_from_front)
    prediction = get_prediction_from_mda(mda_from_front)
    print("backend endpoint sending prediction:", prediction)
    return {
        "prediction" : prediction
    }
