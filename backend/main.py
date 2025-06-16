# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from backend.model import get_prediction_from_mda
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = FastAPI()

@app.on_event("startup")
async def load_model():
    await asyncio.sleep(0.1)
    # Set the cache directory to where the models were downloaded in the Docker build
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    app.state.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", cache_dir=cache_dir)
    app.state.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", cache_dir=cache_dir)
    print("Model and tokenizer loaded from cache.")

class MdaPayload(BaseModel):
    mda_payload: str
    selected_ticker: str

@app.get("/")
def read_root():
    return {"message": "Backend is working!"}

@app.post("/get_prediction_from_mda")
def get_prediction_from_sentiment_stats_proc(payload: MdaPayload):
    print("mda_payload", payload.mda_payload)
    print("selected_ticker", payload.selected_ticker)
    prediction, neutral_dominance, net_sentiment, industry, sentiment_entropy = get_prediction_from_mda(
        payload.mda_payload, app.state.tokenizer, app.state.model, payload.selected_ticker, '2'
    )
    if hasattr(prediction, "__iter__") and not isinstance(prediction, (str, bytes, dict)):
        prediction = prediction[0]
    if hasattr(prediction, "item"):
        prediction = prediction.item()
    print("backend endpoint sending prediction:", prediction)
    return {
        "prediction": prediction,
        "neutral_dominance": neutral_dominance,
        "net_sentiment": net_sentiment,
        "industry": industry,
        "sentiment_entropy": sentiment_entropy
    }
