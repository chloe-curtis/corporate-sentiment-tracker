# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from backend.model import get_prediction_from_mda
# from transformers import AutoTokenizer, AutoModelForSequenceClassification


app = FastAPI()


# @app.on_event("startup")
# async def load_model():
#     # give the HTTP server time to bind first
#     await asyncio.sleep(0.1)
#     app.state.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
#     app.state.model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
#     print("Model and tokenizer loaded successfully.")

class MdaPayload(BaseModel):
    mda_payload: str
    selected_ticker: str


@app.get("/")
def read_root():
    return {"message": "Backend is working!"}


#try async?
@app.post("/get_prediction_from_mda")
def get_prediction_from_sentiment_stats_proc(payload: MdaPayload):
    print("mda_payload", payload.mda_payload)
    print("selected_ticker", payload.selected_ticker)
    prediction, neutral_dominance, net_sentiment, industry, sentiment_entropy = get_prediction_from_mda(
        payload.selected_ticker)#, app.state.tokenizer, app.state.model
    # )
    # Convert numpy array or numpy type to Python int
    if hasattr(prediction, "__iter__") and not isinstance(prediction, (str, bytes, dict)):
        # If it's an array-like, get the first element
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
