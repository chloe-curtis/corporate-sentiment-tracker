from fastapi import FastAPI
from model import get_sentiment_stats_from_text
from model import TEST_MDA

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Backend is working!"}

@app.get("/get_sentiment")
def get_sentiment(text = TEST_MDA):

    print("getting sentiment for text:", text)
    #use harsh's code to get sentiment from text


    return {"sentiment": "positive"}
