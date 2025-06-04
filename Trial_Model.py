import os
import re
import glob
import nltk
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


# ─── 2) Item 7 Extractor ─────────────────────────────────────────────────
def extract_item7_from_10k(filepath: str, skip_chars: int = 18000) -> str:
    with open(raw_data/edgar_filings_2024_QTR4, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    text_to_search = raw_text[skip_chars:]
    start_pattern = re.compile(r"(?m)^ITEM\s+7(?:\s*[\.\-]|\s).*", re.IGNORECASE)
    m_start = start_pattern.search(text_to_search)
    if not m_start:
        return None
    start_idx = m_start.end()

    end_pattern = re.compile(
        r"(?m)^(ITEM\s+7A(?:\s*[\.\-]|\s).*|ITEM\s+8(?:\s*[\.\-]|\s).*)",
        re.IGNORECASE
    )
    m_end = end_pattern.search(text_to_search, pos=start_idx)
    if not m_end:
        return None
    end_idx = m_end.start()

    return text_to_search[start_idx:end_idx].strip()


# ─── 3) Chunking Utility ────────────────────────────────────────────────
def chunk_text_for_finbert(full_text: str, tokenizer, max_tokens: int = 510) -> list[str]:
    sentences = sent_tokenize(full_text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_tokens = tokenizer.tokenize(sent)
        sent_len = len(sent_tokens)
        if current_len + sent_len > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = sent_len
        else:
            current_chunk.append(sent)
            current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ─── 4) Main: Sentiment Function ─────────────────────────────────────────
def get_sentiment_for(
    ticker: str,
    quarter: str,
    raw_folder: str = "raw_data/edgar_filings_2024_QTR4",
    model_name: str = "ProsusAI/finbert"
) -> dict:
    # 4.1) Find the matching 10-K file
    pattern = os.path.join(raw_folder, f"{ticker}_*{quarter}*_10-K.txt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No 10-K file found for {ticker} {quarter} in {raw_folder}/")
    txt_path = matches[0]

    # 4.2) Extract Item 7
    item7_text = extract_item7_from_10k(txt_path, skip_chars=18000)
    if not item7_text:
        raise ValueError(f"Could not extract MD&A (Item 7) from {txt_path}")

    # 4.3) Load FinBERT
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    nlp_pipe  = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # 4.4) Chunk MD&A
    chunks = chunk_text_for_finbert(item7_text, tokenizer, max_tokens=510)

    # 4.5) Run FinBERT on each chunk
    all_pos = []
    all_neu = []
    all_neg = []

    for chunk in tqdm(chunks, desc=f"Scoring {ticker} {quarter}"):
        out = nlp_pipe(chunk)
        scores = {d["label"].lower(): d["score"] for d in out}
        all_pos.append(scores.get("positive", 0.0))
        all_neu.append(scores.get("neutral",  0.0))
        all_neg.append(scores.get("negative", 0.0))

    # 4.6) Aggregate
    avg_pos = sum(all_pos) / len(all_pos)
    avg_neu = sum(all_neu) / len(all_neu)
    avg_neg = sum(all_neg) / len(all_neg)
    net_sent = avg_pos - avg_neg

    result = {
        "ticker":        ticker.upper(),
        "quarter":       quarter.upper(),
        "num_chunks":    len(chunks),
        "avg_positive":  round(avg_pos,  4),
        "avg_neutral":   round(avg_neu,   4),
        "avg_negative":  round(avg_neg,   4),
        "net_sentiment": round(net_sent,  4)
    }

    print(f"\n=== Sentiment for {ticker.upper()} {quarter.upper()} ===")
    print(f"Chunks analyzed : {len(chunks)}")
    print(f"Avg positive    : {result['avg_positive']}")
    print(f"Avg neutral     : {result['avg_neutral']}")
    print(f"Avg negative    : {result['avg_negative']}")
    print(f"Net sentiment   : {result['net_sentiment']}")

    return result
