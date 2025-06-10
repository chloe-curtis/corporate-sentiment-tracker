#extract mda
import re
#finbert imports
import re
import torch
from torch.nn.functional import softmax
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Extract MDA from raw text EMRE
def get_mda_from_text(text, filing_type):
    #filing type is 10k/q
    #item 2 - 10 Q
    #item 7 - 10 K
    if filing_type == "10-K":
        mda = extract_item7_from_10k_text(text)
    else:
        mda = extract_item2_from_10q_text(text)

    return mda

def extract_item2_from_10q_text(text, skip_chars=20000):
    # Skip the first N characters (to avoid TOC and header)
    text_to_search = text[skip_chars:]  # Baş kısmı atla

    # Start and end markers: (daha esnek olması için birkaç varyasyon)
    start_marker = r"item[\s\.]*2[\s\.\:-]+"
    end_marker = r"item[\s\.]*3[\s\.\:-]+"

    # Find start
    start_match = re.search(start_marker, text_to_search, re.IGNORECASE)
    if not start_match:
        return None
    start_idx = start_match.start()

    # Find end
    end_match = re.search(end_marker, text_to_search[start_idx:], re.IGNORECASE)
    if not end_match:
        return None
    end_idx = start_idx + end_match.start()

    # Extract section
    item2_text = text_to_search[start_idx:end_idx].strip()
    return item2_text

def extract_item7_from_10k_text(text, skip_chars=18000):
    # Skip the first N characters (to avoid TOC and header)
    text_to_search = text[skip_chars:]

    # Start and end markers
    start_marker = r"Item\s*7\.*\s*Management.*?Discussion.*?Analysis"
    # End marker could be Item 7A or Item 8, sometimes Item 8 directly follows
    end_marker = r"Item\s*7A\.|Item\s*8\."

    # Find start
    start_match = re.search(start_marker, text_to_search, re.IGNORECASE)
    if not start_match:
        return None
    start_idx = start_match.start()

    # Find end
    end_match = re.search(end_marker, text_to_search[start_idx:], re.IGNORECASE)
    if not end_match:
        return None
    end_idx = start_idx + end_match.start()

    # Extract MD&A section
    item7_text = text_to_search[start_idx:end_idx].strip()
    return item7_text

def get_sentiment_stats_from_text(text_mda, model, tokenizer):
    """
    Break the input text into paragraphs, run each through FinBERT, and compute:
      • count_positive_chunks, count_negative_chunks, count_neutral_chunks
      • max_positive_score, max_negative_score, max_neutral_score
      • sum_positive, sum_negative, sum_neutral
      • avg_positive, avg_negative, avg_neutral

    Returns a dict with all 12 metrics. Assumes logits order is [positive, negative, neutral].

    """

    null_result_dict = {
            "count_positive_chunks": 0,
            "count_negative_chunks": 0,
            "count_neutral_chunks": 0,
            "max_positive_score":    -1,
            "max_negative_score":    -1,
            "max_neutral_score":     -1,
            "sum_positive":          -1,
            "sum_negative":          -1,
            "sum_neutral":           -1,
            "avg_positive":          -1,
            "avg_negative":          -1,
            "avg_neutral":           -1,
        }

    # 1) Skip if empty or too short
    if not isinstance(text_mda, str) or text_mda.strip() == "" or len(text_mda.strip()) < 20:
        # No valid chunks → everything zero or None
        return null_result_dict

    # 2) Split into “paragraphs” by blank lines, drop any < 50 chars
    paragraphs = [
        para.strip()
        for para in re.split(r"\n+", text_mda)
        if len(para.strip()) > 50
    ]

    # 3) For each paragraph, run model and collect (pos, neg, neu)
    chunk_probs = []
    for para in paragraphs:
        encoded = tokenizer(
            para,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            out   = model(**encoded)
            probs = softmax(out.logits, dim=1).squeeze().tolist()
            # probs = [positive_score, negative_score, neutral_score]
            chunk_probs.append({
                "positive": probs[0],
                "negative": probs[1],
                "neutral":  probs[2],
            })

    # 4) If no valid paragraphs, return zeros/None
    if not chunk_probs:
        return null_result_dict

    # 5) Initialize counters, sums, and max trackers
    count_pos = 0
    count_neg = 0
    count_neu = 0
    max_pos = 0.0
    max_neg = 0.0
    max_neu = 0.0
    sum_pos = 0.0
    sum_neg = 0.0
    sum_neu = 0.0

    # 6) Loop over every chunk’s probabilities
    for probs in chunk_probs:
        p = probs["positive"]
        n = probs["negative"]
        u = probs["neutral"]

        # a) increment sums
        sum_pos += p
        sum_neg += n
        sum_neu += u

        # b) track maxima
        if p > max_pos:
            max_pos = p
        if n > max_neg:
            max_neg = n
        if u > max_neu:
            max_neu = u

        # c) count which label is highest for this chunk
        top_label_idx = max(
            enumerate((p, n, u)),
            key=lambda x: x[1]
        )[0]
        if top_label_idx == 0:
            count_pos += 1
        elif top_label_idx == 1:
            count_neg += 1
        else:
            count_neu += 1

    # 7) Compute averages
    num_chunks = len(chunk_probs)
    avg_pos = sum_pos / num_chunks
    avg_neg = sum_neg / num_chunks
    avg_neu = sum_neu / num_chunks

    return {
        "count_positive_chunks": count_pos,
        "count_negative_chunks": count_neg,
        "count_neutral_chunks":  count_neu,
        "max_positive_score":    max_pos,
        "max_negative_score":    max_neg,
        "max_neutral_score":     max_neu,
        "sum_positive":          sum_pos,
        "sum_negative":          sum_neg,
        "sum_neutral":           sum_neu,
        "avg_positive":          avg_pos,
        "avg_negative":          avg_neg,
        "avg_neutral":           avg_neu,
    }
