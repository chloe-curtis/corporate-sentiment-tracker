


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')





import os
import re
import pandas as pd

def extract_item7_from_10k(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Daha esnek regex
    start_pattern = re.compile(r'ITEM\s*7[\.\s\-:]*Management.*?Discussion.*?Analysis', re.IGNORECASE | re.DOTALL)
    end_pattern = re.compile(r'ITEM\s*8[\.\s\-:]', re.IGNORECASE)

    start = start_pattern.search(text)
    end = end_pattern.search(text, start.end()) if start else None

    if start and end:
        item7_text = text[start.start():end.start()]
        return item7_text.strip()
    else:
        return None

folder_path = "."

results = []
for fname in os.listdir(folder_path):
    if fname.endswith(".txt") and "10-K" in fname:
        fpath = os.path.join(folder_path, fname)
        item7 = extract_item7_from_10k(fpath)
        results.append({"filename": fname, "item7": item7})

df = pd.DataFrame(results)
df.to_csv("item7_extracted_10k_only.csv", index=False)
df.head(2)





import re

def extract_item7_from_10k(filepath, skip_chars=18000):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

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





import os
import pandas as pd

folder_path = "."  # Or the path to your directory
results = []

for fname in os.listdir(folder_path):
    if fname.endswith(".txt") and "10-K" in fname:
        fpath = os.path.join(folder_path, fname)
        item7 = extract_item7_from_10k(fpath, skip_chars=18000)  # You can tune skip_chars
        results.append({"filename": fname, "item7": item7})

df = pd.DataFrame(results)
df.to_csv("item7_extracted_10k_only.csv", index=False)
df.head(2)




# İlk satırdaki Item 7 metnini  gösterir
print(df["item7"][1][:])




import re

def extract_item2_from_10q(filepath, skip_chars=20000):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
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



import os
import pandas as pd

folder_path = "."  # veya kendi klasör yolun
results = []

for fname in os.listdir(folder_path):
    if fname.endswith(".txt") and "10-Q" in fname:
        fpath = os.path.join(folder_path, fname)
        item2 = extract_item2_from_10q(fpath, skip_chars=20000)  # İstersen skip_chars'ı ayarlayabilirsin
        results.append({"filename": fname, "item2": item2})

df = pd.DataFrame(results)
df.to_csv("item2_extracted_10q_only.csv", index=False)
df.head(2)





df7 = pd.read_csv("item7_extracted_10k_only.csv")




# İlk satırdaki Item 2 metnini hepsini gosterir.
print(df["item2"][1][:])




import os
import pandas as pd

folder_path = "."  # veya kendi klasör yolun
results = []

for fname in os.listdir(folder_path):
    if fname.endswith(".txt") and "10-Q" in fname:
        fpath = os.path.join(folder_path, fname)
        item2 = extract_item2_from_10q(fpath, skip_chars=20000)  # İstersen skip_chars'ı ayarlayabilirsin
        results.append({"filename": fname, "item2": item2})

df = pd.DataFrame(results)
df.to_csv("item2_extracted_10q_only.csv", index=False)
df.shape





from transformers import pipeline

# Finansal metinler için FinBERT pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")





def analyze_sentiment(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"label": "empty", "score": 0.0}
    # Çok uzun metinlerde ilk 4000 karakter (model limiti için)
    result = sentiment_analyzer(text[:4000])
    return result[0]





print(df.columns.tolist())





import pandas as pd

df7 = pd.read_csv("item7_extracted_10k_only.csv")
print(df7.columns.tolist())   # Sütun isimlerine bak
print(df7.head(2))            # İlk 2 satırı gör





def analyze_sentiment_chunked(text, max_chunk=400):
    """
    Metni max_chunk kelimelik parçalara bölüp her parçanın sentimentini alır.
    Sonuçların ortalamasını döndürür.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"label": "empty", "score": 0.0}

    # Metni kelime bazlı bölelim (token yerine)
    words = text.split()
    chunks = [' '.join(words[i:i+max_chunk]) for i in range(0, len(words), max_chunk)]
    scores = []
    labels = []
    for chunk in chunks:
        try:
            result = sentiment_analyzer(chunk)[0]
            labels.append(result['label'])
            scores.append(result['score'] if result['label'] == 'positive' else -result['score'] if result['label'] == 'negative' else 0)
        except Exception as e:
            continue
    if not scores:
        return {"label": "empty", "score": 0.0}
    # Ortalama skor
    avg_score = sum(scores)/len(scores)
    # Etiket: ortalamadan çıkar (pozitif, negatif, nötr için)
    if avg_score > 0.1:
        label = 'positive'
    elif avg_score < -0.1:
        label = 'negative'
    else:
        label = 'neutral'
    return {"label": label, "score": (avg_score)}





df7['sentiment'] = df7['item7'].apply(analyze_sentiment_chunked)
df7['sentiment_label'] = df7['sentiment'].apply(lambda x: x['label'])
df7['sentiment_score'] = df7['sentiment'].apply(lambda x: x['score'])

df7[['filename', 'sentiment_label', 'sentiment_score']].head()





def analyze_sentiment_chunked(text, max_chunk=400):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"label": "empty", "score": 0.0}

    words = text.split()
    chunks = [' '.join(words[i:i+max_chunk]) for i in range(0, len(words), max_chunk)]
    scores = []
    for chunk in chunks:
        try:
            result = sentiment_analyzer(chunk)[0]
            # Pozitif: +score, Negatif: -score, Neutral: 0
            if result['label'] == 'positive':
                scores.append(result['score'])
            elif result['label'] == 'negative':
                scores.append(-result['score'])
            else:
                scores.append(0)
        except Exception as e:
            continue
    if not scores:
        return {"label": "empty", "score": 0.0}
    avg_score = sum(scores) / len(scores)
    if avg_score > 0.1:
        label = 'positive'
    elif avg_score < -0.1:
        label = 'negative'
    else:
        label = 'neutral'
    return {"label": label, "score": (avg_score)}




import pandas as pd

df2 = pd.read_csv("item2_extracted_10q_only.csv")

df2['sentiment'] = df2['item2'].apply(analyze_sentiment_chunked)
df2['sentiment_label'] = df2['sentiment'].apply(lambda x: x['label'])
df2['sentiment_score'] = df2['sentiment'].apply(lambda x: x['score'])

df2[['filename', 'sentiment_label', 'sentiment_score']].head()
