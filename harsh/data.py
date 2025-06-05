#plan:
# Load raw text from bucket
# Extract MDA from raw text EMRE
# Save MDA as df/csv EMRE
# Upload mda csv into bucket DONE
# Get sentiment from MDA Harsh DONE
# Upload sentiment Harsh

# Orchestrate JOSH / CHLOE

#standard imports for data/files
import pandas as pd
import os

#bucket
from google.cloud import storage
from google.api_core.exceptions import NotFound

#finbert imports
import re
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def upload_mda_csv_to_bucket(year,qtr,csv_path):

    #my_bucket_name
    bucket_name = "sentiment_chloe-curtis"

    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
    except NotFound:
        print(f"Error: The bucket '{bucket_name}' does not exist.")
        return
    except Exception as e:
        print(f"An error occurred during authentication or client initialization: {e}")
        return

    print(f"Uploading mda csv for {year}_{qtr} to bucket '{bucket_name}'...")
    
    filename = f"{year}_{qtr}_mda.csv"
    destination_blob_name = f"mda_extract/{filename}"
    blob = bucket.blob(destination_blob_name)
    
    try:
        blob.upload_from_filename(csv_path)
        print(f"Successfully uploaded '{filename}' to '{destination_blob_name}'.")
        #delete csv?
        
    except Exception as e:
        print(f"Failed to upload '{filename}'. Reason: {e}")

        
def load_df_from_bucket_mda_csv(year,qtr):
    
    bucket_name = "sentiment_chloe-curtis"
    bucket_path = f"gs://{bucket_name}/mda_extract/{year}_{qtr}_mda.csv"
    
    try:
        df = pd.read_csv(bucket_path)
        return df
    
    except Exception as e:
        print(f"Failed to download mda csv for '{bucket_path}'. Reason: {e}")
        return None
    

def get_sentiment_from_text(text_mda, model, tokenizer):
    
    # Skip any empty or extremely short text
    if not isinstance(text_mda, str) or text_mda.strip() == "" or len(text_mda.strip()) < 20:
        sentiment_dict = {
            "avg_positive": None,
            "avg_negative": None,
            "avg_neutral":  None
            }
        return sentiment_dict

    # Split the Management’s Discussion and Analysis into “paragraphs” by blank lines,
    # dropping any short chunk (< 50 characters)
    paragraphs = [
        para.strip()
        for para in re.split(r"\n+", text_mda)
        if len(para.strip()) > 50
    ]

    chunk_sentiments = []
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
            # FinBERT’s output ordering is [positive, negative, neutral]
            chunk_sentiments.append({
                "positive": probs[0],
                "negative": probs[1],
                "neutral":  probs[2],
            })

    if chunk_sentiments:
        avg_pos = sum(d["positive"] for d in chunk_sentiments) / len(chunk_sentiments)
        avg_neg = sum(d["negative"] for d in chunk_sentiments) / len(chunk_sentiments)
        avg_neu = sum(d["neutral"]  for d in chunk_sentiments) / len(chunk_sentiments)
    else:
        # If all “paragraphs” were shorter than 50 chars, leave None
        avg_pos = avg_neg = avg_neu = None
    
    sentiment_dict = {
            "avg_positive": avg_pos,
            "avg_negative": avg_neg,
            "avg_neutral":  avg_neu
            }
    return sentiment_dict
    
def get_sentiment_for_qtr(year,qtr):
    
    # load mda section from bucket
    df = load_df_from_bucket_mda_csv(year,qtr)
    #RENAME COLUMN FOR NOW, RENAME AS PART OF EMRE CODE
    df = df.rename(columns={"item2": "management_discussion"})
    
    #load model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    #extract sentiment for companies
    out_df = df[['cik', 'filename']].copy()
    sentiment_cols = ['avg_positive', 'avg_negative', 'avg_neutral']
    out_df[sentiment_cols] = None
    
    #loop over df
    for idx, row in df.iterrows():
        #if idx < 2:
            
        cik      = row["cik"]
        filename = row["filename"]
        text_mda = row["management_discussion"]

        print(f"extracting sentiment from file: {filename}")

        #extract sentiment
        sentiment_dict = get_sentiment_from_text(text_mda, model, tokenizer)

        for col in sentiment_cols:
            #USING IDX TO LINE THINGS UP...?
            out_df.loc[idx, col] = sentiment_dict[col]

    # save results 
    #save csv local
    out_filename = f"{year}_{qtr}_sentiment.csv"
    out_filepath = os.path.join("sentiment_extract", out_filename)
    out_df.to_csv(out_filepath)
    print("saving sentiment csv:", out_filepath)
    
    # upload csv to bucket
    upload_sentiment_csv_to_bucket(year, qtr, out_filepath)
    
    return out_df #if desired

def upload_sentiment_csv_to_bucket(year,qtr,csv_path):

    #my_bucket_name
    bucket_name = "sentiment_chloe-curtis"

    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
    except NotFound:
        print(f"Error: The bucket '{bucket_name}' does not exist.")
        return
    except Exception as e:
        print(f"An error occurred during authentication or client initialization: {e}")
        return

    print(f"Uploading sentiment csv for {year}_{qtr} to bucket '{bucket_name}'...")
    
    filename = f"{year}_{qtr}_sentiment.csv"
    destination_blob_name = f"sentiment_extract/{filename}"
    blob = bucket.blob(destination_blob_name)
    
    try:
        blob.upload_from_filename(csv_path)
        print(f"Successfully uploaded '{filename}' to '{destination_blob_name}'.")
        #delete csv?
        
    except Exception as e:
        print(f"Failed to upload '{filename}'. Reason: {e}")

# Load raw text from bucket
def get_text_from_bucket(bucket_filepath):
    bucket_name = "sentiment_chloe-curtis"
    #load text from bucket
    try:
        # Initialize the Google Cloud Storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Get a reference to the blob (file)
        blob = bucket.blob(bucket_filepath)

        # Download the content of the blob as a string
        file_content = blob.download_as_text(encoding='utf-8')

        print(f"Successfully pulled text file '{bucket_filepath}' from bucket '{bucket_name}'.")
        return file_content

    except NotFound:
        print(f"Error: File '{bucket_filepath}' not found in bucket '{bucket_name}'.")
        return None
    except Exception as e:
        print(f"An error occurred while pulling the file: {e}")
        return None


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

def get_mda_from_bucket_filepath(bucket_filepath):
    #item 2 - 10 Q
    #item 7 - 10 K
    
    #pull text from bucket
    text = get_text_from_bucket(bucket_filepath)
    
    if "10-K" in bucket_filepath:
        filing_type = "10-K"
    else:
        filing_type = "10-Q"
    
    #extract mda
    mda = get_mda_from_text(text, filing_type)
    
    if not mda:
        print("did not get any mda for:", bucket_filepath)
    
    #return mda
    return mda


# Save MDA as df/csv EMRE
def create_mda_csvs_from_filepaths_df(meta_all_df, year, qtr):
    #year : 2024
    #qtr : "Q4"
    
    #filter meta df on year/qtr
    year_2_digits = str(year)[-2:]
    quarter_slice = f"{qtr}-{year_2_digits}"
    print("year2", year_2_digits)
    print("quarter_slice", quarter_slice)
    
    #get only the quarter we want
    meta_df_slice = meta_all_df[meta_all_df['quarter'] == quarter_slice]

    #TODO get only the companies we want
    
    results = []
    for idx, row in meta_df_slice.iterrows():
        if idx < 50:
            bucket_folder = row['bucket_file_path']
            bucket_filepath = f"{bucket_folder}/{row['filename']}"

            mda = get_mda_from_bucket_filepath(bucket_filepath)

            out_row_dict = {}
            out_row_dict['cik'] = row['cik']
            out_row_dict['filename'] = row['filename']
            out_row_dict['management_discussion'] = mda

            results.append(out_row_dict)
        else:
            break
    
    #create csv name
    csv_name = f"{year}_{qtr}_mda.csv"
    csv_path = os.path.join("mda_extract", csv_name)
    
    mda_df = pd.DataFrame(results)
    mda_df.to_csv(csv_path, index=False)
    
    upload_mda_csv_to_bucket(year, qtr, csv_path)
    
        
        
##run through code test/practice
test_year = 2021
test_qtr = "Q4"
#Harsh funcs
#upload_mda_csv_to_bucket(test_year,test_qtr,"item2_extracted_sec.csv")
# load_df_from_bucket_mda_csv(test_year,test_qtr)
# get_sentiment_for_qtr(test_year,test_qtr)

#Emre funcs
#load meta_df
meta_df = pd.read_csv("louis_df.csv")
# print(meta_df.head())

create_mda_csvs_from_filepaths_df(meta_df, test_year, test_qtr)