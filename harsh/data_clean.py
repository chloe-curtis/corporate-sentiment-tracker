# data_clean.py
#functions to orchestrate the automatic uploading/downloading

#import helper functions
from data_helpers import download_df_from_bq, upload_df_to_bq
from data_helpers import extract_cik_from_filename, is_cik_desired
from data_helpers import get_mda_from_bucket_filepath
from data_utils import get_sentiment_stats_from_text

#funcs needed

#jobs needed
# 0 -- continue to upload sentiment
# 0 - look at all bucket clean paths
# 0 - look at all sentiments already (with path)
# 0 --- download sentiment table, check if

# 1 -- construct global map
# 1a - get filepaths from bucket
# 1b - get cik from filepath
# 1c - get conformed date from text
# 1d - track if it is a cleaned file?
# 1e - upload to BQ

# 2 -- extract sentiment from un-analysed on global map
# 2a - get un analysed ones
# 2b - analyse them
# 2c - upload them


# helper funcs (general)
# download df from bigquery (table name) DONE
# upload df to bigquery (df + table name) DONE
# get text from bucket filepath
# extrct cik + check if it's one we care about

#IMPORTS
#standard imports for data/files
import pandas as pd
import os

#Successfully installed google-cloud-bigquery
#pip install google-cloud-storage
#pip install google-cloud-bigquery-storage
#bucket
from google.cloud import storage
from google.api_core.exceptions import NotFound
## big query
from google.cloud import bigquery

#finbert imports
import re
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#good to know how long things take to run
import time

#TODO get unanalysed filepaths
#change prefix to '2' for non clean data
def get_all_clean_bucket_files_and_upload_to_bq(prefix = "clean_data", upload=True):
    """
        walks through bucket and uploads filepaths, filenames and ciks to BQ
        table: bucket_clean_filepaths
    """

    print("getting all clean bucket files")
    bucket_name = "sentiment_chloe-curtis"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    all_clean_bucket_paths = []
    #get iterator over all bucket paths
    bucket_blobs = bucket.list_blobs(prefix=prefix)
    i=0
    for blob in bucket_blobs:
        #count progress
        i += 1
        if i % 10000 == 0:
            print(i)

        filepath = blob.name

        #get cik from filepath
        cik_from_filepath = extract_cik_from_filename(filepath)

        #stop if not one of the desired companies
        if not is_cik_desired(int(cik_from_filepath)):
            continue

        filename = filepath.split('/')[-1]
        row_dict = {
                "bucket_filepath" : filepath,
                "cik" : cik_from_filepath,
                "filename" : filename
            }
        all_clean_bucket_paths.append(row_dict)

    print(f"found {len(all_clean_bucket_paths)} files, uploading to BQ...")

    bucket_filepaths_df = pd.DataFrame(all_clean_bucket_paths)
    if upload:
        upload_df_to_bq(bucket_filepaths_df, "bucket_clean_filepaths")

    return bucket_filepaths_df

def get_all_bq_extraction_progress(include_mda_text=True, year = None):
    print("getting BQ extraction progress")
    #get all files, unless year specified
    if year:
        print(f"getting year {year}")
        custom_query = f"""
        WITH year_files AS (
            SELECT *,
            CAST (LEFT(filename,4) AS INT64) as year
            FROM `sentiment-lewagon.sentiment_db.bucket_clean_filepaths`
        )
        SELECT * FROM year_files WHERE year = {year}
        """
        bucket_clean_files_df = download_df_from_bq("bucket_clean_filepaths", custom_query=custom_query)
    else:
        bucket_clean_files_df = download_df_from_bq("bucket_clean_filepaths")


    #get tables with progress
    mda_df = download_df_from_bq("MDA")
    sentiment_df = download_df_from_bq("SENTIMENT").drop_duplicates()
    dates_df = download_df_from_bq("final_v3")

    #add some columns
    dates_df = dates_df[['bucket_file_path', 'conformed_period_of_report']].rename(columns={"bucket_file_path" : "bucket_filepath"})
    mda_df['has_mda'] = 1
    sentiment_df['has_sentiment'] = 1

    #setup columns to select
    mda_cols = ['bucket_filepath', 'management_discussion', 'has_mda']
    sent_cols = ['bucket_filepath', 'has_sentiment']

    #merge in relevant info
    bucket_clean_df = bucket_clean_files_df.merge(mda_df[mda_cols], how='left', on='bucket_filepath')
    bucket_clean_df = bucket_clean_df.merge(sentiment_df[sent_cols], how='left', on='bucket_filepath')
    bucket_clean_df = bucket_clean_df.merge(dates_df, how='left', on='bucket_filepath')

    #sort by recent first
    bucket_clean_df = bucket_clean_df.sort_values(by="conformed_period_of_report", ascending=False)

    return bucket_clean_df.reset_index(drop=True)

def extract_missing_mda_and_upload_to_bq(testing=True, chunk_size= 250):
    """
        set testing=False to run for real
    """
    #get clean paths and whether or not they have mda and sentiment
    clean_path_df = get_all_bq_extraction_progress()

    #exclude rows that already have MDA extracted
    mda_to_process_df = clean_path_df[clean_path_df.has_mda != 1]

    #chunks
    total_rows = len(mda_to_process_df)

    if testing:
        print("in testing mode, only doing 10 rows!")
        total_rows = 10

    print(f"extracting {total_rows} rows of mda, in chunks of {chunk_size}")
    start_time = time.time()
    for i in range(0, total_rows, chunk_size):
        start_row = i
        # The end_row for the current chunk.
        # Use min() to ensure it doesn't go beyond the total_rows.
        end_row = min(i + chunk_size, total_rows) - 1 # Inclusive end for range
        print(f"Processing rows from {start_row} to {end_row}")
        chunk_results = []
        for idx, row in mda_to_process_df.iloc[start_row : end_row + 1].iterrows():
            bucket_filepath = row['bucket_filepath']
            mda = get_mda_from_bucket_filepath(bucket_filepath)

            out_row_dict = {}
            out_row_dict['cik'] = int(row['cik'])
            out_row_dict['filename'] = row['filename']
            out_row_dict['bucket_filepath'] = row['bucket_filepath']

            out_row_dict['management_discussion'] = mda

            chunk_results.append(out_row_dict)

        chunk_df = pd.DataFrame(chunk_results)
        #upload chunk to BQ
        upload_df_to_bq(chunk_df, "MDA")

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"finished uploading {total_rows} rows to MDA table.")
    print(f"Time taken: {time_taken/60:.4f} minutes")

#consider specifying year/quarter to compartmentalise work
def get_and_upload_missing_sentiment_to_bq(testing=True, chunk_size=50, running_local = False, year=None):
    print("loading sentiment analysis model")
    #load model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    bq_progress_df = get_all_bq_extraction_progress(year=year)

    #keep only rows without sentiment but with mda
    sentiment_todo = (bq_progress_df.has_sentiment != 1) & (bq_progress_df.has_mda == 1)
    sentiment_to_process_df = bq_progress_df[sentiment_todo].reset_index(drop=True)

    #chunks
    total_rows = len(sentiment_to_process_df)
    if running_local:
        sentiment_to_process_df.sort_values(by="conformed_period_of_report", ascending=True)
        chunk_size = 50
        # total_rows = 40
    if testing:
        chunk_size = 50
        # limit total rows if testing
        total_rows = 5
        print(f"in testing mode, doing {total_rows} rows!")

    print(f"extracting {total_rows} rows of sentiment, in chunks of {chunk_size}")

    start_time = time.time()
    for i in range(0, total_rows, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, total_rows) - 1 # Inclusive end for range
        print(f"Processing chunk rows from {start_row} to {end_row}")

        try:
            chunk_start_time = time.time()
            chunk_rows = []
            for idx, row in sentiment_to_process_df.iloc[start_row : end_row + 1].iterrows():
                #its torture waiting for chunks...
                if idx % 5 == 0:
                    print(f"Working on row with index {idx}")
                out_row_dict = {}

                out_row_dict['cik']  = int(row["cik"])
                out_row_dict['filename'] = row["filename"]
                out_row_dict['bucket_filepath'] = row['bucket_filepath']

                mda_text = row["management_discussion"]
                # print(f"extracting sentiment from file: {out_row_dict['bucket_filepath']}")

                #extract sentiment + put in dict
                sentiment_dict = get_sentiment_stats_from_text(mda_text, model, tokenizer)
                out_row_dict.update(sentiment_dict)

                chunk_rows.append(out_row_dict)
            #end of chunk, upload
            chunk_out_df = pd.DataFrame(chunk_rows)
            upload_df_to_bq(chunk_out_df, "SENTIMENT")
            chunk_end_time = time.time()

            time_taken = chunk_end_time - chunk_start_time
            print(f"uploaded rows {start_row} to {end_row} to bq")
            print(f"Time taken for chunk: {time_taken/60:.4f} minutes")

        except Exception as e:
            print(f"failed on chunk: {start_row} to {end_row}")
            print(f"failed at file: {row['bucket_filepath']}")
            print(e)

    #done! :D
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for all {total_rows}: {time_taken/60:.4f} minutes")

#consider getting sentiment as batch (not row by row)
#9min for 20 rows on louis laptop

#################################

table_name = "core"

#gs://sentiment_chloe-curtis/
sample_bucket_path = "clean_data_2019q1/20190103_10-Q_edgar_data_1702744_0001702744-19-000004.txt"

# test_df = download_df_from_bq(table_name)
# print(test_df.head())
# test_txt = get_text_from_bucket(sample_bucket_path)
# print("test_text:", test_txt)
# test_fps_df = get_all_clean_bucket_files()
# print(test_fps_df)

#look through bucket and get list of filepaths
# upload those to bigquery

#general uploader to bigquery takes a df and a table name?


#load text from filepath
#get mda from text
#get sentiment from text

#upload mda
#generate global map
# test_df = get_all_bq_extraction_progress()

# extract_missing_mda_and_upload_to_bq(testing=False)

if __name__ == "__main__":
    #year =2019 next
    get_and_upload_missing_sentiment_to_bq(year=2020, testing=False)
