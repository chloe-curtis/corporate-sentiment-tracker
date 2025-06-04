##### ALL IMPORTS ######

import os
import re
import requests
import pandas as pd
import json

## FUNCTION takes all files in the folder location
## Table returns: cik, filing_type,	filename, conformed_period_of_report

def parse_filing_folder(folder_path):
    rows = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and ("10-Q" in filename or "10-K" in filename):
            # Extract CIK and filing type from the filename
            cik_match = re.search(r'edgar_data_(\d+)_', filename)
            type_match = re.search(r'10-[QK]', filename)

            cik = cik_match.group(1) if cik_match else None
            filing_type = type_match.group(0) if type_match else None

            # Full path to file
            file_path = os.path.join(folder_path, filename)

            # Read a portion of the file (not entire thing to save time)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(5000)  # Read first 5000 characters
                    # Extract CONFORMED PERIOD OF REPORT
                    period_match = re.search(r'CONFORMED PERIOD OF REPORT:\s*(\d{8})', content)
                    period = period_match.group(1) if period_match else None
            except Exception as e:
                period = None

            rows.append({
                'cik': cik,
                'filing_type': filing_type,
                'filename': filename,
                'conformed_period_of_report': period
            })

    return pd.DataFrame(rows)


## call fucntion
folder_path = "Project" #path will need to be changed
df = parse_filing_folder(folder_path)



#ticker_path = "company_tickers.json"
def tickers(ticker_path):
    # Load JSON data
    with open(ticker_path, 'r') as f:
        ticker_data = json.load(f)
    
    # Convert JSON structure to DataFrame
    fields = ticker_data['fields']
    records = ticker_data['data']
    tickers_df = pd.DataFrame(records, columns=fields)
    
    # Normalize CIKs in both DataFrames to 10-digit strings
    tickers_df['cik'] = tickers_df['cik'].astype(str).str.zfill(10)
    df['cik'] = df['cik'].astype(str).str.zfill(10)
    
    # Rename for clarity
    tickers_df.rename(columns={'name': 'company_name'}, inplace=True)
    
    # Merge your df with the ticker info
    df2 = df.merge(tickers_df, on='cik', how='left')

    ### set to date time format
    df2['conformed_period_of_report'] = pd.to_datetime(df2['conformed_period_of_report'])
    ### remove the day to join on next table
    df2['date_clean'] = df2['conformed_period_of_report'].dt.to_period('M').astype(str)

    
    return df2

ticker_path = "company_tickers.json"  #path will need to be changed
df2 = tickers(ticker_path)


### finds all companys in the S&P by ticker for each month
## then selects only 2023 onwards 

def all_time_tickers(file_path):
    # Read the entire 10-K into a string
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Extract quoted ticker strings
    quoted_lists = re.findall(r'"([^"]+)"', text)
    
    # Convert each comma-separated string into a list of tickers
    ticker_lists = [entry.split(',') for entry in quoted_lists]
    
    # Extract the dates
    dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
    
    # Build a list of (date, ticker) tuples
    records = []
    for date, tickers in zip(dates, ticker_lists):
        for ticker in tickers:
            records.append((date, ticker))
    
    # Create the DataFrame
    df = pd.DataFrame(records, columns=['date', 'ticker'])
    
    df['date'] = pd.to_datetime(df['date'])
    df1 = df[df['date'] > '2021-12-31'].reset_index(drop=True)
    df1['date_clean'] = df1['date'].dt.to_period('M').astype(str)
    df1 = df1.replace("BF.B", "BF-B")
    df1 = df1.replace("BRK.B", "BRK-B")

    return df1

file_path = 'dates_tickers.txt'  # path will need to be chnaged
df1 = all_time_tickers(file_path) 



## if inner join works then company was in S&P 500 in that period
df3 = df1.merge(df2, on=['date_clean','ticker'] , how='inner')