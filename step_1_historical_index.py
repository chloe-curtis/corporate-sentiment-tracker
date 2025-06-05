import re
import pandas as pd

def get_quarter(date_str):
    if date_str in ['2019-01', '2019-02', '2019-03']:
        return 'Q1-19'
    elif date_str in ['2019-04', '2019-05', '2019-06']:
        return 'Q2-19'
    elif date_str in ['2019-07', '2019-08', '2019-09']:
        return 'Q3-19'
    elif date_str in ['2019-10', '2019-11', '2019-12']:
        return 'Q4-19'
    elif date_str in ['2020-01', '2020-02', '2020-03']:
        return 'Q1-20'
    elif date_str in ['2020-04', '2020-05', '2020-06']:
        return 'Q2-20'
    elif date_str in ['2020-07', '2020-08', '2020-09']:
        return 'Q3-20'
    elif date_str in ['2020-10', '2020-11', '2020-12']:
        return 'Q4-20'
    elif date_str in ['2021-01', '2021-02', '2021-03']:
        return 'Q1-21'
    elif date_str in ['2021-04', '2021-05', '2021-06']:
        return 'Q2-21'
    elif date_str in ['2021-07', '2021-08', '2021-09']:
        return 'Q3-21'
    elif date_str in ['2021-10', '2021-11', '2021-12']:
        return 'Q4-21'
    elif date_str in ['2022-01', '2022-02', '2022-03']:
        return 'Q1-22'
    elif date_str in ['2022-04', '2022-05', '2022-06']:
        return 'Q2-22'
    elif date_str in ['2022-07', '2022-08', '2022-09']:
        return 'Q3-22'
    elif date_str in ['2022-10', '2022-11', '2022-12']:
        return 'Q4-22'
    elif date_str in ['2023-01', '2023-02', '2023-03']:
        return 'Q1-23'
    elif date_str in ['2023-04', '2023-05', '2023-06']:
        return 'Q2-23'
    elif date_str in ['2023-07', '2023-08', '2023-09']:
        return 'Q3-23'
    elif date_str in ['2023-10', '2023-11', '2023-12']:
        return 'Q4-23'
    elif date_str in ['2024-01', '2024-02', '2024-03']:
        return 'Q1-24'
    elif date_str in ['2024-04', '2024-05', '2024-06']:
        return 'Q2-24'
    elif date_str in ['2024-07', '2024-08', '2024-09']:
        return 'Q3-24'
    elif date_str in ['2024-10', '2024-11', '2024-12']:
        return 'Q4-24'
    else:
        return 'Unknown'

def historical_data(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Extract quoted ticker strings
    quoted_lists = re.findall(r'"([^"]+)"', text)
    ticker_lists = [entry.split(',') for entry in quoted_lists]
    
    # Extract dates
    dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
    
    # Create (date, ticker) records
    records = []
    for date, tickers in zip(dates, ticker_lists):
        for ticker in tickers:
            records.append((date, ticker))
    
    # Build initial DataFrame
    df = pd.DataFrame(records, columns=['date', 'ticker'])
    df['date'] = pd.to_datetime(df['date'])

    # Filter out pre-2019 data
    df1 = df[df['date'] > '2018-12-31'].reset_index(drop=True)
    
    # Create cleaned month and quarter columns
    df1['date_clean'] = df1['date'].dt.to_period('M').astype(str)
    df1['quarter'] = df1['date_clean'].apply(get_quarter)
    df1['date_clean'] = pd.to_datetime(df1['date_clean'], errors='coerce')

    # Standardize tickers
    df1 = df1.replace("BF.B", "BF-B")
    df1 = df1.replace("BRK.B", "BRK-B")

    # Keep only last month of each quarter
    df1 = df1[df1['date_clean'].dt.month.isin([3, 6, 9, 12])]

    # Step 1: Find the max date per quarter
    max_dates = df1.groupby('quarter')['date'].max().reset_index()
    max_dates = max_dates.rename(columns={'date': 'max_date'})

    # Step 2: Merge max_dates with df1
    df_merged = df1.merge(max_dates, on='quarter')

    # Step 3: Filter rows where date == max_date
    result = df_merged[df_merged['date'] == df_merged['max_date']]

    # Step 4: Select final columns
    result = result[['quarter', 'ticker']].drop_duplicates()

    historical_index = result

    return historical_index
