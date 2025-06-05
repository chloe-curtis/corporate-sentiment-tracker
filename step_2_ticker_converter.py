import json
import pandas as pd

# ticker_path = "company_tickers.json"
def ticker_converter(ticker_path):
    # Load JSON data
    with open(ticker_path, 'r') as f:
        ticker_data = json.load(f)
    
    # Convert JSON structure to DataFrame
    fields = ticker_data['fields']
    records = ticker_data['data']
    tickers_df = pd.DataFrame(records, columns=fields)
    
    # Normalize CIKs to 10-digit strings
    tickers_df['cik'] = tickers_df['cik'].astype(str).str.zfill(10)
    
    # Rename for clarity
    tickers_df.rename(columns={'name': 'company_name'}, inplace=True)

    # Keep only needed columns
    ticker_conversion = tickers_df[['cik', 'ticker', 'company_name']]

    return ticker_conversion


# In[20]:


# folder_path = "Project" #path will need to be changed
# df = parse_filing_folder(folder_path)
ticker_path = "ticker_converter.json"
ticker_conversion = ticker_converter(ticker_path)
ticker_conversion.head()

missing_rows = ticker_conversion[ticker_conversion.isna().any(axis=1)]
missing_rows
