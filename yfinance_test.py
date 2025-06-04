import yfinance as yf
import pandas as pd

#changings
tickers = "WMT"
start_year = 2014
end_year = 2024

ticker = yf.Ticker(tickers)

# Historical Prices
price_data = ticker.history(start=f"{start_year}-01-01", end=f"{end_year}-12-31")

# Earnings Dates
earnings_df = ticker.get_earnings_dates(limit=100)
earnings_dates = earnings_df.loc[
    earnings_df.index.to_series().dt.year.between(start_year, end_year)
].index.normalize()

# Shares Outstanding
shares_outstanding = ticker.info.get("sharesOutstanding")

# --- 1) Market Cap on Earnings Dates ---
market_caps = []
for date in earnings_dates:
    date = pd.to_datetime(date) #normalize()

    # Shift to previous available trading day if it's not in the price data
    while date not in price_data.index:
        date -= pd.Timedelta(days=1)

    close_price = price_data.loc[date]['Close']
    market_cap = close_price * shares_outstanding
    market_caps.append({
        'Date': date.strftime('%Y-%m-%d'),
        'Close Price': round(close_price, 2),
        'Estimated Market Cap': round(market_cap)
    })

market_cap_df = pd.DataFrame(market_caps)

# closing price df
closing_price_df = price_data[['Close']].copy()
closing_price_df.columns = [tickers]
closing_price_df.index = closing_price_df.index.normalize()
closing_price_df.index = closing_price_df.index.date

# --- 3) Latest Market Cap ---
latest_price = price_data['Close'].iloc[-1]
latest_market_cap = latest_price * shares_outstanding

info = ticker.info
sector = info.get("sector", "N/A")
industry = info.get("industry", "N/A")
long_name = info.get("longName", "N/A")
short_name = info.get("shortName", "N/A")

latest_market_cap_df = pd.DataFrame([{
    'Ticker': tickers,
    'Short Name': short_name,
    'Long Name': long_name,
    'Latest Price': round(latest_price, 2),
    'Latest Market Cap': round(latest_market_cap),
    'Sector': sector,
    'Industry': industry
}])

#Market Cap on Earnings Dates for singular ticker
print(market_cap_df)

#Daily Closing Prices - can add column for each ticker
print(closing_price_df.head(n=10))

#Last Market Caps - can add row for each ticker
latest_market_cap_df


import pandas as pd

# URL of the Wikipedia page containing S&P 500 companies
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Read all tables from the Wikipedia page
tables = pd.read_html(url)

# The first table contains the list of S&P 500 companies
sp500_table = tables[0]

# Select relevant columns: Symbol, Security, Date added, and CIK
sp500_df = sp500_table[['Symbol', 'Security', 'Date added', 'CIK']].copy()

# Clean the 'Symbol' column by replacing '.' with '-' to match yfinance format
sp500_df['Symbol'] = sp500_df['Symbol'].str.replace('.', '-', regex=False)

# Display the first few rows
sp500_df[sp500_df["Date added"] > "2015-12-31"].count()

import re
import pandas as pd
file_path = '/Users/chloecurtis/Downloads/dates_tickers.txt'
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


df

q424 = df[df["date"] == "2024-12-23"]
pd.merge(q424)
