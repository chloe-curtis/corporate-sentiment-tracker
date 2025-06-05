base_index = pd.merge(historical_index, ticker_conversion, on = "ticker", how = "left")
### JOIN CAUSED NAS - AS TICKER CONVERSION DIDNT HAVE COMPANY NAME AND CIK
### REMOVE THE NAS FROM THE JOIN
x = base_index.dropna()
## CREATE DF WITH THE NAS
y = base_index[base_index.isnull().any(axis=1)]
## DROP THE COLUMNS WE DONT NEED FROM NA DF
g = y[['quarter','ticker']]
## CSV HAS THE DATA WE NEED
missing_company = pd.read_csv('missing_companies.csv')
### MERGE BACK TO HAVE THE CIX AND COMPANY NAME 
z = pd.merge(g, missing_company, on = "ticker", how = "left")
### PAD THE CIK NUMBERS WITH ZEROS - 10 CHARCTERS
z['cik'] = z['cik'].astype(str).str.zfill(10)
## CONCAT BOTH DFS
end_step_2 = pd.concat([x, z], axis=0, ignore_index=True)