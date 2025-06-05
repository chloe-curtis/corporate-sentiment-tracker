import os
import re
import pandas as pd

## FUNCTION will loop through quarterly folders to return cleaned data
## Table returns: cik, filing_type, filename, conformed_period_of_report

def cloud_clean_data(folder_path):
    # Helper function to convert date to quarter format e.g. q1-22
    def date_to_quarter(date):
        if pd.isnull(date):
            return None
        quarter = (date.month - 1) // 3 + 1
        year_suffix = str(date.year)[-2:]
        return f"Q{quarter}-{year_suffix}"

    rows = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and ("10-Q" in filename or "10-K" in filename):
            # Extract CIK and filing type from the filename
            cik_match = re.search(r'edgar_data_(\d+)_', filename)
            type_match = re.search(r'10-[QK]', filename)
            cik = cik_match.group(1) if cik_match else None
            filing_type = type_match.group(0) if type_match else None
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
                'conformed_period_of_report': period,
            })

    clean_data = pd.DataFrame(rows)
    
    # Convert report date string to datetime
    clean_data["conformed_period_of_report"] = pd.to_datetime(clean_data['conformed_period_of_report'], errors='coerce')
    
    # Derive quarter string in format q1-22
    clean_data["quarter"] = clean_data["conformed_period_of_report"].apply(date_to_quarter)
    
    return clean_data


# In[31]:


cloud_2022q1 = cloud_clean_data("2022q1")


# In[33]:


cloud_2022q1.head()


# In[ ]:




