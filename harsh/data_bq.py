# plan:
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

## big query
from google.cloud import bigquery

#chloe sent
DESIRED_CIKS = [
    1090872,    6201, 1158449,  320193, 1551152,    1800, 1467373,
        796343,    6281,    7084,    8670,  769397, 1002910,    4904,
        874761,    4977,    5272,  922864, 1267238,  354190, 1086222,
        915913, 1097149,  766421,  899051, 1579241,    6951,    2488,
       1037868, 1004434,  318154,  820027, 1053507, 1018724, 1596532,
       1013462,  315293,   91142, 1841666,    2969,  820313, 1521332,
       1035443,  731802,  915912, 1730168,    8818, 1410636,    4962,
        866787,   12927,   70858,   10456,  764478,   10795,   38777,
         14693, 1685040,  875045, 1390777, 1075531, 2012383,   14272,
       1383312, 1067983,  885725,  908255, 1037540,  831001,   23217,
        721371,   18230,  896159, 1374310, 1138118, 1051470,  815097,
        813672, 1306830, 1324404,  759944,  313927, 1043277, 1091667,
       1739940,   20286,   21665,   21076,   28412, 1166691, 1156375,
       1058090,   26172,  811156, 1071739, 1130310,  927628,  711404,
       1163165,  909832, 1024305,   16732, 1530721,  900075, 1108524,
        858877,  277948,  723254, 1058290,   64803,   93410,  715957,
         27904,  315189, 1393612,   29534, 1022079,  882184,  313616,
       1744489, 1297996,  935703,   29905,  940944,  936340, 1326160,
        927066, 1090012, 1688568,  712515, 1065088,   31462, 1047862,
         33185,  827052, 1001250,  915389,   32604,  821189, 1101239,
        906107,   72741,  920522, 1551182,   65984, 1711269, 1099800,
       1109357,  746515, 1324424, 1289490,   37996, 1539838,  815556,
        831259, 1048911, 1031296, 1048695, 1136893,   35527,  850209,
       1124198,   30625,   37785, 1754301,   34903, 1681459, 1262039,
       1659166,   40533,   40545,  882095,   40704,   24741, 1467858,
       1652044,   40987, 1123360, 1121788,  886982,  277135,   45012,
         46080,   49196, 1359841,  860730,  354950,    4447,  874766,
       1501585, 1585689,  793952,  859737,  773840,   46765, 1645590,
         47217,   12659,   48465, 1000228, 1070750,   47111,   49071,
         51143, 1571949,  874716,   51253, 1110803,  879169,   50863,
        896878,   51434,   51644, 1111928, 1478242, 1699150, 1020569,
       1035267,  749251,   49826,  914208,  728535,  833444,   96223,
        779152,  200406, 1043604,   19617,   72333,   55067,   91576,
       1601046, 1637459,  879101,  319201,   55785, 1506307, 1170010,
         21344,   56873,  885639,   60086, 1995807,   58492,  920760,
        920148, 1707925, 1065696,   59478,  936468,   59558,  352541,
         60667,  707549,   92380, 1679273, 1489393,  794367, 1141391,
        912595,  912242, 1048286,   62996,   63276,   63908,  827054,
        927653, 1059556, 1103982, 1613103, 1099219,  789570,  851968,
         63754,  916076,   62709,   66740,  865752,  764180, 1285785,
       1510295,  310158,  895421, 1408198,  789019,   68505,   36270,
       1037646,  723125, 1513761, 1120193,  753308, 1164727, 1065280,
       1111711,  320187,  906709, 1133421, 1021860, 1013871,  702165,
       1002047,   73124,   73309, 1045810,  814453, 1564708,  726728,
       1039684,   29989, 1341439,  898173,  797468,  723531,   75362,
        788784,   77476,   78003, 1126328,   80424,   80661,   76334,
        822416,   75677, 1045609, 1413329,  713676,   77360,  764622,
         79879,  922224, 1585364, 1137774, 1393311, 1534701,   78239,
       1050915, 1633917,  804328, 1604778,  884887,  910606,  872589,
       1281761,  315213,  720005, 1037038,  943819, 1024478,   84839,
        882835,  745732, 1060391, 1034054,  829224,  316709, 1012100,
         89800,   91419,   87347, 1040971,   91440,  883241,   92122,
       1063761,   64040, 1032208, 1881551,   93751, 1137789,   16918,
         93556,    4127, 1601712,  310764,   96021,  732717,   24545,
       1260221, 1385157,   96943,   27419,  109198,   97745, 1116132,
       1526520, 1113169,   86312,  916365,  100493,  946581,   97476,
        217346, 1336917,  100517,   74208,  352915, 1403568,  731766,
          5513,  100885, 1090727, 1067701,   36104, 1403161,  103379,
       1035002, 1396009,  899689, 1442145, 1014473,  875320,  740260,
        732712,  943452, 1000697, 1618921,  106040,  783325,  766704,
         72971,  106640,  823768,  107263,  104169, 1365135,  106535,
       1174922,   72903,   34088,  818479, 1770450, 1524472, 1041061,
       1136869,  109380, 1555280, 1748790, 1755672, 1666700, 1751788,
        202058, 1402057,  320335,  832101, 1336920, 1278021,  906163,
       1283699, 1701605,   52988, 1300514, 1335258, 1373715,  878927,
       1757898,   92230,   11544,  877212, 1590955, 1466258,   12208,
       1783180, 1286681, 1093557,    4281, 1781335,  101829, 1094285,
        860731,  105770, 1370637,   18926,   97210,  945841, 1318605,
       1786842, 1792044, 1590895, 1463101, 1474735, 1280452, 1413447,
        921738,  864749, 1100682, 1821825,  857005,  701985,   79282,
       1682852,  891103,  842023,  858470, 1352010, 1013237, 1419612,
       1868275, 1179929,   72331,  813828, 1140536,    9389,  906345,
       1156039, 1418135, 1326801, 1097864, 1705696, 1437107, 1057352,
       1687229,  947484,   33213, 1274494,  849399, 1004980, 1022671,
       1389170, 1996862,  814547, 1932393, 1145197, 1069183, 1095073,
        798354, 1327567,   31791, 1559720, 1393818, 1140859, 1944048,
       1316835, 1175454, 1725057,  910521,  765880,   48898,  898293,
       1397187, 1375365, 1543151, 1967680, 1535527, 1609711, 1996810,
       1404912, 1964738, 1692819, 2011286, 1571996,  922621, 1321655,
       2005951, 1858681, 1069202, 1811074, 1327811,  815094, 1101215,
       1578845,  899866,  773910, 1790982,  718877,  816284,  804753,
        877890, 1358071, 1001082,  783280, 1015780, 1519751,  354908,
       1132979,   39911,   48039,   40891, 1598014,   54480,   53669,
        101778,  743316, 1623613,   72207, 1492633, 1378946, 1038357,
       1087423, 1047122,  719739,   98246,  721689, 1418091,  203527,
       1339947, 1279363, 1732845, 1168054,  743988, 1424929, 1730175,
       1596783, 1288784
]


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
        sentiment_dict = get_sentiment_stats_from_text(text_mda, model, tokenizer)

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

    #DONE get only the companies we want
    
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


def upload_mda_df_to_bq(df):
    """
        add to mda table
        columns
        CIK
        filename
        Management_discussion
    """
    # required_columns = {"cik", "filename", "management_discussion"}
    
    #manually rename item2 col for sample csv
    # df = df.rename(columns={"item2": "management_discussion"})
    
    
#     if not required_columns.issubset(df.columns.str.lower()):
#         print(f"❌ DataFrame missing required columns. Expected: {required_columns}")
#         return

    try:
        BQ_PROJECT_ID = 'sentiment-lewagon'
        BQ_DATASET_ID = 'sentiment_db'
        BQ_TABLE_ID = 'MDA'
        table_ref = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"

        client = bigquery.Client()

        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            # schema=[
            #     bigquery.SchemaField("cik", "STRING", mode="NULLABLE"),
            #     bigquery.SchemaField("filename", "STRING", mode="REQUIRED"),
            #     bigquery.SchemaField("management_discussion", "STRING", mode="NULLABLE"),
            # ]
        )

        # Ensure correct column order and lowercase names
        # df = df.rename(columns=str.lower)[["cik", "filename", "management_discussion"]]

        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()

        print(f"✅ Uploaded {job.output_rows} rows to {table_ref}")

    except Exception as e:
        print(f"❌ Failed to upload DataFrame to BigQuery: {e}")
    

def extract_cik_from_filename(filename):
    match = re.search(r"_edgar_data_(\d+)_", filename)
    if match:
        return match.group(1)
    else:
        print(f":warning: No CIK found in: {filename}")
        return None 
    
def is_cik_desired(cik):
    return True if cik in DESIRED_CIKS else False

#Orchestrate
#TODO - get all desired bucket filepaths from bucket
def get_relevent_bucket_filepaths(clean=True):
    bucket_name = "sentiment_chloe-curtis"
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # list_blobs returns an iterator, which is efficient for large buckets.
    # We convert it to a list for easier use in this example.
    prefix = "2" #can be used to look through clean or raw
    
    if clean:
        prefix = "clean_data"
    bucket_blobs = bucket.list_blobs(prefix=prefix)

    #sample path
    #gs://sentiment_chloe-curtis/2015q1/20150102_10-K_edgar_data_1022097_0001477932-15-000006.txt
    
    desired_bucket_filepaths = []
    #create df of desired bucket file paths.
    #to contain: cik, 
    
    i=0
    for blob in bucket_blobs:
        filepath = blob.name
        # print(filepath)
        i+= 1
        # if i > 20000:
        #     break
        if i % 10000 == 0:
            print(i)
        
        cik_from_filepath = extract_cik_from_filename(filepath)
        
        if int(cik_from_filepath) in DESIRED_CIKS:
            # print("matched cik:", cik_from_filepath, "on filepath", filepath)
            filename = filepath.split('/')[-1]
            row_dict = {
                "cik" : cik_from_filepath,
                "bucket_filepath" : filepath,
                "filename" : filename
            }
            
            desired_bucket_filepaths.append(row_dict)
    
    bucket_filepaths_df = pd.DataFrame(desired_bucket_filepaths)
    
    print(bucket_filepaths_df. head())
    
    out_csv_path = "bucket_desired_filepaths.csv"
    if clean:
        out_csv_path = "bucket_desired_filepaths_clean.csv"
    
    bucket_filepaths_df.to_csv(out_csv_path, index=False, mode='w')
    
    return desired_bucket_filepaths


#function that takes a bucket filepath and extracts MDA, uploads to BQ
def extract_mda_and_send_to_bq(bucket_filepath):
    
    mda = get_mda_from_bucket_filepath(bucket_filepath)
    
    #upload mda
    
def upload_mda_from_clean_paths():
    #get clean paths
    clean_path_df = pd.read_csv("bucket_desired_filepaths_clean.csv")
    
    #chunks
    max_index = clean_path_df.index.max()
    print("max index", max_index)
    chunk = list(range(max_index))
    
    chunks = [chunk]
    for chunk in chunks:
        
        chunk_results = []
        
        for idx, row in clean_path_df.iloc[chunk].iterrows():
            bucket_filepath = row['bucket_filepath']
            mda = get_mda_from_bucket_filepath(bucket_filepath)

            out_row_dict = {}
            out_row_dict['cik'] = row['cik']
            out_row_dict['filename'] = row['filename']
            out_row_dict['bucket_filepath'] = row['bucket_filepath']           
            
            out_row_dict['management_discussion'] = mda

            chunk_results.append(out_row_dict)
        
        chunk_df = pd.DataFrame(chunk_results)
        #upload chunk to BQ
        upload_mda_df_to_bq(chunk_df)
        
    # upload_mda_df_to_bq(df)

def get_mda_from_bq():
    #Set up the BigQuery client
    client = bigquery.Client()
    query = "SELECT * FROM `sentiment-lewagon.sentiment_db.MDA`"
    df = client.query(query).to_dataframe()
    
    return df

def get_and_upload_sentiment_from_mda_df(mda_df):
       
    #load model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    sentiment_results_list = []
    
    #loop over df
    for idx, row in mda_df.iterrows():
        out_row_dict = {}
            
        out_row_dict['cik']  = row["cik"]
        out_row_dict['filename'] = row["filename"]
        out_row_dict['bucket_filepath'] = row['bucket_filepath']
        
        text_mda = row["management_discussion"]
        print(f"extracting sentiment from file: {out_row_dict['bucket_filepath']}")

        #extract sentiment
        sentiment_dict = get_sentiment_stats_from_text(text_mda, model, tokenizer)
        out_row_dict.update(sentiment_dict)
        
        print(out_row_dict)
        
        sentiment_results_list.append(out_row_dict)
        
    out_df = pd.DataFrame(sentiment_results_list)
    
    #debug
    # for col in out_df.columns:
    #     print(col, out_df[col].apply(type).value_counts())
    # print(out_df.head())
    
    upload_sentiment_df_to_bq(out_df)
        
    return out_df 


def get_and_upload_sentiment_from_mda_df_chunks(mda_df, chunk_size= 50):
       
    #load model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
    #chunk code
    total_rows = len(mda_df)
    
    for i in range(0, total_rows, chunk_size):
        start_row = i
        # The end_row for the current chunk.
        # Use min() to ensure it doesn't go beyond the total_rows.
        end_row = min(i + chunk_size, total_rows) - 1 # Inclusive end for range
        print(f"Processing rows from {start_row} to {end_row}")
        try:
            chunk_rows = []
            #loop over df
            for idx, row in mda_df.iloc[start_row : end_row + 1].iterrows():
                out_row_dict = {}

                out_row_dict['cik']  = row["cik"]
                out_row_dict['filename'] = row["filename"]
                out_row_dict['bucket_filepath'] = row['bucket_filepath']

                text_mda = row["management_discussion"]
                print(f"extracting sentiment from file: {out_row_dict['bucket_filepath']}")

                #extract sentiment
                sentiment_dict = get_sentiment_stats_from_text(text_mda, model, tokenizer)
                out_row_dict.update(sentiment_dict)

                chunk_rows.append(out_row_dict)

            chunk_out_df = pd.DataFrame(chunk_rows)

            upload_sentiment_df_to_bq(chunk_out_df)
            print(f"uploaded rows {start_row} to {end_row} to bq")
            
        except e:
            print(f"failed on chunk: {start_row} to {end_row}")
    
    #done

def upload_sentiment_df_to_bq(df):
    """
        add to mda table
        columns
        CIK
        filename
        Management_discussion
    """

    try:
        BQ_PROJECT_ID = 'sentiment-lewagon'
        BQ_DATASET_ID = 'sentiment_db'
        BQ_TABLE_ID = 'SENTIMENT'
        table_ref = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"

        client = bigquery.Client()

        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
        )

        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()

        print(f"✅ Uploaded {job.output_rows} rows to {table_ref}")

    except Exception as e:
        print(f"❌ Failed to upload DataFrame to BigQuery: {e}")

def upload_sentiment_from_bq_mda(testing=False):
    
    print("loading mda from biq query")
    #get mdas from big query
    mda_df = get_mda_from_bq()
    
    print("got mda df with shape:", mda_df.shape)
    
    if testing:
        mda_df = mda_df.head(15)
        get_and_upload_sentiment_from_mda_df_chunks(mda_df, chunk_size=5)
    else:
        #calculate sentiment
        get_and_upload_sentiment_from_mda_df_chunks(mda_df, chunk_size=50)

        
        
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

# create_mda_csvs_from_filepaths_df(meta_df, test_year, test_qtr)


#load sample mda df
# mda_df = pd.read_csv("mda_extract/2021_Q4_mda_1.csv")
# print(mda_df.dtypes)

#upload to bq
# upload_mda_csv_to_bq(mda_df)



sample_fname = "gs://sentiment_chloe-curtis/2015q1/20150102_10-K_edgar_data_1022097_0001477932-15-000006.txt"

print(extract_cik_from_filename(sample_fname)) #0001477932?


# sample_fname = "20150102_10-K_edgar_data_1022097_0001477932-15-000006.txt"
# print(sample_fname.split('/')[-1])

#get uploaded clean files and save paths to local csv bucket_desired_filepaths_clean
# get_relevent_bucket_filepaths()

# extract and upload mda
# upload_mda_from_clean_paths()

#download mda and calculate sentiment and upload
# upload_sentiment_from_bq_mda(testing=True)

upload_sentiment_from_bq_mda()


