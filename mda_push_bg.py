
def upload_mda_csv_to_bq(df):
    """
        add to mda table
        columns
        CIK
        filename
        Management_discussion
    """
    required_columns = {"cik", "filename", "management_discussion"}
    #manually rename item2 col for sample csv
    df = df.rename(columns={"item2": "management_discussion"})
    
    
    if not required_columns.issubset(df.columns.str.lower()):
        print(f"❌ DataFrame missing required columns. Expected: {required_columns}")
        return

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
    