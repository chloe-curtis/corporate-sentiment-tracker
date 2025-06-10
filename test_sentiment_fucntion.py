from google.cloud import bigquery
import pandas as pd
import numpy as np

def test_sentiment(
    sentiment_table: str = "sentiment-lewagon.sentiment_db.SENTIMENT",
    meta_table: str      = "sentiment-lewagon.sentiment_db.META",
    client: bigquery.Client | None = None,
) -> pd.DataFrame:
    """
    Fetch sentiment & meta tables from BigQuery, enrich them, and return a merged DataFrame.

    Returns
    -------
    pandas.DataFrame
        META ⟕ SENTIMENT, enriched with per-row sentiment and helper columns.
    """
    client = client or bigquery.Client()

    # ────────────────────────────────────────────────────────────
    # 1. SENTIMENT table → df
    # ────────────────────────────────────────────────────────────
    df = (
        client.list_rows(sentiment_table)
        .to_dataframe()
        .drop_duplicates()
    )

    # Ensure consistent naming for merge later
    df.rename(columns={"bucket_filepath": "bucket_file_path"}, inplace=True)

    # Determine overall sentiment
    def _overall(row: pd.Series) -> str:
        max_val = max(
            row["count_positive_chunks"],
            row["count_negative_chunks"],
            row["count_neutral_chunks"],
        )
        if row["count_positive_chunks"] == max_val:
            return "positive"
        elif row["count_negative_chunks"] == max_val:
            return "negative"
        return "neutral"

    df["overall_sentiment"] = df.apply(_overall, axis=1)

    # Ratio and net-sentiment columns
    total = df[["count_positive_chunks", "count_negative_chunks", "count_neutral_chunks"]].sum(axis=1)
    total = total.replace(0, np.nan)  # avoid divide-by-zero

    df["count_pos_over_total_count"]  = df["count_positive_chunks"]  / total
    df["count_neg_over_total_count"]  = df["count_negative_chunks"]  / total
    df["count_neut_over_total_count"] = df["count_neutral_chunks"]   / total
    df["net_sentiment"]               = (df["count_positive_chunks"] - df["count_negative_chunks"]) / total
    df["total_chunks_analysed"]       = total

    # ────────────────────────────────────────────────────────────
    # 2. META table → d
    # ────────────────────────────────────────────────────────────
    d = (
        client.list_rows(meta_table)
        .to_dataframe()
        [["bucket_file_path", "conformed_period_of_report", "quarter", "year", "ticker"]]
    )
    d["quarter_year"] = d["quarter"] + "-" + d["year"].astype(str).str[-2:]

    # ────────────────────────────────────────────────────────────
    # 3. Merge META ← SENTIMENT
    # ────────────────────────────────────────────────────────────
    sentiment_clean_v2 = (
        pd.merge(d, df, on="bucket_file_path", how="left")
        .drop_duplicates()
    )

    return sentiment_clean_v2
