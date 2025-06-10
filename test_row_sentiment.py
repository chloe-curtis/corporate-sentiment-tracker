from google.cloud import bigquery
import pandas as pd
import numpy as np

def test_row_sentiment(
    table_path: str = "sentiment-lewagon.sentiment_db.SENTIMENT",
    client: bigquery.Client | None = None,
) -> pd.DataFrame:
    """
    Fetch a sentiment table from BigQuery, de-duplicate it,
    compute per-row sentiment labels plus several summary columns,
    and return the enriched DataFrame.

    Parameters
    ----------
    table_path : str
        Fully-qualified BigQuery table name.
    client : google.cloud.bigquery.Client | None
        Optionally pass an existing BigQuery client; one is created if omitted.

    Returns
    -------
    pandas.DataFrame
        Enriched sentiment table.
    """
    # 1. BigQuery fetch
    client = client or bigquery.Client()
    df = client.list_rows(table_path).to_dataframe().drop_duplicates()

    # 2. Determine overall sentiment
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

    # 3. Add ratio and net-sentiment columns
    total = df[
        ["count_positive_chunks", "count_negative_chunks", "count_neutral_chunks"]
    ].sum(axis=1)
    total = total.replace(0, np.nan)  # avoid divide-by-zero

    df["count_pos_over_total_count"] = df["count_positive_chunks"] / total
    df["count_neg_over_total_count"] = df["count_negative_chunks"] / total
    df["count_neut_over_total_count"] = df["count_neutral_chunks"] / total
    df["net_sentiment"] = (df["count_positive_chunks"] - df["count_negative_chunks"]) / total
    df["total_chunks_analysed"] = total

    return df
