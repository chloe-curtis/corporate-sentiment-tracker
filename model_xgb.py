import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from google.cloud import bigquery
import matplotlib.pyplot as plt
import numpy as np

# Initialize BigQuery client
client = bigquery.Client()

# financial prices data
#df = pd.read_csv("data_bucket/finance_final.csv")
table_path = "sentiment-lewagon.sentiment_db.STOCK_PRICES"
df = client.list_rows(table_path).to_dataframe()
df = df.drop(['year', 'quarter'], axis=1)
df = df.rename(columns = {"q_tag": "quarter"})

# corresponding sector
#sectors = pd.read_csv("data_bucket/ticker_sector_industry.csv")
table_path = "sentiment-lewagon.sentiment_db.SECTOR"
sectors = client.list_rows(table_path).to_dataframe()

# merge
int_df = df.merge(sectors, how = "left", on = "ticker")

# converting quarter to q_num feature
int_df['q_num'] = int_df['quarter'].str.extract(r'Q([1-4])').astype(str)

# sentiment info data
table_path = "sentiment-lewagon.sentiment_db.SENTIMENT_TRAIN_ALL"
df = client.list_rows(table_path).to_dataframe()
df = df[df['count_positive_chunks']+ df['count_negative_chunks'] + df['count_neutral_chunks'] != 0]
df['neutral_dominance'] = df['count_neut_over_total_count'] > 0.6
sentiment = df[["ticker", "quarter_year", "neutral_dominance", "net_sentiment", "overall_sentiment"]]
sentiment = sentiment.rename(columns = {"quarter_year" : "quarter"})
sentiment = sentiment.drop_duplicates()

# merge sentiment and initial df
df = int_df.merge(sentiment, how = "left", on = ["quarter", "ticker"])
df = df.dropna(axis="index", subset=['net_sentiment'])

# set up model
feature_cols = ['net_sentiment','industry', 'q_num','neutral_dominance']
X = df[feature_cols]
y = df['Up_or_Down'].map({'Down': 0, 'Up': 1})

# test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# setting columns
num_selector = make_column_selector(dtype_include=['float64'])
cat_selector = make_column_selector(dtype_include=['object'])

# pipeline
num_pipeline = make_pipeline(
    MinMaxScaler()
)
cat_pipeline = make_pipeline(
    OneHotEncoder(sparse_output = False, drop='if_binary')
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_selector),
        ('cat', cat_pipeline, cat_selector)
    ]
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        learning_rate = 0.02,
    ))
])

# fitting on training
pipeline.fit(X_train, y_train)

#preprocessor.set_output(transform='pandas')
#preprocessor.fit_transform(X_train)

# y test predict
#y_test_pred = pipeline.predict(X_test)

# function to predict
"""def predict_new_data(pipeline, X_new_row):
    predicted_class = pipeline.predict(X_new_row)[0]
    predicted_proba = pipeline.predict_proba(X_new_row)[0][1]
    return predicted_class, predicted_proba

pred_class, pred_prob = predict_new_data(pipeline, X_new)
print(f"Predicted class: {pred_class}")
print(f"Probability of class 'Up': {pred_prob:.4f}")

# takes df like this
X_new = pd.DataFrame([{
    'net_sentiment': -0.1,
    'industry': 'Auto Manufacturers',
    'q_num': "4",
    'neutral_dominance': False
}])

X_new = X_new.astype({
    'q_num': 'object',
    'neutral_dominance': 'object'
})"""
