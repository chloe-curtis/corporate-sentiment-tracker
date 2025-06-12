#finbert imports
import re
import torch
import numpy as np
#commenting out while installing
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import io
#if bucket later
# from google.cloud import storage
import pickle
import pandas as pd
from google.cloud import bigquery

# #load finbert model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

#not clean... has &#8220
test_mda = """
Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations. Unless the context otherwise requires, the use of the terms &#8220;Best Buy,&#8221; &#8220;we,&#8221; &#8220;us&#8221; and &#8220;our&#8221; refers to Best Buy Co., Inc. and its consolidated subsidiaries. Any references to our website addresses do not constitute incorporation by reference of the information contained on the websites.
Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations (&#8220;MD&#38;A&#8221;) is intended to provide a reader of our financial statements with a narrative from the perspective of our management on our financial condition, results of operations, liquidity and certain other factors that may affect our future results. Unless otherwise noted, transactions and other factors significantly impacting our financial condition, results of operations and liquidity are discussed in order of magnitude. Our MD&#38;A should be read in conjunction with our Annual Report on Form 10-K for the fiscal year ended February 1, 2025 (including the information presented therein under Risk Factors ), as well as our other reports on Forms 10-Q and 8-K and other publicly available information. All amounts herein are unaudited. Overview. We are driven by our purpose to enrich lives through technology and our vision to personalize and humanize technology solutions for every stage of life . We accomplish this by leveraging our combination of tech expertise and a human touch to meet our customers&#8217; everyday needs, whether they come to us online, visit our stores or invite us into their homes. We have two reportable segments: Domestic and International. The Domestic segment is comprised of our operations in all states, districts and territories of the U.S. and our Best Buy Health business. The International segment is comprised of all our operations in Canada. Our fiscal year ends on the Saturday nearest the end of January. Our business, like that of many retailers, is seasonal. A large proportion of our revenue and earnings is generated in the fiscal fourth quarter, which includes the majority of the holiday shopping season. Comparable Sales. Throughout this MD&#38;A, we refer to comparable sales. Comparable sales is a metric used by management to evaluate the performance of our existing stores, websites and call centers by measuring the change in net sales for a particular period over the comparable prior period of equivalent length. Comparable sales includes revenue from stores, websites and call centers operating for at least 14 full months. Revenue from online sales is included in comparable sales and represents sales initiated on a website or app, regardless of whether customers choose to pick up product in store, curbside, at an alternative pick-up location or take delivery direct to their homes. Revenue from acquisitions is included in comparable sales beginning with the first full quarter following the first anniversary of the date of the acquisition. Comparable sales also includes credit card revenue, gift card breakage, commercial sales and sales of merchandise to wholesalers and dealers, as applicable. Revenue from stores closed more than 14 days, including but not limited to relocated, remodeled, expanded and downsized stores, or stores impacted by natural disasters, is excluded from comparable sales until at least 14 full months after reopening. Comparable sales excludes the impact of certain periodic warranty-related profit-share revenue, the effect of fluctuations in foreign currency exchange rates (applicable to our International segment only) and the impact of the 53 rd week (applicable in 53-week fiscal years only). Comparable sales is based on our fiscal calendar and is not adjusted to align calendar weeks. All periods presented apply this methodology consistently. Consistent with our comparable sales policy, revenue from Best Buy Express locations rebranded as a result of our previously announced collaboration with Bell Canada is excluded from our comparable sales calculation until locations have been operating for at least 14 full months. We believe comparable sales is a meaningful supplemental metric for investors to evaluate revenue performance resulting from growth in existing stores, websites and call centers versus the portion resulting from opening new stores or closing existing stores. The method of calculating comparable sales varies across the retail industry. As a result, our method of calculating comparable sales may not be the same as other retailers&#8217; methods. Non-GAAP Financial Measures This MD&#38;A includes financial information prepared in accordance with accounting principles generally accepted in the U.S. (&#8220;GAAP&#8221;), as well as certain non-GAAP financial measures, such as consolidated adjusted operating income, consolidated adjusted operating income rate, consolidated adjusted effective tax rate and consolidated adjusted diluted earnings per share (&#8220;EPS&#8221;). We believe that non-GAAP financial measures, when reviewed in conjunction with GAAP financial measures, provide additional useful information for evaluating current period performance and assessing future performance. For these reasons, internal management reporting, including budgets, forecasts and financial targets used for short-term incentives are based on non-GAAP financial measures. Generally, our non-GAAP financial measures include adjustments for items such as restructuring charges, goodwill and acquired intangible asset impairments, price-fixing settlements, gains and losses on sales of subsidiaries and certain investments, amortization of definite-lived intangible assets associated with acquisitions, certain acquisition-related costs and the tax effect of all such items. In addition, certain other items may be excluded from non-GAAP financial measures when we believe doing so provides greater clarity to management and our investors. We provide reconciliations of the most comparable financial measures presented in accordance with GAAP to presented non-GAAP financial measures that enable investors to understand the adjustments made in arriving at the non-GAAP financial measures and to evaluate performance using the same metrics as management. These non-GAAP financial measures should be considered in addition to, and not superior to or as a substitute for, GAAP financial measures. We strongly encourage investors and shareholders to review our financial statements and publicly filed reports in their entirety and not to rely on any single financial measure. Non-GAAP financial measures may be calculated differently from similarly titled measures used by other companies, thereby limiting their usefulness for comparative purposes. In our discussions of the operating results of our consolidated business and our International segment, we sometimes refer to the impact of changes in foreign currency exchange rates or the impact of foreign currency exchange rate fluctuations, which are references to the differences between the foreign currency exchange rates we use to convert the International segment&#8217;s operating results from local currencies into U.S. dollars for reporting purposes. We also may use the term &#8220;constant currency,&#8221; which represents results adjusted to exclude foreign currency impacts. We calculate those impacts as the difference between the current period results translated using the current period currency exchange rates and using the comparable prior period currency exchange rates. We believe the disclosure of revenue changes in constant currency provides useful supplementary information to investors in light of significant fluctuations in currency rates. Refer to the Non-GAAP Financial Measures section below for detailed reconciliations of items impacting consolidated adjusted operating income, consolidated adjusted effective tax rate and consolidated adjusted diluted EPS in the presented periods. Tariffs.We continue to face significant uncertainty regarding the scope, timing and magnitude of tariffs that may affect the products we sell and the consequent financial impact on our business. While we directly import approximately 2% to 3% of our overall assortment, our complex supply chain is heavily reliant on vendor imports from China, which we currently estimate make up approximately 30% to 35% of the products we purchase, compared to 55% disclosed within our Annual Report on Form 10-K for the fiscal year ended February 1, 2025. This is the result of vendors using production capabilities in multiple countries and leveraging their ability to flex sourcing options as the environment evolves. We currently estimate approximately 25% of the products we purchase are from the U.S. and Mexico. In conjunction with our vendors, we continue to seek to mitigate the impact of tariffs on our business and our customers.
"""
text_mda = test_mda
q_num = '2'
ticker = "BBY"
local_model_path = "model/pipeline.pkl"


def get_sentiment_stats_from_text(text_mda):
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
        print("text too short")
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
        print("no valid paragraphs")
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

    # 8) net sentiment
    net_sentiment = ((count_pos - count_neg) / num_chunks)
    print(net_sentiment)
    # 9) neutral dominance
    neutral_ratio = count_neu / num_chunks
    neutral_dominance = neutral_ratio > 0.6
    print(neutral_dominance)
    # 10) entropy
    #entropy
    entropy = 0.0
    # Create a list of the proportions
    proportions = [count_pos / num_chunks, count_neg / num_chunks, count_neu / num_chunks]
    # Loop through and apply the entropy formula to non-zero proportions
    for p in proportions:
        if p > 0:
            entropy -= p * np.log2(p)

    return {
        # total chunks
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
        "net_sentiment":         net_sentiment,
        "neutral_dominance":     neutral_dominance,
        "sentiment_entropy":     entropy
    }

def load_model_from_local(model_path):
    model_pipe = pickle.load(open(model_path,"rb"))
    return model_pipe

#untested?
def load_pickle_from_bucket(bucket_filepath, bucket_name="sentiment_chloe-curtis"):
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        bucket_filepath = "model/model_pipeline.pkl"
        # Access blob
        blob = bucket.blob(bucket_filepath)

        # Download bytes and load into memory
        pickle_bytes = blob.download_as_bytes()
        model = joblib.load(io.BytesIO(pickle_bytes))

        print(f"✅ Successfully loaded pickle file '{bucket_filepath}' from bucket '{bucket_name}'.")
        return model

    except Exception as e:
        print(f"❌ Failed to load pickle from bucket: {e}")
        return None

def get_X_raw(test_mda):
    stats = get_sentiment_stats_from_text(test_mda)
    industry = get_industry_for_ticker(ticker)
    X_raw = {
        "neutral_dominance": stats.get("neutral_dominance", None),
        "net_sentiment": stats.get("net_sentiment", None),
        "industry": industry,
        "q_num": q_num
    }
    X_new = pd.DataFrame([X_raw])  # wrap dict in a list to make 1-row DataFrame
    X_new = X_new.astype({
        'q_num': 'object',
        'neutral_dominance': 'object'
    })
    return X_new

def get_industry_for_ticker(ticker: str):
    client = bigquery.Client()
    ticker_safe = ticker.replace("'", "''")
    query = f"""
        SELECT ticker, industry
        FROM `sentiment-lewagon.sentiment_db.SECTOR`
        WHERE ticker = '{ticker_safe}'
    """
    df = client.query(query).to_dataframe()
    industry = df.iloc[0]["industry"]
    print(industry)
    return industry


#chloe model
def make_prediction(X_new):
    # "net_sentiment":         net_sentiment,
    # "neutral_dominance":     neutral_dominance,
    # industry
    # quarter
    pipe_model = load_model_from_local(local_model_path)
    prediction = pipe_model.predict(X_new)
    print("prediction", prediction)
    return prediction

def get_prediction_from_mda(mda):
    stats = get_sentiment_stats_from_text(test_mda)
    industry = get_industry_for_ticker(ticker)

    X_raw = {
        "neutral_dominance": stats.get("neutral_dominance", None),
        "net_sentiment": stats.get("net_sentiment", None),
        "industry": industry,
        "q_num": q_num
    }
    X_new = pd.DataFrame([X_raw])  # wrap dict in a list to make 1-row DataFrame
    X_new = X_new.astype({
        'q_num': 'object',
        'neutral_dominance': 'object'
    })

    print("X processed", X_new)

    pipe_model = load_model_from_local(local_model_path)
    prediction = pipe_model.predict(X_new)
    print("prediction", prediction)

    return prediction



TEST_MDA = """
 Item 7. Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations

The following discussion should be read in conjunction with the consolidated financial statements and accompanying notes included in Part II, Item 8 of this Form 10-K. This Item generally discusses 2023 and 2022 items and year-to-year comparisons between 2023 and 2022. Discussions of 2021 items and year-to-year comparisons between 2022 and 2021 are not included, and can be found in &#8220;Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations&#8221; in Part II, Item 7 of the Company&#8217;s Annual Report on Form 10-K for the fiscal year ended September 24, 2022.

Fiscal Period

The Company&#8217;s fiscal year is the 52- or 53-week period that ends on the last Saturday of September. An additional week is included in the first fiscal quarter every five or six years to realign the Company&#8217;s fiscal quarters with calendar quarters, which occurred in the first quarter of 2023. The Company&#8217;s fiscal year 2023 spanned 53 weeks, whereas fiscal years 2022 and 2021 spanned 52 weeks each.

Fiscal Year Highlights

The Company&#8217;s total net sales were $383.3 billion and net income was $97.0 billion during 2023.

The Company&#8217;s total net sales decreased 3% or $11.0 billion during 2023 compared to 2022. The weakness in foreign currencies relative to the U.S. dollar accounted for more than the entire year-over-year decrease in total net sales, which consisted primarily of lower net sales of Mac and iPhone, partially offset by higher net sales of Services.

The Company announces new product, service and software offerings at various times during the year. Significant announcements during fiscal year 2023 included the following:

First Quarter 2023:

&#8226; iPad and iPad Pro;

&#8226; Next-generation Apple TV 4K; and

&#8226; MLS Season Pass, a Major League Soccer subscription streaming service.

Second Quarter 2023:

&#8226; MacBook Pro 14&#8221;, MacBook Pro 16&#8221; and Mac mini; and

&#8226; Second-generation HomePod.

Third Quarter 2023:

&#8226; MacBook Air 15&#8221;, Mac Studio and Mac Pro;

&#8226; Apple Vision Pro&#8482;, the Company&#8217;s first spatial computer featuring its new visionOS&#8482;, expected to be available in early calendar year 2024; and

&#8226; iOS 17, macOS Sonoma, iPadOS 17, tvOS 17 and watchOS 10, updates to the Company&#8217;s operating systems.

Fourth Quarter 2023:

&#8226; iPhone 15, iPhone 15 Plus, iPhone 15 Pro and iPhone 15 Pro Max; and

&#8226; Apple Watch Series 9 and Apple Watch Ultra 2.

In May 2023, the Company announced a new share repurchase program of up to $90 billion and raised its quarterly dividend from $0.23 to $0.24 per share beginning in May 2023. During 2023, the Company repurchased $76.6 billion of its common stock and paid dividends and dividend equivalents of $15.0 billion.

Macroeconomic Conditions

Macroeconomic conditions, including inflation, changes in interest rates, and currency fluctuations, have directly and indirectly impacted, and could in the future materially impact, the Company&#8217;s results of operations and financial condition.

Apple Inc. | 2023 Form 10-K | 20

Segment Operating Performance

The following table shows net sales by reportable segment for 2023, 2022 and 2021 (dollars in millions):

##TABLE_START 2023 Change 2022 Change 2021 Net sales by reportable segment: Americas $ 162,560 (4) % $ 169,658 11 % $ 153,306 Europe 94,294 (1) % 95,118 7 % 89,307 Greater China 72,559 (2) % 74,200 9 % 68,366 Japan 24,257 (7) % 25,977 (9) % 28,482 Rest of Asia Pacific 29,615 1 % 29,375 11 % 26,356 Total net sales $ 383,285 (3) % $ 394,328 8 % $ 365,817 ##TABLE_END

Americas

Americas net sales decreased 4% or $7.1 billion during 2023 compared to 2022 due to lower net sales of iPhone and Mac, partially offset by higher net sales of Services.

Europe

Europe net sales decreased 1% or $824 million during 2023 compared to 2022. The weakness in foreign currencies relative to the U.S. dollar accounted for more than the entire year-over-year decrease in Europe net sales, which consisted primarily of lower net sales of Mac and Wearables, Home and Accessories, partially offset by higher net sales of iPhone and Services.

Greater China

Greater China net sales decreased 2% or $1.6 billion during 2023 compared to 2022. The weakness in the renminbi relative to the U.S. dollar accounted for more than the entire year-over-year decrease in Greater China net sales, which consisted primarily of lower net sales of Mac and iPhone.

Japan

Japan net sales decreased 7% or $1.7 billion during 2023 compared to 2022. The weakness in the yen relative to the U.S. dollar accounted for more than the entire year-over-year decrease in Japan net sales, which consisted primarily of lower net sales of iPhone, Wearables, Home and Accessories and Mac.

Rest of Asia Pacific

Rest of Asia Pacific net sales increased 1% or $240 million during 2023 compared to 2022. The weakness in foreign currencies relative to the U.S. dollar had a significantly unfavorable year-over-year impact on Rest of Asia Pacific net sales. The net sales increase consisted of higher net sales of iPhone and Services, partially offset by lower net sales of Mac and iPad.

Apple Inc. | 2023 Form 10-K | 21

Products and Services Performance

The following table shows net sales by category for 2023, 2022 and 2021 (dollars in millions):

##TABLE_START 2023 Change 2022 Change 2021 Net sales by category: iPhone (1)

$ 200,583 (2) % $ 205,489 7 % $ 191,973 Mac (1)

29,357 (27) % 40,177 14 % 35,190 iPad (1)

28,300 (3) % 29,292 (8) % 31,862 Wearables, Home and Accessories (1)

39,845 (3) % 41,241 7 % 38,367 Services (2)

85,200 9 % 78,129 14 % 68,425 Total net sales $ 383,285 (3) % $ 394,328 8 % $ 365,817 ##TABLE_END

(1) Products net sales include amortization of the deferred value of unspecified software upgrade rights, which are bundled in the sales price of the respective product.

(2) Services net sales include amortization of the deferred value of services bundled in the sales price of certain products.

iPhone

iPhone net sales decreased 2% or $4.9 billion during 2023 compared to 2022 due to lower net sales of non-Pro iPhone models, partially offset by higher net sales of Pro iPhone models.

Mac

Mac net sales decreased 27% or $10.8 billion during 2023 compared to 2022 due primarily to lower net sales of laptops.

iPad

iPad net sales decreased 3% or $1.0 billion during 2023 compared to 2022 due primarily to lower net sales of iPad mini and iPad Air, partially offset by the combined net sales of iPad 9th and 10th generation.

Wearables, Home and Accessories

Wearables, Home and Accessories net sales decreased 3% or $1.4 billion during 2023 compared to 2022 due primarily to lower net sales of Wearables and Accessories.

Services

Services net sales increased 9% or $7.1 billion during 2023 compared to 2022 due to higher net sales across all lines of business.

Apple Inc. | 2023 Form 10-K | 22

Gross Margin

Products and Services gross margin and gross margin percentage for 2023, 2022 and 2021 were as follows (dollars in millions):

##TABLE_START 2023 2022 2021 Gross margin: Products $ 108,803 $ 114,728 $ 105,126 Services 60,345 56,054 47,710 Total gross margin $ 169,148 $ 170,782 $ 152,836 ##TABLE_END

##TABLE_START Gross margin percentage: Products 36.5 % 36.3 % 35.3 % Services 70.8 % 71.7 % 69.7 % Total gross margin percentage 44.1 % 43.3 % 41.8 % ##TABLE_END

Products Gross Margin

Products gross margin decreased during 2023 compared to 2022 due to the weakness in foreign currencies relative to the U.S. dollar and lower Products volume, partially offset by cost savings and a different Products mix.

Products gross margin percentage increased during 2023 compared to 2022 due to cost savings and a different Products mix, partially offset by the weakness in foreign currencies relative to the U.S. dollar and decreased leverage.

Services Gross Margin

Services gross margin increased during 2023 compared to 2022 due primarily to higher Services net sales, partially offset by the weakness in foreign currencies relative to the U.S. dollar and higher Services costs.

Services gross margin percentage decreased during 2023 compared to 2022 due to higher Services costs and the weakness in foreign currencies relative to the U.S. dollar, partially offset by a different Services mix.

The Company&#8217;s future gross margins can be impacted by a variety of factors, as discussed in Part I, Item 1A of this Form 10-K under the heading &#8220;Risk Factors.&#8221; As a result, the Company believes, in general, gross margins will be subject to volatility and downward pressure.

Operating Expenses

Operating expenses for 2023, 2022 and 2021 were as follows (dollars in millions):

##TABLE_START 2023 Change 2022 Change 2021 Research and development $ 29,915 14 % $ 26,251 20 % $ 21,914 Percentage of total net sales 8 % 7 % 6 % Selling, general and administrative $ 24,932 (1) % $ 25,094 14 % $ 21,973 Percentage of total net sales 7 % 6 % 6 % Total operating expenses $ 54,847 7 % $ 51,345 17 % $ 43,887 Percentage of total net sales 14 % 13 % 12 % ##TABLE_END

Research and Development

The year-over-year growth in R&#38;D expense in 2023 was driven primarily by increases in headcount-related expenses.

Selling, General and Administrative

Selling, general and administrative expense was relatively flat in 2023 compared to 2022.

Apple Inc. | 2023 Form 10-K | 23

Provision for Income Taxes

Provision for income taxes, effective tax rate and statutory federal income tax rate for 2023, 2022 and 2021 were as follows (dollars in millions):

##TABLE_START 2023 2022 2021 Provision for income taxes $ 16,741 $ 19,300 $ 14,527 Effective tax rate 14.7 % 16.2 % 13.3 % Statutory federal income tax rate 21 % 21 % 21 % ##TABLE_END

The Company&#8217;s effective tax rate for 2023 and 2022 was lower than the statutory federal income tax rate due primarily to a lower effective tax rate on foreign earnings, the impact of the U.S. federal R&#38;D credit, and tax benefits from share-based compensation, partially offset by state income taxes.

The Company&#8217;s effective tax rate for 2023 was lower compared to 2022 due primarily to a lower effective tax rate on foreign earnings and the impact of U.S. foreign tax credit regulations issued by the U.S. Department of the Treasury in 2022, partially offset by lower tax benefits from share-based compensation.

Liquidity and Capital Resources

The Company believes its balances of cash, cash equivalents and unrestricted marketable securities, which totaled $148.3 billion as of September 30, 2023, along with cash generated by ongoing operations and continued access to debt markets, will be sufficient to satisfy its cash requirements and capital return program over the next 12 months and beyond.

The Company&#8217;s material cash requirements include the following contractual obligations:

Debt

As of September 30, 2023, the Company had outstanding fixed-rate notes with varying maturities for an aggregate principal amount of $106.6 billion (collectively the &#8220;Notes&#8221;), with $9.9 billion payable within 12 months. Future interest payments associated with the Notes total $41.1 billion, with $2.9 billion payable within 12 months.

The Company also issues unsecured short-term promissory notes pursuant to a commercial paper program. As of September 30, 2023, the Company had $6.0 billion of commercial paper outstanding, all of which was payable within 12 months.

"""

if __name__ == "__main__":
    X_new = pd.DataFrame([{
            'net_sentiment': -0.1,
            'industry': 'Auto Manufacturers',
            'q_num': "4",
            'neutral_dominance': False
        }])
    X_new = X_new.astype({
        'q_num': 'object',
        'neutral_dominance': 'object'
    })

    print("test prediction with manual x_new:", make_prediction(X_new))

    print("\n=====\n")
    X_new2 = get_X_raw(test_mda)
    print("test prediction with function made x_new, from test_mda", make_prediction(X_new2))

    #print(get_sentiment_stats_from_text(test_mda))
