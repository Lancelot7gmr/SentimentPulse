"""
In this we read raw Reddit JSON from MongoDB and then
transform it into clean text, score sentiment with VADER, aggregate by day/ticker
Afterwards load to write processed tables back to PostgreSQL
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import pymongo
from sqlalchemy import create_engine, text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONNECTIONS
# ─────────────────────────────────────────────

def get_mongo_col():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGO_DB", "sentiment_stock_db")
    client = pymongo.MongoClient(uri)
    return client[db_name]["reddit_posts"], client

def get_pg_engine():
    h  = os.getenv("PG_HOST", "localhost")
    p  = os.getenv("PG_PORT", "5432")
    u  = os.getenv("PG_USER", "postgres")
    pw = os.getenv("PG_PASSWORD", "password")
    db = os.getenv("PG_DB", "sentiment_stock")
    return create_engine(f"postgresql+psycopg2://{u}:{pw}@{h}:{p}/{db}")

# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
_URL_RE    = re.compile(r"http\S+|www\.\S+")
_TICKER_RE = re.compile(r"\$[A-Z]{1,5}")       
_SPECIAL_RE = re.compile(r"[^a-zA-Z0-9\s!?.,']")
_SPACE_RE   = re.compile(r"\s+")
_EMOJI_RE   = re.compile(
    "[\U0001F600-\U0001F64F"   
    "\U0001F300-\U0001F5FF"    
    "\U0001F680-\U0001F6FF"    
    "\U0001F1E0-\U0001F1FF"    
    "\U00002700-\U000027BF"    
    "]+",
    flags=re.UNICODE
)

# Financial slang mapping
FINANCIAL_SLANG = {
    "moon":    "excellent profit",
    "mooning": "rapidly increasing",
    "ape":     "passionate investor",
    "apes":    "passionate investors",
    "diamond hands": "holding despite losses",
    "paper hands":   "selling too early",
    "yolo":    "high-risk investment",
    "tendies": "profits",
    "rekt":    "suffering large losses",
    "fud":     "fear uncertainty doubt",
    "dd":      "due diligence research",
    "stonks":  "stocks",
    "hodl":    "hold",
    "bears":   "pessimists expecting decline",
    "bulls":   "optimists expecting rise",
    "squeeze": "rapid price increase",
    "dip":     "price decrease opportunity",
    "buy the dip": "purchase during price decrease",
    "short":   "bet against",
    "puts":    "bearish options",
    "calls":   "bullish options",
}

def clean_text(text: str) -> str:
    """
    Cleans a Reddit post title/body for sentiment analysis.
    The major steps involved are namely; make everything in lowercase,
    translate financial slang to plain English, remove URLs which are not relevant
    to sentiment, remove emojis  as VADER doesn't handle them well,
    keep punctuation that carries sentiment such as !?,. etc and finally collapse whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    for slang, meaning in FINANCIAL_SLANG.items():
        text = text.replace(slang, meaning)
    text = _URL_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _SPECIAL_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


# ─────────────────────────────────────────────
# EXTRACT: Read from MongoDB
# ─────────────────────────────────────────────

def extract_reddit_from_mongo(ticker: str = None) -> pd.DataFrame:
    """
    Reads raw Reddit post documents from MongoDB.
    MongoDB stores the raw, messy, semi-structured JSON as it is.
    Once we have cleaned and scored the text, we produce a structured, uniform table.
    The structured output is then Loaded into PostgreSQL for joining with stock prices.
    This Extract-Transform-Load pattern is fundamental to data engineering.
    """
    col, client = get_mongo_col()
    query = {"_fetched_ticker": ticker} if ticker else {}
    projection = {
        "_id": 0,
        "id": 1,
        "title": 1,
        "selftext": 1,
        "score": 1,                  # Reddit upvotes
        "num_comments": 1,
        "upvote_ratio": 1,
        "created_utc": 1,
        "_fetched_ticker": 1,
        "_fetched_subreddit": 1,
        "author": 1,
        "url": 1,
    }
    docs = list(col.find(query, projection))
    client.close()

    if not docs:
        log.warning("No documents found in MongoDB for ticker=%s", ticker)
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    # Convert Unix timestamp to datetime
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df["posted_at"]   = pd.to_datetime(df["created_utc"], unit="s", utc=True, errors="coerce")
    df["date"]        = df["posted_at"].dt.date
    df.rename(columns={"_fetched_ticker": "ticker",
                        "_fetched_subreddit": "subreddit"}, inplace=True)
    log.info("Extracted %d Reddit posts from MongoDB (ticker=%s)", len(df), ticker or "all")
    return df


# ─────────────────────────────────────────────
# TRANSFORM: Sentiment scoring with VADER
# ─────────────────────────────────────────────

vader = SentimentIntensityAnalyzer()

def score_sentiment_vader(text: str) -> dict:
    """
    Scores text using VADER (Valence Aware Dictionary and Sentiment Reasoner).
    VADER is specifically designed for social media text. It understands:
    Capitalisation, punctuation emphasis and negation
    It runs in microseconds per document which is a perfect fit for large
    Reddit datasets.
    
    It then returns overall score from -1 to +1 and find the proportion of
    text in each valence which is then labelled as Positive / Negative / Neutral
    """
    scores = vader.polarity_scores(text)
    compound = scores["compound"]
    label = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
    return {
        "vader_compound":  compound,
        "vader_pos":       scores["pos"],
        "vader_neg":       scores["neg"],
        "vader_neu":       scores["neu"],
        "sentiment_label": label,
    }


def compute_engagement_weight(row: pd.Series) -> float:
    """
    Computes an engagement-weighted importance score for each Reddit post.
    
    NOVELTY: Rather than treating all posts equally, we weight each post's
    sentiment by how much the community engaged with it. A post with 10,000
    upvotes should count more than a post with 2 upvotes.
    
    Weight formula used for this is as follows:
        log(1 + upvotes) * upvote_ratio * log(1 + comments)
        
    Logarithm is used to dampen extreme outliers in this case viral posts
    without ignoring them.
    """
    upvotes  = max(0, row.get("score", 0) or 0)
    comments = max(0, row.get("num_comments", 0) or 0)
    ratio    = max(0.1, row.get("upvote_ratio", 0.5) or 0.5)
    return np.log1p(upvotes) * ratio * np.log1p(comments + 1)


def transform_reddit_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full transformation pipeline on Reddit posts is done by c
    ombining  title with body into one text field and then cleaning the text.
    Sentiment   is nthen scored with VADER. the engagement weight is then computed
    and one row is returned per post with all the features.
    """
    if df.empty:
        return df

    log.info("Transforming %d Reddit posts...", len(df))

    # Combine title and selftext
    df["text_raw"] = (
        df["title"].fillna("") + " " + df["selftext"].fillna("")
    ).str.strip()

    # Clean
    df["text_clean"] = df["text_raw"].apply(clean_text)

    # Score sentiment
    sentiment_scores = df["text_clean"].apply(score_sentiment_vader)
    df = pd.concat([df, pd.DataFrame(list(sentiment_scores))], axis=1)

    # Engagement weight
    df["engagement_weight"] = df.apply(compute_engagement_weight, axis=1)

    # Weighted sentiment: the key feature for our analysis
    df["weighted_sentiment"] = df["vader_compound"] * df["engagement_weight"]

    # Drop rows with no date or no meaningful text
    df = df.dropna(subset=["date", "vader_compound"])
    df = df[df["text_clean"].str.len() > 10]

    log.info("Transformation complete: %d usable posts", len(df))
    return df


# ─────────────────────────────────────────────
# AGGREGATE: Daily sentiment per ticker
# ─────────────────────────────────────────────

def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    this function is used to aggregate individual post-level sentiment into a
    single daily sentiment score per ticker per day.
    
    NOVELTY: We produce TWO sentiment signals:
      1. raw_avg_sentiment   : simple mean of all compound scores for the day
      2. weighted_avg_sentiment: weighted mean where viral posts count more using
                                 engagement_weight as the weight
    
    We also compute How many posts were made that day?, Standard deviation of
    sentiment either a disagreement or controversy and a fraction of posts that
    were positive. These additional features enable richer analysis
    than a simple mean.
    """
    df["date"] = pd.to_datetime(df["date"])

    agg = df.groupby(["ticker", "date"]).apply(lambda g: pd.Series({
        "raw_avg_sentiment":      g["vader_compound"].mean(),
        "weighted_avg_sentiment": np.average(g["vader_compound"],
                                             weights=g["engagement_weight"].clip(lower=0.01)),
        "sentiment_std":          g["vader_compound"].std(),
        "total_engagement":       g["engagement_weight"].sum(),
        "post_count":             len(g),
        "bullish_ratio":          (g["sentiment_label"] == "Positive").mean(),
        "bearish_ratio":          (g["sentiment_label"] == "Negative").mean(),
        "avg_score":              g["score"].mean(),
        "avg_comments":           g["num_comments"].mean(),
    }), include_groups=False).reset_index()

    # Lagged features: yesterday's sentiment (for lead-lag analysis)
    agg = agg.sort_values(["ticker", "date"])
    for lag in [1, 2, 3]:
        agg[f"sentiment_lag_{lag}d"] = (
            agg.groupby("ticker")["weighted_avg_sentiment"].shift(lag)
        )

    # Rolling 7-day average sentiment (smoothed signal)
    agg["sentiment_ma7"] = (
        agg.groupby("ticker")["weighted_avg_sentiment"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    log.info("Aggregated to %d daily sentiment rows", len(agg))
    return agg


# ─────────────────────────────────────────────
# LOAD: Write processed data to PostgreSQL
# ─────────────────────────────────────────────

def load_to_postgres(df: pd.DataFrame, table: str, engine):
    df.to_sql(table, engine, if_exists="replace",
              index=False, method="multi", chunksize=500)
    log.info("Loaded %d rows into PostgreSQL table '%s'", len(df), table)


# ─────────────────────────────────────────────
# MERGE: Combine sentiment + prices
# ─────────────────────────────────────────────

def build_master_dataset(engine) -> pd.DataFrame:
    """
    Creates the master analysis table by joining reddit_sentiment_daily,
    stock_prices and finally news_sentiment which are all joined on
    (ticker, date).
    
    This master table is the foundation for all visualisations and analysis.
    It is stored as 'master_dataset' in PostgreSQL.
    """
    query = """
        SELECT
            sp.ticker,
            sp.date,
            sp.open,
            sp.high,
            sp.low,
            sp.close,
            sp.volume,
            sp.daily_return,
            sp.price_direction,
            sp.rolling_vol_5d,
            rs.raw_avg_sentiment,
            rs.weighted_avg_sentiment,
            rs.sentiment_std,
            rs.post_count,
            rs.bullish_ratio,
            rs.bearish_ratio,
            rs.sentiment_lag_1d,
            rs.sentiment_lag_2d,
            rs.sentiment_lag_3d,
            rs.sentiment_ma7,
            rs.total_engagement,
            ns.ticker_sentiment_score  AS news_sentiment,
            ns.ticker_relevance        AS news_relevance,
            ns.overall_sentiment       AS news_overall
        FROM stock_prices sp
        LEFT JOIN reddit_sentiment_daily rs
            ON sp.ticker = rs.ticker AND sp.date = rs.date
        LEFT JOIN news_sentiment ns
            ON sp.ticker = ns.ticker AND sp.date = ns.date
        ORDER BY sp.ticker, sp.date
    """
    try:
        df = pd.read_sql(query, engine)
        log.info("Built master dataset: %d rows", len(df))
        df.to_sql("master_dataset", engine, if_exists="replace",
                  index=False, method="multi", chunksize=500)
        return df
    except Exception as e:
        log.error("Failed to build master dataset: %s", e)
        return pd.DataFrame()


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (when DBs unavailable)
# ─────────────────────────────────────────────

def generate_full_synthetic_dataset(tickers=None, start="2023-01-01",
                                    end="2024-12-31") -> pd.DataFrame:
    """
    This function generates a complete, realistic synthetic master dataset
    for demonstration.
    This allows the Streamlit dashboard to work without real database connections if not available.
    The synthetic data then simulates realistic stock price movements
    (geometric Brownian motion), correlated sentiment scores (sentiment
    partially drives prices with lag), news sentiment following a similar but
    independent pattern company-specific characteristics which an be seen as
    Tesla being more volatile while Apple being more stable etc.
    """
    if tickers is None:
        tickers = ["TSLA", "AAPL", "GME", "NVDA", "AMZN"]

    # Real-ish starting prices and volatility profiles
    profiles = {
        "TSLA": {"price": 200.0, "vol": 0.035, "drift": 0.0003},
        "AAPL": {"price": 180.0, "vol": 0.015, "drift": 0.0004},
        "GME":  {"price":  20.0, "vol": 0.060, "drift": -0.0001},
        "NVDA": {"price": 400.0, "vol": 0.030, "drift": 0.0008},
        "AMZN": {"price": 130.0, "vol": 0.020, "drift": 0.0003},
    }

    dates = pd.date_range(start, end, freq="B")
    n = len(dates)
    frames = []

    for ticker in tickers:
        p = profiles.get(ticker, {"price": 100.0, "vol": 0.02, "drift": 0.0002})
        rng = np.random.default_rng(abs(hash(ticker)) % (2**31))

        # Geometric Brownian Motion for stock price
        returns = p["drift"] + p["vol"] * rng.standard_normal(n)
        prices  = p["price"] * np.exp(np.cumsum(returns))

        # Sentiment: correlated with returns but noisier and slightly leading
        sentiment_noise  = rng.standard_normal(n)
        sentiment_signal = np.zeros(n)
        sentiment_signal[0] = 0
        for i in range(1, n):
            sentiment_signal[i] = (0.7 * sentiment_signal[i-1]
                                   + 0.3 * returns[i]
                                   + 0.15 * sentiment_noise[i])
        # Normalise to [-1, 1]
        s_min, s_max = sentiment_signal.min(), sentiment_signal.max()
        weighted_sentiment = 2 * (sentiment_signal - s_min) / (s_max - s_min + 1e-9) - 1

        post_count = rng.integers(5, 300, n)
        bullish_ratio = (weighted_sentiment + 1) / 2 * 0.6 + rng.uniform(0.1, 0.3, n)
        bullish_ratio = bullish_ratio.clip(0, 1)

        # News sentiment: loosely correlated with Reddit
        news_sent = (0.5 * weighted_sentiment
                     + 0.5 * rng.standard_normal(n) * 0.2)
        news_sent = np.clip(news_sent, -1, 1)

        # Lag features
        ws_series = pd.Series(weighted_sentiment)
        lag1 = ws_series.shift(1).values
        lag2 = ws_series.shift(2).values
        lag3 = ws_series.shift(3).values
        ma7  = ws_series.rolling(7, min_periods=1).mean().values

        daily_ret = np.diff(prices, prepend=prices[0]) / (np.r_[prices[0], prices[:-1]] + 1e-9) * 100
        vol5d     = pd.Series(daily_ret).rolling(5, min_periods=1).std().values

        frames.append(pd.DataFrame({
            "ticker":                ticker,
            "date":                  dates,
            "open":                  prices * rng.uniform(0.99, 1.0, n),
            "high":                  prices * rng.uniform(1.0, 1.02, n),
            "low":                   prices * rng.uniform(0.98, 1.0, n),
            "close":                 prices,
            "volume":                rng.integers(1_000_000, 50_000_000, n),
            "daily_return":          daily_ret,
            "price_direction":       np.where(daily_ret > 0, 1, -1),
            "rolling_vol_5d":        vol5d,
            "raw_avg_sentiment":     weighted_sentiment * 0.9,
            "weighted_avg_sentiment": weighted_sentiment,
            "sentiment_std":         rng.uniform(0.05, 0.4, n),
            "post_count":            post_count,
            "bullish_ratio":         bullish_ratio,
            "bearish_ratio":         1 - bullish_ratio,
            "sentiment_lag_1d":      lag1,
            "sentiment_lag_2d":      lag2,
            "sentiment_lag_3d":      lag3,
            "sentiment_ma7":         ma7,
            "total_engagement":      post_count * rng.uniform(50, 500, n),
            "news_sentiment":        news_sent,
            "news_relevance":        rng.uniform(0.3, 1.0, n),
            "news_overall":          news_sent * 0.8,
        }))

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    log.info("Generated synthetic dataset: %d rows for %d tickers", len(df), len(tickers))
    return df


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_etl():
    engine = get_pg_engine()

    all_dfs = []
    for ticker in ["TSLA", "AAPL", "GME", "NVDA", "AMZN"]:
        raw_df  = extract_reddit_from_mongo(ticker)
        if raw_df.empty:
            log.warning("No data for %s — skipping", ticker)
            continue
        tx_df   = transform_reddit_sentiment(raw_df)
        all_dfs.append(tx_df)

    if all_dfs:
        full_df  = pd.concat(all_dfs, ignore_index=True)
        daily_df = aggregate_daily_sentiment(full_df)
        load_to_postgres(daily_df, "reddit_sentiment_daily", engine)
        master   = build_master_dataset(engine)
        log.info("ETL complete. Master dataset: %d rows", len(master))
    else:
        log.warning("No Reddit data found. Generating synthetic dataset instead.")
        synthetic = generate_full_synthetic_dataset()
        load_to_postgres(synthetic, "master_dataset", engine)


if __name__ == "__main__":
    run_etl()
