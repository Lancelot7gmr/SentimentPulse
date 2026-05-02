import os
import json
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pymongo
from sqlalchemy import create_engine, text
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TICKERS        = ["TSLA", "AAPL", "GME", "NVDA", "AMZN"]
SUBREDDITS     = ["wallstreetbets", "stocks", "investing"]
START_DATE     = "2023-01-01"
END_DATE       = "2024-12-31"
POSTS_PER_SUB  = 500          

# ─────────────────────────────────────────────
# DATABASE CONNECTIONS
# ─────────────────────────────────────────────

def get_mongo_client():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = pymongo.MongoClient(uri)
    log.info("Connected to MongoDB at %s", uri)
    return client

def get_pg_engine():
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    user = os.getenv("PG_USER", "postgres")
    pw   = os.getenv("PG_PASSWORD", "password")
    db   = os.getenv("PG_DB", "sentiment_stock")
    url  = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    engine = create_engine(url)
    log.info("Connected to PostgreSQL at %s:%s/%s", host, port, db)
    return engine

# ─────────────────────────────────────────────
# 1. REDDIT DATA COLLECTION (Semi-structured)
# ─────────────────────────────────────────────

def fetch_reddit_posts_via_api(subreddit: str, ticker: str, limit: int = 100) -> list[dict]:
    posts = []
    headers = {"User-Agent": os.getenv("REDDIT_USER_AGENT", "SentimentBot/1.0")}
    after   = None   

    while len(posts) < limit:
        batch = min(100, limit - len(posts))
        url   = (f"https://www.reddit.com/r/{subreddit}/search.json"
                 f"?q={ticker}&sort=new&limit={batch}&t=year&restrict_sr=1"
                 + (f"&after={after}" if after else ""))
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning("Reddit fetch failed: %s", e)
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            post = child.get("data", {})
            post["_fetched_ticker"]    = ticker
            post["_fetched_subreddit"] = subreddit
            post["_fetched_at"]        = datetime.utcnow().isoformat()
            posts.append(post)

        after = data.get("data", {}).get("after")
        if not after:
            break
        time.sleep(1.2) 

    log.info("Fetched %d posts from r/%s for %s", len(posts), subreddit, ticker)
    return posts


def store_reddit_in_mongo(posts: list[dict], db_name: str = None):
    client  = get_mongo_client()
    db_name = db_name or os.getenv("MONGO_DB", "sentiment_stock_db")
    db      = client[db_name]
    col     = db["reddit_posts"]

    col.create_index([("id", pymongo.ASCENDING)], unique=True)
    col.create_index([("_fetched_ticker", pymongo.ASCENDING),
                      ("created_utc", pymongo.ASCENDING)])

    inserted = 0
    for post in posts:
        try:
            col.update_one({"id": post["id"]}, {"$set": post}, upsert=True)
            inserted += 1
        except Exception as e:
            log.debug("Mongo insert skip: %s", e)

    log.info("Stored/updated %d Reddit posts in MongoDB", inserted)
    client.close()


def collect_all_reddit(tickers=TICKERS, subreddits=SUBREDDITS):
    for ticker in tickers:
        for sub in subreddits:
            posts = fetch_reddit_posts_via_api(sub, ticker, limit=POSTS_PER_SUB)
            if posts:
                store_reddit_in_mongo(posts)
            time.sleep(2)   


# ─────────────────────────────────────────────
# 2. STOCK PRICE DATA (Structured)
# ─────────────────────────────────────────────

def fetch_stock_prices(tickers: list[str] = TICKERS,
                       start: str = START_DATE,
                       end: str   = END_DATE) -> pd.DataFrame:
    log.info("Downloading stock prices for %s from %s to %s", tickers, start, end)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    frames = []
    for ticker in tickers:
        try:
            if len(tickers) > 1:
                df = raw.xs(ticker, axis=1, level=1).copy()
            else:
                df = raw.copy()
            df.columns = [c.lower() for c in df.columns]
            df["ticker"] = ticker
            df["date"]   = df.index.date
            df["daily_return"]     = df["close"].pct_change() * 100
            df["price_direction"]  = (df["daily_return"] > 0).astype(int).replace({0: -1})
            df["rolling_vol_5d"]   = df["daily_return"].rolling(5).std()
            frames.append(df.reset_index(drop=True))
        except Exception as e:
            log.warning("Could not process %s: %s", ticker, e)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Downloaded %d price rows for %d tickers", len(combined), len(tickers))
    return combined


def store_stock_prices_in_pg(df: pd.DataFrame, engine):
    df_store = df[["date", "ticker", "open", "high", "low", "close",
                   "volume", "daily_return", "price_direction", "rolling_vol_5d"]].copy()
    df_store["date"] = pd.to_datetime(df_store["date"])
    df_store.to_sql("stock_prices", engine, if_exists="replace",
                    index=False, method="multi", chunksize=500)
    with engine.connect() as con:
        con.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_sp_ticker_date "
            "ON stock_prices (ticker, date)"
        ))
        con.commit()
    log.info("Stored %d rows in PostgreSQL table 'stock_prices'", len(df_store))


# ─────────────────────────────────────────────
# 3. NEWS SENTIMENT DATA (Structured)
# ─────────────────────────────────────────────

def fetch_news_sentiment_alpha_vantage(ticker: str, api_key: str) -> list[dict]:
    url = (f"https://www.alphavantage.co/query"
           f"?function=NEWS_SENTIMENT&tickers={ticker}"
           f"&time_from=20230101T0000&limit=1000&apikey={api_key}")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("Alpha Vantage fetch failed for %s: %s", ticker, e)
        return []

    records = []
    for article in data.get("feed", []):
        pub_time = article.get("time_published", "")
        try:
            pub_dt = datetime.strptime(pub_time, "%Y%m%dT%H%M%S")
        except Exception:
            pub_dt = None

        for ts in article.get("ticker_sentiment", []):
            if ts.get("ticker") == ticker:
                records.append({
                    "ticker":               ticker,
                    "date":                 pub_dt.date() if pub_dt else None,
                    "published_at":         pub_dt,
                    "title":                article.get("title", ""),
                    "source":               article.get("source", ""),
                    "overall_sentiment":    article.get("overall_sentiment_score", 0),
                    "overall_sentiment_label": article.get("overall_sentiment_label", ""),
                    "ticker_sentiment_score": float(ts.get("ticker_sentiment_score", 0)),
                    "ticker_relevance":     float(ts.get("relevance_score", 0)),
                })
    log.info("Fetched %d news articles for %s", len(records), ticker)
    return records


def generate_synthetic_news_sentiment(tickers=TICKERS,
                                      start=START_DATE, end=END_DATE) -> pd.DataFrame:
    log.info("Generating synthetic news sentiment (no API key found)")
    dates = pd.date_range(start, end, freq="B")   
    frames = []
    for ticker in tickers:
        rng  = np.random.default_rng(seed=abs(hash(ticker)) % (2**31))
        n    = len(dates)
        score = np.zeros(n)
        score[0] = rng.normal(0, 0.1)
        theta, mu, sigma = 0.1, 0.0, 0.15
        for i in range(1, n):
            score[i] = score[i-1] + theta*(mu - score[i-1]) + sigma*rng.normal()
        score = np.clip(score, -1, 1)
        frames.append(pd.DataFrame({
            "ticker":                  ticker,
            "date":                    dates.date,
            "ticker_sentiment_score":  score,
            "overall_sentiment":       score * 0.9 + rng.normal(0, 0.05, n),
            "ticker_relevance":        rng.uniform(0.3, 1.0, n),
            "source":                  rng.choice(["Reuters","Bloomberg","MarketWatch",
                                                   "CNBC","WSJ"], n),
            "article_count":           rng.integers(1, 15, n),
        }))
    df = pd.concat(frames, ignore_index=True)
    log.info("Generated %d synthetic news rows", len(df))
    return df


def store_news_in_pg(df: pd.DataFrame, engine):
    df["date"] = pd.to_datetime(df["date"])
    df.to_sql("news_sentiment", engine, if_exists="replace",
              index=False, method="multi", chunksize=500)
    with engine.connect() as con:
        con.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_ns_ticker_date "
            "ON news_sentiment (ticker, date)"
        ))
        con.commit()
    log.info("Stored %d rows in PostgreSQL table 'news_sentiment'", len(df))


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_collection():
    log.info("=" * 60)
    log.info("STEP 1/3: Collecting Reddit posts → MongoDB")
    log.info("=" * 60)
    collect_all_reddit()

    log.info("=" * 60)
    log.info("STEP 2/3: Downloading stock prices → PostgreSQL")
    log.info("=" * 60)
    engine = get_pg_engine()
    stock_df = fetch_stock_prices()
    store_stock_prices_in_pg(stock_df, engine)

    log.info("=" * 60)
    log.info("STEP 3/3: Fetching news sentiment → PostgreSQL")
    log.info("=" * 60)
    av_key = os.getenv("ALPHA_VANTAGE_KEY", "")
    if av_key and av_key != "your_alpha_vantage_key_here":
        all_news = []
        for ticker in TICKERS:
            records = fetch_news_sentiment_alpha_vantage(ticker, av_key)
            all_news.extend(records)
            time.sleep(12)   
        if all_news:
            news_df = pd.DataFrame(all_news)
            store_news_in_pg(news_df, engine)
    else:
        log.warning("No Alpha Vantage key found — using synthetic news data")
        news_df = generate_synthetic_news_sentiment()
        store_news_in_pg(news_df, engine)

    log.info("Data collection complete.")


if __name__ == "__main__":
    run_collection()
