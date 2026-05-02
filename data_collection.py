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
    """
    MongoDB is used because Reddit JSON documents have variable schemas —
    some posts have awards, flairs, media etc. that others do not.
    MongoDB stores each post as-is without needing a fixed table schema."""
    
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = pymongo.MongoClient(uri)
    log.info("Connected to MongoDB at %s", uri)
    return client

def get_pg_engine():
    """
    PostgreSQL is used for structured, tabular data (stock prices, news)
    because it supports SQL joins, indexing, and time-series queries efficiently."""
    
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
    """
    Fetches Reddit posts mentioning a ticker from a given subreddit.
    Uses Reddit's public JSON API (no authentication required for read-only).
    
    Each returned post is a raw JSON dictionary — semi-structured because
    the fields present vary post by post. We store these in MongoDB as-is,
    preserving the original structure an it returns a list of post dictionaries.
    """
    posts = []
    headers = {"User-Agent": os.getenv("REDDIT_USER_AGENT", "SentimentBot/1.0")}
    after   = None   # Reddit's pagination cursor

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
            # Add metadata for our own tracking
            post["_fetched_ticker"]    = ticker
            post["_fetched_subreddit"] = subreddit
            post["_fetched_at"]        = datetime.utcnow().isoformat()
            posts.append(post)

        after = data.get("data", {}).get("after")
        if not after:
            break
        time.sleep(1.2)   # be polite to Reddit's rate limit

    log.info("Fetched %d posts from r/%s for %s", len(posts), subreddit, ticker)
    return posts


def store_reddit_in_mongo(posts: list[dict], db_name: str = None):
    """
    Stores raw Reddit post JSON documents into MongoDB.
    Each Reddit post is a JSON object with different fields present depending
    on whether the post has media, awards, flair, etc. Since the MongoDB is a document database: it stores JSON natively and does NOT require
    every document to have the same fields. This makes it perfect for semi-structured data.
    A relational database like PostgreSQL would require us to define all columns upfront
    and would waste space with NULL columns for fields that only some posts have.
    """
    client  = get_mongo_client()
    db_name = db_name or os.getenv("MONGO_DB", "sentiment_stock_db")
    db      = client[db_name]
    col     = db["reddit_posts"]

    # Create index to avoid duplicate inserts and speed up queries
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
            time.sleep(2)   # pause between subreddit requests


# ─────────────────────────────────────────────
# 2. STOCK PRICE DATA (Structured)
# ─────────────────────────────────────────────

def fetch_stock_prices(tickers: list[str] = TICKERS,
                       start: str = START_DATE,
                       end: str   = END_DATE) -> pd.DataFrame:
    """
    Downloads daily OHLCV stock data from Yahoo Finance using the yfinance library.
    
    OHLCV = Open, High, Low, Close, Volume — the standard daily price record.
    We also compute daily_return in which percentage price change day-over-day
    and price_direction where +1 if price went up, -1 if down
    """
    log.info("Downloading stock prices for %s from %s to %s", tickers, start, end)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    frames = []
    for ticker in tickers:
        try:
            # yfinance returns multi-level columns when multiple tickers are given
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
    """
    Stores stock price data in PostgreSQL table 'stock_prices'.
    Stock prices are perfectly structured meaning every row has exactly the same columns.
    PostgreSQL supports time-series queries like windowed rolling averages,
    JOIN with sentiment scores on date+ticker, and efficient indexing on dates.
    The relational model also lets us JOIN stock_prices with news_sentiment on
    (date, ticker) — something much harder to do in MongoDB.
    """
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
    """
    Fetches financial news sentiment from Alpha Vantage's NEWS_SENTIMENT endpoint.
    Alpha Vantage returns JSON with article titles, sources, publication times,
    and pre-computed sentiment scores per ticker mention.
    It is then flattened into a structured table
    because the sentiment scores and metadata are consistent across all articles.
    """
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

        # Each article can mention multiple tickers — extract only our ticker's sentiment
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
    """
    Generates realistic synthetic news sentiment data for demonstration
    when an Alpha Vantage API key is not available.
    This is clearly documented as synthetic.
    The synthetic data uses a random walk with mean-reversion to simulate realistic
    sentiment drift, and seeds the random number generator per ticker for reproducibility.
    """
    log.info("Generating synthetic news sentiment (no API key found)")
    dates = pd.date_range(start, end, freq="B")   # Business days only
    frames = []
    for ticker in tickers:
        rng  = np.random.default_rng(seed=abs(hash(ticker)) % (2**31))
        n    = len(dates)
        # Mean-reverting random walk (Ornstein-Uhlenbeck process)
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
            time.sleep(12)   # Alpha Vantage free tier: 5 calls/min
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
