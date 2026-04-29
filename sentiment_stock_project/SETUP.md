# SentimentPulse — Setup & Run Guide
## NCI MSc Data Analytics · Analytics Programming & Data Visualisation

---

## What This Project Does

This project analyses whether **Reddit social media sentiment** predicts **stock price movements**
for major companies (TSLA, AAPL, GME, NVDA, AMZN).

**Three datasets:**
1. **Reddit posts (JSON → MongoDB)** — semi-structured, engagement-weighted VADER sentiment
2. **Stock prices (CSV → PostgreSQL)** — Yahoo Finance OHLCV data with return features
3. **Financial news (JSON → PostgreSQL)** — Alpha Vantage news sentiment scores

**Pipeline:** Reddit API → MongoDB → ETL (Python/VADER) → PostgreSQL → Streamlit Dashboard

---

## Quickstart (Demo Mode — No Database Needed)

This runs with synthetic data immediately. Perfect for testing the dashboard:

```bash
# 1. Clone / navigate to project directory
cd sentiment_stock_project

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run dashboard/app.py
```

Then open **http://localhost:8501** in your browser.

The dashboard will open with **synthetic demo data** — fully functional for exploration.
Toggle "Connect to PostgreSQL" in the sidebar to use real data once databases are set up.

---

## Understanding Streamlit (Plain English)

Streamlit is a Python library that turns a Python script into an interactive web app.
You don't write any HTML or JavaScript — just Python.

**Key concepts:**
- `streamlit run app.py` starts a local web server on port 8501
- Every time you click a button or move a slider, Streamlit **re-runs the entire script**
- `@st.cache_data` — put this above functions that are slow (like loading data). 
  Streamlit will only run them once and reuse the result on subsequent reruns.
  Think of it as "remember this result, don't recompute it every click."
- `st.session_state` — a dictionary that persists between reruns. 
  Like a global variable that survives when the page re-executes.
- `st.sidebar` — anything inside `with st.sidebar:` appears in the left panel
- `st.columns([2,1])` — splits the page into columns. The numbers are relative widths.
- `st.plotly_chart(fig)` — displays a Plotly chart inline
- `st.dataframe(df)` — displays a pandas DataFrame as an interactive table
- `st.metric("Label", "Value", delta="change")` — shows a KPI card with trend arrow

**The sidebar toggle "Connect to PostgreSQL":**  
- OFF (default) = synthetic demo data, works immediately with no setup
- ON = connects to your local PostgreSQL using credentials in .env

---

## Full Setup (Real Data Collection)

### Step 1: Set Up Databases

**MongoDB** (for Reddit JSON):
```bash
# Mac
brew install mongodb-community && brew services start mongodb-community

# Ubuntu
sudo apt install mongodb && sudo service mongodb start

# Windows: Download installer from mongodb.com/try/download/community
```

**PostgreSQL** (for stock prices and news):
```bash
# Mac
brew install postgresql && brew services start postgresql

# Ubuntu
sudo apt install postgresql && sudo service postgresql start

# Windows: Download from postgresql.org/download/windows

# Create the database:
psql -U postgres -c "CREATE DATABASE sentiment_stock;"
```

### Step 2: Configure Credentials

```bash
cp .env.example .env
# Now edit .env with your actual credentials
```

**Get free API keys:**
- Reddit: Go to https://www.reddit.com/prefs/apps → Create App → Script type
  Copy the client_id (under app name) and client_secret
- Alpha Vantage: https://www.alphavantage.co/support/#api-key (free, instant)

### Step 3: Run the Pipeline

```bash
# Step 3a: Collect data (Reddit → MongoDB, Prices → PostgreSQL, News → PostgreSQL)
python src/data_collection.py

# Step 3b: ETL — clean Reddit text, score sentiment, aggregate, build master table
python src/etl_processing.py

# Step 3c: Run analysis (optional — dashboard computes this live)
python src/analysis.py

# Step 3d: Launch dashboard (connects to real data)
streamlit run dashboard/app.py
```

---

## Dashboard Pages Explained

| Page | What it shows |
|------|--------------|
| 🏠 Overview | KPI cards per stock, dual-axis price+sentiment timeline, scatter plots |
| 📊 Sentiment vs Price | Candlestick + Bollinger Bands + multi-signal sentiment for one ticker |
| 🔬 Correlation & Causality | Pearson/Spearman table, lead-lag heatmap, Granger causality test, regime analysis |
| 🤖 ML Prediction | Random Forest / Logistic / GBM accuracy vs baseline, feature importances, confusion matrix |
| 📰 News vs Social | Reddit vs Alpha Vantage financial news sentiment comparison |
| 🔍 Deep Dive | Rolling 30-day correlation, bullish/bearish split, raw data download |

---

## Project File Structure

```
sentiment_stock_project/
├── dashboard/
│   └── app.py                  ← Streamlit dashboard (run this)
├── src/
│   ├── data_collection.py      ← Fetches Reddit, stock prices, news
│   ├── etl_processing.py       ← Cleans, scores, aggregates, merges data
│   └── analysis.py             ← Correlation, Granger, ML classification
├── data/
│   ├── raw/                    ← CSV exports (optional)
│   └── processed/              ← Processed outputs (optional)
├── notebooks/                  ← Jupyter notebooks for exploration
├── requirements.txt            ← All Python packages
├── .env.example                ← Template for your credentials
└── SETUP.md                    ← This file
```

---

## Key Technical Novelties (for Report)

1. **Engagement-weighted sentiment** — viral posts count more than low-engagement ones
   (log(upvotes) × upvote_ratio × log(comments+1))

2. **Lead-lag analysis** — tests at multiple lags (0–5 days) to find when sentiment
   has the strongest predictive relationship with price

3. **Granger causality test** — formal statistical test of whether sentiment
   Granger-causes returns (beyond simple correlation)

4. **TimeSeriesSplit cross-validation** — correct method for temporal prediction;
   never uses future data to train on past, unlike standard k-fold CV

5. **Rolling 30-day correlation** — shows how the sentiment-price relationship
   changes over time (time-varying analysis)

6. **Sentiment regime analysis** — categorises days into 5 regimes and computes
   the average next-day return in each, revealing non-linear effects

7. **Financial slang expansion** — dictionary that translates "moon", "apes",
   "diamond hands" etc. before VADER scoring, improving accuracy on WSB text

---

## Troubleshooting

**"Module not found" error:**
Make sure you activated the virtual environment and ran `pip install -r requirements.txt`

**"Cannot connect to MongoDB":**
Start MongoDB: `brew services start mongodb-community` (Mac) or `sudo service mongodb start` (Linux)

**"Cannot connect to PostgreSQL":**
Check credentials in .env. Try: `psql -U postgres -c "\l"` to verify PostgreSQL is running.

**Dashboard shows synthetic data even with DB toggle on:**
Check that `master_dataset` table exists in PostgreSQL. Run `python src/etl_processing.py` first.

**Streamlit port already in use:**
`streamlit run dashboard/app.py --server.port 8502`
