"""
dashboard/app.py  —  Main Streamlit entry point.

HOW STREAMLIT WORKS (plain English):
  Streamlit converts this Python script into a web app.
  Every time the user clicks a button or moves a slider, Streamlit
  re-runs the ENTIRE script from top to bottom.
  - st.session_state  : dictionary that survives reruns (like a global store)
  - @st.cache_data    : caches expensive functions so they only run once
  - st.sidebar        : left-hand panel
  - st.columns([2,1]) : splits the page into columns with relative widths

HOW TO RUN:
  cd sentiment_stock_project
  streamlit run dashboard/app.py
  Then open http://localhost:8501 in your browser.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.etl_processing import generate_full_synthetic_dataset
from src.analysis import (compute_correlations, lead_lag_analysis,
                           granger_causality_analysis, run_ml_analysis,
                           compute_sentiment_regimes, train_direction_classifier)

# ── Colour helper ─────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Must be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentPulse",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: #070c18; }
.main .block-container { padding: 2rem 2rem 3rem; max-width: 1440px; }
section[data-testid="stSidebar"] { background: #040810 !important; border-right: 1px solid rgba(255,255,255,0.06); }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span { color: #94a3b8 !important; }
[data-testid="metric-container"] {
    background: #0f1829; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px; padding: 1.1rem 1.3rem;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] { color: #64748b !important; font-size:0.78rem; letter-spacing:0.05em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-family:'Syne',sans-serif; font-size:1.6rem; }
.dash-h1 { font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800; color:#f1f5f9; letter-spacing:-0.04em; line-height:1; margin:0; }
.dash-mono { font-family:'DM Mono',monospace; font-size:0.72rem; color:#00d9a3; letter-spacing:0.18em; text-transform:uppercase; margin-top:4px; }
.sec-title { font-family:'Syne',sans-serif; font-weight:700; font-size:1.05rem; color:#e2e8f0; border-left:3px solid #00d9a3; padding-left:0.55rem; margin:1.2rem 0 0.6rem; }
.card { background:#0f1829; border:1px solid rgba(255,255,255,0.05); border-radius:14px; padding:1.2rem 1.4rem; }
.synth-warn { background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.25); border-radius:8px; padding:0.5rem 1rem; font-family:'DM Mono',monospace; font-size:0.72rem; color:#fbbf24; }
.pill-green { background:rgba(0,217,163,.12); color:#00d9a3; border-radius:20px; padding:2px 10px; font-size:0.78rem; font-weight:500; }
.pill-red   { background:rgba(255,77,109,.12); color:#ff4d6d; border-radius:20px; padding:2px 10px; font-size:0.78rem; font-weight:500; }
.pill-amber { background:rgba(251,191,36,.12); color:#fbbf24; border-radius:20px; padding:2px 10px; font-size:0.78rem; font-weight:500; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TICKER_COLORS = {"TSLA":"#e74c3c","AAPL":"#3b82f6","GME":"#f59e0b",
                 "NVDA":"#00d9a3","AMZN":"#a855f7"}

def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' — Plotly requires rgba for transparency."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
CHART_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,24,41,0.9)",
    font=dict(family="Outfit, sans-serif", color="#94a3b8", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11)),
    hovermode="x unified",
)
# Default margin applied separately — keeping it out of CHART_BG prevents
# "multiple values for margin" when a chart needs a custom margin.
_DEF_MARGIN = dict(l=10, r=10, t=40, b=10)
# Axis style applied separately via update_xaxes/update_yaxes.
# Keeping axes OUT of CHART_BG prevents "multiple values for yaxis" errors
# when update_layout(**CHART_BG, yaxis=..., margin=_DEF_MARGIN) is called.
_AXIS_STYLE = dict(gridcolor="rgba(255,255,255,0.03)", linecolor="rgba(255,255,255,0.08)")
_YAXIS_STYLE = _AXIS_STYLE  # backwards-compat alias

def _apply_axes(fig):
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Generating dataset…")
def load_demo_data():
    df = generate_full_synthetic_dataset()
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=3600)
def cached_correlations(_df):
    return compute_correlations(_df)

@st.cache_data(ttl=3600)
def cached_lead_lag(_df):
    return lead_lag_analysis(_df)

@st.cache_data(ttl=3600)
def cached_granger(_df):
    return granger_causality_analysis(_df)

@st.cache_data(ttl=3600)
def cached_regimes(_df):
    return compute_sentiment_regimes(_df)

@st.cache_data(ttl=3600)
def cached_ml(_df, ticker):
    return train_direction_classifier(_df, ticker)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem; text-align:center;'>
      <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.25rem;color:#e2e8f0;'>📡 SentimentPulse</div>
      <div style='font-family:DM Mono,monospace;font-size:0.62rem;color:#00d9a3;letter-spacing:.18em;margin-top:3px;'>SOCIAL × MARKETS</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    use_db = st.toggle("Connect to PostgreSQL", value=False,
                       help="Requires local PostgreSQL with master_dataset table.")
    
    if use_db:
        pg_pass = st.text_input("PostgreSQL password", type="password",
                                 help="Enter the password for your local postgres user")
        if pg_pass:
            try:
                import os; os.environ["PG_PASSWORD"] = pg_pass
                from src.etl_processing import get_pg_engine
                engine = get_pg_engine()
                raw = pd.read_sql("SELECT * FROM master_dataset", engine)
                raw["date"] = pd.to_datetime(raw["date"])
                is_synth = False
                st.success("DB connected ✓")
            except Exception as ex:
                st.warning(f"DB error: {ex}")
                raw = load_demo_data()
                is_synth = True
        else:
            st.caption("Enter password above to connect")
            raw = load_demo_data()
            is_synth = True
    else:
        raw = load_demo_data()
        is_synth = True

    st.divider()
    st.markdown("**Filters**")
    tickers_all = sorted(raw["ticker"].unique())
    tickers_sel = st.multiselect("Tickers", tickers_all, default=tickers_all)

    d_min = raw["date"].min().date()
    d_max = raw["date"].max().date()
    dr = st.date_input("Date range", (d_min, d_max), d_min, d_max)

    sent_col = st.selectbox("Sentiment signal",
        ["weighted_avg_sentiment","raw_avg_sentiment","sentiment_ma7"],
        format_func=lambda x: {"weighted_avg_sentiment":"Engagement-Weighted (main)",
                                "raw_avg_sentiment":"Raw Average",
                                "sentiment_ma7":"7-Day Moving Avg"}[x])

    st.divider()
    page = st.radio("Navigation", [
        "Overview",
        "Sentiment vs Price",
        "Correlation & Causality",
        "ML Prediction",
        "News vs Social Media",
        "Stock Deep Dive",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("<div style='font-family:DM Mono,monospace;font-size:0.6rem;color:#1e293b;text-align:center;line-height:1.8;'>NCI · MSc Data Analytics<br>Analytics Programming & DV<br>Semester 2 · 2025/26</div>", unsafe_allow_html=True)

# ── Apply filters ─────────────────────────────────────────────────────────────
df = raw[raw["ticker"].isin(tickers_sel)].copy()
if isinstance(dr, (list, tuple)) and len(dr) == 2:
    df = df[(df["date"] >= pd.Timestamp(dr[0])) & (df["date"] <= pd.Timestamp(dr[1]))]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
c1, c2 = st.columns([4, 1])
with c1:
    st.markdown("""
    <div style='padding-top:0.2rem;'>
      <div class='dash-h1'>SentimentPulse</div>
      <div class='dash-mono' style='margin-top:6px;'>Social Media Sentiment × Stock Market · NCI MSc Project</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("<div style='padding-top:0.5rem;'></div>", unsafe_allow_html=True)
    if is_synth:
        st.markdown("<div class='synth-warn'>⚠ DEMO DATA<br>Synthetic — toggle DB for real</div>", unsafe_allow_html=True)
    else:
        st.success("🟢 Live database")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("<div class='sec-title'>Market at a Glance</div>", unsafe_allow_html=True)

    # KPI metrics — one per ticker selected
    cols = st.columns(len(tickers_sel) if tickers_sel else 1)
    for i, ticker in enumerate(tickers_sel):
        tdf = df[df["ticker"] == ticker].sort_values("date")
        if tdf.empty:
            continue
        latest_price   = tdf["close"].iloc[-1]
        price_chg      = tdf["daily_return"].iloc[-1]
        latest_sent    = tdf[sent_col].iloc[-1]
        avg_sent_week  = tdf[sent_col].tail(7).mean()
        sent_trend     = "↑" if latest_sent > avg_sent_week else "↓"
        price_trend    = "↑" if price_chg > 0 else "↓"
        with cols[i]:
            st.metric(
                label=f"{ticker}",
                value=f"${latest_price:,.2f}",
                delta=f"{price_chg:+.2f}%",
            )

    st.write("")

    # Main overview chart: price + sentiment dual-axis for all tickers
    st.markdown("<div class='sec-title'>Price × Sentiment Timeline</div>", unsafe_allow_html=True)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=["Stock Closing Price (USD)", "Reddit Sentiment Score"],
        vertical_spacing=0.05,
    )

    for ticker in tickers_sel:
        tdf = df[df["ticker"] == ticker].sort_values("date")
        color = TICKER_COLORS.get(ticker, "#888")
        # Price line
        fig.add_trace(go.Scatter(
            x=tdf["date"], y=tdf["close"], name=ticker,
            line=dict(color=color, width=1.8),
            legendgroup=ticker, showlegend=True,
        ), row=1, col=1)
        # Sentiment area
        fig.add_trace(go.Scatter(
            x=tdf["date"], y=tdf[sent_col], name=f"{ticker} sentiment",
            fill="tozeroy",
            line=dict(color=color, width=1),
            fillcolor=hex_to_rgba(color, 0.12),
            legendgroup=ticker, showlegend=False,
            opacity=0.8,
        ), row=2, col=1)

    fig.add_hline(y=0, row=2, col=1, line_dash="dash",
                  line_color="rgba(255,255,255,0.2)", line_width=1)
    fig.update_layout(**CHART_BG, height=520,
                      title_text="", showlegend=True,
                      margin=_DEF_MARGIN)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", row=2, col=1, range=[-1, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment distribution heatmap
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("<div class='sec-title'>Daily Sentiment Distribution</div>", unsafe_allow_html=True)
        fig_box = go.Figure()
        for ticker in tickers_sel:
            tdf = df[df["ticker"] == ticker]
            fig_box.add_trace(go.Box(
                y=tdf[sent_col], name=ticker,
                marker_color=TICKER_COLORS.get(ticker, "#888"),
                line_width=1.5, boxmean=True,
            ))
        fig_box.update_layout(**CHART_BG, height=320,
                               title_text="Sentiment score distribution per ticker",
                               yaxis_title="Sentiment Score",
                               margin=_DEF_MARGIN)
        st.plotly_chart(fig_box, use_container_width=True)

    with c_right:
        st.markdown("<div class='sec-title'>Post Volume × Sentiment</div>", unsafe_allow_html=True)
        scatter_df = df[["ticker", sent_col, "post_count", "daily_return"]].dropna()
        fig_sc = px.scatter(
            scatter_df, x=sent_col, y="daily_return",
            color="ticker", size="post_count",
            color_discrete_map=TICKER_COLORS,
            labels={sent_col: "Reddit Sentiment", "daily_return": "Daily Return (%)"},
            opacity=0.6,
        )
        fig_sc.update_layout(**CHART_BG, height=320,
                              title_text="Sentiment vs Return (bubble = post volume)",
                              margin=_DEF_MARGIN)
        fig_sc.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
        fig_sc.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
        st.plotly_chart(fig_sc, use_container_width=True)

    # Research question callout
    st.markdown("""
    <div class='card' style='margin-top:0.5rem; border-left:3px solid #4da6ff;'>
      <div style='font-family:DM Mono,monospace;font-size:0.68rem;color:#4da6ff;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.4rem;'>Research Question</div>
      <div style='font-size:1.05rem;color:#e2e8f0;line-height:1.6;'>
        Does public sentiment expressed on Reddit and financial news platforms
        <strong style='color:#00d9a3;'>predict</strong> short-term stock price movements,
        and does this relationship vary across tickers and sentiment measurement approaches?
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT VS PRICE
# ══════════════════════════════════════════════════════════════════════════════
elif "Sentiment vs Price" in page:
    st.markdown("<div class='sec-title'>Dual-Signal Explorer</div>", unsafe_allow_html=True)
    st.caption("Compare Reddit sentiment and stock price side-by-side for each ticker")

    sel_ticker = st.selectbox("Select ticker for detail view", tickers_sel or ["TSLA"])
    tdf = df[df["ticker"] == sel_ticker].sort_values("date")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.3, 0.25],
        subplot_titles=[
            f"{sel_ticker} Stock Price + Bollinger Bands",
            "Reddit Sentiment (Raw vs Engagement-Weighted vs 7-day MA)",
            "Daily Return (%) × Post Volume"
        ],
        vertical_spacing=0.06,
    )
    color = TICKER_COLORS.get(sel_ticker, "#00d9a3")

    # ── Row 1: Price with Bollinger Bands ──
    close  = tdf["close"]
    ma20   = close.rolling(20, min_periods=1).mean()
    std20  = close.rolling(20, min_periods=1).std().fillna(0)
    upper  = ma20 + 2 * std20
    lower  = ma20 - 2 * std20

    fig.add_trace(go.Candlestick(
        x=tdf["date"], open=tdf["open"], high=tdf["high"],
        low=tdf["low"], close=tdf["close"], name="OHLC",
        increasing_line_color="#00d9a3", decreasing_line_color="#ff4d6d",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=tdf["date"], y=upper, name="Upper BB",
        line=dict(color="rgba(77,166,255,0.4)", width=1, dash="dot"),
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=tdf["date"], y=lower, name="Lower BB",
        fill="tonexty", fillcolor="rgba(77,166,255,0.04)",
        line=dict(color="rgba(77,166,255,0.4)", width=1, dash="dot"),
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=tdf["date"], y=ma20, name="MA20",
        line=dict(color="#fbbf24", width=1.2, dash="dot"),
        showlegend=True), row=1, col=1)

    # ── Row 2: Multi-signal sentiment ──
    if "raw_avg_sentiment" in tdf.columns:
        fig.add_trace(go.Scatter(
            x=tdf["date"], y=tdf["raw_avg_sentiment"], name="Raw sentiment",
            line=dict(color="rgba(148,163,184,0.5)", width=1),
        ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tdf["date"], y=tdf["weighted_avg_sentiment"], name="Weighted sentiment",
        line=dict(color=color, width=2),
    ), row=2, col=1)
    if "sentiment_ma7" in tdf.columns:
        fig.add_trace(go.Scatter(
            x=tdf["date"], y=tdf["sentiment_ma7"], name="MA7 sentiment",
            line=dict(color="#fbbf24", width=1.5, dash="dash"),
        ), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1, line_dash="dash",
                  line_color="rgba(255,255,255,0.15)", line_width=1)

    # ── Row 3: Return bars + post volume line ──
    ret_colors = ["#00d9a3" if r >= 0 else "#ff4d6d"
                  for r in tdf["daily_return"].fillna(0)]
    fig.add_trace(go.Bar(
        x=tdf["date"], y=tdf["daily_return"], name="Daily return %",
        marker_color=ret_colors, opacity=0.7,
    ), row=3, col=1)

    if "post_count" in tdf.columns:
        fig.add_trace(go.Scatter(
            x=tdf["date"], y=tdf["post_count"],
            name="Post count", yaxis="y4",
            line=dict(color="rgba(251,191,36,0.6)", width=1),
        ), row=3, col=1)

    fig.update_layout(**CHART_BG, height=720, showlegend=True,
                      xaxis_rangeslider_visible=False,
                      margin=_DEF_MARGIN)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", row=2, col=1, range=[-1.1, 1.1])
    fig.update_yaxes(title_text="Return %", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment-triggered events heatmap
    st.markdown("<div class='sec-title'>Sentiment Extremes vs Price Reaction</div>", unsafe_allow_html=True)
    extremes = tdf[np.abs(tdf["weighted_avg_sentiment"]) > 0.4].copy()
    extremes["next_return"] = tdf["daily_return"].shift(-1).values[:len(extremes)]
    extremes["direction"] = extremes["weighted_avg_sentiment"].apply(
        lambda x: "📈 Bullish spike" if x > 0 else "📉 Bearish spike")

    if not extremes.empty:
        st.dataframe(
            extremes[["date","weighted_avg_sentiment","daily_return","next_return",
                       "post_count","direction"]].sort_values("date", ascending=False).head(25),
            use_container_width=True,
            column_config={
                "date": "Date",
                "weighted_avg_sentiment": st.column_config.NumberColumn("Sentiment", format="%.3f"),
                "daily_return": st.column_config.NumberColumn("Same-day Return %", format="%.2f%%"),
                "next_return": st.column_config.NumberColumn("Next-day Return %", format="%.2f%%"),
                "post_count": "Reddit Posts",
                "direction": "Signal",
            }
        )
    else:
        st.info("No extreme sentiment events in selected date range.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CORRELATION & CAUSALITY
# ══════════════════════════════════════════════════════════════════════════════
elif "Correlation" in page:
    st.markdown("<div class='sec-title'>Statistical Correlation Analysis</div>", unsafe_allow_html=True)

    with st.spinner("Computing correlations…"):
        corr_df = cached_correlations(df)
        lag_df  = cached_lead_lag(df)
        grang   = cached_granger(df)

    if corr_df.empty:
        st.warning("Not enough data for correlation analysis. Try a wider date range.")
        st.stop()

    # Correlation summary table
    st.markdown("<div class='sec-title'>Pearson & Spearman Correlations — Sentiment → Next-Day Return</div>", unsafe_allow_html=True)

    def sig_badge(p):
        if p < 0.01: return "🟢 p<0.01"
        if p < 0.05: return "🟡 p<0.05"
        return "🔴 n.s."

    corr_display = corr_df.copy()
    corr_display["significance"] = corr_display["pearson_p_next_day"].apply(sig_badge)
    st.dataframe(
        corr_display[["ticker","n_observations","pearson_r_next_day",
                       "pearson_p_next_day","spearman_r_next_day",
                       "spearman_p_next_day","significance"]],
        use_container_width=True,
        column_config={
            "ticker": "Ticker",
            "n_observations": "N",
            "pearson_r_next_day":  st.column_config.NumberColumn("Pearson r", format="%.4f"),
            "pearson_p_next_day":  st.column_config.NumberColumn("Pearson p", format="%.4f"),
            "spearman_r_next_day": st.column_config.NumberColumn("Spearman r", format="%.4f"),
            "spearman_p_next_day": st.column_config.NumberColumn("Spearman p", format="%.4f"),
            "significance": "Significance",
        }
    )

    # Lead-lag heatmap
    st.markdown("<div class='sec-title'>Lead-Lag Analysis — Which Lag Has Highest Correlation?</div>", unsafe_allow_html=True)
    st.caption("Lag 0 = same day · Lag 1 = sentiment today predicts tomorrow's return · etc.")

    if not lag_df.empty:
        pivot = lag_df.pivot(index="ticker", columns="lag_days", values="pearson_r").round(4)
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f"Lag {c}d" for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[[0,"#ff4d6d"],[0.5,"#1e293b"],[1,"#00d9a3"]],
            zmid=0, text=pivot.values.round(3),
            texttemplate="%{text}", textfont=dict(size=12),
            colorbar=dict(title="Pearson r", tickfont=dict(color="#94a3b8")),
        ))
        fig_heat.update_layout(**CHART_BG, height=280,
                                title_text="Pearson r: Sentiment[t] vs Return[t+lag]",
                                xaxis_title="Lag", yaxis_title="Ticker",
                                margin=_DEF_MARGIN)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Line chart version of lead-lag
        fig_lag = go.Figure()
        for ticker in lag_df["ticker"].unique():
            tlag = lag_df[lag_df["ticker"] == ticker]
            fig_lag.add_trace(go.Scatter(
                x=tlag["lag_days"], y=tlag["pearson_r"],
                name=ticker, mode="lines+markers",
                line=dict(color=TICKER_COLORS.get(ticker,"#888"), width=2),
                marker=dict(size=7),
            ))
        fig_lag.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        fig_lag.update_layout(**CHART_BG, height=320,
                               title_text="Correlation by Lag Day",
                               xaxis_title="Lag (days)", yaxis_title="Pearson r",
                               margin=_DEF_MARGIN)
        st.plotly_chart(fig_lag, use_container_width=True)

    # Granger causality
    st.markdown("<div class='sec-title'>Granger Causality Test — Does Sentiment Granger-Cause Returns?</div>", unsafe_allow_html=True)
    st.caption("H₀: Sentiment does NOT help predict returns. Reject H₀ if p < 0.05.")

    if not grang.empty:
        def gc_badge(row):
            return "✅ Rejects H₀" if row["granger_significant"] else "❌ Fails to reject"
        grang["result"] = grang.apply(gc_badge, axis=1)
        st.dataframe(
            grang[["ticker","lag","f_statistic","p_value","result","sent_stationary","ret_stationary"]],
            use_container_width=True,
            column_config={
                "ticker": "Ticker",
                "lag": "Lag",
                "f_statistic": st.column_config.NumberColumn("F-stat", format="%.3f"),
                "p_value": st.column_config.NumberColumn("p-value", format="%.4f"),
                "result": "Granger Result",
                "sent_stationary": "Sentiment Stationary?",
                "ret_stationary": "Returns Stationary?",
            }
        )
    else:
        st.info("Granger test requires ≥30 observations per ticker.")

    # Sentiment regime analysis
    st.markdown("<div class='sec-title'>Sentiment Regime → Next-Day Return</div>", unsafe_allow_html=True)
    st.caption("Average next-day return when today's sentiment falls in each regime")

    with st.spinner("Computing regimes…"):
        reg_df = cached_regimes(df)

    if not reg_df.empty:
        fig_reg = px.bar(
            reg_df, x="sentiment_regime", y="mean_next_return",
            color="ticker", barmode="group",
            color_discrete_map=TICKER_COLORS,
            labels={"sentiment_regime":"Sentiment Regime","mean_next_return":"Mean Next-Day Return (%)"},
            category_orders={"sentiment_regime":["Very Bearish","Bearish","Neutral","Bullish","Very Bullish"]},
        )
        fig_reg.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        fig_reg.update_layout(**CHART_BG, height=360,
                               title_text="Mean Next-Day Return by Sentiment Regime",
                               margin=_DEF_MARGIN)
        st.plotly_chart(fig_reg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif "ML" in page:
    st.markdown("<div class='sec-title'>Machine Learning — Predicting Price Direction</div>", unsafe_allow_html=True)
    st.caption("Can Reddit sentiment alone predict whether a stock goes UP or DOWN tomorrow?")

    sel_ml = st.selectbox("Ticker to train model on", tickers_sel or ["TSLA"])

    with st.spinner(f"Training classifiers for {sel_ml}…"):
        ml_result = cached_ml(df, sel_ml)

    if not ml_result:
        st.warning(f"Insufficient data for {sel_ml}. Select a wider date range.")
        st.stop()

    # Model accuracy comparison
    st.markdown("<div class='sec-title'>Model Accuracy (TimeSeriesSplit CV)</div>", unsafe_allow_html=True)
    baseline = ml_result.get("baseline_accuracy", 0.5)

    model_names = list(ml_result["model_scores"].keys())
    model_acc   = [v["mean_accuracy"] for v in ml_result["model_scores"].values()]
    model_std   = [v["std_accuracy"] for v in ml_result["model_scores"].values()]

    fig_acc = go.Figure()
    # Baseline
    fig_acc.add_hline(y=baseline, line_dash="dot",
                      line_color="#fbbf24", line_width=1.5,
                      annotation_text=f"Baseline (majority class) {baseline:.1%}",
                      annotation_font_color="#fbbf24")
    fig_acc.add_trace(go.Bar(
        x=model_names, y=model_acc,
        error_y=dict(type="data", array=model_std, visible=True,
                     color="rgba(255,255,255,0.4)"),
        marker_color=["#00d9a3","#4da6ff","#a855f7"],
        text=[f"{a:.1%}" for a in model_acc],
        textposition="outside", textfont=dict(color="#e2e8f0"),
    ))
    fig_acc.update_layout(**CHART_BG, height=340,
                           title_text=f"Prediction Accuracy — {sel_ml}",
                           margin=_DEF_MARGIN)
    fig_acc.update_yaxes(tickformat=".0%", range=[0, 1],
                         gridcolor="rgba(255,255,255,0.03)",
                         linecolor="rgba(255,255,255,0.08)",
                         title_text="Accuracy")
    st.plotly_chart(fig_acc, use_container_width=True)

    # Feature importances
    c_fi, c_cm = st.columns(2)
    with c_fi:
        st.markdown("<div class='sec-title'>Feature Importances (Random Forest)</div>", unsafe_allow_html=True)
        imp = ml_result.get("feature_importances", {})
        if imp:
            imp_df = pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())})
            imp_df = imp_df.sort_values("importance", ascending=True)
            fig_imp = go.Figure(go.Bar(
                x=imp_df["importance"], y=imp_df["feature"],
                orientation="h",
                marker=dict(
                    color=imp_df["importance"],
                    colorscale=[[0,"#1e293b"],[1,"#00d9a3"]],
                ),
                text=imp_df["importance"].round(3), textposition="outside",
                textfont=dict(color="#94a3b8"),
            ))
            fig_imp.update_layout(**CHART_BG, height=360,
                                   margin=dict(l=140,r=20,t=30,b=20))
            fig_imp.update_xaxes(title_text="Importance", **_AXIS_STYLE)
            fig_imp.update_yaxes(**_AXIS_STYLE)
            st.plotly_chart(fig_imp, use_container_width=True)

    with c_cm:
        st.markdown("<div class='sec-title'>Confusion Matrix (Random Forest)</div>", unsafe_allow_html=True)
        cm = ml_result.get("confusion_matrix", [[0,0],[0,0]])
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred Down","Pred Up"], y=["Actual Down","Actual Up"],
            colorscale=[[0,"#0f1829"],[1,"#00d9a3"]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=20, color="#e2e8f0"),
            showscale=False,
        ))
        fig_cm.update_layout(**CHART_BG, height=280, margin=_DEF_MARGIN)
        st.plotly_chart(fig_cm, use_container_width=True)

    # Interpretation card
    best_model = max(ml_result["model_scores"],
                     key=lambda k: ml_result["model_scores"][k]["mean_accuracy"])
    best_acc   = ml_result["model_scores"][best_model]["mean_accuracy"]
    lift       = best_acc - baseline

    st.markdown(f"""
    <div class='card' style='border-left: 3px solid #00d9a3; margin-top:0.5rem;'>
      <div style='font-family:DM Mono,monospace;font-size:0.68rem;color:#00d9a3;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem;'>Interpretation</div>
      <div style='color:#e2e8f0;line-height:1.7;'>
        The best performing model (<strong style='color:#00d9a3;'>{best_model}</strong>) achieves 
        <strong style='color:#00d9a3;'>{best_acc:.1%} accuracy</strong> using only Reddit sentiment features,
        compared to a baseline of <strong style='color:#fbbf24;'>{baseline:.1%}</strong>
        (always predicting the majority class). 
        This represents a <strong>+{lift:.1%} lift</strong> over the baseline,
        suggesting Reddit sentiment carries <em>some</em> predictive signal for {sel_ml}.
        A proper trading strategy would need transaction costs, slippage, and out-of-sample testing.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NEWS VS SOCIAL MEDIA
# ══════════════════════════════════════════════════════════════════════════════
elif "News" in page:
    st.markdown("<div class='sec-title'>News Sentiment vs Reddit Sentiment</div>", unsafe_allow_html=True)
    st.caption("Compare institutional news sentiment (Alpha Vantage) against grassroots social sentiment (Reddit)")

    for ticker in tickers_sel:
        tdf = df[df["ticker"] == ticker].sort_values("date")
        if tdf.empty or "news_sentiment" not in tdf.columns:
            continue

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=[f"{ticker} — Reddit vs News Sentiment",
                                            "Price Return for Context"],
                            row_heights=[0.65, 0.35], vertical_spacing=0.06)

        color = TICKER_COLORS.get(ticker, "#888")
        fig.add_trace(go.Scatter(
            x=tdf["date"], y=tdf["weighted_avg_sentiment"],
            name="Reddit (Engagement-Weighted)", line=dict(color=color, width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=tdf["date"], y=tdf["news_sentiment"],
            name="Financial News", line=dict(color="#fbbf24", width=1.5, dash="dash"),
        ), row=1, col=1)
        fig.add_hline(y=0, row=1, col=1, line_dash="dot",
                      line_color="rgba(255,255,255,0.1)")

        ret_colors = ["#00d9a3" if r >= 0 else "#ff4d6d"
                      for r in tdf["daily_return"].fillna(0)]
        fig.add_trace(go.Bar(
            x=tdf["date"], y=tdf["daily_return"],
            name="Daily Return %", marker_color=ret_colors, opacity=0.6,
        ), row=2, col=1)

        fig.update_layout(**CHART_BG, height=400, showlegend=True, margin=_DEF_MARGIN)
        fig.update_yaxes(title_text="Sentiment", row=1, col=1, range=[-1.2, 1.2])
        fig.update_yaxes(title_text="Return %", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation between news and reddit sentiment
    st.markdown("<div class='sec-title'>Agreement Between News & Reddit</div>", unsafe_allow_html=True)
    agree_rows = []
    for ticker in tickers_sel:
        tdf = df[df["ticker"] == ticker].dropna(subset=["weighted_avg_sentiment","news_sentiment"])
        if len(tdf) < 10:
            continue
        from scipy.stats import pearsonr
        r, p = pearsonr(tdf["weighted_avg_sentiment"], tdf["news_sentiment"])
        agree_rows.append({"ticker": ticker, "correlation": round(r, 4), "p_value": round(p, 4),
                           "agree": "High" if r > 0.4 else "Moderate" if r > 0.2 else "Low"})
    if agree_rows:
        agree_df = pd.DataFrame(agree_rows)
        fig_agree = px.bar(agree_df, x="ticker", y="correlation",
                           color="ticker", color_discrete_map=TICKER_COLORS,
                           text="correlation",
                           labels={"correlation": "Pearson r (Reddit vs News)"})
        fig_agree.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_agree.update_layout(**CHART_BG, height=300, showlegend=False,
                                title_text="Correlation: Reddit vs News Sentiment per Ticker",
                                margin=_DEF_MARGIN)
        fig_agree.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        st.plotly_chart(fig_agree, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif "Deep Dive" in page:
    st.markdown("<div class='sec-title'>Single-Ticker Deep Dive</div>", unsafe_allow_html=True)

    sel = st.selectbox("Choose ticker", tickers_sel or ["TSLA"])
    tdf = df[df["ticker"] == sel].sort_values("date").copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Reddit Posts",
                  f"{int(tdf['post_count'].sum()):,}" if "post_count" in tdf else "N/A")
    with c2:
        avg_s = tdf["weighted_avg_sentiment"].mean()
        st.metric("Avg Sentiment", f"{avg_s:.3f}",
                  delta="Positive" if avg_s > 0 else "Negative")
    with c3:
        bulls = tdf["bullish_ratio"].mean() if "bullish_ratio" in tdf else 0
        st.metric("Avg Bullish Ratio", f"{bulls:.1%}")
    with c4:
        corr_df2 = cached_correlations(tdf)
        if not corr_df2.empty:
            r = corr_df2.iloc[0]["pearson_r_next_day"]
            p = corr_df2.iloc[0]["pearson_p_next_day"]
            st.metric("Sentiment→Return r", f"{r:.4f}",
                      delta=f"p={p:.4f}")

    # Rolling correlation over time (NOVELTY)
    st.markdown("<div class='sec-title'>Rolling 30-Day Correlation (Sentiment → Next-Day Return)</div>", unsafe_allow_html=True)
    st.caption("Shows how the sentiment-price relationship evolves over time — novel time-varying analysis")

    tdf["next_return"] = tdf["daily_return"].shift(-1)
    roll_corrs = []
    for i in range(30, len(tdf)):
        window = tdf.iloc[i-30:i][["weighted_avg_sentiment","next_return"]].dropna()
        if len(window) < 10:
            roll_corrs.append(np.nan)
            continue
        from scipy.stats import pearsonr as _pr
        try:
            r, _ = _pr(window["weighted_avg_sentiment"], window["next_return"])
            roll_corrs.append(r)
        except Exception:
            roll_corrs.append(np.nan)

    roll_dates = tdf["date"].iloc[30:].reset_index(drop=True)
    roll_df = pd.DataFrame({"date": roll_dates, "rolling_r": roll_corrs})

    fig_rc = go.Figure()
    pos_mask = roll_df["rolling_r"] >= 0
    neg_mask = roll_df["rolling_r"] < 0
    fig_rc.add_trace(go.Scatter(
        x=roll_df["date"], y=roll_df["rolling_r"].where(pos_mask),
        fill="tozeroy", name="Positive correlation",
        line=dict(color="#00d9a3", width=1.5),
        fillcolor="rgba(0,217,163,0.15)",
    ))
    fig_rc.add_trace(go.Scatter(
        x=roll_df["date"], y=roll_df["rolling_r"].where(neg_mask),
        fill="tozeroy", name="Negative correlation",
        line=dict(color="#ff4d6d", width=1.5),
        fillcolor="rgba(255,77,109,0.15)",
    ))
    fig_rc.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig_rc.update_layout(**CHART_BG, height=300,
                          yaxis_title="Pearson r (30-day rolling)",
                          title_text=f"{sel} — Time-Varying Sentiment-Price Correlation",
                          margin=_DEF_MARGIN)
    st.plotly_chart(fig_rc, use_container_width=True)

    # Bullish/Bearish ratio stacked area
    st.markdown("<div class='sec-title'>Bullish vs Bearish Post Ratio Over Time</div>", unsafe_allow_html=True)
    fig_bull = go.Figure()
    fig_bull.add_trace(go.Scatter(
        x=tdf["date"], y=tdf["bullish_ratio"], name="Bullish",
        fill="tozeroy", line=dict(color="#00d9a3", width=1.5),
        fillcolor="rgba(0,217,163,0.2)",
    ))
    fig_bull.add_trace(go.Scatter(
        x=tdf["date"], y=tdf["bearish_ratio"], name="Bearish",
        fill="tozeroy", line=dict(color="#ff4d6d", width=1.5),
        fillcolor="rgba(255,77,109,0.2)",
    ))
    fig_bull.add_hline(y=0.5, line_dash="dot",
                       line_color="rgba(255,255,255,0.15)")
    fig_bull.update_layout(**CHART_BG, height=280,
                            yaxis_title="Ratio", yaxis_tickformat=".0%",
                            title_text=f"{sel} — Community Sentiment Split",
                            margin=_DEF_MARGIN)
    st.plotly_chart(fig_bull, use_container_width=True)

    # Raw data table
    with st.expander("📋 View raw data for this ticker"):
        st.dataframe(
            tdf.sort_values("date", ascending=False).head(100),
            use_container_width=True
        )
        st.download_button(
            "⬇️ Download CSV",
            data=tdf.to_csv(index=False).encode(),
            file_name=f"{sel}_sentiment_data.csv",
            mime="text/csv",
        )

