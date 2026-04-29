"""
notebooks/exploration.py
========================
Interactive data exploration script.
Run cells in Jupyter or as a plain Python script.

Usage:  python notebooks/exploration.py
Or:     jupyter notebook (then open this file)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.etl_processing import generate_full_synthetic_dataset
from src.analysis import compute_correlations, lead_lag_analysis

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = generate_full_synthetic_dataset()
df["date"] = pd.to_datetime(df["date"])
print(f"Dataset shape: {df.shape}")
print(f"Tickers: {df['ticker'].unique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print()

# ── Basic statistics ──────────────────────────────────────────────────────────
print("=" * 50)
print("SENTIMENT STATISTICS BY TICKER")
print("=" * 50)
summary = df.groupby("ticker").agg(
    mean_sentiment=("weighted_avg_sentiment", "mean"),
    std_sentiment=("weighted_avg_sentiment", "std"),
    mean_daily_return=("daily_return", "mean"),
    std_daily_return=("daily_return", "std"),
    total_posts=("post_count", "sum"),
    avg_bullish=("bullish_ratio", "mean"),
).round(4)
print(summary)
print()

# ── Correlation analysis ──────────────────────────────────────────────────────
print("=" * 50)
print("CORRELATION: SENTIMENT → NEXT-DAY RETURN")
print("=" * 50)
corr = compute_correlations(df)
print(corr[["ticker","pearson_r_next_day","pearson_p_next_day","significant_5pct"]])
print()

# ── Lead-lag ──────────────────────────────────────────────────────────────────
print("=" * 50)
print("LEAD-LAG ANALYSIS")
print("=" * 50)
lag = lead_lag_analysis(df)
print(lag.pivot(index="ticker", columns="lag_days", values="pearson_r").round(4))
print()

# ── Plot correlation heatmap ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SentimentPulse — Exploration Plots", fontsize=14, fontweight="bold")

# Correlation bar chart
ax = axes[0]
colors = ["#00d9a3" if r > 0 else "#ff4d6d" for r in corr["pearson_r_next_day"]]
bars = ax.bar(corr["ticker"], corr["pearson_r_next_day"], color=colors, alpha=0.8)
ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_title("Pearson r: Sentiment → Next-Day Return")
ax.set_ylabel("Pearson r")
ax.set_facecolor("#0f1829")
fig.patch.set_facecolor("#070c18")
for spine in ax.spines.values():
    spine.set_edgecolor("rgba(255,255,255,0.1)")
ax.tick_params(colors="white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")

# Lead-lag heatmap
ax2 = axes[1]
pivot = lag.pivot(index="ticker", columns="lag_days", values="pearson_r")
sns.heatmap(pivot, ax=ax2, cmap="RdYlGn", center=0, annot=True, fmt=".3f",
            linewidths=0.5, cbar_kws={"label": "Pearson r"})
ax2.set_title("Lead-Lag Correlation Matrix")
ax2.set_xlabel("Lag (days)")
ax2.title.set_color("white")

plt.tight_layout()
plt.savefig("data/processed/exploration_plots.png", dpi=150,
            facecolor="#070c18", bbox_inches="tight")
print("Saved exploration_plots.png to data/processed/")
plt.show()
