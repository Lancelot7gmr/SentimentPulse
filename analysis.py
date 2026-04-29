"""
analysis.py
===========
Statistical analysis and machine learning on the master dataset:
  - Pearson & Spearman correlation (sentiment vs price)
  - Lead-lag analysis (does sentiment predict tomorrow's price?)
  - Granger causality test (formal statistical causality)
  - Random Forest classification (predict price direction from sentiment)
  - Sector/ticker comparison

Run: python src/analysis.py
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CORRELATION ANALYSIS
# ─────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson and Spearman correlations between each sentiment signal
    and next-day stock return for every ticker.
    
    WHY TWO CORRELATION MEASURES?
    - Pearson: assumes linear relationship and normally distributed data.
      Good for measuring how much sentiment and returns move together linearly.
    - Spearman: rank-based, makes no distribution assumptions.
      Robust to outliers — better for volatile stocks like GME.
    
    WHY NEXT-DAY RETURN?
    If sentiment today predicts tomorrow's price, it could be used in a trading
    strategy. We test this key hypothesis here.
    """
    results = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").dropna(
            subset=["weighted_avg_sentiment", "daily_return"])

        # Shift price return by -1 to get NEXT day's return
        next_return = grp["daily_return"].shift(-1)

        valid = pd.DataFrame({
            "sentiment": grp["weighted_avg_sentiment"],
            "next_return": next_return,
            "news_sentiment": grp["news_sentiment"] if "news_sentiment" in grp else np.nan,
        }).dropna()

        if len(valid) < 10:
            continue

        # Pearson correlation
        r_p, p_p = stats.pearsonr(valid["sentiment"], valid["next_return"])
        # Spearman correlation
        r_s, p_s = stats.spearmanr(valid["sentiment"], valid["next_return"])
        # Same-day Pearson
        same_valid = grp[["weighted_avg_sentiment", "daily_return"]].dropna()
        r_same, p_same = stats.pearsonr(
            same_valid["weighted_avg_sentiment"], same_valid["daily_return"]
        ) if len(same_valid) > 5 else (np.nan, np.nan)

        results.append({
            "ticker": ticker,
            "n_observations": len(valid),
            "pearson_r_next_day":   round(r_p, 4),
            "pearson_p_next_day":   round(p_p, 4),
            "spearman_r_next_day":  round(r_s, 4),
            "spearman_p_next_day":  round(p_s, 4),
            "pearson_r_same_day":   round(r_same, 4) if not np.isnan(r_same) else None,
            "pearson_p_same_day":   round(p_same, 4) if not np.isnan(p_same) else None,
            "significant_5pct":     p_p < 0.05,
            "significant_1pct":     p_p < 0.01,
        })

    result_df = pd.DataFrame(results)
    log.info("Correlation analysis complete:\n%s", result_df.to_string(index=False))
    return result_df


# ─────────────────────────────────────────────
# LEAD-LAG ANALYSIS
# ─────────────────────────────────────────────

def lead_lag_analysis(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    Computes correlation between sentiment at time t and stock return at time t+lag
    for lags 0 through max_lag days.
    
    NOVELTY: This reveals WHEN sentiment has the strongest relationship with price.
    - lag=0: same-day relationship
    - lag=1: does today's Reddit buzz predict tomorrow's price?
    - lag=2: two-day leading indicator?
    
    A positive correlation at lag=1 but not lag=0 suggests sentiment LEADS price
    (potentially exploitable for trading).
    A positive correlation at lag=0 but not lag=1 suggests price DRIVES sentiment
    (people post more when markets move).
    """
    results = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        sent = grp["weighted_avg_sentiment"].dropna()

        for lag in range(max_lag + 1):
            ret = grp["daily_return"].shift(-lag)
            valid = pd.DataFrame({"s": sent, "r": ret}).dropna()
            if len(valid) < 10:
                continue
            r, p = stats.pearsonr(valid["s"], valid["r"])
            results.append({
                "ticker": ticker,
                "lag_days": lag,
                "pearson_r": round(r, 4),
                "p_value": round(p, 4),
                "significant": p < 0.05,
            })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# GRANGER CAUSALITY TEST
# ─────────────────────────────────────────────

def granger_causality_analysis(df: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """
    Performs the Granger Causality test to formally test whether sentiment
    'Granger-causes' stock returns.
    
    WHAT IS GRANGER CAUSALITY?
    X Granger-causes Y if knowing past values of X improves our forecast of Y
    compared to using only past values of Y. It is NOT true causality, but it
    is a rigorous statistical test of predictive power.
    
    We first test for stationarity (ADF test) — both time series must be stationary
    (no trend) for Granger causality to be valid.
    """
    results = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").dropna(
            subset=["weighted_avg_sentiment", "daily_return"])

        if len(grp) < 30:
            continue

        sent = grp["weighted_avg_sentiment"].values
        ret  = grp["daily_return"].values

        # Stationarity check (ADF test)
        adf_sent = adfuller(sent, autolag="AIC")
        adf_ret  = adfuller(ret,  autolag="AIC")
        sent_stationary = adf_sent[1] < 0.05   # p-value < 0.05 means stationary
        ret_stationary  = adf_ret[1]  < 0.05

        try:
            data = np.column_stack([ret, sent])
            gc_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            for lag, tests in gc_result.items():
                f_stat = tests[0]["ssr_ftest"][0]
                p_val  = tests[0]["ssr_ftest"][1]
                results.append({
                    "ticker": ticker,
                    "lag": lag,
                    "f_statistic": round(f_stat, 4),
                    "p_value": round(p_val, 4),
                    "granger_significant": p_val < 0.05,
                    "sent_stationary": sent_stationary,
                    "ret_stationary":  ret_stationary,
                })
        except Exception as e:
            log.warning("Granger test failed for %s: %s", ticker, e)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# MACHINE LEARNING: PREDICT PRICE DIRECTION
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "weighted_avg_sentiment",
    "sentiment_lag_1d",
    "sentiment_lag_2d",
    "sentiment_lag_3d",
    "sentiment_ma7",
    "bullish_ratio",
    "bearish_ratio",
    "post_count",
    "sentiment_std",
    "news_sentiment",
    "rolling_vol_5d",
]

def train_direction_classifier(df: pd.DataFrame, ticker: str) -> dict:
    """
    Trains and evaluates a Random Forest classifier to predict whether
    a stock's price will go UP or DOWN the next day, using only sentiment
    features (no price history used as input — a fair test of sentiment's value).
    
    NOVELTY: We use TimeSeriesSplit for cross-validation, which respects the
    temporal ordering of data (you cannot train on future data to predict the past).
    Regular k-fold cross-validation would leak future information — TimeSeriesSplit
    is the correct method for any time-series prediction problem.
    
    Models tested:
      1. Random Forest  : ensemble of decision trees, robust to outliers
      2. Logistic Regression : linear baseline
      3. Gradient Boosting : typically best performer on tabular data
    
    Returns accuracy scores, feature importances, and confusion matrix.
    """
    grp = df[df["ticker"] == ticker].sort_values("date").copy()
    grp["target"] = grp["price_direction"].shift(-1)   # predict NEXT day

    # Use only rows where we have both sentiment and a valid target
    feat_cols = [c for c in FEATURE_COLS if c in grp.columns]
    model_df  = grp[feat_cols + ["target", "date"]].dropna()

    if len(model_df) < 50:
        log.warning("Insufficient data for %s classifier (%d rows)", ticker, len(model_df))
        return {}

    X = model_df[feat_cols].values
    y = (model_df["target"] > 0).astype(int).values   # 1=up, 0=down

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        "Random Forest":    RandomForestClassifier(n_estimators=200, max_depth=5,
                                                    random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100,
                                                         max_depth=3, random_state=42),
    }

    scores = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_scaled, y, cv=tscv,
                                    scoring="accuracy")
        scores[name] = {
            "mean_accuracy":  round(cv_scores.mean(), 4),
            "std_accuracy":   round(cv_scores.std(), 4),
            "cv_scores":      cv_scores.tolist(),
        }

    # Fit best model on all data for feature importance
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X_scaled, y)
    importances = dict(zip(feat_cols, rf.feature_importances_.round(4)))

    # Full prediction for confusion matrix
    y_pred = rf.predict(X_scaled)

    result = {
        "ticker": ticker,
        "n_samples": len(model_df),
        "features_used": feat_cols,
        "model_scores": scores,
        "feature_importances": importances,
        "baseline_accuracy": round(max(y.mean(), 1 - y.mean()), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "date_range": [str(model_df["date"].min()), str(model_df["date"].max())],
    }

    log.info(
        "Classifier for %s — RF accuracy: %.3f (baseline: %.3f)",
        ticker,
        scores["Random Forest"]["mean_accuracy"],
        result["baseline_accuracy"],
    )
    return result


def run_ml_analysis(df: pd.DataFrame) -> dict:
    """Runs classifier for all tickers, returns results dict."""
    results = {}
    for ticker in df["ticker"].unique():
        results[ticker] = train_direction_classifier(df, ticker)
    return results


# ─────────────────────────────────────────────
# SENTIMENT REGIME ANALYSIS
# ─────────────────────────────────────────────

def compute_sentiment_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies each day into a sentiment regime:
      Very Bullish / Bullish / Neutral / Bearish / Very Bearish
    
    Then computes average stock return in the FOLLOWING day for each regime.
    This shows which sentiment state has historically been most predictive.
    
    NOVELTY: Regime-based analysis reveals non-linear effects that a simple
    correlation coefficient misses. For example, extreme bearish sentiment
    might actually precede rebounds (contrarian indicator).
    """
    df = df.copy()
    df["sentiment_regime"] = pd.cut(
        df["weighted_avg_sentiment"],
        bins=[-1.01, -0.3, -0.1, 0.1, 0.3, 1.01],
        labels=["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"]
    )

    df["next_day_return"] = df.groupby("ticker")["daily_return"].shift(-1)

    regime_stats = df.groupby(["ticker", "sentiment_regime"], observed=True).agg(
        count=("next_day_return", "count"),
        mean_next_return=("next_day_return", "mean"),
        positive_rate=("price_direction", lambda x: (x.shift(-1) > 0).mean()),
        avg_sentiment=("weighted_avg_sentiment", "mean"),
    ).reset_index()

    return regime_stats


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_full_analysis(df: pd.DataFrame) -> dict:
    """Runs all analyses and returns a dictionary of result DataFrames."""
    log.info("Running full statistical analysis suite...")
    return {
        "correlations":    compute_correlations(df),
        "lead_lag":        lead_lag_analysis(df),
        "granger":         granger_causality_analysis(df),
        "ml_results":      run_ml_analysis(df),
        "regimes":         compute_sentiment_regimes(df),
    }


if __name__ == "__main__":
    # Quick test with synthetic data
    from etl_processing import generate_full_synthetic_dataset
    df = generate_full_synthetic_dataset()
    results = run_full_analysis(df)
    print("\nCorrelation Results:")
    print(results["correlations"])
    print("\nLead-Lag Results:")
    print(results["lead_lag"].head(20))
