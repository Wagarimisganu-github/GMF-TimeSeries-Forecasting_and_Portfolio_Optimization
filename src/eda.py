# eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
from typing import List, Dict, Tuple

IMAGES_DIR = Path("../images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ----- Data cleaning and understanding -----
def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic audit: dtypes and missing values per column."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "non_null": df.notna().sum()
    })


def handle_missing_values(df: pd.DataFrame, method: str = "ffill_bfill") -> pd.DataFrame:
    """
    Handle missing values:
      - 'ffill_bfill': forward-fill then backfill
      - 'interpolate': time-based interpolation (then ffill/bfill)
      - 'drop': drop rows with any missing values
    """
    if method == "ffill_bfill":
        return df.ffill().bfill()
    elif method == "interpolate":
        return df.interpolate(method="time").ffill().bfill()
    elif method == "drop":
        return df.dropna()
    else:
        raise ValueError("method must be one of {'ffill_bfill','interpolate','drop'}")


def get_price_columns(df: pd.DataFrame, prefer_adjusted: bool = True) -> List[str]:
    """
    Return price columns, preferring 'Adj Close_' if available, else 'Close_'.
    Works with normalized columns produced by data_loader.fetch_data.
    """
    adj_cols = [c for c in df.columns if str(c).startswith("Adj Close_")]
    close_cols = [c for c in df.columns if str(c).startswith("Close_")]
    if prefer_adjusted and adj_cols:
        return adj_cols
    return close_cols


# ----- Returns & scaling -----
def calculate_daily_returns(df: pd.DataFrame, prefer_adjusted: bool = True) -> pd.DataFrame:
    """Daily percentage changes from closing prices (prefer adjusted)."""
    price_cols = get_price_columns(df, prefer_adjusted=prefer_adjusted)
    if not price_cols:
        # Return empty DF to avoid crashes upstream; caller can check emptiness
        return pd.DataFrame(index=df.index)
    returns = df[price_cols].pct_change()
    return returns.dropna(how="all")


def zscore_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score scale columns (mean 0, std 1)."""
    if df.empty:
        return df.copy()
    return (df - df.mean()) / df.std(ddof=0)


# ----- Plotting -----
def _resolve_price_col(df: pd.DataFrame, ticker: str, prefer_adjusted: bool = True) -> str:
    # Normalized names: '<Field>_<Ticker>'
    preferred = f"Adj Close_{ticker}" if prefer_adjusted else f"Close_{ticker}"
    if preferred in df.columns:
        return preferred
    # Fallback to other price type
    alt = f"Close_{ticker}" if prefer_adjusted else f"Adj Close_{ticker}"
    if alt in df.columns:
        return alt
    raise KeyError(f"No price column found for ticker '{ticker}'.")


def _resolve_return_col(returns_df: pd.DataFrame, ticker: str) -> str:
    for c in returns_df.columns:
        if c.endswith(f"_{ticker}"):
            return c
    raise KeyError(f"No returns column found for ticker '{ticker}'.")


def plot_prices(df: pd.DataFrame, ticker: str, output_dir=IMAGES_DIR, prefer_adjusted: bool = True):
    """Visualize closing price over time."""
    col = _resolve_price_col(df, ticker, prefer_adjusted=prefer_adjusted)
    plt.figure(figsize=(14, 7))
    df[col].plot()
    plt.title(f"{ticker} Closing Price")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.grid(True)
    plt.savefig(Path(output_dir) / f"{ticker}_price.png", bbox_inches="tight")
    plt.show(); plt.close()


def plot_daily_returns(returns_df: pd.DataFrame, ticker: str, output_dir=IMAGES_DIR):
    """Plot daily percentage change to observe volatility."""
    col = _resolve_return_col(returns_df, ticker)
    plt.figure(figsize=(14, 7))
    returns_df[col].plot()
    plt.title(f"{ticker} Daily Returns")
    plt.xlabel("Date"); plt.ylabel("Daily Return"); plt.grid(True)
    plt.savefig(Path(output_dir) / f"{ticker}_daily_returns.png", bbox_inches="tight")
    plt.show(); plt.close()


def plot_rolling_stats(returns_df: pd.DataFrame, ticker: str, windows=(21, 63, 252), output_dir=IMAGES_DIR):
    """Analyze volatility via rolling means and standard deviations."""
    col = _resolve_return_col(returns_df, ticker)
    plt.figure(figsize=(14, 7))
    returns_df[col].plot(alpha=0.35, label="Daily returns", color="gray")
    for w in windows:
        returns_df[col].rolling(w).mean().plot(label=f"Rolling mean {w}d")
        returns_df[col].rolling(w).std().plot(label=f"Rolling std {w}d")
    plt.title(f"{ticker} Rolling Mean and Std (Volatility)")
    plt.xlabel("Date"); plt.legend(); plt.grid(True)
    plt.savefig(Path(output_dir) / f"{ticker}_rolling_stats.png", bbox_inches="tight")
    plt.show(); plt.close()


# ----- Outliers & extreme days -----
def flag_outliers_zscore(returns_df: pd.DataFrame, z_thresh: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flag significant anomalies using z-scores: mask True where |z| > z_thresh."""
    if returns_df.empty:
        return returns_df.copy(), returns_df.copy()
    z = (returns_df - returns_df.mean()) / returns_df.std(ddof=0)
    mask = z.abs() > z_thresh
    return mask, z


def list_extreme_days(returns_df: pd.DataFrame, q: float = 0.01) -> Dict[str, Dict[str, list]]:
    """Dates for unusually high/low returns using top/bottom quantiles."""
    if returns_df.empty:
        return {"lows": {}, "highs": {}}
    lows = returns_df.le(returns_df.quantile(q))
    highs = returns_df.ge(returns_df.quantile(1 - q))
    return {
        "lows": {c: returns_df.index[lows[c]].tolist() for c in returns_df.columns},
        "highs": {c: returns_df.index[highs[c]].tolist() for c in returns_df.columns},
    }


# ----- Stationarity (seasonality & trends) -----
def analyze_stationarity(series: pd.Series, label: str):
    """ADF test on prices/returns with concise implication for modeling."""
    s = series.dropna()
    if len(s) < 30:
        print(f"\n⚠️ Not enough data points for ADF test on {label}.")
        return
    stat, pvalue, *_ = adfuller(s)
    print(f"\n📊 ADF for {label} | ADF={stat:.4f}, p={pvalue:.4f}")
    if pvalue < 0.05:
        print("✅ Stationary: differencing likely not required for this series.")
    else:
        print("⚠️ Non-stationary: differencing (the 'I' in ARIMA) is required before modeling.")