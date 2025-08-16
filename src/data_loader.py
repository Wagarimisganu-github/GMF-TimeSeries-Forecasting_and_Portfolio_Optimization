# data_loader.py
import yfinance as yf
import pandas as pd

_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns from yfinance. Robustly detect which level
    is field vs ticker and normalize to '<Field>_<Ticker>' (e.g., 'Adj Close_TSLA').
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    new_cols = []
    for col in df.columns:
        # Common case: 2-level Index (ticker, field) or (field, ticker)
        if len(col) == 2:
            a, b = map(str, col)
            if a in _FIELDS and b not in _FIELDS:
                field, ticker = a, b
            elif b in _FIELDS and a not in _FIELDS:
                field, ticker = b, a
            else:
                # Fallback: join as-is
                new_cols.append("_".join(map(str, col)).strip())
                continue
            new_cols.append(f"{field}_{ticker}")
        else:
            new_cols.append("_".join(map(str, col)).strip())

    out = df.copy()
    out.columns = new_cols
    return out


def _normalize_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If columns look like '<Ticker>_<Field>' (e.g., 'TSLA_Adj Close'),
    rename to '<Field>_<Ticker>' for consistency.
    Leaves already-normalized names untouched.
    """
    def normalize_name(name: str) -> str:
        # Already normalized?
        for f in _FIELDS:
            if name.startswith(f + "_"):
                return name
        # Ends with a field? e.g., 'TSLA_Adj Close'
        for f in _FIELDS:
            if name.endswith("_" + f):
                ticker = name[: -(len(f) + 1)]
                return f"{f}_{ticker}"
        return name

    out = df.copy()
    out.columns = [normalize_name(str(c)) for c in out.columns]
    return out


def fetch_data(tickers, start_date, end_date, auto_adjust: bool = False, progress: bool = False) -> pd.DataFrame:
    """
    Extract historical OHLCV data using yfinance for the given tickers.
    Ensures:
      - DatetimeIndex, deduplicated, sorted
      - Columns flattened and normalized to '<Field>_<Ticker>'
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=progress,
        group_by="ticker",  # handle multi-ticker structure
        threads=True,
    )

    # If no data returned, return empty DF
    if isinstance(data, pd.DataFrame) and data.empty:
        return data

    # Flatten and normalize column names
    data = _flatten_columns(data)
    data = _normalize_flat_columns(data)

    # Standardize index: datetime, drop duplicates, sort
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep="first")].sort_index()

    return data


if __name__ == "__main__":
    print("✅ data_loader.py executed successfully")