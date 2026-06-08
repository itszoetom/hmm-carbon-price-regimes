"""reading, cleaning, and preparing functions for the EUA price data

1. read_raw       load the raw download exactly as provided
2. clean_prices   sort by date, restrict the window, drop implausible prices
3. add_log_returns   compute daily log-returns r_t = log(P_t) - log(P_{t-1})
4. validate       produce a plain-language summary so data problems are visible
5. prepare        run the whole pipeline and write the processed file
"""

import numpy as np
import pandas as pd

import params


def read_raw(path=params.RAW_PATH):
    raw = pd.read_csv(path)
    date_col = params.RAW_COLUMNS["date"]
    close_col = params.RAW_COLUMNS["close"]

    missing = [c for c in (date_col, close_col) if c not in raw.columns]
    if missing:
        raise KeyError(f"missing expected columns {missing}, got {list(raw.columns)}")

    frame = pd.DataFrame()
    frame["date"] = pd.to_datetime(raw[date_col], format=params.RAW_DATE_FORMAT)
    # strip thousands separators before converting to float
    price_text = raw[close_col].astype(str).str.replace(",", "", regex=False)
    frame["price"] = pd.to_numeric(price_text, errors="coerce")
    return frame


def clean_prices(frame):
    out = frame.dropna(subset=["price"]).copy()
    out = out.sort_values("date").drop_duplicates(subset="date", keep="last")

    start = pd.Timestamp(params.START_DATE)
    out = out[out["date"] >= start]
    if params.END_DATE is not None:
        out = out[out["date"] <= pd.Timestamp(params.END_DATE)]

    in_bounds = (out["price"] >= params.PRICE_MIN) & (out["price"] <= params.PRICE_MAX)
    out = out[in_bounds]

    return out.reset_index(drop=True)


def add_log_returns(frame):
    out = frame.copy()
    out["log_return"] = np.log(out["price"]).diff()
    out = out.dropna(subset=["log_return"]).reset_index(drop=True)
    return out


def validate(frame):
    """ a dictionary of data-quality checks for the prepared series"""
    gaps = frame["date"].diff().dt.days.dropna()
    flagged = frame[frame["log_return"].abs() > params.RETURN_FLAG_THRESHOLD]
    # unchanged prices give a zero return. a cluster of these can pull a hidden
    # state toward a tiny-variance fit so I count them instead of dropping them
    n_stale = int((frame["log_return"] == 0).sum())

    summary = {
        "n_observations": len(frame),
        "first_date": frame["date"].min(),
        "last_date": frame["date"].max(),
        "min_price": frame["price"].min(),
        "max_price": frame["price"].max(),
        "mean_log_return": frame["log_return"].mean(),
        "std_log_return": frame["log_return"].std(),
        # fat tails and negative skew are why one Gaussian isn't enough
        "skew_log_return": frame["log_return"].skew(),
        "excess_kurtosis_log_return": frame["log_return"].kurtosis(),
        "n_stale_returns": n_stale,
        "pct_stale_returns": round(100.0 * n_stale / len(frame), 2),
        "largest_calendar_gap_days": int(gaps.max()) if len(gaps) else 0,
        "n_flagged_large_returns": len(flagged),
        "flagged_dates": list(flagged["date"].dt.strftime("%Y-%m-%d")),
    }
    return summary


def prepare(save=True):
    frame = add_log_returns(clean_prices(read_raw()))
    if save:
        params.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        frame.to_csv(params.PROCESSED_PATH, index=False)
    return frame


def load_processed(path=params.PROCESSED_PATH):
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame


def _print_summary(summary):
    print("data preparation summary")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    prepared = prepare(save=True)
    _print_summary(validate(prepared))
    print(f"saved processed series to {params.PROCESSED_PATH}")
