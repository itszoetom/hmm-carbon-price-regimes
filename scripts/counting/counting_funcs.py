"""Empirical counting baseline for the regime transition matrix"""

import numpy as np
import pandas as pd

import params
from scripts.plotting import plot_funcs

TRADING_DAYS_PER_YEAR = 252

# regimes ordered calmest to most turbulent so the REGIME_COLORS match severity
REGIME_LABELS = ["low vol", "medium vol", "high vol"]


def label_regimes_by_volatility(frame, window=params.VOL_WINDOW,
                                low_q=params.VOL_LOW_QUANTILE,
                                high_q=params.VOL_HIGH_QUANTILE):
    """Label each day low, medium, or high volatility by rolling-volatility terciles"""
    out = frame.copy()
    vol = out["log_return"].rolling(window).std()
    out["rolling_vol"] = vol

    cut_low = vol.quantile(low_q)
    cut_high = vol.quantile(high_q)

    regime = np.select([vol <= cut_low, vol >= cut_high], [0, 2], default=1).astype(float)
    regime[vol.isna().to_numpy()] = np.nan
    out["regime"] = regime

    out = out.dropna(subset=["regime"]).reset_index(drop=True)
    out["regime"] = out["regime"].astype(int)
    return out, (cut_low, cut_high)


def count_transitions(labels, n_states=len(REGIME_LABELS)):
    """Count transitions between consecutive days. Returns an integer count matrix"""
    counts = np.zeros((n_states, n_states), dtype=int)
    arr = np.asarray(labels)
    for current, nxt in zip(arr[:-1], arr[1:]):
        counts[current, nxt] += 1
    return counts


def transition_matrix(counts):
    """Row-normalize a count matrix into a stochastic transition matrix"""
    row_sums = counts.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    return counts / safe


def mean_run_length(labels, state):
    """Average number of consecutive days spent in a state before leaving it"""
    runs = []
    current = 0
    for value in labels:
        if value == state:
            current += 1
        elif current > 0:
            runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return float(np.mean(runs)) if runs else 0.0


def regime_summary(frame, labels=REGIME_LABELS):
    """Per-regime table: share of days, return moments, price level, mean duration"""
    rows = []
    states = frame["regime"].to_numpy()
    for state, name in enumerate(labels):
        sub = frame[frame["regime"] == state]
        rows.append({
            "regime": name,
            "days": len(sub),
            "share": round(len(sub) / len(frame), 3),
            "mean daily return": round(sub["log_return"].mean(), 5),
            "annualized vol": round(sub["log_return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR), 3),
            "mean price (EUR)": round(sub["price"].mean(), 2),
            "mean duration (days)": round(mean_run_length(states, state), 1),
        })
    return pd.DataFrame(rows)


def transition_frame(matrix, labels=REGIME_LABELS):
    return pd.DataFrame(matrix, index=labels, columns=labels)


def plot_regime_timeline(frame, labels=REGIME_LABELS):
    """Figure: the price path with each day colored by its volatility regime"""
    fig, ax = plot_funcs.new_figure()
    ax.plot(frame["date"], frame["price"], color="0.8", linewidth=0.6, zorder=1)
    for state, name in enumerate(labels):
        sub = frame[frame["regime"] == state]
        ax.scatter(sub["date"], sub["price"], s=4, color=params.REGIME_COLORS[state],
                   label=name, zorder=2)
    ax.set_ylabel("EUA price (EUR per tonne)")
    ax.set_title("Volatility regimes from the counting baseline")
    ax.legend(markerscale=3, loc="upper left")
    return fig
