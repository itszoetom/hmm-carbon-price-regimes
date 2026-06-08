import numpy as np
import pandas as pd

import params
from scripts.plotting import plot_funcs

TRADING_DAYS_PER_YEAR = 252

def summary_table(frame):
    r = frame["log_return"]
    n_stale = int((r == 0).sum())
    stats = {
        "observations": len(frame),
        "first date": frame["date"].min().strftime("%Y-%m-%d"),
        "last date": frame["date"].max().strftime("%Y-%m-%d"),
        "min price (EUR)": round(frame["price"].min(), 2),
        "max price (EUR)": round(frame["price"].max(), 2),
        "mean daily log-return": round(r.mean(), 5),
        "std daily log-return": round(r.std(), 5),
        "annualized volatility": round(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR), 3),
        "skew": round(r.skew(), 3),
        "excess kurtosis": round(r.kurtosis(), 3),
        "min daily log-return": round(r.min(), 4),
        "max daily log-return": round(r.max(), 4),
        "stale days (zero return)": n_stale,
        "stale days percent": round(100.0 * n_stale / len(frame), 2),
    }
    return pd.DataFrame({"statistic": list(stats.keys()), "value": list(stats.values())})


def rolling_volatility(frame, window=params.VOL_WINDOW):
    vol = frame["log_return"].rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return pd.Series(vol.values, index=frame["date"])


def plot_price_with_phases(frame):
    """EUA price figure over the full sample with trading phases shaded"""
    fig, ax = plot_funcs.new_figure()
    plot_funcs.shade_phases(ax, frame["date"].max())
    ax.plot(frame["date"], frame["price"], color="#1f1f1f", linewidth=0.8)
    ax.set_ylabel("EUA price (EUR per tonne)")
    ax.set_title("EU carbon allowance price, front December futures", pad=18)
    return fig


def plot_log_returns(frame):
    """daily log-returns over time, which makes volatility clustering visible"""
    fig, ax = plot_funcs.new_figure()
    ax.plot(frame["date"], frame["log_return"], color="#4C72B0", linewidth=0.5)
    ax.axhline(0, color="0.5", linewidth=0.6)
    ax.set_ylabel("daily log-return")
    ax.set_title("EUA daily log-returns")
    return fig


def plot_rolling_volatility(frame, window=params.VOL_WINDOW):
    """rolling annualized volatility, the visual motivation for regimes"""
    vol = rolling_volatility(frame, window)
    fig, ax = plot_funcs.new_figure()
    plot_funcs.shade_phases(ax, frame["date"].max())
    ax.plot(vol.index, vol.values, color="#C44E52", linewidth=0.9)
    ax.set_ylabel("annualized volatility")
    ax.set_title(f"Rolling {window}-day volatility of EUA log-returns", pad=18)
    return fig


def plot_return_hist_vs_normal(frame):
    """return histogram against a single fitted Normal"""
    r = frame["log_return"]
    mu, sigma = r.mean(), r.std()
    grid = np.linspace(r.min(), r.max(), 400)
    normal_pdf = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    fig, ax = plot_funcs.new_figure(size=params.FIGURE_SIZE_SQUARE)
    ax.hist(r, bins=120, density=True, color="#4C72B0", alpha=0.6, label="observed returns")
    ax.plot(grid, normal_pdf, color="#C44E52", linewidth=1.5, label="single Normal fit")
    ax.set_xlabel("daily log-return")
    ax.set_ylabel("density")
    ax.set_title(f"Returns versus a single Normal (excess kurtosis {round(r.kurtosis(), 1)})")
    ax.legend()
    return fig
