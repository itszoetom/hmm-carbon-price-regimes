"""Viterbi decoding of the most likely hidden regime sequence

Given the fitted HMM, Viterbi finds the single most probable path of latent
regimes behind the observed returns"""
import numpy as np
import pandas as pd

import params
from scripts.plotting import plot_funcs

def decode(frame, saved):
    """align the Viterbi-decoded regime to each date, in calm-to-volatile order"""
    model = saved["model"]
    order = np.asarray(saved["order"])
    X = frame["log_return"].to_numpy().reshape(-1, 1)
    raw_states = model.predict(X)

    # order maps ordered position to raw state so its inverse maps raw to ordered
    raw_to_ordered = np.argsort(order)
    out = frame.copy()
    out["regime"] = raw_to_ordered[raw_states]
    return out


def occupancy_table(decoded, labels):
    counts = decoded["regime"].value_counts().reindex(range(len(labels)), fill_value=0)
    return pd.DataFrame({
        "regime": labels,
        "decoded days": counts.to_numpy(),
        "decoded share": (counts.to_numpy() / len(decoded)).round(3),
    })


def plot_decoded_timeline(decoded, labels):
    """price path colored by the decoded regime each day figure"""
    fig, ax = plot_funcs.new_figure()
    ax.plot(decoded["date"], decoded["price"], color="0.8", linewidth=0.6, zorder=1)
    for state, name in enumerate(labels):
        sub = decoded[decoded["regime"] == state]
        ax.scatter(sub["date"], sub["price"], s=5, color=params.REGIME_COLORS[state],
                   label=name, zorder=2)
    ax.set_ylabel("EUA price (EUR per tonne)")
    ax.set_title("EUA price colored by the Viterbi-decoded regime")
    ax.legend(markerscale=3, loc="upper left")
    return fig
