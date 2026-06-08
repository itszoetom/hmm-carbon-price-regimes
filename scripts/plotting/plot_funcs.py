"""shared plotting helpers so every figure has one consistent style"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import params


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": params.FIGURE_DPI,
        "savefig.dpi": params.FIGURE_DPI,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 12,
        "legend.frameon": False,
    })


def new_figure(size=params.FIGURE_SIZE_WIDE):
    apply_style()
    fig, ax = plt.subplots(figsize=size)
    return fig, ax


def save(fig, name):
    params.FIGURES.mkdir(parents=True, exist_ok=True)
    path = params.FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return path


def transition_heatmap(matrix, labels, title):
    """Heatmap of a transition matrix with the probability annotated in each cell"""
    fig, ax = new_figure(size=params.FIGURE_SIZE_SQUARE)
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.5 else "black", fontsize=10)
    ax.set_xlabel("next state")
    ax.set_ylabel("current state")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="probability")
    return fig


def shade_phases(ax, last_date):
    """shades EU ETS trading phases on a time axis and label them"""
    last = pd.Timestamp(last_date)
    for i, (label, start, end) in enumerate(params.EU_ETS_PHASES):
        start = pd.Timestamp(start)
        end = last if end is None else pd.Timestamp(end)
        if start > last:
            continue
        end = min(end, last)
        color = "0.92" if i % 2 == 0 else "0.97"
        ax.axvspan(start, end, color=color, zorder=0)
        mid = start + (end - start) / 2
        ax.annotate(
            label, xy=(mid, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 2), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color="0.4",
        )
