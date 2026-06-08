"""Baum-Welch (EM) estimation of a Gaussian hidden Markov model"""

import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import params
from scripts.plotting import plot_funcs

TRADING_DAYS_PER_YEAR = 252


def returns_matrix(frame):
    return frame["log_return"].to_numpy().reshape(-1, 1)


def n_free_parameters(n_states):
    """Number of free parameters of a Gaussian HMM with one-dimensional emissions"""
    return n_states * (n_states - 1) + (n_states - 1) + 2 * n_states


def fit_hmm(X, n_states, n_inits=params.N_HMM_INITS, max_iter=params.HMM_MAX_ITER,
            seed=params.RANDOM_SEED):
    """Fit a Gaussian HMM from several random starts and keep the best fit"""
    best_model = None
    best_score = -np.inf
    for offset in range(n_inits):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=max_iter,
            random_state=seed + offset,
        )
        with warnings.catch_warnings():
            # Some random starts fail to converge
            warnings.simplefilter("ignore")
            try:
                model.fit(X)
                score = model.score(X)
            except (ValueError, FloatingPointError):
                continue
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError(f"no Gaussian HMM with {n_states} states converged")
    return best_model, best_score


def model_selection(X, grid=params.N_REGIMES_GRID):
    """Fit one HMM per candidate state count and score by log-likelihood, AIC, BIC"""
    n_obs = len(X)
    rows = []
    models = {}
    for n_states in grid:
        model, log_likelihood = fit_hmm(X, n_states)
        k = n_free_parameters(n_states)
        aic = -2.0 * log_likelihood + 2.0 * k
        bic = -2.0 * log_likelihood + k * np.log(n_obs)
        rows.append({
            "n_states": n_states,
            "log_likelihood": round(log_likelihood, 1),
            "n_parameters": k,
            "AIC": round(aic, 1),
            "BIC": round(bic, 1),
        })
        models[n_states] = model
    return pd.DataFrame(rows), models


def select_best(table):
    """Return the state count with the lowest BIC"""
    return int(table.loc[table["BIC"].idxmin(), "n_states"])


def state_order(model):
    variances = np.asarray(model.covars_).reshape(model.n_components, -1)[:, 0]
    return np.argsort(variances)


def ordered_parameters(model):
    """Returns a dict with the transition matrix, initial distribution, emission
    means and standard deviations, all in the calm-to-volatile state order"""
    order = state_order(model)
    means = model.means_.ravel()[order]
    variances = np.asarray(model.covars_).reshape(model.n_components, -1)[:, 0][order]
    transmat = model.transmat_[np.ix_(order, order)]
    startprob = model.startprob_[order]
    return {
        "order": order,
        "transmat": transmat,
        "startprob": startprob,
        "means": means,
        "stds": np.sqrt(variances),
    }


def state_labels(n_states):
    """Readable labels for an n-state model, ordered calm to volatile"""
    if n_states == 3:
        return ["normal", "stressed", "crashed"]
    return [f"state {i + 1}" for i in range(n_states)]


def emission_table(ordered, weights, labels):
    """Per-state emission table: mean, volatility, and long-run weight

    weights = the stationary distribution, giving the long-run share of time in
    each regime
    """
    rows = []
    for i, name in enumerate(labels):
        rows.append({
            "regime": name,
            "mean daily return": round(float(ordered["means"][i]), 5),
            "daily volatility": round(float(ordered["stds"][i]), 5),
            "annualized volatility": round(float(ordered["stds"][i] * np.sqrt(TRADING_DAYS_PER_YEAR)), 3),
            "stationary weight": round(float(weights[i]), 3),
        })
    return pd.DataFrame(rows)


def _normal_pdf(grid, mean, std):
    """Gaussian density on a grid"""
    return np.exp(-0.5 * ((grid - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def plot_model_selection(table):
    """Figure: AIC and BIC against the number of states, with the BIC choice marked"""
    fig, ax = plot_funcs.new_figure(size=params.FIGURE_SIZE_SQUARE)
    ax.plot(table["n_states"], table["BIC"], marker="o", label="BIC", color="#C44E52")
    ax.plot(table["n_states"], table["AIC"], marker="s", label="AIC", color="#4C72B0")
    best = int(table.loc[table["BIC"].idxmin(), "n_states"])
    ax.axvline(best, color="0.6", linestyle="--", linewidth=1)
    ax.set_xticks(table["n_states"])
    ax.set_xlabel("number of states")
    ax.set_ylabel("information criterion")
    ax.set_title(f"Model selection (lowest BIC at {best} states)")
    ax.legend()
    return fig


def plot_emission_densities(frame, ordered, weights, labels):
    """Figure: return histogram with the stationary-weighted regime densities"""
    r = frame["log_return"].to_numpy()
    grid = np.linspace(r.min(), r.max(), 600)

    fig, ax = plot_funcs.new_figure(size=params.FIGURE_SIZE_SQUARE)
    ax.hist(r, bins=120, density=True, color="0.8", label="observed returns")
    mixture = np.zeros_like(grid)
    for i, name in enumerate(labels):
        component = weights[i] * _normal_pdf(grid, ordered["means"][i], ordered["stds"][i])
        mixture += component
        ax.plot(grid, component, color=params.REGIME_COLORS[i], linewidth=1.2, label=name)
    ax.plot(grid, mixture, color="black", linewidth=1.5, label="mixture")
    ax.set_xlabel("daily log-return")
    ax.set_ylabel("density")
    ax.set_title("HMM regime emission densities")
    ax.legend()
    return fig
