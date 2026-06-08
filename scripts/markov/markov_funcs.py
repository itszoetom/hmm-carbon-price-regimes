"""Finite-state Markov chain analysis of an estimated transition matrix"""

import numpy as np

import params
from scripts.plotting import plot_funcs


def stationary_distribution(P):
    """return the stationary distribution pi solving pi = pi P

    pi is the left eigenvector of P for eigenvalue 1, normalized to sum to one
    For an irreducible aperiodic chain it is unique and gives the long-run
    fraction of time spent in each state
    """
    P = np.asarray(P, dtype=float)
    values, vectors = np.linalg.eig(P.T)
    index = int(np.argmin(np.abs(values - 1.0)))
    vector = np.real(vectors[:, index])
    return vector / vector.sum()


def expected_durations(P):
    """expected number of days spent in each state before leaving"""
    P = np.asarray(P, dtype=float)
    diag = np.diag(P)
    return 1.0 / (1.0 - diag)


def second_eigenvalue_modulus(P):
    """modulus of the second-largest eigenvalue of P"""
    P = np.asarray(P, dtype=float)
    moduli = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    return float(moduli[1]) if len(moduli) > 1 else 0.0


def tv_convergence(P, start, n_steps):
    """Total variation distance between the n-step distribution and pi, for each n"""
    P = np.asarray(P, dtype=float)
    pi = stationary_distribution(P)
    distances = []
    current = np.asarray(start, dtype=float)
    for _ in range(n_steps):
        distances.append(0.5 * np.abs(current - pi).sum())
        current = current @ P
    return np.array(distances), pi


def perturb_entry(P, i, j, value):
    """Set transition probability p(i, j) to value, renormalizing the rest of row i"""
    P = np.asarray(P, dtype=float).copy()
    others = [c for c in range(P.shape[1]) if c != j]
    remaining = 1.0 - value
    current_others = P[i, others].sum()
    if current_others > 0:
        P[i, others] = P[i, others] / current_others * remaining
    else:
        P[i, others] = remaining / len(others)
    P[i, j] = value
    return P


def plot_convergence(distances, lam2, title):
    """Figure: geometric decay of the distance to the stationary distribution"""
    steps = np.arange(len(distances))
    floor = 1e-12
    observed = np.maximum(distances, floor)
    reference = np.maximum(distances[0] * lam2 ** steps, floor)

    fig, ax = plot_funcs.new_figure(size=params.FIGURE_SIZE_SQUARE)
    ax.semilogy(steps, observed, color="#4C72B0", linewidth=1.5, label="total variation distance")
    ax.semilogy(steps, reference, color="#C44E52", linestyle="--", linewidth=1.2,
                label=f"rate of second eigenvalue ({lam2:.3f})")
    ax.set_xlabel("days")
    ax.set_ylabel("distance to stationary distribution")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_sensitivity(values, shares, baseline_value, baseline_share, title):
    """Figure: long-run crashed share as one entering transition probability varies"""
    fig, ax = plot_funcs.new_figure(size=params.FIGURE_SIZE_SQUARE)
    ax.plot(values, shares, color="#C44E52", linewidth=1.5)
    ax.scatter([baseline_value], [baseline_share], color="black", zorder=3,
               label="estimated value")
    ax.set_xlabel("probability of moving from stressed to crashed")
    ax.set_ylabel("long-run share of days crashed")
    ax.set_title(title)
    ax.legend()
    return fig
