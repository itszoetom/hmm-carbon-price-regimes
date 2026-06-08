"""Markov chain analysis of the estimated transition matrices"""

import numpy as np
import pandas as pd

import params
from scripts.markov import markov_funcs
from scripts.plotting import plot_funcs

METHODS = {
    "counting": "table03_transition_empirical.csv",
    "HMM": "table06_transition_hmm.csv",
}


def comparison_table():
    """Build the stationary distribution and expected duration table for both methods"""
    rows = []
    for method, filename in METHODS.items():
        matrix = pd.read_csv(params.TABLES / filename, index_col=0)
        P = matrix.to_numpy()
        pi = markov_funcs.stationary_distribution(P)
        durations = markov_funcs.expected_durations(P)
        lam2 = markov_funcs.second_eigenvalue_modulus(P)
        for state, regime in enumerate(matrix.index):
            rows.append({
                "method": method,
                "regime": regime,
                "stationary probability": round(float(pi[state]), 3),
                "expected duration (days)": round(float(durations[state]), 1),
                "second eigenvalue": round(lam2, 3),
            })
    return pd.DataFrame(rows)


def main():
    table = comparison_table()
    params.TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(params.TABLES / "table07_stationary_durations.csv", index=False)
    (params.TABLES / "table07_stationary_durations.md").write_text(table.to_markdown(index=False))

    hmm_matrix = pd.read_csv(params.TABLES / METHODS["HMM"], index_col=0)
    labels = list(hmm_matrix.index)
    P = hmm_matrix.to_numpy()
    n_states = len(labels)

    start = np.zeros(n_states)
    start[-1] = 1.0
    distances, pi = markov_funcs.tv_convergence(P, start, n_steps=150)
    lam2 = markov_funcs.second_eigenvalue_modulus(P)
    plot_funcs.save(
        markov_funcs.plot_convergence(distances, lam2,
                                      "Convergence to the stationary distribution"),
        "fig10_convergence.png",
    )

    # Sensitivity: vary the stressed to crashed transition (indices n_states-2, n_states-1)
    i, j = n_states - 2, n_states - 1
    baseline_value = float(P[i, j])
    values = np.linspace(0.0, max(0.12, baseline_value * 3), 80)
    shares = []
    for value in values:
        perturbed = markov_funcs.perturb_entry(P, i, j, value)
        shares.append(float(markov_funcs.stationary_distribution(perturbed)[j]))
    plot_funcs.save(
        markov_funcs.plot_sensitivity(values, np.array(shares), baseline_value, float(pi[j]),
                                      "Sensitivity of the long-run crashed share"),
        "fig11_sensitivity.png",
    )

    print("stationary distribution and expected durations:")
    print(table.to_string(index=False))
    print(f"\nHMM mixing: second eigenvalue {lam2:.3f}")
    print("stationary distribution (HMM): {0}".format(dict(zip(labels, pi.round(3)))))


if __name__ == "__main__":
    main()
