"""Fit the Gaussian HMM and select the number of states by BIC"""

import joblib
import pandas as pd

import params
from scripts import data_loader
from scripts.hmm import hmm_funcs
from scripts.markov import markov_funcs
from scripts.plotting import plot_funcs


def main():
    frame = data_loader.load_processed()
    X = hmm_funcs.returns_matrix(frame)

    table, models = hmm_funcs.model_selection(X)
    best_n = hmm_funcs.select_best(table)
    model = models[best_n]

    ordered = hmm_funcs.ordered_parameters(model)
    labels = hmm_funcs.state_labels(best_n)
    pi = markov_funcs.stationary_distribution(ordered["transmat"])
    emissions = hmm_funcs.emission_table(ordered, pi, labels)

    transmat_frame = pd.DataFrame(ordered["transmat"], index=labels, columns=labels)

    params.TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(params.TABLES / "table04_model_selection.csv", index=False)
    (params.TABLES / "table04_model_selection.md").write_text(table.to_markdown(index=False))
    emissions.to_csv(params.TABLES / "table05_emission_params.csv", index=False)
    (params.TABLES / "table05_emission_params.md").write_text(emissions.to_markdown(index=False))
    transmat_frame.to_csv(params.TABLES / "table06_transition_hmm.csv")
    (params.TABLES / "table06_transition_hmm.md").write_text(transmat_frame.round(3).to_markdown())

    plot_funcs.save(hmm_funcs.plot_model_selection(table), "fig07_model_selection.png")
    plot_funcs.save(hmm_funcs.plot_emission_densities(frame, ordered, pi, labels),
                    "fig08_emission_densities.png")
    plot_funcs.save(
        plot_funcs.transition_heatmap(ordered["transmat"], labels,
                                      "HMM transition matrix (Baum-Welch)"),
        "fig09_transition_hmm.png",
    )

    # save the model and state ordering for the decode step
    params.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "order": ordered["order"], "labels": labels, "n_states": best_n},
        params.DATA_PROCESSED / "hmm_selected.joblib",
    )

    print("model selection:")
    print(table.to_string(index=False))
    print(f"\nselected {best_n}-state model, emission parameters:")
    print(emissions.to_string(index=False))
    print("\nHMM transition matrix:")
    print(transmat_frame.round(3).to_string())
    print(f"\nstationary distribution: {dict(zip(labels, pi.round(3)))}")


if __name__ == "__main__":
    main()
