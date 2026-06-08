import params
from scripts import data_loader
from scripts.counting import counting_funcs
from scripts.plotting import plot_funcs

def main():
    frame = data_loader.load_processed()
    labeled, (cut_low, cut_high) = counting_funcs.label_regimes_by_volatility(frame)

    counts = counting_funcs.count_transitions(labeled["regime"].to_numpy())
    matrix = counting_funcs.transition_matrix(counts)
    matrix_frame = counting_funcs.transition_frame(matrix)
    summary = counting_funcs.regime_summary(labeled)

    params.TABLES.mkdir(parents=True, exist_ok=True)
    matrix_frame.to_csv(params.TABLES / "table03_transition_empirical.csv")
    (params.TABLES / "table03_transition_empirical.md").write_text(matrix_frame.round(3).to_markdown())
    summary.to_csv(params.TABLES / "table02_regime_summary_counting.csv", index=False)
    (params.TABLES / "table02_regime_summary_counting.md").write_text(summary.to_markdown(index=False))

    plot_funcs.save(counting_funcs.plot_regime_timeline(labeled), "fig05_regime_timeline.png")
    plot_funcs.save(
        plot_funcs.transition_heatmap(matrix, counting_funcs.REGIME_LABELS,
                                      "Empirical transition matrix (counting)"),
        "fig06_transition_empirical.png",
    )

    print(f"volatility cut points (rolling std): low {cut_low:.4f}, high {cut_high:.4f}")
    print("\nregime summary:")
    print(summary.to_string(index=False))
    print("\nempirical transition matrix:")
    print(matrix_frame.round(3).to_string())


if __name__ == "__main__":
    main()
