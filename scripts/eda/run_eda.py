import params
from scripts import data_loader
from scripts.eda import eda_funcs
from scripts.plotting import plot_funcs

def main():
    frame = data_loader.load_processed()

    params.TABLES.mkdir(parents=True, exist_ok=True)
    table = eda_funcs.summary_table(frame)
    table.to_csv(params.TABLES / "table01_summary_stats.csv", index=False)
    (params.TABLES / "table01_summary_stats.md").write_text(
        table.to_markdown(index=False)
    )

    figures = {
        "fig01_price_phases.png": eda_funcs.plot_price_with_phases(frame),
        "fig02_log_returns.png": eda_funcs.plot_log_returns(frame),
        "fig03_rolling_volatility.png": eda_funcs.plot_rolling_volatility(frame),
        "fig04_return_hist_vs_normal.png": eda_funcs.plot_return_hist_vs_normal(frame),
    }
    for name, fig in figures.items():
        plot_funcs.save(fig, name)

if __name__ == "__main__":
    main()
