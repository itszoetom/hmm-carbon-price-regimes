"""Run Viterbi decoding and produce the headline regime figure"""

import joblib

import params
from scripts import data_loader
from scripts.decode import decode_funcs
from scripts.plotting import plot_funcs


def main():
    saved = joblib.load(params.DATA_PROCESSED / "hmm_selected.joblib")
    labels = saved["labels"]

    frame = data_loader.load_processed()
    decoded = decode_funcs.decode(frame, saved)

    decoded[["date", "price", "log_return", "regime"]].to_csv(
        params.DATA_PROCESSED / "decoded_regimes.csv", index=False
    )

    occupancy = decode_funcs.occupancy_table(decoded, labels)
    params.TABLES.mkdir(parents=True, exist_ok=True)
    occupancy.to_csv(params.TABLES / "table08_decoded_occupancy.csv", index=False)
    (params.TABLES / "table08_decoded_occupancy.md").write_text(occupancy.to_markdown(index=False))

    plot_funcs.save(decode_funcs.plot_decoded_timeline(decoded, labels),
                    "fig12_decoded_regimes.png")

    print("decoded regime occupancy:")
    print(occupancy.to_string(index=False))


if __name__ == "__main__":
    main()
