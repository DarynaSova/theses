"""Create a simple barplot of sequence counts per dataset."""
from pathlib import Path

import altair as alt
import pandas as pd
from biotrainer_core.input_files import read_FASTA

from al_simulation_container import ALSimulatorDataset

OUTPUT_DIR = Path("dataset_statistics")


def main() -> None:
    rows = []
    seen_datasets = set()

    datasets = [*ALSimulatorDataset.all(), ALSimulatorDataset.EXOTOX]
    for dataset in datasets:
        dataset_name = dataset.base_name()
        if dataset_name in seen_datasets:
            continue
        seen_datasets.add(dataset_name)
        if dataset_name in {"MELTOME_MAXIMIZE", "MELTOME_MINIMIZE"}:
            dataset_name = "MELTOME"
            if dataset_name in seen_datasets:
                continue
            seen_datasets.add(dataset_name)
        rows.append({
            "dataset": dataset_name,
            "sequence_count": len(read_FASTA(dataset.to_path())),
        })

    frame = pd.DataFrame(rows).sort_values("sequence_count", ascending=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "dataset_sequence_counts.csv", index=False)

    chart = (
        alt.Chart(frame)
        .mark_bar(color="#2F6690")
        .encode(
            y=alt.Y("dataset:N", title="Dataset", sort="-x"),
            x=alt.X("sequence_count:Q", title="Number of sequences"),
            tooltip=["dataset", "sequence_count"],
        )
        .properties(
            title="Number of protein sequences per dataset",
            width=700,
            height=350,
        )
    )

    chart.save(OUTPUT_DIR / "dataset_sequence_counts.html")
    try:
        chart.save(OUTPUT_DIR / "dataset_sequence_counts.png")
    except Exception as exc:
        print(f"PNG export unavailable; HTML was saved: {exc}")

    print(frame.to_string(index=False))
    print(f"Wrote outputs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
