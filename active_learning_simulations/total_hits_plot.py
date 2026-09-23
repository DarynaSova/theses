"""Create an interactive Plotly heatmap comparison of total AL hits.

Run with:
    uv run python total_hits_plot.py

Output:
    total_hits.html
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from al_simulation_container import ALSimulatorDataset


RESULT_DIRS = [
    Path("simulation_results_updated"),
    Path("simulation_results_scl"),
    Path("simulation_results_dms"),
    Path("simulation_v1_results_final"),
]
OUTPUT_PATH = Path("total_hits.html")
CONFIGS = ("OFF", "0.5", "1.0", "2.0")
SCALES = ("LOW", "MEDIUM")
MODELS = ("GP", "FNN", "Random")
EMBEDDER_NAMES = {
    "one_hot_encoding": "OHE",
    "facebook/esm2_t6_8M_UR50D": "ESM2 8M",
}
MODEL_NAMES = {
    "GAUSSIAN_PROCESS": "GP",
    "FNN_MCD": "FNN",
    "RANDOM": "Random",
}
WEIGHTED_PATTERN = re.compile(
    r"compressed_dashboard_data_disorder_on_(?P<method>.+)_w(?P<weight>0\.5|1\.0|2\.0)\.json$"
)


def _records_from_file(path: Path, method: str, config: str) -> list[dict]:
    payload = json.loads(path.read_text())
    records = []
    for experiment in payload["experiments"]:
        dataset_id = ALSimulatorDataset(experiment["dataset_id"])
        scale = dataset_id.scale()
        if scale not in SCALES:
            continue
        campaigns = experiment.get("per_sim_is_success", [])
        if not campaigns:
            continue
        records.append({
            "dataset": dataset_id.base_name(),
            "scale": scale,
            "model": MODEL_NAMES.get(experiment["model"], experiment["model"]),
            "embedder": EMBEDDER_NAMES.get(experiment["embedder"], experiment["embedder"]),
            "method": method,
            "config": config,
            "hits": experiment.get("aggregated_hits", 0),
            "campaigns": len(campaigns),
            "source": path.parent.name,
        })
    return records


def _read_records() -> list[dict]:
    records = []
    for results_dir in RESULT_DIRS:
        weighted_files = []
        for path in results_dir.glob("compressed_dashboard_data_disorder_on_*_w*.json"):
            match = WEIGHTED_PATTERN.match(path.name)
            if match is None:
                continue
            method = match.group("method")
            weight = match.group("weight")
            weighted_files.append(method)
            records.extend(_records_from_file(path, method, weight))

        baseline_path = results_dir / "compressed_dashboard_data_disorder_off.json"
        if baseline_path.exists():
            for method in sorted(set(weighted_files)):
                records.extend(_records_from_file(baseline_path, method, "OFF"))
    return records


def _aggregate_records(records: list[dict]) -> dict[tuple, dict]:
    grouped = defaultdict(lambda: {"hits": 0, "campaigns": 0, "sources": set()})
    for record in records:
        key = (
            record["dataset"], record["scale"], record["model"],
            record["method"], record["config"], record["embedder"],
        )
        grouped[key]["hits"] += record["hits"]
        grouped[key]["campaigns"] += record["campaigns"]
        grouped[key]["sources"].add(record["source"])
    return grouped


def _build_figure(records: list[dict]) -> tuple[go.Figure, list[tuple[str, str]]]:
    grouped = _aggregate_records(records)
    embedder_methods = sorted({
        (key[5], key[3]) for key in grouped if key[3] != "baseline"
    })
    datasets = sorted({key[0] for key in grouped})
    if not embedder_methods or not datasets:
        raise ValueError("No weighted aggregation results were found.")

    figure = make_subplots(
        rows=len(SCALES),
        cols=len(datasets),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.035,
        vertical_spacing=0.12,
        subplot_titles=datasets,
    )
    trace_groups = []
    max_hits = max(point["hits"] for point in grouped.values())

    for embedder, method in embedder_methods:
        for scale_idx, scale in enumerate(SCALES, start=1):
            for dataset_idx, dataset in enumerate(datasets, start=1):
                z_values = []
                text_values = []
                customdata = []
                for model in MODELS:
                    row_values = []
                    row_text = []
                    row_customdata = []
                    for config in CONFIGS:
                        point = grouped.get((dataset, scale, model, method, config, embedder))
                        if point is None:
                            row_values.append(None)
                            row_text.append("")
                            row_customdata.append([0, 0, ""])
                        else:
                            row_values.append(point["hits"])
                            row_text.append(str(point["hits"]))
                            row_customdata.append([
                                point["hits"], point["campaigns"],
                                ", ".join(sorted(point["sources"])),
                            ])
                    z_values.append(row_values)
                    text_values.append(row_text)
                    customdata.append(row_customdata)

                figure.add_trace(go.Heatmap(
                    z=z_values,
                    x=list(CONFIGS),
                    y=list(MODELS),
                    text=text_values,
                    texttemplate="%{text}",
                    textfont={"size": 13},
                    customdata=customdata,
                    colorscale="Viridis",
                    zmin=0,
                    zmax=max_hits,
                    colorbar={"title": "Total<br>hits", "len": 0.35},
                    hovertemplate=(
                        "Dataset: " + dataset + "<br>"
                        "Embedder: " + embedder + "<br>"
                        "Budget: " + scale + "<br>"
                        "Model: %{y}<br>"
                        "Configuration: %{x}<br>"
                        "Total hits: %{z}<br>"
                        "Campaigns: %{customdata[1]}<br>"
                        "Source: %{customdata[2]}<extra></extra>"
                    ),
                    visible=(embedder, method) == embedder_methods[0],
                    showscale=scale_idx == 1 and dataset_idx == len(datasets),
                ), row=scale_idx, col=dataset_idx)
                trace_groups.append((embedder, method))

    buttons = [{
        "label": f"{embedder} | {method}",
        "method": "update",
        "args": [
            {"visible": [trace_group == (embedder, method) for trace_group in trace_groups]},
            {"title": f"Total hits | Aggregation method: {method} | Embedder: {embedder}"},
        ],
    } for embedder, method in embedder_methods]

    first_embedder, first_method = embedder_methods[0]
    figure.update_layout(
        title=f"Total hits | Aggregation method: {first_method} | Embedder: {first_embedder}",
        height=700,
        width=max(1100, 260 * len(datasets)),
        margin={"l": 80, "r": 40, "b": 80, "t": 130},
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0,
            "y": 1.16,
        }],
    )
    for row_idx, scale in enumerate(SCALES, start=1):
        figure.update_yaxes(title_text=scale, row=row_idx, col=1)
    figure.update_xaxes(title_text="Configuration (OFF = baseline)", row=len(SCALES), col=1)
    return figure, embedder_methods


def main() -> None:
    records = _read_records()
    grouped = _aggregate_records(records)
    weighted_keys = [key for key in grouped if key[4] != "OFF"]
    missing_baselines = [
        key for key in weighted_keys
        if (key[0], key[1], key[2], key[3], "OFF", key[5]) not in grouped
    ]
    if missing_baselines:
        raise ValueError(f"Missing Disorder-off baselines for {len(missing_baselines)} weighted result groups.")

    figure, _ = _build_figure(records)
    figure.write_html(OUTPUT_PATH, include_plotlyjs=True)
    print(f"Wrote {OUTPUT_PATH} using {len(records)} result records.")
    print("Baseline: OFF is included as the leftmost column in every method view.")


if __name__ == "__main__":
    main()
