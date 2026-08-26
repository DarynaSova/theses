"""Merge compressed dashboard runs into a single file for cross-run comparison.

Creates one compressed JSON containing the experiments of all runs found under
``simulation_v1_results`` (excluding the vanilla runs), so that every disorder
variant can be compared side-by-side on the dashboard
(Comparison tab -> "Compare by Surrogate Model", and the cross-dataset
"Average Hits" / "Unique Hits" plots).

The surrogate model name of each experiment is suffixed with a short run label
(e.g. "FNN_MCD [mean_w1.0]") so the variants appear as separate groups/bars.
Experiment names are suffixed as well to avoid selectbox collisions.
Embedder names are left untouched so that projections still load.

Usage:
    uv run python merge_compressed_runs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("simulation_v1_results")
OUTPUT_FILE = RESULTS_DIR / "compressed_dashboard_data_merged_disorder.json"
MERGED_RUN_NAME = "merged_disorder_comparison"

# Runs whose name contains any of these substrings are excluded from the merge
EXCLUDED_RUN_SUBSTRINGS = ("vanilla",)

# Short labels used to suffix surrogate model / experiment names
RUN_LABELS = {
    "baseline_no_disorder": "baseline",
    "fixedscores_with_disorder_mean_weight_1.0": "fixscores_mean_w1.0",
    "with_disorder_frac_above_5_weight_1.0": "frac>5_w1.0",
    "with_disorder_frac_above_5_weight_2.0": "frac>5_w2.0",
    "with_disorder_mean_weight_1.0": "mean_w1.0",
    "with_disorder_mean_weight_2.0": "mean_w2.0",
    "with_disorder_p90_weight_1.0": "p90_w1.0",
}


def _run_label(run_name: str) -> str:
    """Short display label for a run, with a sensible fallback for new runs."""
    if run_name in RUN_LABELS:
        return RUN_LABELS[run_name]
    return run_name.replace("with_disorder_", "").replace("weight_", "w")


def main() -> int:
    run_files = sorted(
        p for p in RESULTS_DIR.rglob("compressed_dashboard_data_*.json")
        if p != OUTPUT_FILE
    )
    if not run_files:
        print(f"No compressed_dashboard_data_*.json files found under {RESULTS_DIR}")
        return 1

    merged_experiments = []
    included_runs = []
    skipped_runs = []

    for path in run_files:
        data = json.loads(path.read_text())
        run_name = data.get("run_name", path.stem)

        if run_name == MERGED_RUN_NAME:
            continue  # Do not re-merge our own output
        if any(sub in run_name for sub in EXCLUDED_RUN_SUBSTRINGS):
            skipped_runs.append(run_name)
            continue

        label = _run_label(run_name)
        experiments = data.get("experiments", [])
        for exp in experiments:
            exp = dict(exp)
            exp["model"] = f"{exp['model']} [{label}]"
            exp["name"] = f"{exp['name']} [{label}]"
            merged_experiments.append(exp)

        included_runs.append((run_name, label, len(experiments)))

    if not merged_experiments:
        print("No experiments left after exclusions - nothing to merge.")
        return 1

    merged = {"run_name": MERGED_RUN_NAME, "experiments": merged_experiments}
    OUTPUT_FILE.write_text(json.dumps(merged, indent=2))

    print(f"Merged {len(merged_experiments)} experiments from {len(included_runs)} runs:")
    for run_name, label, n_exps in included_runs:
        print(f"  - {run_name} -> [{label}] ({n_exps} experiments)")
    for run_name in skipped_runs:
        print(f"  - skipped: {run_name}")
    print(f"\nWritten to: {OUTPUT_FILE}")
    print("Select 'merged_disorder_comparison' in the dashboard sidebar and use "
          "'Compare by Surrogate Model' to compare disorder variants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
