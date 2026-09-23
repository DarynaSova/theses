"""Extract raw simulation JSON files from simulation_v1_results.tar.zst directly in memory
and compress them into dashboard JSON files under simulation_v1_results/ without expanding
gigabytes of raw files onto disk.
"""
from __future__ import annotations

import json
import subprocess
import tarfile
import sys
from pathlib import Path

# Add current dir to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

from al_simulation_container import ALSimulatorDataset
from al_simulator import (
    DashboardCompressedData,
    DashboardExperimentData,
    DashboardSingleSimulationData,
)

TAR_ZST_PATH = Path("/Users/darynasova/Documents/theses/simulation_v1_results.tar.zst")
RESULTS_DIR = Path("/Users/darynasova/Documents/theses/active_learning_simulations/simulation_v1_results")


def extract_disorder_label(fname: str) -> str:
    """Extract disorder configuration label from filename.
    
    e.g. 'al_sim_AMYLASE_LOW_..._disorder_on_mean_w1.0.json' -> 'disorder_on_mean_w1.0'
    'al_sim_AMYLASE_LOW_..._disorder_off.json' -> 'disorder_off'
    """
    if "_disorder_" in fname:
        suffix = "disorder_" + fname.split("_disorder_")[-1].replace(".json", "")
        return suffix
    return "cluster_run"


def process_raw_json(fname: str, json_results: list) -> DashboardExperimentData:
    """Fast dict parsing of raw simulation result JSON."""
    first_ssr = json_results[0]
    dataset_id = ALSimulatorDataset(first_ssr['dataset_id'])
    
    embedder_name = first_ssr['al_campaign_config']['embedder_name']
    
    model_name = first_ssr['al_campaign_config']['model_type']
    if isinstance(model_name, dict):
        model_name = model_name.get('value', str(model_name))
    
    potential_hits = first_ssr['simulation_result']['potential_hits']
    
    opt_mode = first_ssr['al_campaign_config']['optimization_mode']
    if isinstance(opt_mode, dict):
        opt_mode = opt_mode.get('value', str(opt_mode))

    is_discrete = opt_mode in ['DISCRETE', 'discrete']
    n_hits_thresh = first_ssr['al_simulation_config']['convergence_config']['n_hits']

    summary = {
        'embedder_name': embedder_name,
        'model_type': model_name,
        'optimization_mode': opt_mode,
        'n_simulations': len(json_results),
        'n_successful': sum(
            1 for ssr in json_results
            if sum(len(it) for it in (ssr['simulation_result'].get('iteration_hits') or [])) >= n_hits_thresh
        ),
        'n_hits_threshold': n_hits_thresh,
        'is_discrete': is_discrete,
        'discrete_targets': first_ssr['al_campaign_config'].get('discrete_targets'),
    }

    per_sim_n_hits = [
        list(map(len, ssr['simulation_result'].get('iteration_hits') or []))
        for ssr in json_results
    ]
    per_sim_metrics_total = [
        [m['mean'] for m in (ssr['simulation_result'].get('iteration_metrics_total') or [])]
        for ssr in json_results
    ]
    per_sim_metrics_suggestions = [
        [m['mean'] for m in (ssr['simulation_result'].get('iteration_metrics_suggestions') or [])]
        for ssr in json_results
    ]
    per_sim_is_success = [
        sum(len(it) for it in (ssr['simulation_result'].get('iteration_hits') or [])) >= n_hits_thresh
        for ssr in json_results
    ]

    single_sims = []
    for ssr in json_results:
        is_succ = sum(len(it) for it in (ssr['simulation_result'].get('iteration_hits') or [])) >= n_hits_thresh
        seed = ssr['al_campaign_config']['seed']
        sim_data = DashboardSingleSimulationData(
            label=f'seed={seed} | {model_name} | success={is_succ}',
            is_success=is_succ,
            seed=seed,
            stop_reasons=ssr['simulation_result'].get('stop_reasons') or ['None'],
            iteration_metrics_total=[m['mean'] for m in (ssr['simulation_result'].get('iteration_metrics_total') or [])],
            iteration_metrics_suggestions=[m['mean'] for m in (ssr['simulation_result'].get('iteration_metrics_suggestions') or [])],
            iteration_hits=list(ssr['simulation_result'].get('iteration_hits') or []),
            iteration_consecutive_failures=list(ssr['simulation_result'].get('iteration_consecutive_failures') or []),
            iteration_results_count=len(ssr['simulation_result'].get('iteration_results') or []),
            n_hits_threshold=n_hits_thresh,
        )
        single_sims.append(sim_data)

    aggregated_hits = sum(sum(len(it) for it in (ssr['simulation_result'].get('iteration_hits') or [])) for ssr in json_results)
    aggregated_suggestions = sum(sum(len(it.get('suggestions', [])) for it in (ssr['simulation_result'].get('iteration_results') or [])) for ssr in json_results)

    return DashboardExperimentData(
        name=fname,
        dataset_id=dataset_id,
        embedder=embedder_name,
        model=model_name,
        summary=summary,
        aggregated_hits=aggregated_hits,
        aggregated_suggestions=aggregated_suggestions,
        potential_hits=potential_hits,
        per_sim_n_hits=per_sim_n_hits,
        per_sim_metrics_total=per_sim_metrics_total,
        per_sim_metrics_suggestions=per_sim_metrics_suggestions,
        per_sim_is_success=per_sim_is_success,
        single_sims=single_sims,
    )


def main():
    if not TAR_ZST_PATH.exists():
        print(f"Error: {TAR_ZST_PATH} not found!")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Opening archive stream from {TAR_ZST_PATH}...")
    proc = subprocess.Popen(["zstd", "-dc", str(TAR_ZST_PATH)], stdout=subprocess.PIPE)
    
    experiments_by_run = {}
    processed_count = 0

    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".json")):
                continue
            
            fname = Path(member.name).name
            if "compressed_dashboard" in fname:
                continue

            processed_count += 1
            if processed_count % 50 == 0 or processed_count == 1:
                print(f"[{processed_count}] Processing {fname}...")

            f = tar.extractfile(member)
            json_results = json.load(f)

            exp_data = process_raw_json(fname, json_results)

            run_label = extract_disorder_label(fname)
            if run_label not in experiments_by_run:
                experiments_by_run[run_label] = []
            experiments_by_run[run_label].append(exp_data)

    print(f"\nCompleted processing {processed_count} files across {len(experiments_by_run)} disorder configurations.")
    print("Writing compressed dashboard files...")

    for run_name, exps in sorted(experiments_by_run.items()):
        compressed = DashboardCompressedData(run_name=run_name, experiments=exps)
        output_path = RESULTS_DIR / f"compressed_dashboard_data_{run_name}.json"
        output_path.write_text(compressed.model_dump_json(indent=4))
        print(f"  - Wrote {output_path.name} ({len(exps)} experiments)")

    print("\nRunning merge_compressed_runs.py...")
    import merge_compressed_runs
    merge_compressed_runs.main()

    print("\nAll done! Compressed and merged cluster simulation results are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    sys.exit(main())
