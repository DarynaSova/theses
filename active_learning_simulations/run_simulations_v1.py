import argparse
import itertools

from pathlib import Path
from biotrainer_core.input_files import read_FASTA
from pydantic import BaseModel, Field
from biocentral_api import ActiveLearningModelType, CommonEmbedder, BiocentralAPI

from al_compress_reports import compress_reports
from al_simulation_container import ALSimulatorDataset
from al_simulator import ActiveLearningMultipleSimulationResult, get_simulator


class ExperimentConstants:
    n_rounds: int = 5
    result_dir: Path = Path("simulation_v1_results/")
    projection_dir: Path = Path("simulation_v1_projections/")


def _format_weight(weight: float) -> str:
    # Keep filenames stable/short, e.g. 0.5 -> "0.5", 1.0 -> "1.0", 2.0 -> "2.0"
    return f"{weight:g}" if weight != int(weight) else f"{weight:.1f}"


def build_disorder_configs():
    """All disorder configurations to sweep in a single invocation.

    One disabled baseline, plus every (aggregation method x weight) combination
    with disorder enabled. Each entry gets a unique, filesystem-safe label used
    to keep raw/compressed result files of different configs from colliding.
    """
    configs = [{
        "label": "disorder_off",
        "enabled": False,
        "aggregation_method": "mean",  # unused while disabled, kept for a stable/valid config
        "weight": 1.0,
    }]
    for method in ("mean", "frac_above_5", "frac_above_7", "p90", "median"):
        for weight in (0.5, 1.0, 2.0):
            configs.append({
                "label": f"disorder_on_{method}_w{_format_weight(weight)}",
                "enabled": True,
                "aggregation_method": method,
                "weight": weight,
            })
    return configs


DISORDER_CONFIGS = build_disorder_configs()


class ExperimentParametersV1(BaseModel):
    dataset_id: ALSimulatorDataset = Field(description="Dataset to use for the simulation")
    embedder_name: str = Field(description="Name of the embedder to use")
    model_type: ActiveLearningModelType = Field(description="Type of the model to use")
    disorder_label: str = Field(description="Unique label identifying the disorder configuration")
    disorder_enabled: bool = Field(description="Whether disorder is included in the acquisition function")
    disorder_aggregation_method: str = Field(description="Aggregation method for per-residue disorder scores")
    disorder_weight: float = Field(description="Weight (gamma) applied to disorder in the acquisition function")

    def to_file_name(self):
        embedder_name = self.embedder_name.replace("/", "-")
        # disorder_label is included so raw result files from different disorder
        # configurations never collide/get mistakenly reused across runs.
        return f"al_sim_{self.dataset_id.name}_{embedder_name}_{self.model_type.value}_{self.disorder_label}.json"


def _create_experiment_params(dataset_ids=None, embedder_names=None, model_types=None, disorder_configs=None):
    experiment_params = []
    dataset_ids = dataset_ids if dataset_ids is not None else ALSimulatorDataset.all()  # EXOTOX excluded on purpose
    embedder_names = embedder_names if embedder_names is not None else [
        CommonEmbedder.ESM_8M.value,
        CommonEmbedder.ESM2_650M.value,
        CommonEmbedder.ONE_HOT_ENCODING.value,
        CommonEmbedder.LENGTH_EMBEDDER.value,
        CommonEmbedder.RANDOM_EMBEDDER.value,
        CommonEmbedder.BLOSUM62.value,
        CommonEmbedder.ProtT5.value,
    ]
    model_types = model_types if model_types is not None else [
        ActiveLearningModelType.GAUSSIAN_PROCESS, ActiveLearningModelType.FNN_MCD,
        ActiveLearningModelType.RANDOM,
    ]
    disorder_configs = disorder_configs if disorder_configs is not None else DISORDER_CONFIGS

    for dataset_id, embedder_name, model_type, disorder_cfg in itertools.product(
        dataset_ids, embedder_names, model_types, disorder_configs
    ):
        experiment_params.append(ExperimentParametersV1(
            dataset_id=dataset_id,
            embedder_name=embedder_name,
            model_type=model_type,
            disorder_label=disorder_cfg["label"],
            disorder_enabled=disorder_cfg["enabled"],
            disorder_aggregation_method=disorder_cfg["aggregation_method"],
            disorder_weight=disorder_cfg["weight"],
        ))
    return experiment_params


def _run_experiment(experiment_params: ExperimentParametersV1):
    if not ExperimentConstants.result_dir.exists():
        ExperimentConstants.result_dir.mkdir(parents=True, exist_ok=True)

    save_dir = ExperimentConstants.result_dir / experiment_params.to_file_name()
    use_save = True
    if use_save and save_dir.exists():
        sim_result = ActiveLearningMultipleSimulationResult.from_json(save_dir)
    else:
        al_simulator = get_simulator(
            experiment_params.dataset_id,
            use_disorder=experiment_params.disorder_enabled,
            disorder_method=experiment_params.disorder_aggregation_method,
            disorder_weight=experiment_params.disorder_weight,
        )
        print(f"Running simulation for {experiment_params}..")
        sim_result = al_simulator.simulate(model_type=experiment_params.model_type,
                                           embedder_name=experiment_params.embedder_name,
                                           n_rounds=ExperimentConstants.n_rounds)
        sim_result.save(save_dir)
    sim_result.print_stats()


def _create_projection(experiment_params: ExperimentParametersV1):
    embedder_name = experiment_params.embedder_name
    # LOW/MEDIUM variants share the same underlying sequences/fasta file, so the
    # projection is computed once per base dataset instead of once per scale.
    dataset_base_name = experiment_params.dataset_id.base_name()
    projection_name = f"projection_result_{dataset_base_name}_{embedder_name}.json"
    projection_path = ExperimentConstants.projection_dir / projection_name
    if projection_path.exists():
        print("Projection already exists. Skipping...")
        return

    sequence_data = read_FASTA(experiment_params.dataset_id.to_path())
    biocentral_api = BiocentralAPI(local_only=True)
    projection_result = biocentral_api.project(embedder_name=experiment_params.embedder_name,
                                               method="pca",
                                               sequence_data=sequence_data,
                                               projection_config={"n_components": "2"}).run()

    if not ExperimentConstants.projection_dir.exists():
        ExperimentConstants.projection_dir.mkdir(parents=True, exist_ok=True)

    with open(projection_path, "w") as f:
        f.write(projection_result.model_dump_json())

    print(f"Projection saved to {projection_path}!")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the active learning simulation matrix.")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a small subset (1 dataset, 1 embedder, 1 model, 2 disorder configs, 1 round) "
             "to quickly verify the wiring locally before a full cluster run.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.smoke_test:
        print("Running in --smoke-test mode: a small subset of the full matrix.")
        ExperimentConstants.n_rounds = 1
        experiment_params = _create_experiment_params(
            dataset_ids=[ALSimulatorDataset.SCL_LOW, ALSimulatorDataset.SCL_MEDIUM],
            embedder_names=[CommonEmbedder.ONE_HOT_ENCODING.value],
            model_types=[ActiveLearningModelType.RANDOM],
            disorder_configs=[DISORDER_CONFIGS[0], DISORDER_CONFIGS[1]],
        )
    else:
        experiment_params = _create_experiment_params()

    used_disorder_labels = sorted({p.disorder_label for p in experiment_params})
    print(f"Prepared {len(experiment_params)} experiment configurations "
          f"across {len(used_disorder_labels)} disorder configuration(s).")

    for experiment_param in experiment_params:
        _run_experiment(experiment_param)

    print("All simulations completed. Compressing reports per disorder configuration...")
    for disorder_label in used_disorder_labels:
        # Restrict each compression pass to its own disorder configuration's raw files,
        # so raw results from different runs coexisting on disk never get mixed together.
        compress_reports(run_name=disorder_label, file_glob=f"al_sim_*_{disorder_label}.json")

    print("Reports compressed. Creating projections...")
    seen_projections = set()
    for experiment_param in experiment_params:
        projection_key = (experiment_param.dataset_id.base_name(), experiment_param.embedder_name)
        if projection_key in seen_projections:
            continue
        seen_projections.add(projection_key)
        _create_projection(experiment_param)
    print("Projections created. Exiting with success.")


if __name__ == "__main__":
    main()
