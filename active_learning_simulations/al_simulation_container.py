from enum import Enum, auto
from typing import Dict, Optional


class ALSimulatorDataset(Enum):
    # Renamed in place (ordinal values 1-6 unchanged) so older compressed result
    # files, which store the ordinal rather than the name, still deserialize correctly.
    MELTOME_MAXIMIZE_LOW = auto()
    MELTOME_MINIMIZE_LOW = auto()
    SCL_LOW = auto()
    AMYLASE_LOW = auto()
    PHOT_LOW = auto()
    EXOTOX = auto()  # kept only so historic results still load; excluded from all()

    # New medium-budget variants: same underlying data, different simulation config
    MELTOME_MAXIMIZE_MEDIUM = auto()
    MELTOME_MINIMIZE_MEDIUM = auto()
    SCL_MEDIUM = auto()
    AMYLASE_MEDIUM = auto()
    PHOT_MEDIUM = auto()

    @staticmethod
    def all():
        """Datasets used for new simulation runs (EXOTOX excluded on purpose)."""
        return [
            ALSimulatorDataset.MELTOME_MAXIMIZE_LOW,
            ALSimulatorDataset.MELTOME_MINIMIZE_LOW,
            ALSimulatorDataset.SCL_LOW,
            ALSimulatorDataset.AMYLASE_LOW,
            ALSimulatorDataset.PHOT_LOW,
            ALSimulatorDataset.MELTOME_MAXIMIZE_MEDIUM,
            ALSimulatorDataset.MELTOME_MINIMIZE_MEDIUM,
            ALSimulatorDataset.SCL_MEDIUM,
            ALSimulatorDataset.AMYLASE_MEDIUM,
            ALSimulatorDataset.PHOT_MEDIUM,
        ]

    def base_name(self) -> str:
        """Underlying dataset identity, without the LOW/MEDIUM scale suffix."""
        if self.name.endswith("_LOW"):
            return self.name[: -len("_LOW")]
        if self.name.endswith("_MEDIUM"):
            return self.name[: -len("_MEDIUM")]
        return self.name  # EXOTOX has no scale suffix

    def scale(self) -> Optional[str]:
        """'LOW', 'MEDIUM', or None (EXOTOX, which predates the scale split)."""
        if self.name.endswith("_LOW"):
            return "LOW"
        if self.name.endswith("_MEDIUM"):
            return "MEDIUM"
        return None

    def n_start(self) -> int:
        """Number of initial training sequences, depending on the dataset scale."""
        return 96 if self.scale() == "MEDIUM" else 10

    def to_path(self, path_override: Optional[Dict[str, str]] = None) -> str:
        path = None
        if path_override:
            path = path_override.get(self.name) or path_override.get(self.base_name())
        if path is not None:
            return path
        default_dict = {
            "MELTOME_MAXIMIZE": "datasets/biotrainer_meltome_mixed_max2000.fasta",
            "MELTOME_MINIMIZE": "datasets/biotrainer_meltome_mixed_max2000.fasta",
            "SCL": "datasets/scl_max2000.fasta",
            "AMYLASE": "datasets/amylase_pet_max2000.fasta",
            "PHOT": "datasets/PHOT_CHLRE_Chen_2023_max2000.fasta",
            "EXOTOX": "datasets/exotox_merged_max2000.fasta",
        }
        return default_dict[self.base_name()]