
from pathlib import Path
from typing import Dict, List, Optional
import statistics


def parse_disorder_fasta(fasta_path: Path) -> Dict[str, List[float]]:
    """
    Parse a disorder FASTA file into a dict of seq_id -> list of per-residue scores.
    
    Args:
        fasta_path: Path to the disorder FASTA file
        
    Returns:
        Dict mapping seq_id to list of per-residue disorder values (floats)
    """
    result = {}
    current_seq_id = None
    
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith(">"):
                # Extract seq_id from header (e.g., ">P0C1V1 TARGET=EXOTOXIN" -> "P0C1V1")
                header = line[1:]  # Remove ">"
                current_seq_id = header.split()[0]  # Take first token as seq_id
            else:
                # Parse comma-separated values
                if current_seq_id is not None:
                    try:
                        residue_scores = [float(val.strip()) for val in line.split(",")]
                        result[current_seq_id] = residue_scores
                    except ValueError as e:
                        print(f"Warning: Could not parse disorder values for {current_seq_id}: {e}")
    
    return result


def mean_disorder(residue_scores: List[float]) -> float:
    """Calculate mean disorder across residues."""
    return statistics.mean(residue_scores) if residue_scores else 0.0


def median_disorder(residue_scores: List[float]) -> float:
    """Calculate median disorder across residues."""
    return statistics.median(residue_scores) if residue_scores else 0.0


def max_disorder(residue_scores: List[float]) -> float:
    """Get maximum disorder value."""
    return max(residue_scores) if residue_scores else 0.0


def min_disorder(residue_scores: List[float]) -> float:
    """Get minimum disorder value."""
    return min(residue_scores) if residue_scores else 0.0


def std_disorder(residue_scores: List[float]) -> float:
    """Calculate standard deviation of disorder."""
    if len(residue_scores) < 2:
        return 0.0
    return statistics.stdev(residue_scores)


def p90_disorder(residue_scores: List[float]) -> float:
    """Get 90th percentile of disorder values."""
    if not residue_scores:
        return 0.0
    sorted_scores = sorted(residue_scores)
    idx = int(0.9 * len(sorted_scores))
    return sorted_scores[min(idx, len(sorted_scores) - 1)]


def p10_disorder(residue_scores: List[float]) -> float:
    """Get 10th percentile of disorder values."""
    if not residue_scores:
        return 0.0
    sorted_scores = sorted(residue_scores)
    idx = int(0.1 * len(sorted_scores))
    return sorted_scores[idx]


def frac_above_threshold(residue_scores: List[float], threshold: float = 5.0) -> float:
    """Calculate fraction of residues with disorder > threshold."""
    if not residue_scores:
        return 0.0
    return sum(1 for s in residue_scores if s > threshold) / len(residue_scores)


def frac_below_threshold(residue_scores: List[float], threshold: float = 3.0) -> float:
    """Calculate fraction of residues with disorder < threshold."""
    if not residue_scores:
        return 0.0
    return sum(1 for s in residue_scores if s < threshold) / len(residue_scores)


# Registry of available aggregation methods
AGGREGATION_METHODS = {
    "mean": mean_disorder,
    "median": median_disorder,
    "max": max_disorder,
    "min": min_disorder,
    "std": std_disorder,
    "p90": p90_disorder,
    "p10": p10_disorder,
    "frac_above_5": lambda x: frac_above_threshold(x, threshold=5.0),
    "frac_above_7": lambda x: frac_above_threshold(x, threshold=7.0),
    "frac_below_3": lambda x: frac_below_threshold(x, threshold=3.0),
    "frac_below_5": lambda x: frac_below_threshold(x, threshold=5.0),
}


def aggregate_residue_scores(
    residue_scores: List[float], method: str = "mean"
) -> float:
    """
    Compute a single aggregate score from per-residue values.
    
    Args:
        residue_scores: List of per-residue disorder values
        method: Aggregation method (key in AGGREGATION_METHODS)
        
    Returns:
        Single float representing the aggregate disorder
    """
    if method not in AGGREGATION_METHODS:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Available: {list(AGGREGATION_METHODS.keys())}"
        )
    
    return AGGREGATION_METHODS[method](residue_scores)


def load_and_aggregate_disorder(
    dataset_fasta_path: Path, disorder_fasta_path: Path, method: str = "mean"
) -> Dict[str, float]:
    """
    Load per-residue disorder from file and aggregate using specified method.
    
    Args:
        dataset_fasta_path: Path to the main dataset FASTA (for validation only)
        disorder_fasta_path: Path to the disorder FASTA file
        method: Aggregation method name
        
    Returns:
        Dict mapping seq_id -> aggregated disorder score (float)
    """
    if not disorder_fasta_path.exists():
        print(f"Warning: Disorder file not found at {disorder_fasta_path}")
        return {}
    
    disorder_raw = parse_disorder_fasta(disorder_fasta_path)
    disorder_agg = {
        seq_id: aggregate_residue_scores(residue_vals, method=method)
        for seq_id, residue_vals in disorder_raw.items()
    }
    
    return disorder_agg


def get_available_methods() -> List[str]:
    """Return list of available aggregation methods."""
    return list(AGGREGATION_METHODS.keys())
