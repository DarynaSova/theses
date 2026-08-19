from typing import List
from junban import PipelineStep
import torch

from ..al_context import ALContext

from .....server_management import ActiveLearningResult


class AcquisitionStep(PipelineStep[ALContext]):
    def _check_entry_assumptions(self, context: ALContext) -> bool:
        assert context.desirability is not None
        assert context.uncertainty is not None
        return True

    def _check_exit_assumptions(self, context: ALContext) -> bool:
        assert context.scores is not None
        return True

    def get_start_message(self) -> str:
        return "Running scoring via acquisition function..."

    def get_end_message(self) -> str:
        return "Samples scored."

    @staticmethod
    def _upper_confidence_bound(desirability, uncertainty, beta, disorder=None, gamma: float = 0.0):
        """
        Calculate acquisition score combining desirability, uncertainty, and disorder.
        
        Formula: acquisition = desirability + beta * uncertainty + gamma * disorder
        
        Args:
            desirability: Proximity to target (higher is better)
            uncertainty: Model uncertainty (higher = more exploration)
            beta: Exploitation-exploration coefficient
            disorder: Per-sequence disorder aggregates (optional)
            gamma: Weight for disorder term
            
        Returns:
            Acquisition scores as tensor
        """
        acquisition = desirability + beta * uncertainty
        if disorder is not None and gamma != 0.0:
            acquisition = acquisition + gamma * disorder
        return acquisition

    def _aggregate_disorder_residues(self, residue_scores: List[float], method: str) -> float:
        """
        Aggregate per-residue disorder scores using specified method.
        
        Args:
            residue_scores: List of per-residue disorder values
            method: Aggregation method (mean, median, max, p90, frac_above_5, etc.)
            
        Returns:
            Single float aggregated disorder score
        """
        if not residue_scores:
            return 0.0
        
        # Import methods from disorder_loader at runtime to avoid circular dependency
        try:
            from disorder_loader import aggregate_residue_scores
            return aggregate_residue_scores(residue_scores, method=method)
        except Exception as e:
            # Fallback to mean if import fails
            print(f"Warning: Could not aggregate disorder using method '{method}': {e}. Falling back to mean.")
            return sum(residue_scores) / len(residue_scores) if residue_scores else 0.0

    def _execute(self, context: ALContext) -> ALContext:
        # Calculate acquisition scores
        beta = context.coefficient
        desirability = context.desirability
        uncertainty = context.uncertainty
        
        # Read disorder config from context
        use_disorder = getattr(context, "use_disorder_in_acquisition", False)
        aggregation_method = getattr(context, "disorder_aggregation_method", "mean")
        gamma = getattr(context, "disorder_weight", 1.0)
        
        # Build disorder tensor from inference_data raw values
        disorder_list = []
        for key in context.inference_data.keys():
            seq = context.inference_data[key]
            disorder_score = 0.0
            
            if use_disorder:
                try:
                    attrs = getattr(seq, "attributes", None)
                    if isinstance(attrs, dict):
                        residue_scores = attrs.get("disorder_residues", None)
                        if residue_scores is None:
                            residue_scores = attrs.get("DISORDER_RESIDUES", None)
                    else:
                        residue_scores = getattr(seq, "disorder_residues", None)
                        if residue_scores is None:
                            residue_scores = getattr(seq, "DISORDER_RESIDUES", None)
                    
                    if residue_scores is not None and isinstance(residue_scores, (list, tuple)):
                        # Aggregate raw per-residue values using specified method
                        disorder_score = self._aggregate_disorder_residues(
                            residue_scores, method=aggregation_method
                        )
                except Exception as e:
                    print(f"Warning: Could not compute disorder for {key}: {e}")
                    disorder_score = 0.0
            
            disorder_list.append(disorder_score)
        
        # Create the disorder tensor once after the loop
        disorder_tensor = torch.tensor(
            disorder_list,
            dtype=getattr(context.uncertainty, "dtype", torch.float32)
            if context.uncertainty is not None
            else torch.float32,
        )

        if use_disorder:
            print(
                f"[ACQ] use_disorder={use_disorder}, method={aggregation_method}, gamma={gamma}, "
                f"disorder_len={len(disorder_list)}, disorder_sample={disorder_list[:5]}"
            )

        acquisition_scores = self._upper_confidence_bound(
            desirability, uncertainty, beta, disorder=disorder_tensor if use_disorder else None, gamma=gamma
        )

        context.scores = acquisition_scores

        # Create AL results
        results: List[ActiveLearningResult] = []
        for idx, key in enumerate(context.inference_data.keys()):
            sid = key
            pred = str(context.predictions[idx])
            uncertainty_val = context.uncertainty[idx].item()
            score = context.scores[idx].item()
            al_result = ActiveLearningResult(
                entity_id=sid, prediction=pred, uncertainty=uncertainty_val, score=score
            )
            results.append(al_result)

        context.al_results = results
        return context
