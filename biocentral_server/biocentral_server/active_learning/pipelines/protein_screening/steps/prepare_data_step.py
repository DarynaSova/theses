from junban import PipelineStep

from ..screening_pipeline_context import ScreeningPipelineContext


class PrepareDataStep(PipelineStep[ScreeningPipelineContext]):
    def _check_entry_assumptions(self, context: ScreeningPipelineContext) -> bool:
        assert len(context.embeddings) > 0
        return True

    def _check_exit_assumptions(self, context: ScreeningPipelineContext) -> bool:
        assert context.training_data is not None
        assert context.inference_data is not None
        assert len(context.training_data) + len(context.inference_data) == len(
            context.embeddings
        )
        return True

    def get_start_message(self) -> str:
        return "Preparing data..."

    def get_end_message(self) -> str:
        return "Data prepared."

    def _execute(self, context: ScreeningPipelineContext) -> ScreeningPipelineContext:
        id2emb = {embd.get_hash(): embd.embedding for embd in context.embeddings}
        train_data = {}
        inference_data = {}
        for data_point in context.al_iteration_data:
            if data_point.get_set() == "pred":
                inference_data[data_point.seq_id] = data_point.copy_with_embedding(
                    id2emb[data_point.get_hash()]
                )
            else:
                train_data[data_point.seq_id] = data_point.copy_with_embedding(
                    id2emb[data_point.get_hash()]
                )

        # Preserve attributes (e.g., disorder_residues) from original iteration_data
        # in case copy_with_embedding does not preserve them.
        for original in context.al_iteration_data:
            sid = original.seq_id
            original_attrs = getattr(original, "attributes", None) or {}

            for target_dict in (inference_data, train_data):
                if sid not in target_dict:
                    continue

                copied = target_dict[sid]
                copied_attrs = getattr(copied, "attributes", None)
                if copied_attrs is None:
                    copied.attributes = {}
                    copied_attrs = copied.attributes

                for key in ("disorder_residues", "DISORDER_RESIDUES"):
                    if key in original_attrs and key not in copied_attrs:
                        copied_attrs[key] = original_attrs[key]

        context.training_data = train_data
        context.inference_data = inference_data

        return context
