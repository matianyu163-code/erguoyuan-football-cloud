"""Fail-closed validation for model input bundles before dry-run planning."""

from __future__ import annotations

from dataclasses import dataclass

from erguoyuan_football.prediction.model_input_adapter import ModelInputBundle


@dataclass(frozen=True)
class ModelInputValidation:
    """Explicit bundle validity and any reason that blocks its use."""

    status: str
    reason: str | None


class ModelInputValidator:
    """Reject synthetic, future, unlineaged, or category-overlapping data."""

    def validate(self, bundle: ModelInputBundle, *, production: bool = True
                 ) -> ModelInputValidation:
        """Check the evidence-bearing records without fitting a model."""
        rows = bundle.training_dataset.matches
        if production and bundle.training_dataset.dataset_kind != "REAL":
            return ModelInputValidation("INVALID", "PRODUCTION_INPUT_REJECTED")
        if any(not row.source or row.source.upper() in {"MOCK", "DEMO", "SYNTHETIC_TEST"}
               for row in rows):
            return ModelInputValidation("INVALID", "TEST_OR_UNSOURCED_INPUT_REJECTED")
        for row in rows:
            if row.kickoff_time >= bundle.cutoff:
                return ModelInputValidation("INVALID", "FUTURE_SAMPLE_REJECTED")
            if row.completed_at is None or row.completed_at >= bundle.cutoff:
                return ModelInputValidation("INVALID", "RESULT_NOT_KNOWN_AT_CUTOFF")
            if row.as_of_time is None or row.as_of_time > bundle.cutoff:
                return ModelInputValidation("INVALID", "EVIDENCE_AFTER_CUTOFF")
            if not row.source or row.data_version is None:
                return ModelInputValidation("INVALID", "EVIDENCE_LINEAGE_MISSING")
        if not bundle.evidence_ids and rows:
            return ModelInputValidation("INVALID", "EVIDENCE_LINEAGE_MISSING")
        direct = {row.match_id for row in (*bundle.direct_samples, *bundle.competition_samples,
                                           *bundle.comparable_samples)}
        prior = {row.match_id for row in bundle.prior_samples}
        if direct & prior:
            return ModelInputValidation("INVALID", "DIRECT_PRIOR_OVERLAP")
        return ModelInputValidation("VALID", None)

