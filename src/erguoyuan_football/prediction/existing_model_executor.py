"""Bridge readiness-approved bundles into the existing model lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from erguoyuan_football.contracts.common import Availability, ExecutionStatus
from erguoyuan_football.contracts.predictions import ProbabilityVector
from erguoyuan_football.data.availability import (
    AvailabilityItem,
    DataAvailabilityReport,
)
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.registry import ModelRegistry
from erguoyuan_football.prediction.model_execution_engine import ModelExecutionResult
from erguoyuan_football.prediction.model_execution_planner import ExecutionPlanEntry
from erguoyuan_football.prediction.model_input_adapter import ModelInputBundle


class ExistingModelExecutor:
    """Run existing code only for planned models, with no prediction fallbacks."""

    def __init__(self, match: Fixture, snapshot: PredictionSnapshot,
                 training_cutoff: datetime, config: ModelConfig,
                 *, registry: ModelRegistry | None = None,
                 artifacts: Mapping[str, BaseFootballModel] | None = None) -> None:
        self.match = match
        self.snapshot = snapshot
        self.training_cutoff = training_cutoff
        self.config = config
        self.registry = registry or ModelRegistry()
        self.artifacts = dict(artifacts or {})

    def __call__(self, entry: ExecutionPlanEntry,
                 bundle: ModelInputBundle) -> ModelExecutionResult:
        """Fit/predict from the supplied real bundle or use a prevalidated artifact."""
        if self.snapshot.match_data_snapshot != self.match:
            raise ValueError("EXECUTOR_SNAPSHOT_FIXTURE_MISMATCH")
        if entry.mode == "PREDICT_ARTIFACT":
            model = self.artifacts.get(entry.model_id)
            if model is None or not model.fitted or model.trained_until is None:
                raise ValueError("VALIDATED_MODEL_ARTIFACT_NOT_LOADED")
            if model.trained_until > self.snapshot.prediction_time:
                raise ValueError("ARTIFACT_TRAINING_CUTOFF_AFTER_PREDICTION")
        elif entry.mode == "FIT_AND_PREDICT":
            model = self.registry.get_model(entry.model_id)
            fit_config = self.config
            if entry.requirements.training_min_matches is not None:
                fit_config = self.config.model_copy(update={
                    "min_matches": entry.requirements.training_min_matches,
                })
            if entry.requirements.training_min_team_matches is not None:
                fit_config = fit_config.model_copy(update={
                    "min_team_matches": entry.requirements.training_min_team_matches,
                })
            model.fit(bundle.training_dataset, self.training_cutoff, fit_config)
            if entry.model_id == "BAYESIAN_HIERARCHICAL_V1":
                setter = getattr(model, "set_prediction_prior", None)
                if setter is None or bundle.bayesian_prior is None:
                    raise ValueError("BAYESIAN_PRIOR_NOT_AVAILABLE_OR_CONSUMABLE")
                setter(bundle.bayesian_prior, direct_count=len(bundle.direct_samples))
        else:
            raise ValueError("BLOCKED_MODEL_CANNOT_BE_EXECUTED")
        snapshot = self._snapshot_with_lineage(bundle)
        prediction = model.predict(self.match, snapshot)
        if prediction.execution_status != ExecutionStatus.SUCCESS:
            raise RuntimeError(prediction.reason or prediction.execution_status.value)
        if (prediction.p_home is None or prediction.p_draw is None
                or prediction.p_away is None):
            raise ValueError("MODEL_SUCCESS_WITHOUT_PROBABILITIES")
        vector = ProbabilityVector(p_home=prediction.p_home, p_draw=prediction.p_draw,
                                  p_away=prediction.p_away)
        return ModelExecutionResult(entry.model_id, "EXECUTED", vector,
            {"model_version": prediction.model_version,
             "trained_until": prediction.trained_until.isoformat()
                 if prediction.trained_until else None,
             "prediction_snapshot_id": prediction.prediction_snapshot_id,
             "lambda_home": prediction.lambda_home,
             "lambda_away": prediction.lambda_away,
             "score_matrix": prediction.score_matrix,
             "metadata": prediction.metadata},
            "HIGH" if not bundle.degraded else "VERY_HIGH", "PENDING")

    def _snapshot_with_lineage(self, bundle: ModelInputBundle) -> PredictionSnapshot:
        """Attach evidence IDs for data-completeness checks without adding fake rows."""
        if self.snapshot.data_completeness is not None:
            return self.snapshot
        evidence = bundle.evidence_ids
        available = AvailabilityItem(availability=Availability.AVAILABLE,
            reason="PIT_TRAINING_EVIDENCE_ATTACHED", evidence_ids=evidence)
        missing = AvailabilityItem(availability=Availability.UNAVAILABLE,
            reason="NO_EVIDENCE_IN_MODEL_INPUT")
        report = DataAvailabilityReport(match_id=self.match.match_id, items={
            "historical_results": available if bundle.training_dataset.matches else missing,
            "historical_goals": available if bundle.training_dataset.matches else missing,
        })
        return self.snapshot.model_copy(update={"data_completeness": report})
