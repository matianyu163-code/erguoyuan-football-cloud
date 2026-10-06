"""Dynamic Bayesian Poisson V1 model wrapper."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from datetime import datetime
from typing import Any, ClassVar

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.dynamic_bayes.diagnostics import ModelHealthReport
from erguoyuan_football.models.dynamic_bayes.inference import DynamicBayesianInference
from erguoyuan_football.models.dynamic_bayes.state_model import PreMatchTeamState
from erguoyuan_football.models.point_in_time import (
    FAILURE_CODE_TRAINING_CUTOFF_AFTER_PREDICTION,
    FutureHistoryLeakageError,
    validate_history_point_in_time,
)
from erguoyuan_football.models.training import InsufficientData, TrainingDataset

LOGGER = logging.getLogger(__name__)


class CoreDynamicBayesianPoissonModel(BaseFootballModel):
    """Sequential Bayesian state-space Poisson model with a strict PIT boundary.

    The inference adapter uses a Poisson likelihood and sequential Laplace
    posterior updates. It is intentionally self-contained for Python 3.11;
    no deterministic form heuristic or synthetic fallback is used.
    """

    model_id = "DYNAMIC_BAYESIAN_POISSON_V1"
    model_name = "Dynamic Bayesian Poisson"
    model_version = "1.0.0"
    required_data = ("historical_goals",)
    optional_data: ClassVar[tuple[str, ...]] = ()
    allows_unseen_teams: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self.inference: DynamicBayesianInference | None = None
        self.health_report: ModelHealthReport | None = None
        self._state_history: tuple[PreMatchTeamState, ...] = ()
        self._active_prediction_time: datetime | None = None

    @property
    def state_history(self) -> tuple[PreMatchTeamState, ...]:
        """Pre-match states persisted for audit and future state queries."""
        return self._state_history

    def _fit(self, data: TrainingDataset) -> None:
        if self.trained_until is None:
            raise ValueError("DYNAMIC_TRAINING_CUTOFF_REQUIRED")
        inference = DynamicBayesianInference(
            self.config, model_id=self.model_id, model_version=self.model_version,
        )
        diagnostics = inference.fit(data, self.trained_until)
        self.inference = inference
        self.health_report = ModelHealthReport.from_diagnostics(diagnostics)
        if self.health_report.status == "FAILED":
            self.metadata.update({
                "inference_method": "SEQUENTIAL_LAPLACE_POISSON",
                "model_health": self.health_report.model_dump(mode="json"),
                "state_history_count": len(inference.state_store.snapshot()),
            })
            raise RuntimeError("DYNAMIC_DIAGNOSTICS_FAILED")
        self._state_history = inference.state_store.snapshot()
        self.metadata.update({
            "inference_method": "SEQUENTIAL_LAPLACE_POISSON",
            "state_process": self.config.dynamic_state_process,
            "time_index_version": self.config.dynamic_time_index,
            "dynamic_sigma_attack": self.config.dynamic_sigma_attack,
            "dynamic_sigma_defence": self.config.dynamic_sigma_defence,
            "dynamic_season_transition_weight": self.config.dynamic_season_transition_weight,
            "posterior_draws": self.config.dynamic_posterior_draws,
            "training_time_seconds": diagnostics.sampling_time_seconds,
            "model_health": self.health_report.model_dump(mode="json"),
            "state_history_count": len(self._state_history),
        })

    def _predict_values(self, match: Fixture) -> dict[str, Any]:
        """Produce primitive values for BaseFootballModel compatibility."""
        if self.inference is None or self._active_prediction_time is None:
            raise RuntimeError("DYNAMIC_INFERENCE_NOT_READY")
        return self.inference.posterior_prediction(match, self._active_prediction_time).values

    def predict(self, match: Fixture, prediction_snapshot: PredictionSnapshot, *,
                history: Iterable[Any] | None = None) -> ModelPrediction:
        """Predict against exactly one frozen snapshot and isolate failures."""
        started = time.perf_counter()
        try:
            if history is not None:
                validate_history_point_in_time(history, prediction_snapshot.prediction_time)
            self.validate_prediction_input(match, prediction_snapshot)
            if self.inference is None or self.health_report is None:
                raise ValueError("DYNAMIC_INFERENCE_NOT_READY")
            if self.health_report.status == "FAILED":
                raise ValueError("DYNAMIC_MODEL_HEALTH_FAILED")
            self._active_prediction_time = prediction_snapshot.prediction_time
            values = self._predict_values(match)
            warnings = () if self.health_report.status == "GOOD" else ("MODEL_HEALTH_WARNING",)
            values = {**values, "warnings": warnings}
            return self.prediction_record(
                match, prediction_snapshot, ExecutionStatus.SUCCESS, values=values,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except InsufficientData as error:
            LOGGER.info("%s unavailable: %s", self.model_id, error)
            return self.prediction_record(
                match, prediction_snapshot, ExecutionStatus.UNAVAILABLE, reason=str(error),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except FutureHistoryLeakageError as error:
            LOGGER.error("%s prediction blocked by PIT guard: %s", self.model_id, error)
            return self.prediction_record(
                match, prediction_snapshot, ExecutionStatus.FAILED, reason=str(error),
                failure_code=FAILURE_CODE_TRAINING_CUTOFF_AFTER_PREDICTION,
                audit_trained_until=error.trained_until,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as error:
            LOGGER.exception("%s prediction failed", self.model_id)
            return self.prediction_record(
                match, prediction_snapshot, ExecutionStatus.FAILED, reason=f"{type(error).__name__}:{error}",
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
