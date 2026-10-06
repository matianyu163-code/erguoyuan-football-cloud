"""Independent model runner with per-model failure isolation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from erguoyuan_football.contracts.common import Availability, ExecutionStatus
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.models.artifact import cache_key
from erguoyuan_football.models.artifact_index import ArtifactIndex
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.point_in_time import (
    FAILURE_CODE_TRAINING_CUTOFF_AFTER_PREDICTION,
)
from erguoyuan_football.models.registry import ModelRegistry
from erguoyuan_football.models.training import InsufficientData, TrainingDataset

LOGGER = logging.getLogger(__name__)


class BaseModelPredictionBundle:
    """Independent base-model results, including null-valued unavailable/failed results."""

    def __init__(self, predictions: tuple[ModelPrediction, ...]):
        self.predictions = predictions

    def by_model(self) -> dict[str, ModelPrediction]:
        """Return a detached map for comparison reporting."""
        return {prediction.model_id: prediction for prediction in self.predictions}

    @property
    def dependency_tags(self) -> dict[str, tuple[str, ...]]:
        """Expose upstream information dependencies for later META double-counting guards."""
        return {prediction.model_id: prediction.dependency_tags for prediction in self.predictions}


class ModelRunner:
    """Prepare and run each model independently; no prediction is passed between models."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def _unavailable(self, model: Any, match: Fixture, snapshot: PredictionSnapshot, reason: str) -> ModelPrediction:
        return model.prediction_record(match, snapshot, ExecutionStatus.UNAVAILABLE, reason=reason)

    def run(self, match: Fixture, snapshot: PredictionSnapshot, training_data: TrainingDataset,
            *, trained_until, config: ModelConfig | None = None,
            model_ids: tuple[str, ...] | None = None,
            artifact_root: str | Path | None = None) -> BaseModelPredictionBundle:
        """Fit and predict each model, continuing if another model raises."""
        chosen = model_ids or self.registry.list_models()
        active_config = config or ModelConfig()
        outputs: list[ModelPrediction] = []
        for model_id in chosen:
            model = self.registry.get_model(model_id)
            artifact_index = ArtifactIndex(Path(artifact_root) / "artifact_index.duckdb") if artifact_root is not None else None
            try:
                model_config = model.prepare_config(active_config)
                required_data = model.required_data_for(model_config)
                if trained_until > snapshot.prediction_time:
                    outputs.append(model.prediction_record(
                        match, snapshot, ExecutionStatus.FAILED,
                        reason="POINT_IN_TIME_GUARD_V1 training cutoff is after prediction_time.",
                        failure_code=FAILURE_CODE_TRAINING_CUTOFF_AFTER_PREDICTION,
                        audit_trained_until=trained_until,
                    ))
                    continue
                report = snapshot.data_completeness
                if required_data and report is None:
                    raise InsufficientData("MISSING_DATA_AVAILABILITY_REPORT")
                missing = tuple(name for name in required_data
                                if report is None or report.items.get(name) is None
                                or report.items[name].availability != Availability.AVAILABLE)
                if missing:
                    raise InsufficientData("MISSING_REQUIRED_DATA:" + ",".join(missing))
                prepared_training = model.prepare_training_data(training_data, model_config)
                expected_data_hash = prepared_training.window(trained_until, model_config.training_window).data_hash
                cache_directory = None
                if artifact_root is not None:
                    indexed_path = artifact_index.find_model(model.model_id, model.model_version, trained_until,
                        model_config.config_hash, expected_data_hash) if artifact_index else None
                    identity = cache_key(model.model_id, model.model_version, trained_until.isoformat(),
                                         expected_data_hash, model_config.config_hash)
                    cache_directory = Path(indexed_path) if indexed_path else Path(artifact_root) / model.model_id / identity
                    if (cache_directory / "manifest.json").is_file():
                        try:
                            cached = type(model).load(cache_directory)
                            if (cached.trained_until != trained_until or cached.training_data_hash != expected_data_hash
                                    or cached.config.config_hash != model_config.config_hash):
                                raise ValueError("ARTIFACT_CACHE_LINEAGE_MISMATCH")
                            model = cached
                        except (OSError, ValueError) as error:
                            LOGGER.info("ignoring invalid %s cache: %s", model_id, error)
                if not model.fitted:
                    model.fit(prepared_training, trained_until, model_config)
                    if cache_directory is not None:
                        artifact = model.save(cache_directory)
                        if artifact_index:
                            artifact_index.register_model(artifact)
                outputs.append(model.predict(match, snapshot))
            except InsufficientData as error:
                outputs.append(self._unavailable(model, match, snapshot, str(error)))
            except Exception as error:  # one model cannot break independent siblings
                LOGGER.exception("base model %s failed", model_id)
                outputs.append(model.prediction_record(match, snapshot, ExecutionStatus.FAILED,
                                                       reason=f"{type(error).__name__}:{error}"))
            finally:
                if artifact_index:
                    artifact_index.close()
        return BaseModelPredictionBundle(tuple(outputs))
