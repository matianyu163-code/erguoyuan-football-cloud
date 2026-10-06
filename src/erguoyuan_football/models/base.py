"""Independent fit/predict lifecycle shared by goal and rating wrappers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import joblib

from erguoyuan_football.contracts.common import (
    Availability,
    ExecutionStatus,
    ImplementationType,
    utc,
)
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.models.artifact import (
    ModelArtifact,
    dependency_versions,
    file_hash,
)
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.point_in_time import (
    FAILURE_CODE_TRAINING_CUTOFF_AFTER_PREDICTION,
    FutureHistoryLeakageError,
)
from erguoyuan_football.models.training import (
    InsufficientData,
    TrainingDataset,
    TrainingDatasetValidator,
)

LOGGER = logging.getLogger(__name__)


class BaseFootballModel(ABC):
    """No model can observe another model's outputs through this interface."""

    model_id: ClassVar[str]
    model_name: ClassVar[str]
    model_version: ClassVar[str] = "1.0.0"
    implementation_type: ClassVar[str] = "REAL_IMPLEMENTATION"
    required_data: ClassVar[tuple[str, ...]] = ("historical_goals",)
    optional_data: ClassVar[tuple[str, ...]] = ()
    supports_score_matrix: ClassVar[bool] = True
    supports_expected_goals: ClassVar[bool] = True
    supports_1x2: ClassVar[bool] = True
    allows_unseen_teams: ClassVar[bool] = False
    allows_multiple_competitions: ClassVar[bool] = False
    allows_senior_national_multi_competition: ClassVar[bool] = False

    def __init__(self) -> None:
        """Construct an unfitted model with no fallback state."""
        self.fitted = False
        self.config = ModelConfig()
        self.trained_until: datetime | None = None
        self.training_data_hash = ""
        self.training_ids: frozenset[str] = frozenset()
        self.teams: frozenset[str] = frozenset()
        self.competition_id = ""
        self.sources: tuple[str, ...] = ()
        self.metadata: dict[str, Any] = {}

    def validate_training_data(self, data: TrainingDataset, trained_until: datetime) -> None:
        """Validate before windowing; no invalid future row is silently dropped."""
        TrainingDatasetValidator().validate(data, trained_until, max_score=self.config.max_score)
        if data.dataset_kind == "SYNTHETIC_TEST" and not self.config.allow_test_data:
            raise InsufficientData("SYNTHETIC_DATA_FORBIDDEN_IN_PRODUCTION")

    def fit(self, training_data: TrainingDataset, trained_until: datetime, config: ModelConfig) -> Self:
        """Fit real parameters or raise; never leave a stale fitted model after failure."""
        self.fitted = False
        self.config = config
        at = utc(trained_until)
        self.validate_training_data(training_data, at)
        data = training_data.window(at, config.training_window)
        if len(data.matches) < config.min_matches:
            raise InsufficientData(f"INSUFFICIENT_DATA:{len(data.matches)}<{config.min_matches}")
        leagues = {row.competition_id for row in data.matches}
        team_ids = {team for row in data.matches
                    for team in (row.home_team_id, row.away_team_id)}
        senior_national_scope = (
            self.allows_senior_national_multi_competition
            and bool(team_ids)
            and all(team.startswith("NATIONAL_") and team.endswith("_M_SENIOR")
                    for team in team_ids)
        )
        if len(leagues) != 1 and not (
                self.allows_multiple_competitions or senior_national_scope):
            raise InsufficientData("V1_REQUIRES_ONE_COMPETITION_PER_FIT")
        counts = Counter(team for row in data.matches for team in (row.home_team_id, row.away_team_id))
        if len(counts) < config.min_team_matches:
            raise InsufficientData("INSUFFICIENT_TEAM_COVERAGE")
        self.trained_until = at
        self.training_data_hash = data.data_hash
        self.training_ids = frozenset(row.match_id for row in data.matches)
        self.teams = frozenset(counts)
        self.competition_id = next(iter(leagues)) if len(leagues) == 1 else "MULTI_COMPETITION"
        self.competition_ids = frozenset(leagues)
        self.sources = tuple(sorted({row.source for row in data.matches}))
        self.metadata = {"dataset_kind": data.dataset_kind, "training_sample_count": len(data.matches),
                         "team_match_counts": dict(counts),
                         "training_team_coverage": len(counts),
                         "minimum_training_team_coverage": config.min_team_matches,
                         "xi": config.time_decay,
                         "training_window_days": config.training_window, "training_data_hash": data.data_hash,
                         "config_hash": config.config_hash, "competition_id": self.competition_id,
                         "temporal_mode": data.temporal_mode,
                         "training_assumptions": list(data.assumptions),
                         "competition_ids": sorted(leagues),
                         "competition_hierarchy": data.competition_hierarchy,
                         "team_hierarchy": data.team_hierarchy}
        self._fit(data)
        self.fitted = True
        return self

    @abstractmethod
    def _fit(self, data: TrainingDataset) -> None:
        """Fit this implementation, accessing only its own training rows."""

    def prepare_config(self, config: ModelConfig) -> ModelConfig:
        """Allow a model-specific typed config while preserving the common runner contract."""
        return config

    def required_data_for(self, config: ModelConfig) -> tuple[str, ...]:
        """Return mode-dependent input requirements without executing model code."""
        return self.required_data

    def prepare_training_data(self, data: TrainingDataset, config: ModelConfig) -> TrainingDataset:
        """Apply a declared ablation before hashing and fitting; default models keep all inputs."""
        return data

    def validate_prediction_input(self, match: Fixture, snapshot: PredictionSnapshot) -> None:
        """Require a matching frozen fixture and a strictly legal training lineage."""
        if not self.fitted or self.trained_until is None:
            raise InsufficientData("MODEL_NOT_FITTED")
        if match != snapshot.match_data_snapshot or match.match_id != snapshot.match_id:
            raise ValueError("SNAPSHOT_MATCH_MISMATCH")
        if self.trained_until > snapshot.prediction_time:
            raise FutureHistoryLeakageError(
                "POINT_IN_TIME_GUARD_V1 training cutoff is after prediction_time.",
                trained_until=self.trained_until,
            )
        if not snapshot.prediction_time < match.kickoff_time:
            raise ValueError("TRAINING_OR_PREDICTION_TIME_LEAKAGE")
        if match.match_id in self.training_ids:
            raise ValueError("IN_SAMPLE: target match used in training")
        if match.competition_id not in getattr(self, "competition_ids", {self.competition_id}):
            raise InsufficientData("UNTRAINED_COMPETITION")
        if not self.allows_unseen_teams and not {match.home_team_id, match.away_team_id} <= self.teams:
            raise InsufficientData("UNSEEN_TEAM")
        if match.neutral_venue is None:
            raise InsufficientData("UNKNOWN_NEUTRAL_VENUE")

    @abstractmethod
    def _predict_values(self, match: Fixture) -> dict[str, Any]:
        """Return primitive CORE fields, not third-party objects."""

    def prediction_record(self, match: Fixture, snapshot: PredictionSnapshot, status: ExecutionStatus,
                          *, reason: str | None = None, elapsed_ms: float = 0,
                          values: dict[str, Any] | None = None, failure_code: str | None = None,
                          audit_trained_until: datetime | None = None,
                          dependency_tags: tuple[str, ...] = ()) -> ModelPrediction:
        """Build successful or null-valued failure records with the same audit contract."""
        fields = dict(values or {})
        extra = fields.pop("metadata", {})
        data_sources = set(self.sources) | {snapshot.match_data_snapshot.source}
        for group in (snapshot.market_snapshot, snapshot.team_stats_snapshot, snapshot.xg_snapshot,
                      snapshot.lineup_snapshot, snapshot.injury_snapshot, snapshot.external_snapshot,
                      snapshot.historical_results):
            data_sources.update(item.source if hasattr(item, "source") else item.result.source for item in group)
        if snapshot.market_goal_features is not None:
            data_sources.update(snapshot.market_goal_features.source_ids)
        audit_cutoff = audit_trained_until if audit_trained_until is not None else self.trained_until
        if status != ExecutionStatus.FAILED and audit_cutoff is not None and audit_cutoff > snapshot.prediction_time:
            audit_cutoff = None
        return ModelPrediction(
            match_id=match.match_id, prediction_snapshot_id=snapshot.prediction_snapshot_id,
            model_id=self.model_id, model_version=self.model_version,
            implementation_type=ImplementationType(self.implementation_type),
            training_end_time=audit_cutoff, trained_until=audit_cutoff,
            prediction_time=snapshot.prediction_time, input_data_version=snapshot.input_data_version,
            data_source=tuple(sorted(data_sources)),
            data_status=Availability.AVAILABLE if status == ExecutionStatus.SUCCESS else Availability.UNAVAILABLE,
            execution_status=status, reason=reason, failure_code=failure_code, execution_time_ms=elapsed_ms,
            dependency_tags=dependency_tags,
            metadata={**self.metadata, **extra,
                      "training_match_ids_hash": hashlib.sha256(json.dumps(
                          sorted(self.training_ids), ensure_ascii=False,
                          separators=(",", ":")).encode()).hexdigest(),
                      "production_status": "NOT_READY_FOR_PRODUCTION"}, **fields,
        )

    def predict(self, match: Fixture, prediction_snapshot: PredictionSnapshot) -> ModelPrediction:
        """Isolate prediction failures without inventing fallback probabilities."""
        start = time.perf_counter()
        try:
            self.validate_prediction_input(match, prediction_snapshot)
            values = self._predict_values(match)
            return self.prediction_record(match, prediction_snapshot, ExecutionStatus.SUCCESS,
                                          values=values, elapsed_ms=(time.perf_counter() - start) * 1000)
        except InsufficientData as error:
            LOGGER.info("%s unavailable: %s", self.model_id, error)
            return self.prediction_record(match, prediction_snapshot, ExecutionStatus.UNAVAILABLE, reason=str(error))
        except FutureHistoryLeakageError as error:
            LOGGER.error("%s prediction blocked by PIT guard: %s", self.model_id, error)
            return self.prediction_record(
                match, prediction_snapshot, ExecutionStatus.FAILED, reason=str(error),
                failure_code=FAILURE_CODE_TRAINING_CUTOFF_AFTER_PREDICTION,
                audit_trained_until=error.trained_until,
            )
        except Exception as error:
            LOGGER.exception("%s prediction failed", self.model_id)
            return self.prediction_record(match, prediction_snapshot, ExecutionStatus.FAILED,
                                          reason=f"{type(error).__name__}:{error}")

    def predict_many(self, matches: list[Fixture], prediction_snapshot: list[PredictionSnapshot]) -> list[ModelPrediction]:
        """Every match needs its own snapshot; mismatched batch sizes are an error."""
        if len(matches) != len(prediction_snapshot):
            raise ValueError("one snapshot per match required")
        return [self.predict(match, snapshot) for match, snapshot in zip(matches, prediction_snapshot, strict=True)]

    def get_metadata(self) -> dict[str, Any]:
        """Return model capabilities and training metadata as a detached dictionary."""
        return {**self.metadata, "model_id": self.model_id, "model_name": self.model_name,
                "model_version": self.model_version, "implementation_type": self.implementation_type,
                "trained_until": self.trained_until.isoformat() if self.trained_until else None,
                "required_data": self.required_data, "optional_data": self.optional_data,
                "supports_score_matrix": self.supports_score_matrix, "supports_expected_goals": self.supports_expected_goals,
                "supports_1x2": self.supports_1x2, "fitted": self.fitted}

    def save(self, path: str | Path) -> ModelArtifact:
        """Persist only fitted models; atomic payload replacement plus checksum manifest."""
        if not self.fitted or self.trained_until is None:
            raise InsufficientData("cannot save unfitted model")
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".joblib", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            joblib.dump(self, temporary)
            os.replace(temporary, directory / "model.joblib")
        finally:
            temporary.unlink(missing_ok=True)
        dynamic = getattr(self, "inference", None) is not None
        artifact = ModelArtifact(model_id=self.model_id, model_version=self.model_version,
            trained_until=self.trained_until, training_data_hash=self.training_data_hash,
            config_hash=self.config.config_hash, artifact_path=str(directory.resolve()),
            payload_sha256=file_hash(directory / "model.joblib"), dependencies=dependency_versions(),
            state_process=self.config.dynamic_state_process if dynamic else None,
            time_index_version=self.config.dynamic_time_index if dynamic else None,
            posterior_method=getattr(self, "metadata", {}).get("inference_method") if dynamic else None,
            sampling_diagnostics=getattr(self, "metadata", {}).get("model_health", {}) if dynamic else {},
            state_history_count=getattr(self, "metadata", {}).get("state_history_count") if dynamic else None)
        manifest_temp = directory / "manifest.json.tmp"
        manifest_temp.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        os.replace(manifest_temp, directory / "manifest.json")
        return artifact

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load only locally trusted artifacts; never deserialize user-supplied pickle."""
        directory = Path(path)
        artifact = ModelArtifact.model_validate_json((directory / "manifest.json").read_text(encoding="utf-8"))
        if artifact.dependencies != dependency_versions() or artifact.model_version != cls.model_version:
            raise ValueError("ARTIFACT_VERSION_MISMATCH")
        if file_hash(directory / "model.joblib") != artifact.payload_sha256:
            raise ValueError("ARTIFACT_CHECKSUM_MISMATCH")
        model = joblib.load(directory / "model.joblib")
        if not isinstance(model, cls) or not model.fitted or model.model_id != artifact.model_id:
            raise ValueError("ARTIFACT_MODEL_MISMATCH")
        if (model.training_data_hash, model.config.config_hash, model.trained_until) != (
                artifact.training_data_hash, artifact.config_hash, artifact.trained_until):
            raise ValueError("ARTIFACT_LINEAGE_MISMATCH")
        return cast(Self, model)
