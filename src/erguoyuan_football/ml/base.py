"""Shared lifecycle for official XGBoost/CatBoost multiclass wrappers."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from erguoyuan_football.contracts.common import (
    Availability,
    ExecutionStatus,
    ImplementationType,
)
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.ml.artifacts import (
    MLArtifact,
    current_code_version,
    file_sha256,
)
from erguoyuan_football.ml.class_order import ProbabilityClassOrderGuard
from erguoyuan_football.ml.config import MLConfig
from erguoyuan_football.ml.drift import inference_drift_warnings, training_drift_profile
from erguoyuan_football.ml.feature_lineage import OOSFeatureValidator
from erguoyuan_football.ml.labels import CLASS_MAPPING, CLASS_ORDER
from erguoyuan_football.ml.missing import feature_frame
from erguoyuan_football.ml.schemas import (
    FeatureDriftReport,
    FeatureMode,
    FeatureSchema,
    MLDataset,
    MLFeatureVector,
    ModelFamily,
    stable_hash,
)

LOGGER = logging.getLogger(__name__)


class MLDependencyUnavailable(RuntimeError):
    """Official tree library is absent; a surrogate must never run under its name."""


class CoreMLModel(ABC):
    model_id: ClassVar[str]
    model_name: ClassVar[str]
    model_version: ClassVar[str] = "1.0.0"
    implementation_type: ClassVar[str] = "REAL_IMPLEMENTATION"
    family: ClassVar[ModelFamily]
    library_name: ClassVar[str]
    model_extension: ClassVar[str]
    supports_score_matrix: ClassVar[bool] = False
    supports_expected_goals: ClassVar[bool] = False
    supports_1x2: ClassVar[bool] = True

    def __init__(self) -> None:
        self.fitted = False
        self.estimator: Any = None
        self.schema: FeatureSchema | None = None
        self.config: MLConfig | None = None
        self.dataset_hash = ""
        self.trained_until: datetime | None = None
        self.training_match_ids: frozenset[str] = frozenset()
        self.training_metadata: dict[str, Any] = {}
        self.drift_report: FeatureDriftReport | None = None

    @abstractmethod
    def _new_estimator(self, config: MLConfig) -> Any:
        """Instantiate only the official library estimator."""

    @abstractmethod
    def _fit_estimator(self, train_x: Any, train_y: np.ndarray,
                       validation_x: Any, validation_y: np.ndarray,
                       config: MLConfig) -> Any:
        """Fit on train and early-stop only on chronological validation."""

    @abstractmethod
    def _best_metadata(self) -> dict[str, Any]:
        """Return native best-iteration/score diagnostics."""

    @abstractmethod
    def _save_native(self, path: Path) -> None:
        """Write the official library's stable model format."""

    @abstractmethod
    def _load_native(self, path: Path) -> Any:
        """Load the official library's stable model format."""

    def fit(self, train: MLDataset, validation: MLDataset, config: MLConfig) -> Self:
        """Fit from OOS-only features without access to a final test partition."""
        self.fitted = False
        if config.family != self.family or train.feature_schema.model_family != self.family:
            raise ValueError("ML_MODEL_FAMILY_MISMATCH")
        if train.feature_schema != validation.feature_schema or train.feature_schema.mode != validation.feature_schema.mode:
            raise ValueError("FEATURE_SCHEMA_MISMATCH")
        if train.dataset_kind == "SYNTHETIC_TEST" and not config.allow_test_data:
            raise ValueError("SYNTHETIC_DATA_FORBIDDEN_IN_PRODUCTION")
        if train.dataset_kind == "REAL" and any(
                item.source_id == "SYNTHETIC_TEST" for row in (*train.rows, *validation.rows)
                for item in row.vector.feature_lineage):
            raise ValueError("SYNTHETIC_SOURCE_CANNOT_BE_LAUNDERED_AS_REAL")
        if train.dataset_kind != validation.dataset_kind:
            raise ValueError("ML_DATASET_KIND_MISMATCH")
        if len(train.rows) < config.min_train_rows:
            raise ValueError("INSUFFICIENT_ML_TRAINING_ROWS")
        if train.rows[-1].vector.kickoff_time >= validation.rows[0].vector.kickoff_time:
            raise ValueError("ML_VALIDATION_NOT_AFTER_TRAIN")
        if max(row.label_available_at for row in train.rows) > min(
                row.vector.prediction_time for row in validation.rows):
            raise ValueError("TRAIN_LABEL_UNAVAILABLE_AT_VALIDATION")
        ids = [row.vector.match_id for row in (*train.rows, *validation.rows)]
        if len(ids) != len(set(ids)):
            raise ValueError("ML_TRAIN_VALIDATION_OVERLAP")
        validator = OOSFeatureValidator()
        for row in (*train.rows, *validation.rows):
            validator.validate_training_vector(row.vector)
        labels = {int(row.label) for row in train.rows}
        if labels != set(CLASS_ORDER):
            raise ValueError("ML_TRAIN_REQUIRES_ALL_THREE_CLASSES")
        train_x = feature_frame(tuple(row.vector for row in train.rows), train.feature_schema)
        val_x = feature_frame(tuple(row.vector for row in validation.rows), validation.feature_schema)
        train_y = np.asarray([int(row.label) for row in train.rows], dtype=int)
        val_y = np.asarray([int(row.label) for row in validation.rows], dtype=int)
        self.schema = train.feature_schema
        self.config = config
        self.estimator = self._fit_estimator(train_x, train_y, val_x, val_y, config)
        ProbabilityClassOrderGuard.reorder(np.asarray(self.estimator.predict_proba(val_x)),
                                          self.estimator.classes_)
        self.dataset_hash = train.manifest.data_hash + ":" + validation.manifest.data_hash
        self.trained_until = max(row.label_available_at for row in (*train.rows, *validation.rows))
        self.training_match_ids = frozenset(ids)
        self.drift_report = training_drift_profile(tuple(row.vector for row in train.rows), train.feature_schema)
        self.training_metadata = {
            "dataset_kind": train.dataset_kind,
            "feature_version": train.manifest.feature_version,
            "feature_count": len(train.feature_schema.feature_names),
            "feature_mode": train.feature_schema.mode.value,
            "uses_market": train.feature_schema.mode == FeatureMode.WITH_MARKET,
            "best_iteration": self._best_metadata().get("best_iteration"),
            "best_score": self._best_metadata().get("best_score"),
            "training_rows": len(train.rows),
            "training_start": train.manifest.start_time.isoformat(),
            "training_end": train.manifest.end_time.isoformat(),
            "validation_period": (validation.manifest.start_time.isoformat(),
                                  validation.manifest.end_time.isoformat()),
            "model_config_hash": config.config_hash,
            "dataset_hash": self.dataset_hash,
            "training_match_ids_hash": stable_hash(sorted(ids)),
            "class_mapping": CLASS_MAPPING,
            "calibration_status": "NOT_APPLIED",
            "production_status": "NOT_READY_FOR_PRODUCTION",
        }
        self.fitted = True
        return self

    def _record(self, vector: MLFeatureVector, status: ExecutionStatus, *,
                probabilities: tuple[float, float, float] | None = None,
                reason: str | None = None, failure_code: str | None = None,
                elapsed_ms: float = 0.0, warnings: tuple[str, ...] = (),
                is_oos: bool = False) -> ModelPrediction:
        sources = tuple(sorted({item.source_id for item in vector.feature_lineage} or {"ML_FEATURE_STORE"}))
        base_ids = tuple(sorted({item.prediction.model_id for item in vector.base_prediction_evidence}))
        market_ids = tuple(sorted({dependency for item in vector.feature_lineage
                                   if item.source_type == "MARKET" for dependency in item.dependency_ids}))
        metadata = {**self.training_metadata,
            "uses_xg": bool(vector.features.get("xg_available")),
            "uses_base_models": bool(base_ids), "base_model_ids": base_ids,
            "market_dependency_ids": market_ids,
            "feature_schema_hash": vector.feature_schema_hash,
            "feature_data_hash": vector.feature_data_hash,
            "library_version": self._library_version(),
        }
        return ModelPrediction(match_id=vector.match_id,
            prediction_snapshot_id=vector.prediction_snapshot_id,
            model_id=self.model_id, model_version=self.model_version,
            implementation_type=ImplementationType.REAL_IMPLEMENTATION,
            training_end_time=self.trained_until, trained_until=self.trained_until,
            prediction_time=vector.prediction_time, input_data_version=vector.input_data_version,
            p_home=probabilities[0] if probabilities is not None else None,
            p_draw=probabilities[1] if probabilities is not None else None,
            p_away=probabilities[2] if probabilities is not None else None,
            data_source=sources, data_status=Availability.AVAILABLE if status == ExecutionStatus.SUCCESS
            else Availability.UNAVAILABLE, execution_status=status, reason=reason,
            failure_code=failure_code, execution_time_ms=elapsed_ms,
            is_oos=is_oos and status == ExecutionStatus.SUCCESS, warnings=warnings,
            dependency_tags=("MARKET_RAW", "MARKET_CONSENSUS") if market_ids else (),
            metadata=metadata)

    def predict(self, vector: MLFeatureVector, *, oos: bool = False,
                production: bool = False, network_gate_status: str | None = None,
                market_gate_status: str | None = None) -> ModelPrediction:
        start = time.perf_counter()
        if not self.fitted or self.schema is None or self.config is None:
            return self._record(vector, ExecutionStatus.UNAVAILABLE, reason="ML_MODEL_NOT_FITTED")
        if self.trained_until is None or self.trained_until > vector.prediction_time:
            return self._record(vector, ExecutionStatus.FAILED, reason="ML_TRAINING_AFTER_PREDICTION",
                                failure_code="TRAINING_CUTOFF_AFTER_PREDICTION")
        if production and self.training_metadata.get("dataset_kind") != "REAL":
            return self._record(vector, ExecutionStatus.UNAVAILABLE,
                                reason="SYNTHETIC_DATA_FORBIDDEN_IN_PRODUCTION")
        if production and (network_gate_status != "PASS" or market_gate_status != "PASS"):
            return self._record(vector, ExecutionStatus.UNAVAILABLE,
                                reason="PREDICTION_BLOCKED:ENHANCED_ONLY_GATE")
        if vector.feature_schema_hash != self.schema.schema_hash or vector.feature_mode != self.schema.mode:
            return self._record(vector, ExecutionStatus.FAILED, reason="FEATURE_SCHEMA_MISMATCH",
                                failure_code="FEATURE_SCHEMA_MISMATCH")
        if vector.match_id in self.training_match_ids:
            return self._record(vector, ExecutionStatus.FAILED, reason="IN_SAMPLE_ML_PREDICTION",
                                failure_code="IN_SAMPLE_ML_PREDICTION")
        if self.schema.mode == FeatureMode.WITH_MARKET and vector.features.get("market_available") != 1.0:
            return self._record(vector, ExecutionStatus.UNAVAILABLE, reason="MISSING_REQUIRED_MARKET_DATA")
        try:
            MLFeatureVector.model_validate(vector.model_dump())
            matrix = feature_frame((vector,), self.schema)
            probabilities = ProbabilityClassOrderGuard.reorder(
                np.asarray(self.estimator.predict_proba(matrix)), self.estimator.classes_)
            warnings = inference_drift_warnings(vector, self.drift_report) if self.drift_report else ()
            return self._record(vector, ExecutionStatus.SUCCESS,
                probabilities=(float(probabilities[0, 0]), float(probabilities[0, 1]),
                               float(probabilities[0, 2])),
                elapsed_ms=(time.perf_counter() - start) * 1000,
                warnings=warnings, is_oos=oos)
        except ValueError as error:
            LOGGER.warning("%s input failed: %s", self.model_id, error)
            return self._record(vector, ExecutionStatus.FAILED, reason=str(error),
                                failure_code="ML_PREDICTION_VALIDATION_FAILED")
        except Exception as error:
            LOGGER.exception("%s prediction failed", self.model_id)
            return self._record(vector, ExecutionStatus.FAILED,
                                reason=f"{type(error).__name__}:{error}")

    def predict_many(self, vectors: tuple[MLFeatureVector, ...], *, oos: bool = False,
                     production: bool = False, network_gate_status: str | None = None,
                     market_gate_status: str | None = None) -> tuple[ModelPrediction, ...]:
        """Reuse one loaded estimator for a whole day; isolate per-match failures."""
        return tuple(self.predict(vector, oos=oos, production=production,
            network_gate_status=network_gate_status, market_gate_status=market_gate_status)
            for vector in vectors)

    def feature_importance(self) -> dict[str, float]:
        """Diagnostic importance, never a causal interpretation."""
        if not self.fitted or self.schema is None:
            raise ValueError("ML_MODEL_NOT_FITTED")
        values = np.asarray(self.estimator.feature_importances_, dtype=float)
        return dict(zip(self.schema.feature_names, (float(item) for item in values), strict=True))

    @classmethod
    def _library_version(cls) -> str:
        try:
            return importlib.metadata.version(cls.library_name)
        except importlib.metadata.PackageNotFoundError:
            return "NOT_INSTALLED"

    def save(self, path: str | Path) -> MLArtifact:
        if not self.fitted or self.schema is None or self.config is None or self.trained_until is None:
            raise ValueError("ML_MODEL_NOT_FITTED")
        if self.drift_report is None:
            raise ValueError("ML_DRIFT_PROFILE_MISSING")
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        filename = "model" + self.model_extension
        with tempfile.NamedTemporaryFile(dir=directory, suffix=self.model_extension, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            self._save_native(temporary)
            os.replace(temporary, directory / filename)
        finally:
            temporary.unlink(missing_ok=True)
        artifact = MLArtifact(model_id=self.model_id, model_version=self.model_version,
            library_name=self.library_name, library_version=self._library_version(),
            feature_version=self.training_metadata["feature_version"],
            feature_schema_hash=self.schema.schema_hash, feature_schema=self.schema,
            config_hash=self.config.config_hash, config_payload=self.config.model_dump(mode="json"),
            dataset_hash=self.dataset_hash, trained_until=self.trained_until,
            class_mapping=CLASS_MAPPING, training_match_ids=tuple(sorted(self.training_match_ids)),
            training_metadata=self.training_metadata, drift_report=self.drift_report,
            artifact_path=str(directory.resolve()), model_file=filename,
            payload_sha256=file_sha256(directory / filename))
        temp_manifest = directory / "manifest.json.tmp"
        temp_manifest.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temp_manifest, directory / "manifest.json")
        return artifact

    @classmethod
    def load(cls, path: str | Path, *, expected_schema: FeatureSchema | None = None,
             expected_config_hash: str | None = None) -> Self:
        directory = Path(path)
        artifact = MLArtifact.model_validate_json((directory / "manifest.json").read_text(encoding="utf-8"))
        if (artifact.model_id != cls.model_id or artifact.model_version != cls.model_version or
                artifact.library_name != cls.library_name or
                artifact.library_version != cls._library_version() or
                artifact.code_version != current_code_version() or
                (expected_schema is not None and expected_schema.schema_hash != artifact.feature_schema_hash) or
                (expected_config_hash is not None and expected_config_hash != artifact.config_hash)):
            raise ValueError("ML_ARTIFACT_VERSION_SCHEMA_OR_CONFIG_MISMATCH")
        config = MLConfig.model_validate(artifact.config_payload)
        if config.config_hash != artifact.config_hash:
            raise ValueError("ML_ARTIFACT_CONFIG_HASH_MISMATCH")
        model_path = directory / artifact.model_file
        if file_sha256(model_path) != artifact.payload_sha256:
            raise ValueError("ML_ARTIFACT_CHECKSUM_MISMATCH")
        loaded = cls()
        loaded.estimator = loaded._load_native(model_path)
        loaded.schema = artifact.feature_schema
        loaded.config = config
        loaded.dataset_hash = artifact.dataset_hash
        loaded.trained_until = artifact.trained_until
        loaded.training_match_ids = frozenset(artifact.training_match_ids)
        loaded.training_metadata = artifact.training_metadata
        loaded.drift_report = artifact.drift_report
        loaded.fitted = True
        if {int(value) for value in loaded.estimator.classes_} != set(CLASS_ORDER):
            raise ValueError("ML_ARTIFACT_CLASS_MAPPING_MISMATCH")
        return loaded
