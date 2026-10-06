"""Frozen ML feature, dataset and diagnostic contracts."""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import Contract, Identifier, UTCTime, now
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.ml.labels import ResultClass


class FeatureMode(StrEnum):
    NO_MARKET = "NO_MARKET"
    WITH_MARKET = "WITH_MARKET"


class ModelFamily(StrEnum):
    XGBOOST = "XGBOOST"
    CATBOOST = "CATBOOST"


class FeatureLineage(Contract):
    feature_names: tuple[Identifier, ...]
    source_type: Identifier
    source_id: Identifier
    as_of_time: UTCTime
    retrieved_at: UTCTime
    data_version: Identifier
    dependency_ids: tuple[Identifier, ...] = ()


class BasePredictionEvidence(Contract):
    prediction: ModelPrediction
    training_match_ids: tuple[Identifier, ...]
    base_prediction_oos: bool
    verified_training_ids_hash: Identifier | None = None
    verified_training_match_count: int | None = Field(default=None, ge=1)
    verification_method: Literal["REAL_OOS_STORE_RECOMPUTED"] | None = None


class FeatureSchema(Contract):
    schema_version: Identifier = "ML_FEATURE_SCHEMA_V1"
    feature_names: tuple[Identifier, ...]
    feature_types: dict[str, Literal["numeric", "categorical"]]
    categorical_features: tuple[Identifier, ...] = ()
    numeric_features: tuple[Identifier, ...] = ()
    nullable_features: tuple[Identifier, ...] = ()
    mode: FeatureMode
    model_family: ModelFamily

    @model_validator(mode="after")
    def validate_columns(self) -> FeatureSchema:
        names = self.feature_names
        if not names or len(set(names)) != len(names):
            raise ValueError("FEATURE_SCHEMA_DUPLICATE_OR_EMPTY")
        if set(self.feature_types) != set(names):
            raise ValueError("FEATURE_SCHEMA_TYPES_MISMATCH")
        if set(self.numeric_features) | set(self.categorical_features) != set(names):
            raise ValueError("FEATURE_SCHEMA_GROUPS_MISMATCH")
        if set(self.numeric_features) & set(self.categorical_features):
            raise ValueError("FEATURE_SCHEMA_OVERLAP")
        if not set(self.nullable_features) <= set(names):
            raise ValueError("FEATURE_SCHEMA_NULLABLE_UNKNOWN")
        if any(self.feature_types[name] != "numeric" for name in self.numeric_features):
            raise ValueError("FEATURE_SCHEMA_NUMERIC_TYPE_MISMATCH")
        if any(self.feature_types[name] != "categorical" for name in self.categorical_features):
            raise ValueError("FEATURE_SCHEMA_CATEGORY_TYPE_MISMATCH")
        if self.model_family == ModelFamily.XGBOOST and self.categorical_features:
            raise ValueError("XGBOOST_V1_NUMERIC_ONLY")
        return self

    @property
    def schema_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class MLFeatureVector(Contract):
    match_id: Identifier
    prediction_snapshot_id: Identifier
    input_data_version: Identifier
    prediction_time: UTCTime
    kickoff_time: UTCTime
    feature_version: Identifier
    feature_schema_hash: Identifier
    feature_data_hash: Identifier
    feature_mode: FeatureMode
    model_family: ModelFamily
    features: dict[str, float | str | None]
    feature_lineage: tuple[FeatureLineage, ...]
    base_prediction_evidence: tuple[BasePredictionEvidence, ...] = ()

    @model_validator(mode="after")
    def point_in_time(self) -> MLFeatureVector:
        if self.prediction_time >= self.kickoff_time:
            raise ValueError("ML_FEATURE_AFTER_KICKOFF")
        for item in self.feature_lineage:
            if max(item.as_of_time, item.retrieved_at) > self.prediction_time:
                raise ValueError("FUTURE_FEATURE_REJECTED")
            if self.feature_mode == FeatureMode.NO_MARKET and item.source_type == "MARKET":
                raise ValueError("NO_MARKET_HAS_MARKET_LINEAGE")
        if any(isinstance(value, float) and not math.isfinite(value)
               for value in self.features.values() if value is not None):
            raise ValueError("NONFINITE_FEATURE")
        content = self.model_dump(mode="json", exclude={"feature_data_hash"})
        if self.feature_data_hash != stable_hash(content):
            raise ValueError("FEATURE_DATA_HASH_MISMATCH")
        return self


class MLTrainingRow(Contract):
    vector: MLFeatureVector
    label: ResultClass
    result_id: Identifier
    label_available_at: UTCTime
    competition_id: Identifier
    season: str | None = None

    @model_validator(mode="after")
    def label_timing(self) -> MLTrainingRow:
        if self.label_available_at <= self.vector.kickoff_time:
            raise ValueError("LABEL_AVAILABLE_BEFORE_MATCH")
        return self


class MLDatasetManifest(Contract):
    dataset_id: Identifier = Field(default_factory=lambda: str(uuid4()))
    feature_version: Identifier
    row_count: int = Field(ge=1)
    start_time: UTCTime
    end_time: UTCTime
    competitions: tuple[Identifier, ...]
    feature_schema_hash: Identifier
    data_hash: Identifier
    created_at: UTCTime = Field(default_factory=now)


class MLDataset(Contract):
    manifest: MLDatasetManifest
    feature_schema: FeatureSchema
    rows: tuple[MLTrainingRow, ...]
    dataset_kind: Literal["REAL", "SYNTHETIC_TEST"]

    @model_validator(mode="after")
    def dataset_consistency(self) -> MLDataset:
        if len(self.rows) != self.manifest.row_count or len({row.vector.match_id for row in self.rows}) != len(self.rows):
            raise ValueError("ML_DATASET_DUPLICATE_OR_COUNT_MISMATCH")
        if self.feature_schema.schema_hash != self.manifest.feature_schema_hash:
            raise ValueError("ML_DATASET_SCHEMA_MISMATCH")
        if any(row.vector.feature_schema_hash != self.feature_schema.schema_hash for row in self.rows):
            raise ValueError("ML_DATASET_VECTOR_SCHEMA_MISMATCH")
        if stable_hash([row.model_dump(mode="json") for row in self.rows]) != self.manifest.data_hash:
            raise ValueError("ML_DATASET_HASH_MISMATCH")
        return self


class FeatureDriftReport(Contract):
    feature_schema_hash: Identifier
    numeric_distribution: dict[str, dict[str, float | None]]
    categorical_known_values: dict[str, tuple[str, ...]]
    sample_size: int = Field(ge=1)
    warnings: tuple[str, ...] = ()


def stable_hash(value: Any) -> str:
    """Hash JSON-compatible content without process-dependent dictionary ordering."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str).encode()).hexdigest()
