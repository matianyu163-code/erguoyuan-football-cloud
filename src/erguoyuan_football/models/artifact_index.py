"""Explicit DuckDB artifact index; cache lookup never recursively scans directories."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import duckdb
from pydantic import Field

from erguoyuan_football.contracts.common import Contract, UTCTime, now


class FeatureArtifact(Contract):
    """Versioned point-in-time feature materialization metadata."""

    feature_set_id: str
    feature_version: str
    as_of_time: UTCTime
    data_hash: str
    config_hash: str
    path: str
    created_at: UTCTime = Field(default_factory=now)


class ArtifactIndex:
    """Indexed model and feature artifact lookup backed by DuckDB."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.connection = duckdb.connect(self.path)
        self.connection.execute("""CREATE TABLE IF NOT EXISTS model_artifacts (
            model_id VARCHAR, model_version VARCHAR, trained_until TIMESTAMPTZ, config_hash VARCHAR,
            data_hash VARCHAR, artifact_path VARCHAR, created_at TIMESTAMPTZ,
            PRIMARY KEY(model_id, model_version, trained_until, config_hash, data_hash))""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS feature_artifacts (
            feature_set_id VARCHAR, feature_version VARCHAR, as_of_time TIMESTAMPTZ,
            data_hash VARCHAR, config_hash VARCHAR, path VARCHAR, created_at TIMESTAMPTZ,
            PRIMARY KEY(feature_set_id, feature_version, as_of_time, config_hash, data_hash))""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS ml_artifacts (
            model_id VARCHAR, model_version VARCHAR, trained_until TIMESTAMPTZ,
            config_hash VARCHAR, dataset_hash VARCHAR, schema_hash VARCHAR,
            artifact_path VARCHAR, created_at TIMESTAMPTZ,
            PRIMARY KEY(model_id,model_version,trained_until,config_hash,dataset_hash,schema_hash))""")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def register_model(self, artifact) -> None:
        self.connection.execute("""INSERT OR REPLACE INTO model_artifacts VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [artifact.model_id, artifact.model_version, artifact.trained_until, artifact.config_hash,
             artifact.training_data_hash, artifact.artifact_path, artifact.created_at])

    def find_model(self, model_id: str, model_version: str, trained_until,
                   config_hash: str, data_hash: str) -> str | None:
        row = self.connection.execute("""SELECT artifact_path FROM model_artifacts WHERE model_id = ?
            AND model_version = ? AND trained_until = ? AND config_hash = ? AND data_hash = ?""",
            [model_id, model_version, trained_until, config_hash, data_hash]).fetchone()
        return str(row[0]) if row else None

    def register_ml_model(self, artifact) -> None:
        """Index a native XGBoost/CatBoost artifact without directory scanning."""
        self.connection.execute("""INSERT OR REPLACE INTO ml_artifacts VALUES
            (?,?,?,?,?,?,?,?)""", [artifact.model_id, artifact.model_version,
            artifact.trained_until, artifact.config_hash, artifact.dataset_hash,
            artifact.feature_schema_hash, artifact.artifact_path, artifact.created_at])

    def find_ml_model(self, model_id: str, model_version: str, trained_until,
                      config_hash: str, dataset_hash: str, schema_hash: str) -> str | None:
        row = self.connection.execute("""SELECT artifact_path FROM ml_artifacts
            WHERE model_id=? AND model_version=? AND trained_until=?
            AND config_hash=? AND dataset_hash=? AND schema_hash=?""",
            [model_id, model_version, trained_until, config_hash, dataset_hash,
             schema_hash]).fetchone()
        return str(row[0]) if row else None

    def register_feature(self, artifact: FeatureArtifact) -> None:
        self.connection.execute("""INSERT OR REPLACE INTO feature_artifacts VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [artifact.feature_set_id, artifact.feature_version, artifact.as_of_time, artifact.data_hash,
             artifact.config_hash, artifact.path, artifact.created_at])

    def find_feature(self, feature_set_id: str, feature_version: str, as_of_time,
                     config_hash: str, data_hash: str) -> FeatureArtifact | None:
        row = self.connection.execute("""SELECT feature_set_id, feature_version, as_of_time,
            data_hash, config_hash, path, created_at FROM feature_artifacts WHERE feature_set_id = ?
            AND feature_version = ? AND as_of_time = ? AND config_hash = ? AND data_hash = ?""",
            [feature_set_id, feature_version, as_of_time, config_hash, data_hash]).fetchone()
        return FeatureArtifact.model_validate(dict(zip(
            ("feature_set_id", "feature_version", "as_of_time", "data_hash", "config_hash", "path", "created_at"),
            row, strict=True))) if row else None
