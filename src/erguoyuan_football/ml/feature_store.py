"""Append-only DuckDB store for frozen ML vectors and immutable dataset manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import duckdb

from erguoyuan_football.ml.schemas import MLDataset, MLFeatureVector


class MLFeatureStore:
    """Separate Phase 7 tables; existing point-in-time storage remains untouched."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.connection = duckdb.connect(str(path))
        self.connection.execute("SET TimeZone='UTC'")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS ml_feature_vectors (
            feature_data_hash VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            prediction_snapshot_id VARCHAR NOT NULL, prediction_time TIMESTAMPTZ NOT NULL,
            feature_mode VARCHAR NOT NULL, model_family VARCHAR NOT NULL,
            feature_schema_hash VARCHAR NOT NULL, payload JSON NOT NULL)""")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_ml_feature_snapshot "
                                "ON ml_feature_vectors(prediction_snapshot_id, feature_mode, model_family)")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS ml_datasets (
            data_hash VARCHAR PRIMARY KEY, dataset_id VARCHAR NOT NULL,
            feature_schema_hash VARCHAR NOT NULL, row_count INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL, payload JSON NOT NULL)""")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def append_vector(self, vector: MLFeatureVector) -> None:
        """Same hash may repeat identically; a changed vector gets a new hash."""
        valid = MLFeatureVector.model_validate(vector.model_dump())
        existing = self.connection.execute("SELECT payload FROM ml_feature_vectors WHERE feature_data_hash=?",
                                           [valid.feature_data_hash]).fetchone()
        if existing is not None:
            if existing[0] != valid.model_dump_json():
                raise ValueError("ML_FEATURE_HASH_CONTENT_CONFLICT")
            return
        self.connection.execute("INSERT INTO ml_feature_vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
            valid.feature_data_hash, valid.match_id, valid.prediction_snapshot_id,
            valid.prediction_time, valid.feature_mode.value, valid.model_family.value,
            valid.feature_schema_hash, valid.model_dump_json(),
        ])

    def append_many(self, vectors: tuple[MLFeatureVector, ...]) -> None:
        self.connection.execute("BEGIN TRANSACTION")
        try:
            for vector in vectors:
                self.append_vector(vector)
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def load_vector(self, feature_data_hash: str) -> MLFeatureVector:
        row = self.connection.execute("SELECT payload FROM ml_feature_vectors WHERE feature_data_hash=?",
                                      [feature_data_hash]).fetchone()
        if row is None:
            raise KeyError(feature_data_hash)
        return MLFeatureVector.model_validate_json(row[0])

    def save_dataset(self, dataset: MLDataset) -> None:
        valid = MLDataset.model_validate(dataset.model_dump())
        existing = self.connection.execute("SELECT payload FROM ml_datasets WHERE data_hash=?",
                                           [valid.manifest.data_hash]).fetchone()
        if existing is not None:
            saved = MLDataset.model_validate_json(existing[0])
            if (saved.rows, saved.feature_schema, saved.dataset_kind) != (
                    valid.rows, valid.feature_schema, valid.dataset_kind):
                raise ValueError("ML_DATASET_HASH_CONTENT_CONFLICT")
            return
        self.connection.execute("INSERT INTO ml_datasets VALUES (?, ?, ?, ?, ?, ?)", [
            valid.manifest.data_hash, valid.manifest.dataset_id,
            valid.manifest.feature_schema_hash, valid.manifest.row_count,
            valid.manifest.created_at, valid.model_dump_json(),
        ])

    def load_dataset(self, data_hash: str) -> MLDataset:
        row = self.connection.execute("SELECT payload FROM ml_datasets WHERE data_hash=?",
                                      [data_hash]).fetchone()
        if row is None:
            raise KeyError(data_hash)
        return MLDataset.model_validate_json(row[0])
