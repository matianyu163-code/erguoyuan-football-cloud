"""Append-only execution evidence for planned, blocked and invoked models."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ModelExecutionRecord:
    """Auditable model call boundary, including the exact sample lineage."""

    execution_record_id: str
    plan_id: str
    model_name: str
    model_version: str
    status: str
    started_at: datetime
    finished_at: datetime
    input_evidence_ids: tuple[str, ...]
    sample_counts: dict[str, int]
    prior_types: tuple[str, ...]
    cutoff: datetime
    training_cutoff: datetime
    degraded: bool
    output_valid: bool
    block_reason: str | None
    error_code: str | None
    input_schema_version: str
    sample_hierarchy_version: str
    prior_parameter_source: str | None = None
    prior_strength: float | None = None
    prior_derived_from: tuple[str, ...] = ()
    feature_provenance: dict[str, tuple[str, ...]] | None = None
    data_origin: str = "UNKNOWN"
    sample_window: str = "UNKNOWN"
    training_status: str = "UNKNOWN"
    inference_status: str = "UNKNOWN"
    execution_mode: str = "REAL_DRY_RUN"
    output_probabilities: dict[str, float] | None = None
    calibrated: bool = False
    bayesian_diagnostic: dict[str, Any] | None = None

    @classmethod
    def create(cls, *, plan_id: str, model_name: str, model_version: str,
               status: str, started_at: datetime, finished_at: datetime,
               input_evidence_ids: tuple[str, ...] = (),
               sample_counts: dict[str, int] | None = None,
               prior_types: tuple[str, ...] = (), cutoff: datetime,
               training_cutoff: datetime, degraded: bool = False,
               output_valid: bool = False, block_reason: str | None = None,
               error_code: str | None = None,
               input_schema_version: str = "MODEL_INPUT_BUNDLE_V1",
               sample_hierarchy_version: str = "SAMPLE_HIERARCHY_V1",
               prior_parameter_source: str | None = None,
               prior_strength: float | None = None,
               prior_derived_from: tuple[str, ...] = (),
               feature_provenance: dict[str, tuple[str, ...]] | None = None,
               data_origin: str = "UNKNOWN", sample_window: str = "UNKNOWN",
               training_status: str = "UNKNOWN", inference_status: str = "UNKNOWN",
               execution_mode: str = "REAL_DRY_RUN",
               output_probabilities: dict[str, float] | None = None,
               calibrated: bool = False,
               bayesian_diagnostic: dict[str, Any] | None = None,
               ) -> ModelExecutionRecord:
        """Create an immutable evidence record with a unique stable identifier."""
        return cls(str(uuid4()), plan_id, model_name, model_version, status,
                   started_at, finished_at, tuple(sorted(set(input_evidence_ids))),
                   dict(sample_counts or {}), prior_types, cutoff, training_cutoff,
                   degraded, output_valid, block_reason, error_code,
                   input_schema_version, sample_hierarchy_version,
                   prior_parameter_source, prior_strength, prior_derived_from,
                   feature_provenance, data_origin, sample_window,
                   training_status, inference_status, execution_mode,
                   dict(output_probabilities) if output_probabilities is not None else None,
                   calibrated, bayesian_diagnostic)


def _json_value(value: Any) -> Any:
    """Convert record values to deterministic JSON-compatible primitives."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


class ModelExecutionStore:
    """SQLite append-only ledger; no prediction artifact overwrite is involved."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path))
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS model_execution_records (
                execution_record_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                record_json TEXT NOT NULL
            )
        """)
        self._connection.commit()

    def save(self, record: ModelExecutionRecord) -> None:
        """Append one unique execution record."""
        value = json.dumps(_json_value(asdict(record)), sort_keys=True,
                           ensure_ascii=False, allow_nan=False)
        with self._connection:
            self._connection.execute(
                "INSERT INTO model_execution_records VALUES (?, ?, ?, ?, ?, ?)",
                (record.execution_record_id, record.plan_id, record.model_name,
                 record.status, record.started_at.isoformat(), value))

    def records(self, *, plan_id: str | None = None) -> tuple[dict[str, Any], ...]:
        """Read saved records in insertion order, optionally by plan."""
        if plan_id is None:
            rows = self._connection.execute(
                "SELECT record_json FROM model_execution_records ORDER BY rowid").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT record_json FROM model_execution_records WHERE plan_id=? ORDER BY rowid",
                (plan_id,)).fetchall()
        return tuple(json.loads(row[0]) for row in rows)

    def close(self) -> None:
        """Close the ledger connection."""
        self._connection.close()
