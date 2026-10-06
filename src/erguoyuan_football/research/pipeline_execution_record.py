"""Append-only SQLite audit for Phase 14.1 pipeline stages."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from erguoyuan_football.research.production_match_data import PipelineStageResult


@dataclass(frozen=True)
class PipelineExecutionRecord:
    """One persisted stage result with evidence lineage."""

    pipeline_id: str
    match_id: str | None
    stage: PipelineStageResult


class PipelineExecutionStore:
    """Persist stage records without replacing prior runs."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute("""CREATE TABLE IF NOT EXISTS pipeline_stage_records(
            record_id TEXT PRIMARY KEY, pipeline_id TEXT NOT NULL,
            match_id TEXT, stage TEXT NOT NULL, status TEXT NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT NOT NULL,
            record_json TEXT NOT NULL)""")
        self._db.commit()

    def append(self, record: PipelineExecutionRecord) -> None:
        """Append a stage result in an atomic SQLite transaction."""
        stage = record.stage
        payload = json.dumps(asdict(record), default=_json_default,
                             sort_keys=True, ensure_ascii=False, allow_nan=False)
        with self._db:
            self._db.execute("INSERT INTO pipeline_stage_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(uuid4()), record.pipeline_id, record.match_id, stage.stage,
                stage.status, stage.started_at.isoformat(), stage.finished_at.isoformat(), payload))

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the owned SQLite connection for related append-only audit tables."""
        return self._db

    def stage_count(self, pipeline_id: str | None = None) -> int:
        """Count persisted stages, optionally for one pipeline run."""
        if pipeline_id is None:
            row = self._db.execute("SELECT COUNT(*) FROM pipeline_stage_records").fetchone()
        else:
            row = self._db.execute(
                "SELECT COUNT(*) FROM pipeline_stage_records WHERE pipeline_id=?",
                (pipeline_id,)).fetchone()
        return int(row[0])

    def close(self) -> None:
        """Release the SQLite connection."""
        self._db.close()


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"UNSERIALIZABLE_AUDIT_VALUE:{type(value).__name__}")
