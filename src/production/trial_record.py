"""Append-only SQLite records for production trial attempts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class TrialPredictionRecord:
    """Auditable trial run; output may explicitly contain unavailable stages."""

    prediction_id: str
    created_time: datetime
    match_name: str
    competition: str | None
    source_type: str
    models_used: tuple[str, ...]
    input_snapshot: dict[str, Any]
    prediction_output: dict[str, Any]
    final_result: dict[str, Any] | None = None
    review_status: str = "NOT_REVIEWED"

    @classmethod
    def create(cls, **values: Any) -> TrialPredictionRecord:
        """Create an immutable audit row with a unique ID and UTC time."""
        return cls(prediction_id=str(uuid4()), created_time=datetime.now(UTC), **values)


class TrialRecordStore:
    """Persist records with INSERT-only behavior; no update/delete API is exposed."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path)
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("""CREATE TABLE IF NOT EXISTS trial_prediction_records (
            prediction_id TEXT PRIMARY KEY,
            created_time TEXT NOT NULL,
            match_name TEXT NOT NULL,
            competition TEXT,
            source_type TEXT NOT NULL,
            models_used TEXT NOT NULL,
            input_snapshot TEXT NOT NULL,
            prediction_output TEXT NOT NULL,
            actual_result TEXT,
            review_status TEXT NOT NULL
        )""")
        self._connection.execute("""CREATE TABLE IF NOT EXISTS trial_record_annotations (
            annotation_id TEXT PRIMARY KEY,
            prediction_id TEXT NOT NULL REFERENCES trial_prediction_records(prediction_id),
            created_time TEXT NOT NULL,
            operator TEXT NOT NULL,
            note TEXT NOT NULL
        )""")
        self._connection.executescript("""
            CREATE TRIGGER IF NOT EXISTS trial_prediction_records_no_update
            BEFORE UPDATE ON trial_prediction_records
            BEGIN SELECT RAISE(ABORT, 'TRIAL_RECORDS_APPEND_ONLY'); END;
            CREATE TRIGGER IF NOT EXISTS trial_prediction_records_no_delete
            BEFORE DELETE ON trial_prediction_records
            BEGIN SELECT RAISE(ABORT, 'TRIAL_RECORDS_APPEND_ONLY'); END;
            CREATE TRIGGER IF NOT EXISTS trial_record_annotations_no_update
            BEFORE UPDATE ON trial_record_annotations
            BEGIN SELECT RAISE(ABORT, 'TRIAL_ANNOTATIONS_APPEND_ONLY'); END;
            CREATE TRIGGER IF NOT EXISTS trial_record_annotations_no_delete
            BEFORE DELETE ON trial_record_annotations
            BEGIN SELECT RAISE(ABORT, 'TRIAL_ANNOTATIONS_APPEND_ONLY'); END;
        """)
        self._connection.commit()

    def append(self, record: TrialPredictionRecord) -> None:
        """Append one row; duplicate IDs fail instead of replacing prior evidence."""
        if record.created_time.tzinfo is None or record.created_time.utcoffset() is None:
            raise ValueError("TRIAL_RECORD_TIMEZONE_REQUIRED")
        with self._connection:
            self._connection.execute("""INSERT INTO trial_prediction_records VALUES
                (?,?,?,?,?,?,?,?,?,?)""", (
                record.prediction_id,
                record.created_time.astimezone(UTC).isoformat(),
                record.match_name,
                record.competition,
                record.source_type,
                json.dumps(record.models_used, ensure_ascii=False),
                json.dumps(record.input_snapshot, ensure_ascii=False, sort_keys=True),
                json.dumps(record.prediction_output, ensure_ascii=False, sort_keys=True),
                json.dumps(record.final_result, ensure_ascii=False, sort_keys=True)
                if record.final_result is not None else None,
                record.review_status,
            ))

    def list_records(self) -> tuple[dict[str, Any], ...]:
        """Return persisted records without modifying their append-only history."""
        rows = self._connection.execute("""SELECT prediction_id,created_time,match_name,
            competition,source_type,models_used,input_snapshot,prediction_output,
            actual_result,review_status FROM trial_prediction_records ORDER BY rowid""").fetchall()
        return tuple({
            "prediction_id": row[0], "time": row[1], "created_time": row[1],
            "match": row[2], "match_name": row[2],
            "competition": row[3], "source_type": row[4], "models_used": json.loads(row[5]),
            "data_snapshot": json.loads(row[6]),
            "input_snapshot": json.loads(row[6]), "prediction_output": json.loads(row[7]),
            "actual_result": json.loads(row[8]) if row[8] is not None else None,
            "final_result": json.loads(row[8]) if row[8] is not None else None,
            "review_status": row[9],
        } for row in rows)

    def append_annotation(self, *, prediction_id: str, note: str,
                          operator: str) -> str:
        """Append a correction note without rewriting an earlier trial record."""
        if not note.strip() or not operator.strip():
            raise ValueError("TRIAL_ANNOTATION_NOTE_AND_OPERATOR_REQUIRED")
        exists = self._connection.execute(
            "SELECT 1 FROM trial_prediction_records WHERE prediction_id=?",
            (prediction_id,),
        ).fetchone()
        if exists is None:
            raise ValueError("TRIAL_ANNOTATION_TARGET_NOT_FOUND")
        annotation_id = str(uuid4())
        with self._connection:
            self._connection.execute("""INSERT INTO trial_record_annotations
                (annotation_id,prediction_id,created_time,operator,note)
                VALUES (?,?,?,?,?)""", (
                annotation_id, prediction_id, datetime.now(UTC).isoformat(),
                operator.strip(), note.strip(),
            ))
        return annotation_id

    def close(self) -> None:
        """Close the SQLite connection."""
        self._connection.close()
