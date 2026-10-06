"""Append-only persistence for match source classification evidence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from erguoyuan_football.match_source.match_source import MatchSourceType


@dataclass(frozen=True)
class MatchSourceAudit:
    """One deterministic source classification and its evidence."""

    source_type: MatchSourceType
    created_time: datetime
    confidence: float
    evidence: tuple[str, ...]
    audit_id: str

    def __post_init__(self) -> None:
        if self.created_time.tzinfo is None or self.created_time.utcoffset() is None:
            raise ValueError("MATCH_SOURCE_AUDIT_TIMEZONE_REQUIRED")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("MATCH_SOURCE_CONFIDENCE_OUT_OF_RANGE")
        if not self.audit_id or not self.evidence:
            raise ValueError("MATCH_SOURCE_AUDIT_EVIDENCE_REQUIRED")


class MatchSourceAuditStore:
    """SQLite append-only source audit, optionally sharing an existing connection."""

    def __init__(self, database: str | Path | sqlite3.Connection) -> None:
        self._owns_connection = not isinstance(database, sqlite3.Connection)
        self._db = (database if isinstance(database, sqlite3.Connection)
                    else sqlite3.connect(database))
        self._db.execute("""CREATE TABLE IF NOT EXISTS match_source_audit (
            audit_id TEXT PRIMARY KEY, source_type TEXT NOT NULL,
            created_time TEXT NOT NULL, confidence REAL NOT NULL,
            evidence TEXT NOT NULL)""")
        self._db.commit()

    def append(self, audit: MatchSourceAudit) -> None:
        """Store a classification without replacing earlier request evidence."""
        with self._db:
            self._db.execute("""INSERT INTO match_source_audit
                (audit_id,source_type,created_time,confidence,evidence)
                VALUES (?,?,?,?,?)""",
                (audit.audit_id, audit.source_type.value,
                 audit.created_time.astimezone(UTC).isoformat(), audit.confidence,
                 json.dumps(audit.evidence, ensure_ascii=False)))

    def list(self) -> tuple[MatchSourceAudit, ...]:
        """Return persisted audit records in insertion order."""
        rows = self._db.execute("""SELECT audit_id,source_type,created_time,
            confidence,evidence FROM match_source_audit ORDER BY rowid""").fetchall()
        return tuple(MatchSourceAudit(MatchSourceType(row[1]),
            datetime.fromisoformat(row[2]), float(row[3]), tuple(json.loads(row[4])), row[0])
            for row in rows)

    def close(self) -> None:
        """Close the database only when this store owns its connection."""
        if self._owns_connection:
            self._db.close()


def new_audit(source_type: MatchSourceType, *, confidence: float,
              evidence: tuple[str, ...], created_time: datetime | None = None
              ) -> MatchSourceAudit:
    """Build an immutable audit record with a unique ID and UTC timestamp."""
    return MatchSourceAudit(source_type, created_time or datetime.now(UTC),
                            confidence, evidence, str(uuid4()))
