"""Append-only SQLite evidence store, separate from prediction databases."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.time_utils import parse_utc, utc_iso


class EvidenceStore:
    """Persist sourced research without changing existing DuckDB schemas."""

    def __init__(
        self,
        path: Path | str,
        sources: SourceRegistry,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.sources = sources
        self.clock = clock
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path))
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS research_evidence (
                evidence_id TEXT PRIMARY KEY,
                data_type TEXT NOT NULL,
                value_json TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_url TEXT NOT NULL,
                published_time TEXT NOT NULL,
                fetched_time TEXT NOT NULL,
                as_of_time TEXT NOT NULL,
                confidence TEXT NOT NULL
            )
        """)
        columns = {row[1] for row in self._connection.execute(
            "PRAGMA table_info(research_evidence)").fetchall()}
        for name, sql_type in (
            ("provider_id", "TEXT"), ("source_tier", "INTEGER"),
            ("observed_time", "TEXT"), ("match_key", "TEXT"),
            ("claim_type", "TEXT"),
        ):
            if name not in columns:
                self._connection.execute(
                    f"ALTER TABLE research_evidence ADD COLUMN {name} {sql_type}"
                )
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS research_evidence_task_links (
                evidence_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                linked_at TEXT NOT NULL,
                PRIMARY KEY (evidence_id, task_id),
                FOREIGN KEY (evidence_id) REFERENCES research_evidence(evidence_id)
            )
        """)
        self._connection.commit()

    def close(self) -> None:
        """Close the research-only store."""
        self._connection.close()

    def save(self, evidence: EvidenceRecord) -> None:
        """Append a record only when its source and endpoint are registered."""
        source = self.sources.validate_result_url(evidence.source_id, evidence.source_url)
        if evidence.source_tier is not None and evidence.source_tier != source.tier:
            raise ValueError("EVIDENCE_SOURCE_TIER_MISMATCH")
        try:
            value_json = json.dumps(evidence.value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("EVIDENCE_VALUE_NOT_JSON") from error
        with self._connection:
            self._connection.execute(
                """INSERT INTO research_evidence
                   (evidence_id,data_type,value_json,source_id,source_url,published_time,
                    fetched_time,as_of_time,confidence,provider_id,source_tier,
                    observed_time,match_key,claim_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (evidence.evidence_id, evidence.data_type, value_json,
                 evidence.source_id, evidence.source_url, evidence.published_time or "",
                 evidence.fetched_time, evidence.as_of_time, evidence.confidence,
                 evidence.provider_id, evidence.source_tier, evidence.observed_time,
                 evidence.match_key, evidence.claim_type),
            )

    @staticmethod
    def _from_row(row: tuple[Any, ...]) -> EvidenceRecord:
        return EvidenceRecord(
            row[0], row[1], json.loads(row[2]), row[3], row[5] or None,
            row[6], row[8], row[4], row[7], row[9] or "", row[10],
            row[11], row[12], row[13],
        )

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        """Read a stored evidence record by ID."""
        row = self._connection.execute(
            "SELECT * FROM research_evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        return self._from_row(row) if row is not None else None

    def available_at(self, prediction_time: datetime) -> tuple[EvidenceRecord, ...]:
        """Return only records retrieved and published by the target time."""
        target = prediction_time
        if target.tzinfo is None or target.utcoffset() is None:
            raise ValueError("UTC_TIMESTAMP_REQUIRED")
        rows = self._connection.execute("SELECT * FROM research_evidence").fetchall()
        evidence = (self._from_row(row) for row in rows)
        return tuple(item for item in evidence
                     if self._available_at(item, target))

    @staticmethod
    def _available_at(item: EvidenceRecord, target: datetime) -> bool:
        times = [parse_utc(item.fetched_time), parse_utc(item.as_of_time)]
        if item.published_time is not None:
            times.append(parse_utc(item.published_time))
        if item.observed_time is not None:
            times.append(parse_utc(item.observed_time))
        return max(times) <= target

    def link_to_task(
        self,
        evidence_id: str,
        task_id: str,
    ) -> None:
        """Explicitly associate evidence with one deterministic research task."""
        evidence = self.get(evidence_id)
        if evidence is None:
            raise KeyError(f"EVIDENCE_NOT_FOUND:{evidence_id}")
        if not task_id:
            raise ValueError("TASK_ID_REQUIRED")
        linked = self.clock()
        if parse_utc(evidence.fetched_time) > linked:
            raise ValueError("LINK_BEFORE_EVIDENCE_RETRIEVAL")
        with self._connection:
            self._connection.execute(
                "INSERT INTO research_evidence_task_links VALUES (?, ?, ?)",
                (evidence_id, task_id, utc_iso(linked)),
            )

    def for_task(self, task_id: str, as_of_time: datetime) -> tuple[EvidenceRecord, ...]:
        """Read only explicitly linked evidence available at the requested time."""
        if as_of_time.tzinfo is None or as_of_time.utcoffset() is None:
            raise ValueError("UTC_TIMESTAMP_REQUIRED")
        rows = self._connection.execute("""
            SELECT e.* , l.linked_at FROM research_evidence e
            JOIN research_evidence_task_links l ON e.evidence_id = l.evidence_id
            WHERE l.task_id = ? ORDER BY e.evidence_id
        """, (task_id,)).fetchall()
        available: list[EvidenceRecord] = []
        for row in rows:
            evidence = self._from_row(row[:-1])
            if (parse_utc(row[-1]) <= as_of_time
                    and self._available_at(evidence, as_of_time)):
                available.append(evidence)
        return tuple(available)
