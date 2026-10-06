"""Persistent research response cache and append-only request audit."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from erguoyuan_football.web_research.providers.provider_result import ProviderResult
from erguoyuan_football.web_research.search.search_task import SearchTask
from erguoyuan_football.web_research.time_utils import parse_utc, utc_iso


@dataclass(frozen=True)
class ResearchAuditRecord:
    """One cache hit, provider result or classified provider failure."""

    task_id: str
    source_id: str
    query: str
    requested_at: str
    outcome: str
    cache_hit: bool
    fetched_time: str | None = None
    error_code: str | None = None
    request_id: str | None = None
    provider_id: str | None = None
    endpoint_id: str | None = None
    finished_at: str | None = None
    http_status: int | None = None
    retry_count: int = 0
    cache_status: str = "MISS"


class ResearchCache:
    """SQLite history of successful responses; never backdates retrieval times."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path))
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS research_cache (
                source_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response_json TEXT NOT NULL,
                fetched_time TEXT NOT NULL,
                as_of_time TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                PRIMARY KEY (source_id, task_id, saved_at)
            );
            CREATE TABLE IF NOT EXISTS research_request_audit (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                query TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                outcome TEXT NOT NULL,
                cache_hit INTEGER NOT NULL,
                fetched_time TEXT,
                error_code TEXT
            );
        """)
        existing = {row[1] for row in self._connection.execute(
            "PRAGMA table_info(research_request_audit)").fetchall()}
        for column, sql_type in (
            ("request_id", "TEXT"), ("provider_id", "TEXT"),
            ("endpoint_id", "TEXT"), ("finished_at", "TEXT"),
            ("http_status", "INTEGER"), ("retry_count", "INTEGER"),
            ("cache_status", "TEXT"),
        ):
            if column not in existing:
                self._connection.execute(
                    f"ALTER TABLE research_request_audit ADD COLUMN {column} {sql_type}"
                )
        self._connection.commit()

    def close(self) -> None:
        """Close the research-only cache and audit database."""
        self._connection.close()

    def get(self, source_id: str, task: SearchTask,
            at: datetime) -> ProviderResult | None:
        """Return a still-fresh response that was already retrieved by `at`."""
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("UTC_TIMESTAMP_REQUIRED")
        rows = self._connection.execute("""
            SELECT query,response_json,fetched_time,as_of_time,saved_at,expires_at
            FROM research_cache WHERE source_id = ? AND task_id = ?
            ORDER BY rowid DESC
        """, (source_id, task.task_id)).fetchall()
        for query, payload, fetched, as_of, saved, expires in rows:
            if query != task.query:
                continue
            if (max(parse_utc(fetched), parse_utc(as_of), parse_utc(saved)) <= at
                    < parse_utc(expires)):
                record = json.loads(payload)
                return ProviderResult(**record)
        return None

    def latest(self, source_id: str, task: SearchTask,
               at: datetime) -> tuple[ProviderResult, str, int] | None:
        """Return latest PIT-visible cache, explicitly fresh or stale."""
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("UTC_TIMESTAMP_REQUIRED")
        rows = self._connection.execute("""
            SELECT query,response_json,fetched_time,as_of_time,saved_at,expires_at
            FROM research_cache WHERE source_id = ? AND task_id = ? ORDER BY rowid DESC
        """, (source_id, task.task_id)).fetchall()
        for query, payload, fetched, as_of, saved, expires in rows:
            if query != task.query or max(parse_utc(fetched), parse_utc(as_of),
                                           parse_utc(saved)) > at:
                continue
            status = "CACHE_FRESH" if at < parse_utc(expires) else "CACHE_STALE"
            age = max(0, int((at - parse_utc(saved)).total_seconds()))
            return ProviderResult(**json.loads(payload)), status, age
        return None

    def put(self, source_id: str, task: SearchTask, result: ProviderResult,
            saved_at: datetime, ttl: timedelta) -> None:
        """Append an auditable response version with a bounded freshness window."""
        if not result.success or result.as_of_time is None:
            raise ValueError("SUCCESSFUL_SOURCED_RESULT_REQUIRED")
        if ttl.total_seconds() <= 0:
            raise ValueError("POSITIVE_CACHE_TTL_REQUIRED")
        if parse_utc(result.fetched_time) > saved_at:
            raise ValueError("RETRIEVAL_AFTER_CACHE_SAVE")
        from dataclasses import asdict

        with self._connection:
            self._connection.execute(
                "INSERT INTO research_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (source_id, task.task_id, task.query,
                 json.dumps(asdict(result), ensure_ascii=False, sort_keys=True, allow_nan=False),
                 result.fetched_time, result.as_of_time, utc_iso(saved_at),
                 utc_iso(saved_at + ttl)),
            )

    def record_audit(self, record: ResearchAuditRecord) -> None:
        """Append request metadata; no credentials or raw response body are logged."""
        parse_utc(record.requested_at)
        with self._connection:
            self._connection.execute(
                """INSERT INTO research_request_audit
                   (task_id,source_id,query,requested_at,outcome,cache_hit,fetched_time,
                    error_code,request_id,provider_id,endpoint_id,finished_at,http_status,
                    retry_count,cache_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record.task_id, record.source_id, record.query, record.requested_at,
                 record.outcome, int(record.cache_hit), record.fetched_time, record.error_code,
                 record.request_id, record.provider_id, record.endpoint_id,
                 record.finished_at, record.http_status, record.retry_count,
                 record.cache_status),
            )

    def audit_records(self) -> tuple[ResearchAuditRecord, ...]:
        """Read append-only research decisions in insertion order."""
        rows = self._connection.execute("""
            SELECT task_id,source_id,query,requested_at,outcome,cache_hit,fetched_time,
                   error_code,request_id,provider_id,endpoint_id,finished_at,http_status,
                   retry_count,cache_status
            FROM research_request_audit ORDER BY audit_id
        """).fetchall()
        return tuple(ResearchAuditRecord(row[0], row[1], row[2], row[3], row[4],
                                         bool(row[5]), row[6], row[7], row[8], row[9],
                                         row[10], row[11], row[12], row[13] or 0,
                                         row[14] or "MISS") for row in rows)
