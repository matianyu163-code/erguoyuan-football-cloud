"""Persistent, cutoff-aware cache for provider season responses."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from erguoyuan_football.research.live_data.openligadb import SeasonBatch


class HistoricalResultCache:
    """Keep original provider observations; cached retrieval times are never rebased."""

    def __init__(self, path: Path | str, *, completed_ttl_days: int = 3650) -> None:
        if completed_ttl_days < 1:
            raise ValueError("COMPLETED_RESULT_TTL_MUST_BE_POSITIVE")
        self.completed_ttl = timedelta(days=completed_ttl_days)
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute("""CREATE TABLE IF NOT EXISTS historical_season_cache_v2 (
            provider_id TEXT NOT NULL, competition_id TEXT NOT NULL,
            season INTEGER NOT NULL, retrieved_at TEXT NOT NULL,
            source_url TEXT NOT NULL, http_status INTEGER NOT NULL, latency_ms REAL NOT NULL,
            matches_json TEXT NOT NULL,
            PRIMARY KEY(provider_id, competition_id, season))""")
        self._db.commit()

    def put(self, provider_id: str, competition_id: str, batch: SeasonBatch) -> None:
        """Persist source response without changing its observed timestamp."""
        if batch.season is None:
            raise ValueError("CACHE_SEASON_REQUIRED")
        if batch.retrieved_at.tzinfo is None or batch.retrieved_at.utcoffset() is None:
            raise ValueError("CACHE_RETRIEVAL_TIME_MUST_BE_AWARE")
        payload = json.dumps(batch.matches, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, separators=(",", ":"))
        with self._db:
            self._db.execute("""INSERT INTO historical_season_cache_v2 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider_id, competition_id, season) DO UPDATE SET
                retrieved_at=excluded.retrieved_at, source_url=excluded.source_url,
                http_status=excluded.http_status, latency_ms=excluded.latency_ms,
                matches_json=excluded.matches_json""",
                (provider_id, competition_id, batch.season,
                 batch.retrieved_at.astimezone(UTC).isoformat(),
                 batch.source_url, batch.http_status, batch.latency_ms, payload))

    def get(self, provider_id: str, competition_id: str, season: int, *, cutoff: datetime,
            now: datetime | None = None) -> SeasonBatch | None:
        """Return only a still-valid cache entry actually observed by the cutoff."""
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("UTC_CUTOFF_REQUIRED")
        row = self._db.execute("""SELECT retrieved_at, source_url, http_status,
            latency_ms, matches_json FROM historical_season_cache_v2
            WHERE provider_id=? AND competition_id=? AND season=?""",
            (provider_id, competition_id, season)).fetchone()
        if row is None:
            return None
        retrieved_at = datetime.fromisoformat(row[0])
        current = now or datetime.now(UTC)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("UTC_NOW_REQUIRED")
        if retrieved_at > cutoff or current - retrieved_at > self.completed_ttl:
            return None
        values = json.loads(row[4])
        if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
            raise ValueError("HISTORICAL_CACHE_PAYLOAD_INVALID")
        return SeasonBatch(tuple(values), retrieved_at, row[1], int(row[2]),
                           float(row[3]), season)

    def close(self) -> None:
        """Close persistent cache storage."""
        self._db.close()
