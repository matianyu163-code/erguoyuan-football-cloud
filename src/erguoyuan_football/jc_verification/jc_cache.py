"""SQLite cache for dated JC provider responses and verification outcomes."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from erguoyuan_football.jc_verification.jc_schema import JCMatch


class JCMatchCache:
    """Persist fetched match rows with same-day TTL and historical read-only policy."""

    def __init__(self, path: str | Path, *, ttl: timedelta = timedelta(minutes=15)) -> None:
        if ttl <= timedelta(0):
            raise ValueError("JC_CACHE_TTL_MUST_BE_POSITIVE")
        self.path = str(path)
        self.ttl = ttl
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS jc_match_cache (
                date TEXT NOT NULL, match_id TEXT NOT NULL, home_team TEXT NOT NULL,
                away_team TEXT NOT NULL, competition TEXT NOT NULL, kickoff TEXT NOT NULL,
                status TEXT NOT NULL, source TEXT NOT NULL, checked_at TEXT NOT NULL,
                expire_at TEXT NOT NULL, evidence_id TEXT NOT NULL,
                market_types TEXT NOT NULL, provider_tier INTEGER NOT NULL,
                competition_id TEXT,
                PRIMARY KEY(date, match_id, source))""")
            columns = {row[1] for row in connection.execute(
                "PRAGMA table_info(jc_match_cache)").fetchall()}
            if "competition_id" not in columns:
                connection.execute("ALTER TABLE jc_match_cache ADD COLUMN competition_id TEXT")
            connection.execute("""CREATE TABLE IF NOT EXISTS jc_match_cache_days (
                date TEXT NOT NULL, source TEXT NOT NULL, checked_at TEXT NOT NULL,
                expire_at TEXT NOT NULL, complete_coverage INTEGER NOT NULL,
                PRIMARY KEY(date, source))""")

    def get(self, match_date: date, *, source: str | None = None,
            now: datetime | None = None
            ) -> tuple[JCMatch, ...] | None:
        """Read a fresh same-day cache; historical cache is always read-only."""
        current = _aware_utc(now or datetime.now(UTC))
        cache_date = date.fromisoformat(match_date.isoformat())
        with self._connect() as connection:
            day_query = "SELECT expire_at FROM jc_match_cache_days WHERE date=?"
            day_params: tuple[str, ...] = (cache_date.isoformat(),)
            if source is not None:
                day_query += " AND source=?"
                day_params += (source,)
            day_rows = connection.execute(day_query, day_params).fetchall()
            query = """SELECT match_id,home_team,away_team,competition,
                kickoff,source,evidence_id,market_types,provider_tier,expire_at,competition_id
                FROM jc_match_cache WHERE date=?"""
            params: tuple[str, ...] = (cache_date.isoformat(),)
            if source is not None:
                query += " AND source=?"
                params += (source,)
            rows = connection.execute(query + " ORDER BY source,match_id", params).fetchall()
        if not day_rows:
            return None
        if cache_date >= current.date() and all(
            datetime.fromisoformat(row[0]) <= current for row in day_rows
        ):
            return None
        if cache_date >= current.date() and any(
            datetime.fromisoformat(row[0]) <= current for row in day_rows
        ):
            return None
        if not rows:
            return ()
        return tuple(JCMatch(row[0], row[1], row[2], row[3], datetime.fromisoformat(row[4]),
                             tuple(json.loads(row[7])), row[5], row[6], row[8], row[10])
                     for row in rows)

    def put(self, match_date: date, matches: tuple[JCMatch, ...], *,
            checked_at: datetime, source: str | None = None,
            complete_coverage: bool = False) -> None:
        """Store a provider's complete response; caller must not cache failures."""
        checked = _aware_utc(checked_at)
        expires = checked + self.ttl
        sources = {row.source for row in matches}
        if source:
            sources.add(source)
        if len(sources) != 1:
            raise ValueError("JC_CACHE_RESPONSE_SOURCE_REQUIRED")
        response_source = next(iter(sources))
        with self._connect() as connection:
            connection.execute("DELETE FROM jc_match_cache WHERE date=? AND source=?",
                               (match_date.isoformat(), response_source))
            connection.execute("""INSERT OR REPLACE INTO jc_match_cache_days
                (date,source,checked_at,expire_at,complete_coverage) VALUES (?,?,?,?,?)""",
                (match_date.isoformat(), response_source, checked.isoformat(),
                 expires.isoformat(), int(complete_coverage)))
            for row in matches:
                if row.kickoff_time.astimezone(UTC).date() != match_date:
                    raise ValueError("JC_CACHE_MATCH_DATE_MISMATCH")
                if row.source != response_source:
                    raise ValueError("JC_CACHE_RESPONSE_SOURCE_MISMATCH")
                connection.execute("""INSERT OR REPLACE INTO jc_match_cache
                    (date,match_id,home_team,away_team,competition,kickoff,status,source,
                     checked_at,expire_at,evidence_id,market_types,provider_tier,competition_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (match_date.isoformat(), row.match_id, row.home_team, row.away_team,
                     row.competition, row.kickoff_time.isoformat(), "AVAILABLE", row.source,
                     checked.isoformat(), expires.isoformat(), row.evidence_id,
                     json.dumps(row.market_types), row.provider_tier, row.competition_id))

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("UTC_TIMEZONE_REQUIRED")
    return value.astimezone(UTC)
