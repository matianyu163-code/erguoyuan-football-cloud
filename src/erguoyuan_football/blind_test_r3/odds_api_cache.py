"""Append-only R3 cache and source evidence for The Odds API responses."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.store import canonical_bytes, sha256


class OddsApiResponseCache:
    """Keep actual response bodies by hash; never store the API key or URL query."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS odds_api_responses ("
            "response_id TEXT PRIMARY KEY, endpoint TEXT NOT NULL, sport_key TEXT, "
            "regions TEXT, fetched_at TEXT NOT NULL, expires_at TEXT NOT NULL, "
            "payload_sha256 TEXT NOT NULL, payload TEXT NOT NULL, "
            "quota_remaining INTEGER, quota_used INTEGER, quota_last_cost INTEGER)")
        for operation in ("UPDATE", "DELETE"):
            self.connection.execute(f"CREATE TRIGGER IF NOT EXISTS odds_api_no_{operation.lower()} "
                f"BEFORE {operation} ON odds_api_responses "
                "BEGIN SELECT RAISE(ABORT, 'APPEND_ONLY'); END")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_odds_api_cache "
            "ON odds_api_responses(endpoint, sport_key, regions, fetched_at)")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def put(self, *, endpoint: str, sport_key: str | None, regions: str | None,
            fetched_at: datetime, ttl_seconds: int, payload: Any,
            quota_remaining: int | None, quota_used: int | None,
            quota_last_cost: int | None) -> dict[str, Any]:
        if fetched_at.tzinfo is None or ttl_seconds < 0:
            raise ValueError("ODDS_API_CACHE_TIME_INVALID")
        fetched = fetched_at.astimezone(UTC)
        payload_bytes = canonical_bytes(payload)
        content_hash = sha256(payload_bytes)
        response_id = sha256(canonical_bytes({"endpoint": endpoint, "sport_key": sport_key,
            "regions": regions, "fetched_at": fetched.isoformat(),
            "payload_sha256": content_hash}))
        expires = fetched + timedelta(seconds=ttl_seconds)
        with self.connection:
            self.connection.execute("INSERT INTO odds_api_responses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (response_id, endpoint, sport_key, regions, fetched.isoformat(),
                 expires.isoformat(), content_hash, payload_bytes.decode("utf-8"),
                 quota_remaining, quota_used, quota_last_cost))
        return {"response_id": response_id, "payload_sha256": content_hash,
                "fetched_at": fetched, "expires_at": expires,
                "quota_remaining": quota_remaining, "quota_used": quota_used,
                "quota_last_cost": quota_last_cost, "payload": payload}

    def fresh(self, *, endpoint: str, sport_key: str | None,
              regions: str | None, at: datetime) -> dict[str, Any] | None:
        if at.tzinfo is None:
            raise ValueError("ODDS_API_CACHE_TIMEZONE_REQUIRED")
        instant = at.astimezone(UTC).isoformat()
        row = self.connection.execute("SELECT response_id,payload_sha256,fetched_at,"
            "expires_at,quota_remaining,quota_used,quota_last_cost,payload "
            "FROM odds_api_responses WHERE endpoint=? AND sport_key IS ? AND regions IS ? "
            "AND fetched_at<=? AND expires_at>? ORDER BY fetched_at DESC,response_id DESC LIMIT 1",
            (endpoint, sport_key, regions, instant, instant)).fetchone()
        if row is None:
            return None
        payload = json.loads(row[7])
        if sha256(canonical_bytes(payload)) != row[1]:
            raise ValueError("ODDS_API_CACHE_HASH_INVALID")
        return {"response_id": row[0], "payload_sha256": row[1],
                "fetched_at": datetime.fromisoformat(row[2]),
                "expires_at": datetime.fromisoformat(row[3]),
                "quota_remaining": row[4], "quota_used": row[5],
                "quota_last_cost": row[6], "payload": payload}
