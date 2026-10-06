"""Independent append-only SQLite market time series for R3 blind tests."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.store import canonical_bytes, sha256


class R3MarketStore:
    """Persist canonical market rows without touching trial or Production data."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        schema = {
            "market_quotes": "quote_id TEXT PRIMARY KEY, fixture_id TEXT NOT NULL, "
                "source TEXT NOT NULL, bookmaker TEXT NOT NULL, market_type TEXT NOT NULL, "
                "selection TEXT NOT NULL, odds REAL NOT NULL, fetched_at TEXT NOT NULL, "
                "payload_sha256 TEXT NOT NULL, payload TEXT NOT NULL",
            "market_snapshots": "snapshot_id TEXT PRIMARY KEY, fixture_id TEXT NOT NULL, "
                "captured_at TEXT NOT NULL, kickoff_time TEXT NOT NULL, is_pre_match INTEGER NOT NULL, "
                "status TEXT NOT NULL, "
                "snapshot_sha256 TEXT NOT NULL, payload TEXT NOT NULL",
            "market_snapshot_quotes": "snapshot_id TEXT NOT NULL REFERENCES market_snapshots(snapshot_id), "
                "quote_id TEXT NOT NULL REFERENCES market_quotes(quote_id), "
                "PRIMARY KEY(snapshot_id, quote_id)",
            "market_movement": "movement_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL "
                "REFERENCES market_snapshots(snapshot_id), payload TEXT NOT NULL",
            "model_market_comparison": "comparison_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL "
                "REFERENCES market_snapshots(snapshot_id), model_snapshot_id TEXT NOT NULL, "
                "prediction_id TEXT NOT NULL, payload TEXT NOT NULL",
        }
        for table, fields in schema.items():
            self.connection.execute(f"CREATE TABLE IF NOT EXISTS {table} ({fields})")
            for operation in ("UPDATE", "DELETE"):
                self.connection.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()} "
                    f"BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'APPEND_ONLY'); END")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_r3_market_asof "
                                "ON market_snapshots(fixture_id, captured_at)")
        self.connection.commit()

    def append(self, result: dict[str, Any], *, prediction_id: str,
               model_snapshot_id: str) -> str:
        snapshot = result["market_snapshot"]
        snapshot_id = result["market_snapshot_id"]
        if snapshot["market_snapshot_id"] != snapshot_id or (
            sha256(canonical_bytes(snapshot)) != result["market_snapshot_sha256"]
        ):
            raise ValueError("R3_MARKET_SNAPSHOT_HASH_INVALID")
        captured_at = datetime.fromisoformat(result["prediction_time"])
        kickoff = datetime.fromisoformat(result["kickoff_time"])
        if captured_at.tzinfo is None or kickoff.tzinfo is None or captured_at >= kickoff:
            raise ValueError("R3_MARKET_SNAPSHOT_NOT_PREMATCH")
        comparison_id = sha256(canonical_bytes({"prediction_id": prediction_id,
            "snapshot_id": snapshot_id, "model_snapshot_id": model_snapshot_id}))[:32]
        with self.connection:
            for quote in result["quote_payloads"]:
                payload = canonical_bytes(quote).decode("utf-8")
                existing = self.connection.execute("SELECT payload FROM market_quotes WHERE quote_id=?",
                    (quote["quote_id"],)).fetchone()
                if existing is not None:
                    if existing[0] != payload:
                        raise ValueError("R3_MARKET_QUOTE_ID_CONFLICT")
                    continue
                self.connection.execute("INSERT INTO market_quotes VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (quote["quote_id"], quote["match_id"], quote["provider_id"],
                     quote["bookmaker_id"], quote["market_type"], quote["selection"],
                     float(quote["odds_decimal"]), quote["retrieved_at"],
                     sha256(canonical_bytes(quote)), payload))
            self.connection.execute("INSERT INTO market_snapshots VALUES (?,?,?,?,?,?,?,?)",
                (snapshot_id, result["fixture_id"], captured_at.astimezone(UTC).isoformat(),
                 kickoff.astimezone(UTC).isoformat(), 1, result["market_status"],
                 result["market_snapshot_sha256"], canonical_bytes(snapshot).decode("utf-8")))
            for quote in result["quote_payloads"]:
                self.connection.execute("INSERT INTO market_snapshot_quotes VALUES (?,?)",
                                        (snapshot_id, quote["quote_id"]))
            for movement in result["movement_payloads"]:
                self.connection.execute("INSERT INTO market_movement VALUES (?,?,?)",
                    (movement["movement_id"], snapshot_id,
                     canonical_bytes(movement).decode("utf-8")))
            self.connection.execute("INSERT INTO model_market_comparison VALUES (?,?,?,?,?)",
                (comparison_id, snapshot_id, model_snapshot_id, prediction_id,
                 canonical_bytes({"model_probabilities": result["model_probabilities"],
                     "market_no_vig": result["market_no_vig"],
                     "model_market_edge_pp": result["model_market_edge_pp"],
                     "fusion": result["fusion"]}).decode("utf-8")))
        return comparison_id

    def snapshots_at(self, fixture_id: str, at: datetime) -> list[dict[str, Any]]:
        if at.tzinfo is None:
            raise ValueError("R3_MARKET_ASOF_TIMEZONE_REQUIRED")
        rows = self.connection.execute("SELECT snapshot_id, captured_at, status, payload "
            "FROM market_snapshots WHERE fixture_id=? AND captured_at<=? "
            "ORDER BY captured_at, snapshot_id", (fixture_id, at.astimezone(UTC).isoformat())).fetchall()
        return [{"snapshot_id": row[0], "captured_at": row[1], "status": row[2],
                 "snapshot": json.loads(row[3])} for row in rows]
