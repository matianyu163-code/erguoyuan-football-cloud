"""Append rendered JC observations to the existing R3 market.sqlite database."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.market_store import R3MarketStore
from erguoyuan_football.blind_test_r3.store import canonical_bytes, sha256


class JCRenderedMarketStore:
    """Market-only pages and fixtures never require an R3 prediction ID."""

    def __init__(self, path: Path) -> None:
        self.market = R3MarketStore(path)
        self.connection = self.market.connection
        schema = {
            "jc_rendered_pages": "page_id TEXT PRIMARY KEY, endpoint TEXT NOT NULL, "
                "fetched_at TEXT NOT NULL, url TEXT NOT NULL, http_status INTEGER NOT NULL, "
                "dom_sha256 TEXT NOT NULL, screenshot_sha256 TEXT NOT NULL, payload TEXT NOT NULL",
            "jc_rendered_snapshots": "snapshot_id TEXT PRIMARY KEY, page_id TEXT NOT NULL "
                "REFERENCES jc_rendered_pages(page_id), jc_match_key TEXT NOT NULL, "
                "jc_match_number TEXT NOT NULL, captured_at TEXT NOT NULL, "
                "kickoff_time TEXT NOT NULL, status TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL, "
                "payload TEXT NOT NULL",
            "jc_rendered_quotes": "quote_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL "
                "REFERENCES jc_rendered_snapshots(snapshot_id), market_type TEXT NOT NULL, "
                "selection TEXT NOT NULL, handicap INTEGER, bonus REAL NOT NULL, "
                "raw_implied REAL NOT NULL, no_vig REAL NOT NULL",
            "jc_rendered_movement": "movement_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL "
                "REFERENCES jc_rendered_snapshots(snapshot_id), payload TEXT NOT NULL",
        }
        for table, columns in schema.items():
            self.connection.execute(f"CREATE TABLE IF NOT EXISTS {table} ({columns})")
            for operation in ("UPDATE", "DELETE"):
                self.connection.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()} "
                    f"BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'APPEND_ONLY'); END")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_jc_rendered_match_time "
            "ON jc_rendered_snapshots(jc_match_key,captured_at)")
        self.connection.commit()

    def close(self) -> None:
        self.market.close()

    def append_page(self, metadata: dict[str, Any], *, endpoint: str) -> str:
        captured = datetime.fromisoformat(metadata["fetched_at"])
        if captured.tzinfo is None or metadata["http_status"] != 200:
            raise ValueError("JC_RENDERED_PAGE_INVALID")
        page_id = "JCPAGE-" + sha256(canonical_bytes({"endpoint": endpoint,
            "fetched_at": captured.isoformat(), "dom_sha256": metadata["dom_sha256"]}))[:32]
        payload = canonical_bytes(metadata).decode("utf-8")
        with self.connection:
            row = self.connection.execute("SELECT payload FROM jc_rendered_pages WHERE page_id=?",
                                          (page_id,)).fetchone()
            if row is not None:
                if row[0] != payload:
                    raise ValueError("JC_RENDERED_PAGE_ID_CONFLICT")
                return page_id
            self.connection.execute("INSERT INTO jc_rendered_pages VALUES (?,?,?,?,?,?,?,?)",
                (page_id, endpoint, captured.astimezone(UTC).isoformat(),
                 metadata["url"], metadata["http_status"], metadata["dom_sha256"],
                 metadata["screenshot_sha256"], payload))
        return page_id

    def latest(self, jc_match_key: str, *, before: datetime) -> dict[str, Any] | None:
        if before.tzinfo is None:
            raise ValueError("JC_RENDERED_TIMEZONE_REQUIRED")
        row = self.connection.execute("SELECT snapshot_id,payload FROM jc_rendered_snapshots "
            "WHERE jc_match_key=? AND captured_at<? ORDER BY captured_at DESC LIMIT 1",
            (jc_match_key, before.astimezone(UTC).isoformat())).fetchone()
        return {"snapshot_id": row[0], "payload": json.loads(row[1])} if row else None

    def append_snapshot(self, snapshot: dict[str, Any]) -> str:
        captured = datetime.fromisoformat(snapshot["captured_at"])
        kickoff = datetime.fromisoformat(snapshot["kickoff_time"])
        if captured.tzinfo is None or kickoff.tzinfo is None or captured >= kickoff:
            raise ValueError("JC_RENDERED_NOT_PREMATCH")
        snapshot_id = snapshot["snapshot_id"]
        payload = canonical_bytes(snapshot).decode("utf-8")
        digest = sha256(payload.encode())
        with self.connection:
            row = self.connection.execute("SELECT payload FROM jc_rendered_snapshots "
                                          "WHERE snapshot_id=?", (snapshot_id,)).fetchone()
            if row is not None:
                if row[0] != payload:
                    raise ValueError("JC_RENDERED_SNAPSHOT_ID_CONFLICT")
                return digest
            self.connection.execute("INSERT INTO jc_rendered_snapshots VALUES (?,?,?,?,?,?,?,?,?)",
                (snapshot_id, snapshot["page_id"], snapshot["jc_match_key"],
                 snapshot["jc_match_number"], captured.astimezone(UTC).isoformat(),
                 kickoff.astimezone(UTC).isoformat(), snapshot["status"], digest, payload))
            for market_type, item in (("JC_SPF", snapshot.get("spf")),
                                      ("JC_RQSPF", snapshot.get("rqspf"))):
                if item is None:
                    continue
                for side in ("HOME", "DRAW", "AWAY"):
                    quote_id = sha256(canonical_bytes({"snapshot_id": snapshot_id,
                        "market_type": market_type, "selection": side}))
                    self.connection.execute("INSERT INTO jc_rendered_quotes VALUES (?,?,?,?,?,?,?,?)",
                        (quote_id, snapshot_id, market_type, side, item["handicap"],
                         item["raw_bonus"][side], item["raw_implied"][side],
                         item["no_vig"][side]))
            if snapshot.get("movement") is not None:
                movement_id = sha256(canonical_bytes({"snapshot_id": snapshot_id,
                    "movement": snapshot["movement"]}))
                self.connection.execute("INSERT INTO jc_rendered_movement VALUES (?,?,?)",
                    (movement_id, snapshot_id,
                     canonical_bytes(snapshot["movement"]).decode("utf-8")))
        return digest

    def snapshots_for_number(self, jc_match_number: str, *, before: datetime) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT payload FROM jc_rendered_snapshots "
            "WHERE jc_match_number=? AND captured_at<=? ORDER BY captured_at DESC",
            (jc_match_number, before.astimezone(UTC).isoformat())).fetchall()
        return [json.loads(row[0]) for row in rows]
