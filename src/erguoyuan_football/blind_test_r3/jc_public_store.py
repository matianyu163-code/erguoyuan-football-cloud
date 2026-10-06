"""Append-only public-page evidence and separate JC market time series."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.store import canonical_bytes, sha256, write_once


class JCPublicStore:
    """Keep raw HTML and interpreted rows separate from R3 model records."""

    def __init__(self, path: Path, raw_directory: Path) -> None:
        self.path = path.resolve()
        self.raw_directory = raw_directory.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_directory.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        schema = {
            "jc_public_pages": "page_id TEXT PRIMARY KEY, endpoint TEXT NOT NULL, "
                "url TEXT NOT NULL, fetched_at TEXT NOT NULL, http_status INTEGER NOT NULL, "
                "body_sha256 TEXT NOT NULL, transport_sha256 TEXT NOT NULL, "
                "html_path TEXT NOT NULL, parser_version TEXT NOT NULL",
            "jc_market_snapshots": "snapshot_id TEXT PRIMARY KEY, prediction_id TEXT NOT NULL, "
                "fixture_id TEXT NOT NULL, jc_match_number TEXT NOT NULL, "
                "captured_at TEXT NOT NULL, kickoff_time TEXT NOT NULL, "
                "source_type TEXT NOT NULL, source_quality TEXT NOT NULL, "
                "status TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL, payload TEXT NOT NULL",
            "jc_market_quotes": "quote_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL "
                "REFERENCES jc_market_snapshots(snapshot_id), market_type TEXT NOT NULL, "
                "selection TEXT NOT NULL, handicap INTEGER, bonus REAL NOT NULL, "
                "raw_implied REAL NOT NULL, no_vig REAL NOT NULL",
            "jc_market_movement": "movement_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL "
                "REFERENCES jc_market_snapshots(snapshot_id), payload TEXT NOT NULL",
        }
        self.connection.execute("PRAGMA foreign_keys=ON")
        for table, columns in schema.items():
            self.connection.execute(f"CREATE TABLE IF NOT EXISTS {table} ({columns})")
            for operation in ("UPDATE", "DELETE"):
                self.connection.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()} "
                    f"BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'APPEND_ONLY'); END")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_jc_pages "
                                "ON jc_public_pages(endpoint, fetched_at)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_jc_snapshots "
                                "ON jc_market_snapshots(fixture_id, captured_at)")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def append_page(self, *, endpoint: str, url: str, fetched_at: datetime,
                    http_status: int, body: str, transport_sha256: str,
                    parser_version: str) -> dict[str, Any]:
        if fetched_at.tzinfo is None or http_status != 200:
            raise ValueError("JC_PAGE_NOT_VALID_HTTP_200")
        fetched = fetched_at.astimezone(UTC)
        content = body.encode("utf-8")
        body_hash = sha256(content)
        page_id = sha256(canonical_bytes({"endpoint": endpoint, "url": url,
            "fetched_at": fetched.isoformat(), "body_sha256": body_hash}))
        path = self.raw_directory / f"{page_id}.html"
        write_once(path, content)
        with self.connection:
            self.connection.execute("INSERT INTO jc_public_pages VALUES (?,?,?,?,?,?,?,?,?)",
                (page_id, endpoint, url, fetched.isoformat(), http_status,
                 body_hash, transport_sha256, str(path), parser_version))
        return {"page_id": page_id, "endpoint": endpoint, "url": url,
                "fetched_at": fetched, "http_status": http_status,
                "body_sha256": body_hash, "transport_sha256": transport_sha256,
                "html_path": str(path), "parser_version": parser_version,
                "body": body}

    def fresh_page(self, endpoint: str, *, at: datetime,
                   ttl_seconds: int) -> dict[str, Any] | None:
        if at.tzinfo is None or ttl_seconds < 0:
            raise ValueError("JC_PAGE_CACHE_TIME_INVALID")
        row = self.connection.execute("SELECT page_id,url,fetched_at,http_status,body_sha256,"
            "transport_sha256,html_path,parser_version FROM jc_public_pages "
            "WHERE endpoint=? AND fetched_at<=? ORDER BY fetched_at DESC LIMIT 1",
            (endpoint, at.astimezone(UTC).isoformat())).fetchone()
        if row is None:
            return None
        fetched = datetime.fromisoformat(row[2])
        if (at.astimezone(UTC) - fetched).total_seconds() >= ttl_seconds:
            return None
        path = Path(row[6]).resolve()
        if not path.is_relative_to(self.raw_directory):
            raise ValueError("JC_PAGE_CACHE_PATH_OUTSIDE_ROOT")
        content = path.read_bytes()
        if sha256(content) != row[4]:
            raise ValueError("JC_PAGE_CACHE_HASH_INVALID")
        return {"page_id": row[0], "endpoint": endpoint, "url": row[1],
                "fetched_at": fetched, "http_status": row[3],
                "body_sha256": row[4], "transport_sha256": row[5],
                "html_path": str(path), "parser_version": row[7],
                "body": content.decode("utf-8")}

    def latest_snapshot(self, fixture_id: str, *, before: datetime) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT snapshot_id,payload FROM jc_market_snapshots "
            "WHERE fixture_id=? AND captured_at<? AND status='AVAILABLE' "
            "ORDER BY captured_at DESC,snapshot_id DESC LIMIT 1",
            (fixture_id, before.astimezone(UTC).isoformat())).fetchone()
        return {"snapshot_id": row[0], "payload": json.loads(row[1])} if row else None

    def append_market(self, payload: dict[str, Any]) -> str:
        captured = datetime.fromisoformat(payload["captured_at"])
        kickoff = datetime.fromisoformat(payload["kickoff_time"])
        if captured.tzinfo is None or kickoff.tzinfo is None or captured >= kickoff:
            raise ValueError("JC_MARKET_NOT_PREMATCH")
        snapshot_id = payload["snapshot_id"]
        digest = sha256(canonical_bytes(payload))
        with self.connection:
            self.connection.execute("INSERT INTO jc_market_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (snapshot_id, payload["prediction_id"], payload["fixture_id"],
                 payload["jc_match_number"], captured.astimezone(UTC).isoformat(),
                 kickoff.astimezone(UTC).isoformat(), payload["source_type"],
                 payload["source_quality"], payload["status"], digest,
                 canonical_bytes(payload).decode("utf-8")))
            for market_type, item in (("JC_SPF", payload.get("spf")),
                                      ("JC_RQSPF", payload.get("rqspf"))):
                if item is None:
                    continue
                for selection in ("HOME", "DRAW", "AWAY"):
                    quote_id = sha256(canonical_bytes({"snapshot_id": snapshot_id,
                        "market_type": market_type, "selection": selection}))
                    self.connection.execute("INSERT INTO jc_market_quotes VALUES (?,?,?,?,?,?,?,?)",
                        (quote_id, snapshot_id, market_type, selection,
                         item.get("handicap"), item["raw_bonus"][selection],
                         item["raw_implied"][selection], item["no_vig"][selection]))
            movement = payload.get("movement")
            if movement is not None:
                movement_id = sha256(canonical_bytes({"snapshot_id": snapshot_id,
                    "movement": movement}))
                self.connection.execute("INSERT INTO jc_market_movement VALUES (?,?,?)",
                    (movement_id, snapshot_id, canonical_bytes(movement).decode("utf-8")))
        return digest
