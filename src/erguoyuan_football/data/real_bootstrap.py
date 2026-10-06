"""Auditable OpenFootball import. Historical rows are not backdated to enable OOS."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import duckdb


class DataSourceLicenseClass(StrEnum):
    PUBLIC_DOMAIN = "PUBLIC_DOMAIN"
    OPEN_DATA_WITH_TERMS = "OPEN_DATA_WITH_TERMS"
    AUTHORIZED_API = "AUTHORIZED_API"
    USER_SUPPLIED_LICENSED = "USER_SUPPLIED_LICENSED"
    PUBLIC_API = "PUBLIC_API"
    UNVERIFIED = "UNVERIFIED"
    PROHIBITED = "PROHIBITED"


class TimestampPrecision(StrEnum):
    EXACT_UTC = "EXACT_UTC"
    LOCAL_TIME_UNKNOWN_ZONE = "LOCAL_TIME_UNKNOWN_ZONE"
    DATE_ONLY = "DATE_ONLY"


@dataclass(frozen=True)
class ImportSummary:
    source_file: str
    source_commit: str
    raw_hash: str
    raw_matches: int
    promoted: int
    quarantined: int
    repeated: int
    status: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


class RealDataWarehouse:
    """Separate Phase 8 RAW/STAGING/CANONICAL tables in the existing DuckDB file."""

    def __init__(self, path: str | Path):
        self.connection = duckdb.connect(str(path))
        self.connection.execute("SET TimeZone='UTC'")
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self) -> None:
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS real_raw_sources (
                raw_hash VARCHAR PRIMARY KEY, source VARCHAR NOT NULL, repository VARCHAR NOT NULL,
                source_commit VARCHAR NOT NULL, source_path VARCHAR NOT NULL,
                license_class VARCHAR NOT NULL, retrieved_at TIMESTAMPTZ NOT NULL,
                content JSON NOT NULL);
            CREATE TABLE IF NOT EXISTS real_staging_matches (
                staging_id VARCHAR PRIMARY KEY, raw_hash VARCHAR NOT NULL,
                source_match_index INTEGER NOT NULL, source_match_id VARCHAR NOT NULL,
                status VARCHAR NOT NULL, reason VARCHAR, content JSON NOT NULL);
            CREATE TABLE IF NOT EXISTS real_canonical_competitions (
                competition_id VARCHAR PRIMARY KEY, competition_name VARCHAR NOT NULL,
                source VARCHAR NOT NULL);
            CREATE TABLE IF NOT EXISTS real_canonical_teams (
                team_id VARCHAR PRIMARY KEY, competition_id VARCHAR NOT NULL,
                team_name VARCHAR NOT NULL, source VARCHAR NOT NULL);
            CREATE TABLE IF NOT EXISTS real_canonical_matches (
                match_id VARCHAR PRIMARY KEY, competition_id VARCHAR NOT NULL,
                season_id VARCHAR NOT NULL, home_team_id VARCHAR NOT NULL,
                away_team_id VARCHAR NOT NULL, match_date DATE NOT NULL,
                kickoff_time_utc TIMESTAMPTZ, source_local_time VARCHAR,
                timestamp_precision VARCHAR NOT NULL, status VARCHAR NOT NULL,
                home_goals INTEGER, away_goals INTEGER,
                source_id VARCHAR NOT NULL, source_priority INTEGER NOT NULL,
                source_commit VARCHAR NOT NULL, raw_hash VARCHAR NOT NULL,
                retrieved_at TIMESTAMPTZ NOT NULL, as_of_time TIMESTAMPTZ NOT NULL,
                payload JSON NOT NULL);
            CREATE TABLE IF NOT EXISTS real_data_quarantine (
                quarantine_id VARCHAR PRIMARY KEY, raw_hash VARCHAR NOT NULL,
                source_match_id VARCHAR NOT NULL, reason VARCHAR NOT NULL,
                payload JSON NOT NULL);
            CREATE TABLE IF NOT EXISTS real_oos_predictions (
                prediction_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
                model_id VARCHAR NOT NULL, model_version VARCHAR NOT NULL,
                trained_until TIMESTAMPTZ NOT NULL, prediction_time TIMESTAMPTZ NOT NULL,
                kickoff_time TIMESTAMPTZ NOT NULL, p_home DOUBLE NOT NULL,
                p_draw DOUBLE NOT NULL, p_away DOUBLE NOT NULL,
                prediction_snapshot_id VARCHAR NOT NULL, training_data_hash VARCHAR NOT NULL,
                data_origin VARCHAR NOT NULL CHECK(data_origin='REAL'), is_oos BOOLEAN NOT NULL,
                payload JSON NOT NULL);
        """)

    def coverage(self) -> list[dict[str, Any]]:
        """Counts are based on promoted records, never declared provider coverage."""
        rows = self.connection.execute("""
            SELECT competition_id, season_id, count(*) fixtures,
              count(*) FILTER (WHERE status='FINISHED') results,
              count(*) FILTER (WHERE timestamp_precision='EXACT_UTC') exact_utc,
              count(*) FILTER (WHERE timestamp_precision='LOCAL_TIME_UNKNOWN_ZONE') unknown_zone,
              count(*) FILTER (WHERE timestamp_precision='DATE_ONLY') date_only
            FROM real_canonical_matches GROUP BY 1,2 ORDER BY 1,2
        """).fetchall()
        keys = ("competition_id", "season_id", "fixtures", "results", "exact_utc", "unknown_zone", "date_only")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def promote_openfootball(self, source_file: str | Path, *, repository: str,
                             source_commit: str, competition_id: str, season_id: str,
                             license_class: DataSourceLicenseClass = DataSourceLicenseClass.PUBLIC_DOMAIN,
                             retrieved_at: datetime | None = None,
                             from_date: date | None = None, to_date: date | None = None,
                             dry_run: bool = False) -> ImportSummary:
        """Validate, stage and transactionally promote a pinned local JSON file."""
        path = Path(source_file)
        if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
            raise ValueError("OPENFOOTBALL_COMMIT_MUST_BE_FULL_SHA")
        if repository != "https://github.com/openfootball/football.json":
            raise ValueError("UNVERIFIED_REPOSITORY")
        if license_class != DataSourceLicenseClass.PUBLIC_DOMAIN:
            raise ValueError("OPENFOOTBALL_LICENSE_NOT_VERIFIED")
        raw = path.read_bytes()
        raw_hash = hashlib.sha256(raw).hexdigest()
        document = json.loads(raw.decode("utf-8"))
        matches = document.get("matches")
        if not isinstance(matches, list) or not isinstance(document.get("name"), str):
            raise TypeError("INVALID_OPENFOOTBALL_SCHEMA")
        at = retrieved_at or _utc_now()
        if at.tzinfo is None:
            raise ValueError("RETRIEVED_AT_MUST_HAVE_TIMEZONE")
        at = at.astimezone(UTC)
        promoted = quarantined = repeated = 0
        self.connection.execute("BEGIN TRANSACTION")
        try:
            old = self.connection.execute("SELECT content FROM real_raw_sources WHERE raw_hash=?", [raw_hash]).fetchone()
            if old is None:
                self.connection.execute("INSERT INTO real_raw_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [raw_hash, "OPENFOOTBALL", repository, source_commit, str(path),
                     license_class.value, at, raw.decode("utf-8")])
            self.connection.execute("INSERT INTO real_canonical_competitions VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                                    [competition_id, document["name"], "OPENFOOTBALL"])
            for index, item in enumerate(matches):
                if isinstance(item, dict) and isinstance(item.get("date"), str):
                    try:
                        match_day = date.fromisoformat(item["date"])
                    except ValueError:
                        match_day = None
                    if match_day is not None and ((from_date is not None and match_day < from_date) or
                                                  (to_date is not None and match_day > to_date)):
                        continue
                source_id = f"{competition_id}:{season_id}:{index}"
                staging_id = _stable_id("OPENFOOTBALL", source_commit, source_id, raw_hash)
                reason = self._validate_item(item)
                self.connection.execute("INSERT INTO real_staging_matches VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                    [staging_id, raw_hash, index, source_id, "QUARANTINED" if reason else "VALIDATED",
                     reason, json.dumps(item, ensure_ascii=False)])
                if reason:
                    self._quarantine(staging_id, raw_hash, source_id, reason, item)
                    quarantined += 1
                    continue
                home = item["team1"].strip()
                away = item["team2"].strip()
                match_date_str = item["date"]
                match_id = _stable_id("OPENFOOTBALL", competition_id, season_id, match_date_str,
                                      home.casefold(), away.casefold())
                home_id = _stable_id("OPENFOOTBALL_TEAM", competition_id, home.casefold())
                away_id = _stable_id("OPENFOOTBALL_TEAM", competition_id, away.casefold())
                score_field = item.get("score")
                score = score_field.get("ft") if isinstance(score_field, dict) else score_field
                home_goals, away_goals = (score if score is not None else (None, None))
                existing = self.connection.execute("SELECT home_goals, away_goals, raw_hash FROM real_canonical_matches WHERE match_id=?",
                                                   [match_id]).fetchone()
                if existing is not None:
                    if (existing[0], existing[1]) != (home_goals, away_goals):
                        self._quarantine(staging_id, raw_hash, source_id, "SCORE_CONFLICT", item)
                        quarantined += 1
                    else:
                        repeated += 1
                    continue
                self.connection.execute("DELETE FROM real_data_quarantine WHERE quarantine_id=? AND reason='INVALID_SCORE'",
                                        [staging_id])
                self.connection.execute("UPDATE real_staging_matches SET status='VALIDATED', reason=NULL WHERE staging_id=?",
                                        [staging_id])
                for team_id, name in ((home_id, home), (away_id, away)):
                    self.connection.execute("INSERT INTO real_canonical_teams VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
                                            [team_id, competition_id, name, "OPENFOOTBALL"])
                local_time = item.get("time")
                precision = (TimestampPrecision.LOCAL_TIME_UNKNOWN_ZONE if local_time else
                             TimestampPrecision.DATE_ONLY)
                # OpenFootball does not declare a timezone for the clock field.
                # No UTC kickoff or historical source-publication time is invented.
                self.connection.execute("""INSERT INTO real_canonical_matches VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [match_id, competition_id, season_id, home_id, away_id, match_date_str, None,
                     local_time, precision.value, "FINISHED" if score is not None else "SCHEDULED",
                     home_goals, away_goals, source_id, 1, source_commit,
                     raw_hash, at, at, json.dumps(item, ensure_ascii=False)])
                promoted += 1
            if dry_run:
                self.connection.execute("ROLLBACK")
            else:
                self.connection.execute("COMMIT")
        except (ValueError, TypeError, duckdb.Error, OSError):
            self.connection.execute("ROLLBACK")
            raise
        return ImportSummary(str(path), source_commit, raw_hash, len(matches), promoted,
                             quarantined, repeated, "DRY_RUN" if dry_run else "COMMITTED")

    def _quarantine(self, staging_id: str, raw_hash: str, source_id: str,
                    reason: str, item: Any) -> None:
        self.connection.execute("INSERT INTO real_data_quarantine VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            [staging_id, raw_hash, source_id, reason, json.dumps(item, ensure_ascii=False)])

    @staticmethod
    def _validate_item(item: Any) -> str | None:
        if not isinstance(item, dict):
            return "INVALID_RECORD"
        if not all(isinstance(item.get(key), str) and item[key].strip() for key in ("date", "team1", "team2")):
            return "MISSING_DATE_OR_TEAM"
        if item["team1"].casefold().strip() == item["team2"].casefold().strip():
            return "IDENTICAL_TEAMS"
        try:
            date.fromisoformat(item["date"])
        except ValueError:
            return "INVALID_DATE"
        local_time = item.get("time")
        if local_time is not None and (not isinstance(local_time, str) or
                                       not re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", local_time) or
                                       int(local_time[:2]) > 23):
            return "INVALID_LOCAL_TIME"
        score = item.get("score")
        if score is not None:
            if not isinstance(score, (dict, list)):
                return "INVALID_SCORE"
            ft = score.get("ft") if isinstance(score, dict) else score
            if ft is not None and (not isinstance(ft, list) or len(ft) != 2 or
                                  any(type(goal) is not int or goal < 0 or goal > 30 for goal in ft)):
                return "INVALID_SCORE"
        return None
