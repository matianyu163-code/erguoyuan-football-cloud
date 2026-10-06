"""Verified UTC enrichment kept separate from immutable OpenFootball raw data."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self

import duckdb
from pydantic import model_validator

from erguoyuan_football.contracts.common import Contract, Identifier, UTCTime, utc

if TYPE_CHECKING:
    from erguoyuan_football.data.external_providers import ProviderResult


class KickoffEnrichmentRecord(Contract):
    """Externally sourced exact UTC kickoff with full matching and retrieval provenance."""

    match_id: Identifier
    provider_id: Identifier
    source_event_id: Identifier
    original_datetime: str
    enriched_kickoff_utc: UTCTime
    timezone_source: Identifier
    timestamp_precision: Literal["EXACT_UTC"] = "EXACT_UTC"
    match_confidence: Literal["EXACT", "HIGH_CONFIDENCE", "AMBIGUOUS", "NOT_FOUND"]
    verified: bool
    retrieved_at: UTCTime
    content_hash: Identifier

    @model_validator(mode="after")
    def verified_only_for_unique_link(self) -> KickoffEnrichmentRecord:
        if self.verified != (self.match_confidence in {"EXACT", "HIGH_CONFIDENCE"}):
            raise ValueError("only exact or high-confidence unique links can be verified")
        if utc(self.enriched_kickoff_utc).utcoffset().total_seconds() != 0:
            raise ValueError("enriched kickoff must be UTC")
        return self


class KickoffEnrichmentProvider(Protocol):
    """Provider boundary for licensed API or user supplied, verified fixture files."""

    provider_id: str

    def fetch_matches(self, competition: str, season: int) -> ProviderResult:
        """Return provider status, records and retrieval/as-of provenance."""


def prepare_phase8_1_migration(db_path: str | Path) -> Path | None:
    """Back up the Phase 8 database once, then apply additive schema migrations."""
    path = Path(db_path)
    backup: Path | None = None
    if str(path) != ":memory:" and path.exists():
        with duckdb.connect(str(path), read_only=True) as current:
            applied = current.execute("SELECT 1 FROM information_schema.tables "
                                      "WHERE table_name='phase8_1_migrations'").fetchone()
        if applied is None:
            backup = path.with_name(f"{path.stem}.phase8_1-pre-migration-{datetime.now(UTC):%Y%m%dT%H%M%SZ}{path.suffix}")
            shutil.copy2(path, backup)
    with duckdb.connect(str(path)) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS phase8_1_migrations (
                migration_id VARCHAR PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL);
            CREATE TABLE IF NOT EXISTS kickoff_enrichment_records (
                enrichment_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
                provider_id VARCHAR NOT NULL, source_event_id VARCHAR NOT NULL,
                original_datetime VARCHAR NOT NULL, enriched_kickoff_utc TIMESTAMPTZ NOT NULL,
                timezone_source VARCHAR NOT NULL, timestamp_precision VARCHAR NOT NULL,
                confidence VARCHAR NOT NULL, verified BOOLEAN NOT NULL,
                retrieved_at TIMESTAMPTZ NOT NULL, content_hash VARCHAR NOT NULL, payload JSON NOT NULL,
                UNIQUE(provider_id, source_event_id));
            CREATE TABLE IF NOT EXISTS temporal_evidence (
                data_id VARCHAR PRIMARY KEY, data_class VARCHAR NOT NULL,
                event_time TIMESTAMPTZ, event_date DATE, as_of_time TIMESTAMPTZ,
                retrieved_at TIMESTAMPTZ NOT NULL, source_id VARCHAR NOT NULL,
                reconstruction_method VARCHAR, temporal_quality VARCHAR NOT NULL,
                eligible_for_training BOOLEAN NOT NULL, eligible_for_prediction BOOLEAN NOT NULL,
                reason_codes JSON NOT NULL, recorded_at TIMESTAMPTZ NOT NULL);
            CREATE TABLE IF NOT EXISTS real_canonical_team_aliases (
                alias_id VARCHAR PRIMARY KEY, team_id VARCHAR NOT NULL, alias VARCHAR NOT NULL,
                language VARCHAR NOT NULL, source VARCHAR NOT NULL,
                confidence DOUBLE NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                created_at TIMESTAMPTZ NOT NULL, UNIQUE(team_id, alias, source));
            CREATE TABLE IF NOT EXISTS kickoff_enrichment_review_queue (
                review_id VARCHAR PRIMARY KEY, provider_id VARCHAR NOT NULL,
                source_event_id VARCHAR NOT NULL, reason VARCHAR NOT NULL,
                candidates JSON NOT NULL, payload JSON NOT NULL,
                UNIQUE(provider_id, source_event_id));
            CREATE TABLE IF NOT EXISTS real_date_safe_snapshots (
                prediction_snapshot_id VARCHAR PRIMARY KEY, match_date DATE NOT NULL,
                temporal_mode VARCHAR NOT NULL CHECK(temporal_mode='DATE_SAFE_BATCH'),
                source_commit VARCHAR NOT NULL, training_data_hash VARCHAR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL, data_lineage JSON NOT NULL);
            CREATE TABLE IF NOT EXISTS real_oos_prediction_snapshots (
                prediction_snapshot_id VARCHAR PRIMARY KEY, competition_id VARCHAR NOT NULL,
                match_date DATE NOT NULL, temporal_mode VARCHAR NOT NULL
                    CHECK(temporal_mode IN ('EXACT_UTC','DATE_SAFE_BATCH')),
                prediction_time TIMESTAMPTZ NOT NULL, training_data_hash VARCHAR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL, data_lineage JSON NOT NULL);
            CREATE TABLE IF NOT EXISTS real_oos_predictions_v2 (
                prediction_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
                competition_id VARCHAR NOT NULL, season_id VARCHAR NOT NULL,
                model_id VARCHAR NOT NULL, model_version VARCHAR NOT NULL,
                training_cutoff TIMESTAMPTZ NOT NULL, prediction_time TIMESTAMPTZ NOT NULL,
                prediction_temporal_mode VARCHAR NOT NULL,
                kickoff_time_if_known TIMESTAMPTZ, match_date DATE NOT NULL,
                p_home DOUBLE NOT NULL CHECK(p_home BETWEEN 0 AND 1),
                p_draw DOUBLE NOT NULL CHECK(p_draw BETWEEN 0 AND 1),
                p_away DOUBLE NOT NULL CHECK(p_away BETWEEN 0 AND 1),
                is_oos BOOLEAN NOT NULL CHECK(is_oos), data_origin VARCHAR NOT NULL CHECK(data_origin='REAL'),
                training_match_count INTEGER NOT NULL CHECK(training_match_count > 0),
                training_data_hash VARCHAR NOT NULL,
                config_hash VARCHAR NOT NULL, prediction_snapshot_id VARCHAR NOT NULL,
                timestamp_precision VARCHAR NOT NULL, payload JSON NOT NULL,
                CHECK(abs(p_home+p_draw+p_away-1)<1e-6),
                CHECK(training_cutoff<=prediction_time),
                CHECK(timestamp_precision=prediction_temporal_mode),
                UNIQUE(match_id, model_id, model_version, training_data_hash, config_hash));
            CREATE TABLE IF NOT EXISTS real_oos_metrics (
                metric_id VARCHAR PRIMARY KEY, model_id VARCHAR NOT NULL,
                temporal_mode VARCHAR NOT NULL, competition_id VARCHAR NOT NULL,
                season_id VARCHAR NOT NULL, sample_size INTEGER NOT NULL,
                status VARCHAR NOT NULL, metrics JSON NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                UNIQUE(model_id, temporal_mode, competition_id, season_id));
            CREATE TABLE IF NOT EXISTS real_canonical_predictions_v2 (
                prediction_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
                prediction_snapshot_id VARCHAR NOT NULL, probability_stage VARCHAR NOT NULL,
                temporal_mode VARCHAR NOT NULL, data_origin VARCHAR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL, payload JSON NOT NULL);
        """)
        connection.execute("INSERT INTO phase8_1_migrations VALUES ('PHASE8_1_V1', ?) "
                           "ON CONFLICT DO NOTHING", [datetime.now(UTC)])
        connection.execute("INSERT INTO phase8_1_migrations VALUES ('PHASE8_1_V2', ?) "
                           "ON CONFLICT DO NOTHING", [datetime.now(UTC)])
    return backup


class KickoffEnrichmentLayer:
    """Link external events using competition, season, teams and neighboring date."""

    def __init__(self, db_path: str | Path):
        prepare_phase8_1_migration(db_path)
        self.connection = duckdb.connect(str(db_path))

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _key(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value.casefold())
        ascii_key = "".join(char for char in normalized if not unicodedata.combining(char))
        tokens = re.findall(r"[a-z0-9]+", ascii_key)
        return " ".join(token for token in tokens if token not in {"fc", "cf", "afc", "club"})

    def resolve(self, event: dict, *, provider_id: str, retrieved_at: datetime,
                competition_map: dict[str, str]) -> KickoffEnrichmentRecord | None:
        """Persist only unique exact/high-confidence matches; ambiguous events queue for review."""
        required = {"competition", "season", "home", "away", "utcDate", "id"}
        if not required <= event.keys():
            raise ValueError("KICKOFF_EVENT_SCHEMA_INVALID")
        kickoff = datetime.fromisoformat(str(event["utcDate"]))
        if kickoff.tzinfo is None:
            raise ValueError("KICKOFF_EVENT_MUST_BE_TIMEZONE_AWARE")
        kickoff = kickoff.astimezone(UTC)
        retrieved = utc(retrieved_at)
        competition_id = competition_map.get(str(event["competition"]))
        if competition_id is None:
            self._review(provider_id, str(event["id"]), "UNKNOWN_COMPETITION", (), event)
            return None
        year = int(event["season"])
        candidates = self.connection.execute("""
            SELECT m.match_id, m.match_date, h.team_name, a.team_name,
                   m.home_team_id, m.away_team_id
            FROM real_canonical_matches m
            JOIN real_canonical_teams h ON h.team_id=m.home_team_id
            JOIN real_canonical_teams a ON a.team_id=m.away_team_id
            WHERE m.competition_id=? AND m.season_id=?
              AND m.match_date BETWEEN ? AND ?
        """, [competition_id, f"{year}-{(year + 1) % 100:02d}",
                kickoff.date() - timedelta(days=1), kickoff.date() + timedelta(days=1)]).fetchall()
        aliases: dict[str, set[str]] = {}
        if candidates:
            team_ids = sorted({team_id for row in candidates for team_id in (row[4], row[5])})
            placeholders = ",".join("?" for _ in team_ids)
            alias_rows = self.connection.execute(
                f"SELECT team_id,alias FROM real_canonical_team_aliases WHERE team_id IN ({placeholders})",
                team_ids).fetchall()
            for team_id, alias in alias_rows:
                aliases.setdefault(team_id, set()).add(self._key(alias))
        home_key, away_key = self._key(str(event["home"])), self._key(str(event["away"]))
        team_candidates = [row for row in candidates
            if (self._key(row[2]) == home_key or home_key in aliases.get(row[4], set()))
            and (self._key(row[3]) == away_key or away_key in aliases.get(row[5], set()))]
        if not team_candidates:
            self._review(provider_id, str(event["id"]), "NO_MATCHING_FIXTURE", (), event)
            return None
        same_day = [row for row in team_candidates if row[1] == kickoff.date()]
        selected = same_day if same_day else team_candidates
        if len(selected) != 1:
            self._review(provider_id, str(event["id"]), "AMBIGUOUS_MATCH", tuple(row[0] for row in selected), event)
            return None
        match_id, source_date, home_name, away_name, _, _ = selected[0]
        names_exact = self._key(home_name) == home_key and self._key(away_name) == away_key
        confidence = "EXACT" if source_date == kickoff.date() and names_exact else "HIGH_CONFIDENCE"
        raw = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        record = KickoffEnrichmentRecord(match_id=match_id, provider_id=provider_id,
            source_event_id=str(event["id"]), original_datetime=str(event["utcDate"]),
            enriched_kickoff_utc=kickoff, timezone_source=provider_id,
            match_confidence=confidence, verified=True, retrieved_at=retrieved,
            content_hash=hashlib.sha256(raw.encode()).hexdigest())
        self.connection.execute("""INSERT INTO kickoff_enrichment_records VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(provider_id, source_event_id) DO NOTHING""",
            [str(uuid.uuid4()), match_id, provider_id, record.source_event_id,
             record.original_datetime, kickoff, record.timezone_source, record.timestamp_precision,
             confidence, True, retrieved, record.content_hash, record.model_dump_json()])
        return record

    def _review(self, provider_id: str, event_id: str, reason: str,
                candidates: tuple[str, ...], event: dict) -> None:
        payload = json.dumps(event, ensure_ascii=False, sort_keys=True)
        review_id = hashlib.sha256(f"{provider_id}|{event_id}".encode()).hexdigest()
        self.connection.execute("""INSERT INTO kickoff_enrichment_review_queue VALUES
            (?, ?, ?, ?, ?, ?) ON CONFLICT(provider_id, source_event_id) DO NOTHING""",
            [review_id, provider_id, event_id, reason, json.dumps(candidates), payload])
