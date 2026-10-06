"""Real historical event loader and additive Phase 10 DuckDB persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Self

from erguoyuan_football.context.schemas import (
    ContextAssessment,
    HistoricalMatchEvent,
)
from erguoyuan_football.data.connection import connect_project_duckdb
from erguoyuan_football.ml.schemas import stable_hash


@dataclass(frozen=True)
class RealTournamentContextDataset:
    """Verified immutable results used by a batch of context reconstructions."""

    events: tuple[HistoricalMatchEvent, ...]
    data_hash: str
    source: str


def load_real_context_events(db_path: str | Path, *, before_date: date) -> RealTournamentContextDataset:
    """Load only real canonical completed results preceding the latest target date."""
    connection = connect_project_duckdb(db_path, read_only=True)
    try:
        connection.execute("SET TimeZone='UTC'")
        rows = connection.execute("""SELECT match_id, competition_id, season_id,
            home_team_id, away_team_id, match_date, home_goals, away_goals, source_id,
            retrieved_at, as_of_time FROM real_canonical_matches
            WHERE status='FINISHED' AND home_goals IS NOT NULL AND away_goals IS NOT NULL
              AND match_date < ?
            ORDER BY competition_id, season_id, match_date, match_id""", [before_date]).fetchall()
    finally:
        connection.close()
    events = tuple(HistoricalMatchEvent(
        match_id=row[0], competition_id=row[1], season_id=row[2],
        home_team_id=row[3], away_team_id=row[4], match_date=row[5],
        home_goals=row[6], away_goals=row[7], source=row[8],
        retrieved_at=row[9], as_of_time=row[10],
    ) for row in rows)
    return RealTournamentContextDataset(events=events,
        data_hash=stable_hash([event.model_dump(mode="json") for event in events]),
        source="real_canonical_matches:EVENT_IMMUTABLE")


class ContextStore:
    """Additive, idempotent Phase 10 tables; does not rewrite Phase 9 records."""

    def __init__(self, db_path: str | Path) -> None:
        self.connection = connect_project_duckdb(db_path)
        self.connection.execute("SET TimeZone='UTC'")
        self.connection.execute("SET threads=2")
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def transaction(self) -> None:
        """Begin an explicit transaction for an idempotent batch commit."""
        self.connection.execute("BEGIN TRANSACTION")

    def commit(self) -> None:
        """Commit the current context-only transaction."""
        self.connection.execute("COMMIT")

    def rollback(self) -> None:
        """Roll back the current context-only transaction."""
        self.connection.execute("ROLLBACK")

    def _initialize(self) -> None:
        self.connection.execute("""CREATE TABLE IF NOT EXISTS context_tournament_states (
            state_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            prediction_snapshot_id VARCHAR NOT NULL, as_of_time TIMESTAMPTZ NOT NULL,
            payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS context_standings_snapshots (
            snapshot_id VARCHAR PRIMARY KEY, competition_id VARCHAR NOT NULL,
            season_id VARCHAR NOT NULL, as_of_date DATE NOT NULL, payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS context_schedule_contexts (
            context_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            prediction_snapshot_id VARCHAR NOT NULL, payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS context_feature_vectors (
            feature_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            prediction_snapshot_id VARCHAR NOT NULL, feature_schema_hash VARCHAR NOT NULL,
            payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS context_adjustments (
            adjustment_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            prediction_snapshot_id VARCHAR NOT NULL, status VARCHAR NOT NULL,
            payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS context_adjustment_ledger (
            ledger_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            prediction_snapshot_id VARCHAR NOT NULL, adjustment_applied BOOLEAN NOT NULL,
            payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS context_assessments (
            assessment_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            prediction_snapshot_id VARCHAR NOT NULL, context_status VARCHAR NOT NULL,
            payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS context_lineup_evidence (
            evidence_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            prediction_snapshot_id VARCHAR NOT NULL, source_id VARCHAR NOT NULL,
            retrieved_at TIMESTAMPTZ NOT NULL, payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS context_injury_evidence (
            evidence_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            prediction_snapshot_id VARCHAR NOT NULL, source_id VARCHAR NOT NULL,
            retrieved_at TIMESTAMPTZ NOT NULL, payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS context_canonical_outputs (
            run_id VARCHAR NOT NULL, prediction_id VARCHAR NOT NULL, match_id VARCHAR NOT NULL,
            prediction_snapshot_id VARCHAR NOT NULL, context_status VARCHAR NOT NULL,
            payload JSON NOT NULL, PRIMARY KEY(run_id, prediction_id))""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS context_run_manifests (
            run_id VARCHAR PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL,
            dataset_hash VARCHAR NOT NULL, target_count INTEGER NOT NULL,
            status VARCHAR NOT NULL, payload JSON NOT NULL)""")

    def save_assessment(self, assessment: ContextAssessment, canonical_payload: dict[str, Any],
                        *, run_id: str) -> None:
        """Persist each evidence and probability layer idempotently by stable identity."""
        assessment_id = stable_hash({
            "match_id": assessment.match_id,
            "prediction_snapshot_id": assessment.prediction_snapshot_id,
            "context_status": assessment.context_status.value,
            "tournament_state": assessment.tournament_state.model_dump(mode="json"),
            "incentive": assessment.incentive.model_dump(mode="json"),
            "schedule": assessment.schedule.model_dump(mode="json"),
            "data_availability": assessment.data_availability.model_dump(mode="json"),
            "lineup_evidence": [item.model_dump(mode="json") for item in assessment.lineup_evidence],
            "injury_evidence": [item.model_dump(mode="json") for item in assessment.injury_evidence],
            "feature_vector": assessment.feature_vector.model_dump(mode="json"),
            "probability_layers": (assessment.adjusted_probability.base_p_home,
                assessment.adjusted_probability.base_p_draw, assessment.adjusted_probability.base_p_away,
                assessment.adjusted_probability.final_p_home, assessment.adjusted_probability.final_p_draw,
                assessment.adjusted_probability.final_p_away),
        })[:32]
        self.connection.execute("INSERT INTO context_assessments VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(assessment_id) DO NOTHING", [assessment_id, assessment.match_id,
            assessment.prediction_snapshot_id, assessment.context_status.value,
            assessment.model_dump_json()])
        for lineup_item in assessment.lineup_evidence:
            evidence_id = stable_hash((assessment.match_id, assessment.prediction_snapshot_id,
                                       lineup_item.model_dump(mode="json")))[:32]
            self.connection.execute("INSERT INTO context_lineup_evidence VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(evidence_id) DO NOTHING", [evidence_id, assessment.match_id,
                assessment.prediction_snapshot_id, lineup_item.source_id, lineup_item.retrieved_at,
                lineup_item.model_dump_json()])
        for injury_item in assessment.injury_evidence:
            evidence_id = stable_hash((assessment.match_id, assessment.prediction_snapshot_id,
                                       injury_item.model_dump(mode="json")))[:32]
            self.connection.execute("INSERT INTO context_injury_evidence VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(evidence_id) DO NOTHING", [evidence_id, assessment.match_id,
                assessment.prediction_snapshot_id, injury_item.source_id, injury_item.retrieved_at,
                injury_item.model_dump_json()])
        state_id = stable_hash((assessment.match_id, assessment.prediction_snapshot_id, "TOURNAMENT"))[:32]
        snapshot = assessment.tournament_state.standings_state
        self.connection.execute("INSERT INTO context_tournament_states VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(state_id) DO NOTHING", [state_id, assessment.match_id,
            assessment.prediction_snapshot_id, assessment.tournament_state.as_of_time,
            assessment.tournament_state.model_dump_json()])
        if snapshot is not None:
            snapshot_id = stable_hash(snapshot.model_dump(mode="json"))[:32]
            self.connection.execute("INSERT INTO context_standings_snapshots VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(snapshot_id) DO NOTHING", [snapshot_id, snapshot.competition_id,
                snapshot.season_id, snapshot.as_of_date, snapshot.model_dump_json()])
        schedule_id = stable_hash((assessment.match_id, assessment.prediction_snapshot_id,
                                    "SCHEDULE"))[:32]
        self.connection.execute("INSERT INTO context_schedule_contexts VALUES (?, ?, ?, ?) "
            "ON CONFLICT(context_id) DO NOTHING", [schedule_id, assessment.match_id,
            assessment.prediction_snapshot_id, assessment.schedule.model_dump_json()])
        features = assessment.feature_vector
        feature_id = stable_hash((assessment.match_id, assessment.prediction_snapshot_id,
                                  features.feature_schema_hash))[:32]
        self.connection.execute("INSERT INTO context_feature_vectors VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(feature_id) DO NOTHING", [feature_id, assessment.match_id,
            assessment.prediction_snapshot_id, features.feature_schema_hash,
            features.model_dump_json()])
        adjusted = assessment.adjusted_probability
        adjustment_id = adjusted.prediction_id
        self.connection.execute("INSERT INTO context_adjustments VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(adjustment_id) DO NOTHING", [adjustment_id, assessment.match_id,
            assessment.prediction_snapshot_id, assessment.context_status.value,
            adjusted.model_dump_json()])
        ledger = assessment.ledger
        self.connection.execute("INSERT INTO context_adjustment_ledger VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(ledger_id) DO NOTHING", [ledger.ledger_id, assessment.match_id,
            assessment.prediction_snapshot_id, ledger.adjustment_applied, ledger.model_dump_json()])
        self.connection.execute("INSERT INTO context_canonical_outputs VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id, prediction_id) DO NOTHING", [run_id,
            canonical_payload["prediction_id"], assessment.match_id, assessment.prediction_snapshot_id,
            assessment.context_status.value, json.dumps(canonical_payload, ensure_ascii=False)])

    def save_run(self, *, run_id: str, created_at: Any, dataset_hash: str,
                 target_count: int, status: str, payload: dict[str, Any]) -> None:
        self.connection.execute("INSERT INTO context_run_manifests VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO NOTHING", [run_id, created_at, dataset_hash, target_count,
            status, json.dumps(payload, ensure_ascii=False)])

    def table_counts(self) -> dict[str, int]:
        tables = ("context_tournament_states", "context_standings_snapshots",
                  "context_schedule_contexts", "context_feature_vectors",
                  "context_adjustments", "context_adjustment_ledger", "context_assessments",
                  "context_lineup_evidence", "context_injury_evidence",
                  "context_canonical_outputs")
        result: dict[str, int] = {}
        for table in tables:
            row = self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()
            if row is None:
                raise ValueError(f"CONTEXT_TABLE_COUNT_UNAVAILABLE:{table}")
            result[table] = int(row[0])
        return result

    def load_run(self, run_id: str) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]] | None:
        row = self.connection.execute(
            "SELECT payload FROM context_run_manifests WHERE run_id=?", [run_id]).fetchone()
        if row is None:
            return None
        if row[0] is None:
            raise ValueError("CONTEXT_RUN_MANIFEST_PAYLOAD_MISSING")
        manifest = json.loads(row[0])
        outputs = self.connection.execute("SELECT payload FROM context_canonical_outputs "
            "WHERE run_id=? ORDER BY prediction_id", [run_id]).fetchall()
        return manifest, tuple(json.loads(item[0]) for item in outputs)
