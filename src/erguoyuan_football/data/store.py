"""Append-only DuckDB repository with parameterized writes and Parquet export."""

import json
from pathlib import Path

import duckdb

from erguoyuan_football.contracts.common import Contract, utc
from erguoyuan_football.data.schemas import (
    Competition,
    CompetitionAlias,
    Fixture,
    InjurySnapshot,
    LineupSnapshot,
    MatchResult,
    MetricSnapshot,
    OddsSnapshot,
    OptaSnapshot,
    Team,
    TeamAlias,
)

SNAPSHOT_TYPES: dict[str, type[Contract]] = {
    "odds_snapshots": OddsSnapshot,
    "team_stats_snapshots": MetricSnapshot, "xg_snapshots": MetricSnapshot,
    "lineup_snapshots": LineupSnapshot, "injury_snapshots": InjurySnapshot,
    "opta_snapshots": OptaSnapshot,
}


def normalize(value: str) -> str:
    """Exact case/space normalization; deliberately no fuzzy matching."""
    return " ".join(value.casefold().split())


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.connection = duckdb.connect(str(path))
        self.connection.execute("SET TimeZone='UTC'")
        self._initialize()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.connection.close()

    def _initialize(self):
        existing = {row[0] for row in self.connection.execute("SHOW TABLES").fetchall()}
        if "market_snapshots" in existing:
            columns = {row[1] for row in self.connection.execute(
                "PRAGMA table_info('market_snapshots')").fetchall()}
            if "snapshot_id" in columns and "market_snapshot_id" not in columns:
                legacy_name = "market_snapshots_legacy_phase2"
                if legacy_name not in existing:
                    self.connection.execute(f"ALTER TABLE market_snapshots RENAME TO {legacy_name}")
        self.connection.execute("""
        CREATE TABLE IF NOT EXISTS teams (team_id VARCHAR PRIMARY KEY, team_name VARCHAR NOT NULL, payload JSON NOT NULL);
        CREATE TABLE IF NOT EXISTS team_aliases (
            alias VARCHAR NOT NULL, normalized_alias VARCHAR NOT NULL,
            team_id VARCHAR REFERENCES teams(team_id), language VARCHAR NOT NULL, source VARCHAR NOT NULL,
            confidence DOUBLE CHECK(confidence BETWEEN 0 AND 1), created_at TIMESTAMPTZ NOT NULL,
            payload JSON NOT NULL, PRIMARY KEY(alias, team_id, source, created_at));
        CREATE TABLE IF NOT EXISTS competitions (
            competition_id VARCHAR PRIMARY KEY, competition_name VARCHAR NOT NULL,
            country VARCHAR, competition_type VARCHAR, season_format VARCHAR, tier INTEGER, payload JSON NOT NULL);
        CREATE TABLE IF NOT EXISTS competition_aliases (
            alias VARCHAR NOT NULL, normalized_alias VARCHAR NOT NULL,
            competition_id VARCHAR REFERENCES competitions(competition_id), language VARCHAR NOT NULL,
            source VARCHAR NOT NULL, confidence DOUBLE CHECK(confidence BETWEEN 0 AND 1),
            created_at TIMESTAMPTZ NOT NULL, payload JSON NOT NULL,
            PRIMARY KEY(alias, competition_id, source, created_at));
        CREATE TABLE IF NOT EXISTS matches (
            match_id VARCHAR NOT NULL, data_version VARCHAR NOT NULL, lottery_match_no VARCHAR,
            competition_id VARCHAR REFERENCES competitions(competition_id),
            home_team_id VARCHAR REFERENCES teams(team_id), away_team_id VARCHAR REFERENCES teams(team_id),
            kickoff_time TIMESTAMPTZ NOT NULL, source VARCHAR NOT NULL,
            retrieved_at TIMESTAMPTZ NOT NULL, as_of_time TIMESTAMPTZ NOT NULL, payload JSON NOT NULL,
            PRIMARY KEY(match_id, data_version));
        CREATE TABLE IF NOT EXISTS match_requests (
            request_id VARCHAR PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, input_type VARCHAR NOT NULL,
            match_id VARCHAR, resolution_status VARCHAR NOT NULL, payload JSON NOT NULL);
        CREATE TABLE IF NOT EXISTS match_results (
            result_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL, home_goals INTEGER NOT NULL,
            away_goals INTEGER NOT NULL, completed_at TIMESTAMPTZ NOT NULL, source VARCHAR NOT NULL,
            retrieved_at TIMESTAMPTZ NOT NULL, as_of_time TIMESTAMPTZ NOT NULL,
            data_version VARCHAR NOT NULL, payload JSON NOT NULL, UNIQUE(match_id, data_version));
        CREATE TABLE IF NOT EXISTS prediction_snapshots (
            prediction_snapshot_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL, prediction_time TIMESTAMPTZ NOT NULL, payload JSON NOT NULL);
        """)
        for table in SNAPSHOT_TYPES:
            self.connection.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
                snapshot_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL, source VARCHAR NOT NULL,
                retrieved_at TIMESTAMPTZ NOT NULL, as_of_time TIMESTAMPTZ NOT NULL,
                data_version VARCHAR NOT NULL, availability VARCHAR NOT NULL, payload JSON NOT NULL)""")
        self._migrate_legacy_market_quotes()
        self.connection.execute("""CREATE TABLE IF NOT EXISTS odds_quotes (
            quote_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL, provider_id VARCHAR NOT NULL,
            bookmaker_id VARCHAR NOT NULL, market_type VARCHAR NOT NULL, selection VARCHAR NOT NULL,
            line_quarters INTEGER, source_time TIMESTAMPTZ, retrieved_at TIMESTAMPTZ NOT NULL,
            as_of_time TIMESTAMPTZ, is_live BOOLEAN NOT NULL, is_suspended BOOLEAN NOT NULL,
            schema_version VARCHAR NOT NULL, content_hash VARCHAR NOT NULL, payload JSON NOT NULL)""")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_odds_quotes_match_asof ON odds_quotes(match_id, as_of_time, retrieved_at)")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS market_event_bindings (
            binding_id VARCHAR PRIMARY KEY, provider_id VARCHAR NOT NULL,
            source_event_id VARCHAR NOT NULL, match_id VARCHAR NOT NULL,
            orientation VARCHAR NOT NULL, checked_at TIMESTAMPTZ NOT NULL,
            payload JSON NOT NULL)""")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_market_event_identity "
                                "ON market_event_bindings(provider_id, source_event_id)")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS market_snapshots (
            market_snapshot_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            prediction_time TIMESTAMPTZ NOT NULL, created_at TIMESTAMPTZ NOT NULL,
            source_count INTEGER NOT NULL, bookmaker_count INTEGER NOT NULL,
            quote_count INTEGER NOT NULL, quality_status VARCHAR NOT NULL, payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS market_consensus (
            consensus_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            market_snapshot_id VARCHAR NOT NULL, prediction_time TIMESTAMPTZ NOT NULL,
            market_type VARCHAR NOT NULL, line_quarters INTEGER, payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS market_movements (
            movement_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            as_of_time TIMESTAMPTZ NOT NULL, market_type VARCHAR NOT NULL,
            line_quarters INTEGER, payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS market_quality_reports (
            report_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            prediction_time TIMESTAMPTZ NOT NULL, quality_status VARCHAR NOT NULL, payload JSON NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS market_implied_goals (
            market_snapshot_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
            prediction_time TIMESTAMPTZ NOT NULL, availability VARCHAR NOT NULL, payload JSON NOT NULL)""")
        for table in ("model_predictions", "oos_predictions", "final_predictions"):
            self.connection.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
                prediction_id VARCHAR PRIMARY KEY, match_id VARCHAR NOT NULL,
                prediction_snapshot_id VARCHAR REFERENCES prediction_snapshots(prediction_snapshot_id),
                prediction_time TIMESTAMPTZ NOT NULL, payload JSON NOT NULL)""")

    def _migrate_legacy_market_quotes(self) -> None:
        """Copy the Phase 2 alias table's immutable quote rows into odds_snapshots."""
        exists = self.connection.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name='market_snapshots_legacy_phase2'"
        ).fetchone()
        if not exists:
            return
        rows = self.connection.execute("SELECT snapshot_id, match_id, source, retrieved_at, as_of_time, "
            "data_version, availability, payload FROM market_snapshots_legacy_phase2").fetchall()
        for row in rows:
            self.connection.execute("INSERT INTO odds_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                                    "ON CONFLICT(snapshot_id) DO NOTHING", list(row))

    @property
    def tables(self) -> tuple[str, ...]:
        return tuple(row[0] for row in self.connection.execute("SHOW TABLES").fetchall())

    def add_team(self, value: Team):
        self.connection.execute("INSERT INTO teams VALUES (?, ?, ?)",
                                [value.team_id, value.team_name, value.model_dump_json()])

    def add_team_alias(self, value: TeamAlias):
        self.connection.execute("INSERT INTO team_aliases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                [value.alias, normalize(value.alias), value.team_id, value.language,
                                 value.source, value.confidence, value.created_at, value.model_dump_json()])

    def add_competition(self, value: Competition):
        self.connection.execute("INSERT INTO competitions VALUES (?, ?, ?, ?, ?, ?, ?)",
                                [value.competition_id, value.competition_name, value.country,
                                 value.competition_type, value.season_format, value.tier, value.model_dump_json()])

    def add_competition_alias(self, value: CompetitionAlias):
        self.connection.execute("INSERT INTO competition_aliases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                [value.alias, normalize(value.alias), value.competition_id, value.language,
                                 value.source, value.confidence, value.created_at, value.model_dump_json()])

    def alias_ids(self, name: str, at, *, competition=False, min_confidence=1.0) -> tuple[str, ...]:
        table, column = ("competition_aliases", "competition_id") if competition else ("team_aliases", "team_id")
        rows = self.connection.execute(
            f"SELECT DISTINCT {column} FROM {table} WHERE normalized_alias=? AND created_at<=? AND confidence>=? ORDER BY {column}",
            [normalize(name), utc(at), min_confidence]).fetchall()
        return tuple(row[0] for row in rows)

    def team(self, team_id: str) -> Team:
        row = self.connection.execute("SELECT payload FROM teams WHERE team_id=?", [team_id]).fetchone()
        if not row:
            raise KeyError(team_id)
        return Team.model_validate_json(row[0])

    def competition(self, competition_id: str) -> Competition:
        row = self.connection.execute("SELECT payload FROM competitions WHERE competition_id=?", [competition_id]).fetchone()
        if not row:
            raise KeyError(competition_id)
        return Competition.model_validate_json(row[0])

    def add_fixture(self, value: Fixture):
        if self.connection.execute("SELECT 1 FROM matches WHERE match_id=? AND as_of_time=? AND retrieved_at=?",
                                   [value.match_id, value.as_of_time, value.retrieved_at]).fetchone():
            raise ValueError("fixture version at identical timestamps already exists")
        self.connection.execute("INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [value.match_id, value.data_version, value.lottery_match_no, value.competition_id,
                                 value.home_team_id, value.away_team_id, value.kickoff_time, value.source,
                                 value.retrieved_at, value.as_of_time, value.model_dump_json()])

    def fixtures_at(self, at) -> tuple[Fixture, ...]:
        # Filter versions before ranking, so a later schedule correction cannot leak backwards.
        rows = self.connection.execute("""SELECT payload FROM matches
            WHERE as_of_time<=? AND retrieved_at<=?
            QUALIFY row_number() OVER (PARTITION BY match_id ORDER BY as_of_time DESC, retrieved_at DESC, data_version DESC)=1
            ORDER BY match_id""", [utc(at), utc(at)]).fetchall()
        return tuple(Fixture.model_validate_json(row[0]) for row in rows)

    def fixture_at(self, match_id, at) -> Fixture:
        for fixture in self.fixtures_at(at):
            if fixture.match_id == match_id:
                return fixture
        raise ValueError("fixture unavailable at prediction time")

    def _require_match(self, match_id):
        if not self.connection.execute("SELECT 1 FROM matches WHERE match_id=? LIMIT 1", [match_id]).fetchone():
            raise ValueError("unknown match_id")

    def save_request(self, value):
        from erguoyuan_football.input.schemas import MatchRequest
        value = MatchRequest.model_validate(value.model_dump())
        self.connection.execute("INSERT INTO match_requests VALUES (?, ?, ?, ?, ?, ?)",
                                [value.request_id, value.created_at, value.input_type.value,
                                 value.match_id, value.resolution_status.value, value.model_dump_json()])

    def add_snapshot(self, table: str, value):
        if table == "market_snapshots":
            # Backward-compatible Phase 2 API: this alias still accepts legacy OddsSnapshot records.
            table = "odds_snapshots"
        if table not in SNAPSHOT_TYPES:
            raise ValueError("not an input snapshot table")
        value = SNAPSHOT_TYPES[table].model_validate(value.model_dump())
        self._require_match(value.match_id)
        if hasattr(value, "team_id"):
            rows = self.connection.execute("SELECT home_team_id, away_team_id FROM matches WHERE match_id=?", [value.match_id]).fetchall()
            if not any(value.team_id in row for row in rows):
                raise ValueError("snapshot team does not belong to match")
        self.connection.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                [value.snapshot_id, value.match_id, value.source, value.retrieved_at,
                                 value.as_of_time, value.data_version, value.availability.value, value.model_dump_json()])

    def snapshots_at(self, table: str, match_id: str, prediction_time):
        if table == "market_snapshots":
            table = "odds_snapshots"
        if table not in SNAPSHOT_TYPES:
            raise ValueError("not an input snapshot table")
        fixture = self.fixture_at(match_id, prediction_time)
        at = utc(prediction_time)
        if at >= fixture.kickoff_time:
            raise ValueError("prediction must precede kickoff")
        rows = self.connection.execute(f"""SELECT payload FROM {table}
            WHERE match_id=? AND as_of_time<=? AND retrieved_at<=?
            ORDER BY as_of_time, retrieved_at, snapshot_id""", [match_id, at, at]).fetchall()
        return tuple(SNAPSHOT_TYPES[table].model_validate_json(row[0]) for row in rows)

    def add_odds_quote(self, quote) -> None:
        """Append a canonical quote; primary-key conflicts are rejected rather than updated."""
        from erguoyuan_football.markets.schemas import OddsQuote

        quote = OddsQuote.model_validate(quote.model_dump())
        self._require_match(quote.match_id)
        self.connection.execute("INSERT INTO odds_quotes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            quote.quote_id, quote.match_id, quote.provider_id, quote.bookmaker_id, quote.market_type.value,
            quote.selection.value, quote.line_quarters, quote.source_time, quote.retrieved_at,
            quote.as_of_time, quote.is_live, quote.is_suspended, quote.schema_version,
            quote.content_hash, quote.model_dump_json(),
        ])

    def save_market_event_binding(self, binding) -> None:
        """Append verified source-event identity; reject reuse for another fixture or orientation."""
        from erguoyuan_football.markets.identity import EventBinding

        binding = EventBinding.model_validate(binding.model_dump())
        self._require_match(binding.match_id)
        rows = self.connection.execute(
            "SELECT match_id, orientation FROM market_event_bindings "
            "WHERE provider_id=? AND source_event_id=?",
            [binding.provider_id, binding.source_event_id],
        ).fetchall()
        if any((match_id, orientation) != (binding.match_id, binding.orientation.value)
               for match_id, orientation in rows):
            raise ValueError("PROVIDER_EVENT_REBOUND_TO_DIFFERENT_FIXTURE_OR_DIRECTION")
        quote_matches = self.connection.execute(
            "SELECT DISTINCT match_id FROM odds_quotes WHERE provider_id=? "
            "AND json_extract_string(payload, '$.source_event_id')=?",
            [binding.provider_id, binding.source_event_id],
        ).fetchall()
        if any(match_id != binding.match_id for (match_id,) in quote_matches):
            raise ValueError("PROVIDER_EVENT_QUOTE_ALREADY_BOUND_TO_OTHER_FIXTURE")
        existing = self.connection.execute("SELECT payload FROM market_event_bindings WHERE binding_id=?",
                                           [binding.binding_id]).fetchone()
        if existing is None:
            self.connection.execute("INSERT INTO market_event_bindings VALUES (?, ?, ?, ?, ?, ?, ?)", [
                binding.binding_id, binding.provider_id, binding.source_event_id, binding.match_id,
                binding.orientation.value, binding.checked_at, binding.model_dump_json(),
            ])
        elif existing[0] != binding.model_dump_json():
            raise ValueError("MARKET_EVENT_BINDING_ID_CONTENT_CONFLICT")

    def odds_quotes_at(self, match_id: str, prediction_time):
        """Read only source-timestamped and retrieved quotes available by the forecast cutoff."""
        from erguoyuan_football.markets.schemas import OddsQuote

        at = utc(prediction_time)
        fixture = self.fixture_at(match_id, at)
        if at >= fixture.kickoff_time:
            raise ValueError("prediction must precede kickoff")
        rows = self.connection.execute("""SELECT payload FROM odds_quotes
            WHERE match_id=? AND as_of_time IS NOT NULL AND as_of_time<=? AND retrieved_at<=?
            ORDER BY as_of_time, retrieved_at, quote_id""", [match_id, at, at]).fetchall()
        return tuple(OddsQuote.model_validate_json(row[0]) for row in rows)

    def save_market_snapshot(self, snapshot) -> None:
        """Persist one immutable point-in-time market freeze and its quote identities."""
        self._require_match(snapshot.match_id)
        persisted_rows = self.connection.execute(
            "SELECT quote_id, payload FROM odds_quotes WHERE match_id=? AND as_of_time<=? AND retrieved_at<=?",
            [snapshot.match_id, utc(snapshot.prediction_time), utc(snapshot.prediction_time)]).fetchall()
        persisted = {quote_id: payload for quote_id, payload in persisted_rows}
        by_quote_id = {quote.quote_id: quote for quote in snapshot.quotes}
        if set(snapshot.included_quote_ids) - persisted.keys():
            raise ValueError("MARKET_SNAPSHOT_REFERENCES_UNPERSISTED_OR_FUTURE_QUOTES")
        if any(persisted[quote_id] != by_quote_id[quote_id].model_dump_json()
               for quote_id in snapshot.included_quote_ids):
            raise ValueError("MARKET_SNAPSHOT_QUOTE_CONTENT_MISMATCH")
        self.connection.execute("INSERT INTO market_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            snapshot.market_snapshot_id, snapshot.match_id, snapshot.prediction_time, snapshot.created_at,
            snapshot.source_count, snapshot.bookmaker_count, snapshot.quote_count,
            snapshot.quality_status.value, snapshot.model_dump_json(),
        ])

    def save_market_consensus(self, consensus) -> None:
        """Append a consensus row tied to a persisted snapshot."""
        row = self.connection.execute("SELECT match_id, prediction_time, payload FROM market_snapshots "
                                      "WHERE market_snapshot_id=?", [consensus.market_snapshot_id]).fetchone()
        if not row:
            raise ValueError("UNKNOWN_MARKET_SNAPSHOT")
        if (row[0], row[1]) != (consensus.match_id, consensus.prediction_time):
            raise ValueError("MARKET_CONSENSUS_SNAPSHOT_LINEAGE_MISMATCH")
        snapshot_payload = json.loads(row[2])
        if not set(consensus.quote_ids) <= set(snapshot_payload["included_quote_ids"]):
            raise ValueError("MARKET_CONSENSUS_REFERENCES_QUOTES_OUTSIDE_SNAPSHOT")
        self.connection.execute("INSERT INTO market_consensus VALUES (?, ?, ?, ?, ?, ?, ?)", [
            consensus.consensus_id, consensus.match_id, consensus.market_snapshot_id,
            consensus.prediction_time, consensus.market_type.value, consensus.line_quarters,
            consensus.model_dump_json(),
        ])

    def save_market_movement(self, movement) -> None:
        """Append a market movement diagnostic without rewriting history."""
        self._require_match(movement.match_id)
        quote_ids = [movement.current_quote_id]
        if movement.opening_quote_id is not None:
            quote_ids.append(movement.opening_quote_id)
        quote_ids = list(dict.fromkeys(quote_ids))
        rows = self.connection.execute("SELECT quote_id, match_id, as_of_time FROM odds_quotes "
                                       "WHERE quote_id IN (" + ",".join("?" for _ in quote_ids) + ")",
                                       quote_ids).fetchall()
        if len(rows) != len(quote_ids) or any(row[1] != movement.match_id for row in rows):
            raise ValueError("MARKET_MOVEMENT_QUOTE_LINEAGE_MISMATCH")
        if any(row[2] > movement.as_of_time for row in rows):
            raise ValueError("MARKET_MOVEMENT_FUTURE_QUOTE")
        self.connection.execute("INSERT INTO market_movements VALUES (?, ?, ?, ?, ?, ?)", [
            movement.movement_id, movement.match_id, movement.as_of_time,
            movement.market_type.value, movement.line_quarters, movement.model_dump_json(),
        ])

    def save_market_quality(self, quality) -> None:
        """Append a quality report to its match timeline."""
        self._require_match(quality.match_id)
        self.connection.execute("INSERT INTO market_quality_reports VALUES (?, ?, ?, ?, ?)", [
            quality.report_id, quality.match_id, quality.prediction_time,
            quality.status.value, quality.model_dump_json(),
        ])

    def save_market_implied_goals(self, features) -> None:
        """Append the goal-fit record under the unique market-snapshot identity."""
        row = self.connection.execute("SELECT match_id, prediction_time FROM market_snapshots "
                                      "WHERE market_snapshot_id=?", [features.market_snapshot_id]).fetchone()
        if not row or (row[0], row[1]) != (features.match_id, features.prediction_time):
            raise ValueError("MARKET_GOALS_SNAPSHOT_LINEAGE_MISMATCH")
        self.connection.execute("INSERT INTO market_implied_goals VALUES (?, ?, ?, ?, ?)", [
            features.market_snapshot_id, features.match_id, features.prediction_time,
            features.availability.value, features.model_dump_json(),
        ])

    def market_goal_features_at(self, prediction_time, *, match_ids: tuple[str, ...] = ()):
        """Return latest available, fully retrieved market goal features at a cutoff."""
        from erguoyuan_football.markets.schemas import MarketGoalFeatures

        if not match_ids:
            return ()
        at = utc(prediction_time)
        placeholders = ",".join("?" for _ in match_ids)
        rows = self.connection.execute(
            f"SELECT payload FROM market_implied_goals WHERE match_id IN ({placeholders}) "
            "AND prediction_time<=? AND availability='AVAILABLE' ORDER BY prediction_time, market_snapshot_id",
            [*match_ids, at],
        ).fetchall()
        latest: dict[tuple[str, str], MarketGoalFeatures] = {}
        for (payload,) in rows:
            feature = MarketGoalFeatures.model_validate_json(payload)
            if feature.as_of_time is None or feature.retrieved_at is None:
                continue
            if max(feature.as_of_time, feature.retrieved_at) > at:
                continue
            key = (feature.match_id, feature.prediction_horizon.value)
            previous = latest.get(key)
            if previous is None or feature.prediction_time > previous.prediction_time:
                latest[key] = feature
        return tuple(latest[key] for key in sorted(latest))

    def market_snapshot_at(self, match_id: str, prediction_time):
        """Return a canonical market freeze only when it was created at this exact cutoff."""
        from erguoyuan_football.markets.schemas import MarketSnapshot

        row = self.connection.execute("SELECT payload FROM market_snapshots "
            "WHERE match_id=? AND prediction_time=? ORDER BY created_at DESC, market_snapshot_id DESC LIMIT 1",
            [match_id, utc(prediction_time)]).fetchone()
        return MarketSnapshot.model_validate_json(row[0]) if row else None

    def market_consensus_at(self, market_snapshot_id: str):
        """Load every consensus row attached to the requested immutable freeze."""
        from erguoyuan_football.markets.schemas import MarketConsensus

        rows = self.connection.execute("SELECT payload FROM market_consensus WHERE market_snapshot_id=? "
                                       "ORDER BY market_type, line_quarters", [market_snapshot_id]).fetchall()
        return tuple(MarketConsensus.model_validate_json(row[0]) for row in rows)

    def confirmed_opening_quotes_at(self, match_id: str, prediction_time):
        """Load only explicitly confirmed opening quotes available by this forecast cutoff."""
        return tuple(quote for quote in self.odds_quotes_at(match_id, prediction_time)
                     if quote.is_opening_confirmed)

    def add_result(self, value: MatchResult):
        self._require_match(value.match_id)
        fixture = self.fixture_at(value.match_id, max(value.retrieved_at, value.as_of_time))
        if value.completed_at <= fixture.kickoff_time:
            raise ValueError("result completion must follow kickoff")
        self.connection.execute("INSERT INTO match_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [value.result_id, value.match_id, value.home_goals, value.away_goals,
                                 value.completed_at, value.source, value.retrieved_at, value.as_of_time,
                                 value.data_version, value.model_dump_json()])

    def results_at(self, prediction_time) -> tuple[MatchResult, ...]:
        at = utc(prediction_time)
        rows = self.connection.execute("""SELECT payload FROM match_results
            WHERE as_of_time<=? AND retrieved_at<=? AND completed_at<?
            QUALIFY row_number() OVER (PARTITION BY match_id ORDER BY as_of_time DESC, retrieved_at DESC, data_version DESC)=1
            ORDER BY match_id""", [at, at, at]).fetchall()
        return tuple(MatchResult.model_validate_json(row[0]) for row in rows)

    def save_prediction_snapshot(self, value):
        from erguoyuan_football.data.availability import build_report
        from erguoyuan_football.data.snapshots import PredictionSnapshot
        value = PredictionSnapshot.model_validate(value.model_dump())
        if value.data_completeness != build_report(value):
            raise ValueError("availability report must match frozen evidence")
        if self.fixture_at(value.match_id, value.prediction_time) != value.match_data_snapshot:
            raise ValueError("snapshot fixture is not the stored PIT version")
        if value.market_goal_features is not None:
            persisted_goal = self.connection.execute(
                "SELECT payload FROM market_implied_goals WHERE market_snapshot_id=?",
                [value.market_goal_features.market_snapshot_id],
            ).fetchone()
            if not persisted_goal or persisted_goal[0] != value.market_goal_features.model_dump_json():
                raise ValueError("market goal features are not persisted or were altered")
        if value.canonical_market_snapshot is not None:
            persisted_market = self.market_snapshot_at(value.match_id, value.prediction_time)
            if persisted_market != value.canonical_market_snapshot:
                raise ValueError("canonical market snapshot is not persisted or was altered")
            persisted_consensus = self.market_consensus_at(persisted_market.market_snapshot_id)
            if persisted_consensus != value.canonical_market_consensus:
                raise ValueError("canonical market consensus is not persisted or was altered")
        elif value.canonical_market_consensus:
            raise ValueError("market consensus cannot exist without its canonical snapshot")
        persisted_openings = {quote.quote_id: quote for quote in self.odds_quotes_at(
            value.match_id, value.prediction_time) if quote.is_opening_confirmed}
        if any(persisted_openings.get(quote.quote_id) != quote for quote in value.opening_market_quotes):
            raise ValueError("opening market quotes are not persisted or were altered")
        groups = ((value.market_snapshot, ("odds_snapshots",)),
                  (value.team_stats_snapshot, ("team_stats_snapshots",)), (value.xg_snapshot, ("xg_snapshots",)),
                  (value.lineup_snapshot, ("lineup_snapshots",)), (value.injury_snapshot, ("injury_snapshots",)),
                  (value.external_snapshot, ("opta_snapshots",)))
        for records, tables in groups:
            stored = {item.snapshot_id: item for table in tables
                      for item in self.snapshots_at(table, value.match_id, value.prediction_time)}
            if any(stored.get(item.snapshot_id) != item for item in records):
                raise ValueError("snapshot evidence is not persisted or was altered")
        stored_results = {item.result_id: item for item in self.results_at(value.prediction_time)}
        for item in value.historical_results:
            if stored_results.get(item.result.result_id) != item.result or self.fixture_at(item.fixture.match_id, value.prediction_time) != item.fixture:
                raise ValueError("historical evidence is not the stored PIT version")
        self.connection.execute("INSERT INTO prediction_snapshots VALUES (?, ?, ?, ?, ?)",
                                [value.prediction_snapshot_id, value.match_id, value.created_at,
                                 value.prediction_time, value.model_dump_json()])

    def load_prediction_snapshot(self, identifier):
        from erguoyuan_football.data.snapshots import PredictionSnapshot
        row = self.connection.execute("SELECT payload FROM prediction_snapshots WHERE prediction_snapshot_id=?", [identifier]).fetchone()
        if not row:
            raise KeyError(identifier)
        return PredictionSnapshot.model_validate_json(row[0])

    def save_prediction(self, value, *, prediction_id: str, table="model_predictions"):
        from erguoyuan_football.contracts.predictions import (
            CorePrediction,
            ModelPrediction,
        )
        if table not in {"model_predictions", "oos_predictions", "final_predictions"}:
            raise ValueError("invalid prediction table")
        cls = CorePrediction if table == "final_predictions" else ModelPrediction
        value = cls.model_validate(value.model_dump())
        snapshot = self.load_prediction_snapshot(value.prediction_snapshot_id)
        if value.match_id != snapshot.match_id or value.prediction_time != snapshot.prediction_time:
            raise ValueError("prediction does not match frozen snapshot")
        if isinstance(value, ModelPrediction) and value.input_data_version != snapshot.input_data_version:
            raise ValueError("prediction input version does not match frozen snapshot")
        if isinstance(value, CorePrediction) and any(p.input_data_version != snapshot.input_data_version
                for p in value.base_model_predictions + value.external_predictions):
            raise ValueError("CORE child prediction input version does not match frozen snapshot")
        if table == "oos_predictions" and (not value.is_oos or value.training_end_time is None
                                          or value.training_end_time >= value.prediction_time):
            raise ValueError("OOS requires explicit provenance and earlier training cutoff")
        self.connection.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?)",
                                [prediction_id, value.match_id, value.prediction_snapshot_id,
                                 value.prediction_time, value.model_dump_json()])

    def export_parquet(self, table: str, destination: str | Path) -> Path:
        if table not in self.tables:
            raise ValueError("unknown table")
        target = Path(destination).resolve()
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Relation API handles the path, never interpolate user paths into SQL.
        self.connection.table(table).write_parquet(str(target))
        return target

    def schema_description(self):
        return {table: self.connection.execute(f"DESCRIBE {table}").fetchall() for table in self.tables}
