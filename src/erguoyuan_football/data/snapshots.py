"""Immutable, self-contained prediction inputs with conservative PIT checks."""

import hashlib
from uuid import uuid4

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import Contract, UTCTime, now, utc
from erguoyuan_football.data.availability import DataAvailabilityReport, build_report
from erguoyuan_football.data.schemas import (
    Fixture,
    InjurySnapshot,
    LineupSnapshot,
    MatchResult,
    MetricSnapshot,
    OddsSnapshot,
    OptaSnapshot,
)
from erguoyuan_football.markets.schemas import (
    MarketConsensus,
    MarketGoalFeatures,
    MarketSnapshot,
    OddsQuote,
)


class HistoricalResult(Contract):
    fixture: Fixture
    result: MatchResult

    @model_validator(mode="after")
    def identity(self):
        if self.fixture.match_id != self.result.match_id or self.result.completed_at <= self.fixture.kickoff_time:
            raise ValueError("historical result does not match fixture")
        return self


class PredictionSnapshot(Contract):
    prediction_snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    match_id: str
    created_at: UTCTime = Field(default_factory=now)
    prediction_time: UTCTime
    match_data_snapshot: Fixture
    market_snapshot: tuple[OddsSnapshot, ...] = ()
    canonical_market_snapshot: MarketSnapshot | None = None
    canonical_market_consensus: tuple[MarketConsensus, ...] = ()
    opening_market_quotes: tuple[OddsQuote, ...] = ()
    market_goal_features: MarketGoalFeatures | None = None
    team_stats_snapshot: tuple[MetricSnapshot, ...] = ()
    xg_snapshot: tuple[MetricSnapshot, ...] = ()
    lineup_snapshot: tuple[LineupSnapshot, ...] = ()
    injury_snapshot: tuple[InjurySnapshot, ...] = ()
    external_snapshot: tuple[OptaSnapshot, ...] = ()
    historical_results: tuple[HistoricalResult, ...] = ()
    data_completeness: DataAvailabilityReport | None = None

    @property
    def input_data_version(self) -> str:
        # Includes frozen values, not mutable catalog references.
        payload = self.model_dump_json(exclude={"prediction_snapshot_id", "created_at", "data_completeness"})
        return hashlib.sha256(payload.encode()).hexdigest()

    @model_validator(mode="after")
    def point_in_time(self):
        fixture = self.match_data_snapshot
        if fixture.match_id != self.match_id or self.prediction_time >= fixture.kickoff_time:
            raise ValueError("prediction must refer to this fixture before kickoff")
        if fixture.as_of_time > self.prediction_time or fixture.retrieved_at > self.prediction_time:
            raise ValueError("future fixture version")
        for group in (self.market_snapshot, self.team_stats_snapshot, self.xg_snapshot,
                      self.lineup_snapshot, self.injury_snapshot, self.external_snapshot):
            for item in group:
                if item.match_id != self.match_id:
                    raise ValueError("cross-match snapshot")
                if max(item.as_of_time, item.retrieved_at) > self.prediction_time:
                    raise ValueError("future input cannot enter snapshot")
                if hasattr(item, "team_id") and item.team_id not in {fixture.home_team_id, fixture.away_team_id}:
                    raise ValueError("snapshot has unrelated team")
        for history in self.historical_results:
            if history.result.match_id == self.match_id:
                raise ValueError("target result cannot be a feature")
            if max(history.result.as_of_time, history.result.retrieved_at,
                   history.fixture.as_of_time, history.fixture.retrieved_at) > self.prediction_time:
                raise ValueError("future historical evidence")
            if history.result.completed_at >= self.prediction_time:
                raise ValueError("history must finish before prediction")
        if self.data_completeness and self.data_completeness.match_id != self.match_id:
            raise ValueError("availability report for wrong match")
        if self.canonical_market_snapshot is not None:
            frozen = self.canonical_market_snapshot
            if (frozen.match_id, frozen.prediction_time, frozen.kickoff_time) != (
                self.match_id, self.prediction_time, fixture.kickoff_time
            ):
                raise ValueError("canonical market snapshot does not match prediction cutoff")
        for consensus in self.canonical_market_consensus:
            if self.canonical_market_snapshot is None or (
                consensus.match_id, consensus.market_snapshot_id, consensus.prediction_time
            ) != (self.match_id, self.canonical_market_snapshot.market_snapshot_id, self.prediction_time):
                raise ValueError("market consensus is detached from the frozen market snapshot")
        for quote in self.opening_market_quotes:
            if (quote.match_id != self.match_id or not quote.is_opening_confirmed or quote.as_of_time is None
                    or max(utc(quote.as_of_time), utc(quote.retrieved_at)) > self.prediction_time):
                raise ValueError("opening quote violates prediction-time availability")
        if self.market_goal_features is not None:
            feature = self.market_goal_features
            if feature.match_id != self.match_id or feature.prediction_time != self.prediction_time:
                raise ValueError("market goal features do not match frozen prediction identity")
            if feature.availability == "AVAILABLE" and (
                feature.as_of_time is None or feature.retrieved_at is None
                or max(feature.as_of_time, feature.retrieved_at) > self.prediction_time
            ):
                raise ValueError("future market goal features cannot enter prediction snapshot")
        return self


def latest(records):
    """Keep latest revision per semantic stream; never mix bookmakers or market phases."""
    streams = {}
    for record in records:
        if isinstance(record, OddsSnapshot):
            key = (record.source, record.bookmaker, record.market_type, record.phase, record.line, record.selection)
        else:
            key = (record.source, getattr(record, "team_id", None), getattr(record, "product", None))
        existing = streams.get(key)
        if (existing and (existing.as_of_time, existing.retrieved_at) == (record.as_of_time, record.retrieved_at)
                and existing.model_dump(exclude={"snapshot_id"}) != record.model_dump(exclude={"snapshot_id"})):
            raise ValueError("conflicting snapshot revisions at identical timestamps")
        if existing is None or (record.as_of_time, record.retrieved_at) > (existing.as_of_time, existing.retrieved_at):
            streams[key] = record
    return tuple(sorted(streams.values(), key=lambda item: item.snapshot_id))


class SnapshotService:
    def __init__(self, store):
        self.store = store

    def create(self, match_id: str, prediction_time) -> PredictionSnapshot:
        at = utc(prediction_time)
        connection = self.store.connection
        connection.execute("BEGIN TRANSACTION")
        try:
            fixture = self.store.fixture_at(match_id, at)
            if at >= fixture.kickoff_time:
                raise ValueError("prediction must precede kickoff")
            get = lambda table: self.store.snapshots_at(table, match_id, at)
            history = []
            for result in self.store.results_at(at):
                past = self.store.fixture_at(result.match_id, at)
                if {past.home_team_id, past.away_team_id} & {fixture.home_team_id, fixture.away_team_id}:
                    history.append(HistoricalResult(fixture=past, result=result))
            canonical_market = self.store.market_snapshot_at(match_id, at)
            snapshot = PredictionSnapshot(
                match_id=match_id, prediction_time=at, match_data_snapshot=fixture,
                market_snapshot=latest(get("odds_snapshots")),
                canonical_market_snapshot=canonical_market,
                canonical_market_consensus=self.store.market_consensus_at(canonical_market.market_snapshot_id)
                    if canonical_market is not None else (),
                opening_market_quotes=self.store.confirmed_opening_quotes_at(match_id, at),
                market_goal_features=next((item for item in self.store.market_goal_features_at(
                    at, match_ids=(match_id,)) if item.prediction_time == at), None),
                team_stats_snapshot=latest(get("team_stats_snapshots")), xg_snapshot=latest(get("xg_snapshots")),
                lineup_snapshot=latest(get("lineup_snapshots")), injury_snapshot=latest(get("injury_snapshots")),
                external_snapshot=latest(get("opta_snapshots")), historical_results=tuple(history))
            # Competition metadata is not versioned yet; do not claim PIT hierarchy availability.
            report = build_report(snapshot)
            snapshot = PredictionSnapshot.model_validate({**snapshot.model_dump(), "data_completeness": report})
            self.store.save_prediction_snapshot(snapshot)
            connection.execute("COMMIT")
            return snapshot
        except Exception:
            connection.execute("ROLLBACK")
            raise
