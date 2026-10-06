"""Canonical catalog and versioned input records, not prediction models."""

from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import (
    Availability,
    Contract,
    Identifier,
    NonNegative,
    Probability,
    UTCTime,
)


class Team(Contract):
    team_id: Identifier
    team_name: Identifier


class TeamAlias(Contract):
    alias: Identifier
    team_id: Identifier
    language: Identifier
    source: Identifier
    confidence: Probability
    created_at: UTCTime


class Competition(Contract):
    competition_id: Identifier
    competition_name: Identifier
    country: str | None = None
    competition_type: str | None = None
    season_format: str | None = None
    tier: int | None = Field(default=None, ge=1)


class CompetitionAlias(Contract):
    alias: Identifier
    competition_id: Identifier
    language: Identifier
    source: Identifier
    confidence: Probability
    created_at: UTCTime


class Fixture(Contract):
    match_id: Identifier
    lottery_match_no: str | None = None
    competition_id: Identifier
    home_team_id: Identifier
    away_team_id: Identifier
    kickoff_time: UTCTime
    source: Identifier
    retrieved_at: UTCTime
    as_of_time: UTCTime
    data_version: Identifier
    season: str | None = None
    neutral_venue: bool | None = None

    @model_validator(mode="after")
    def different_teams(self):
        if self.home_team_id == self.away_team_id:
            raise ValueError("fixture teams must differ")
        return self


class SnapshotRecord(Contract):
    snapshot_id: Identifier = Field(default_factory=lambda: str(uuid4()))
    match_id: Identifier
    source: Identifier
    retrieved_at: UTCTime
    as_of_time: UTCTime
    data_version: Identifier
    availability: Availability = Availability.AVAILABLE
    reason: str | None = None

    @model_validator(mode="after")
    def unavailable_reason(self):
        if self.availability == Availability.UNAVAILABLE and not self.reason:
            raise ValueError("UNAVAILABLE requires reason")
        return self


class OddsSnapshot(SnapshotRecord):
    bookmaker: Identifier
    market_type: Literal["1X2", "ASIAN_HANDICAP", "OVER_UNDER", "SPORTS_LOTTERY"]
    phase: Literal["OPEN", "CURRENT", "CLOSE"]
    selection: Identifier
    line: float | None = None
    odds: float | None = Field(default=None, gt=1, allow_inf_nan=False)
    raw_implied_probability: Probability | None = None
    devig_probability: Probability | None = None

    @model_validator(mode="after")
    def market_fields(self):
        if self.availability == Availability.AVAILABLE and self.odds is None:
            raise ValueError("available odds require a real price")
        if self.availability == Availability.UNAVAILABLE and any(value is not None for value in
                (self.odds, self.raw_implied_probability, self.devig_probability)):
            raise ValueError("unavailable odds and probabilities must be null")
        choices = {"1X2": {"HOME", "DRAW", "AWAY"},
                   "SPORTS_LOTTERY": {"HOME", "DRAW", "AWAY"},
                   "ASIAN_HANDICAP": {"HOME", "AWAY"}, "OVER_UNDER": {"OVER", "UNDER"}}
        if self.selection not in choices[self.market_type]:
            raise ValueError("invalid selection for market")
        if self.market_type in {"ASIAN_HANDICAP", "OVER_UNDER"} and self.line is None:
            raise ValueError("line required")
        if self.source == "USER_SCREENSHOT" and self.phase == "CLOSE":
            raise ValueError("screenshot odds cannot establish closing odds")
        if self.raw_implied_probability is not None and abs(self.raw_implied_probability - 1 / self.odds) > 1e-8:
            raise ValueError("raw implied probability inconsistent with odds")
        return self


class MetricSnapshot(SnapshotRecord):
    team_id: Identifier
    metrics: dict[str, NonNegative] = Field(min_length=1)


class LineupSnapshot(SnapshotRecord):
    team_id: Identifier
    player_ids: tuple[Identifier, ...] = Field(min_length=1)
    confirmed: bool


class InjurySnapshot(SnapshotRecord):
    team_id: Identifier
    # Empty means an explicitly sourced report of no injuries, not missing data.
    player_ids: tuple[Identifier, ...]


class OptaSnapshot(SnapshotRecord):
    product: Literal["POWER_RANKING", "XG", "XGA", "TEAM_STATS", "PLAYER_STATS", "PREDICTION"]
    payload: dict | None = None
    access_basis: Literal["PUBLIC_PAGE", "LICENSED_API"] | None = None

    @model_validator(mode="after")
    def authentic_external(self):
        if self.availability == Availability.AVAILABLE:
            if not self.payload or not self.access_basis:
                raise ValueError("available Opta requires sourced payload and lawful access basis")
        elif self.payload is not None:
            raise ValueError("unavailable Opta cannot contain fabricated payload")
        return self


class MatchResult(Contract):
    result_id: Identifier = Field(default_factory=lambda: str(uuid4()))
    match_id: Identifier
    home_goals: int = Field(ge=0)
    away_goals: int = Field(ge=0)
    completed_at: UTCTime
    source: Identifier
    retrieved_at: UTCTime
    as_of_time: UTCTime
    data_version: Identifier

    @model_validator(mode="after")
    def result_time(self):
        if self.as_of_time < self.completed_at:
            raise ValueError("result cannot be known before completion")
        return self
