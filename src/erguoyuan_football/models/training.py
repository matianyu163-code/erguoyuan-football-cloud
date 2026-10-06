"""Versioned training rows and fail-closed, point-in-time validation."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import (
    Contract,
    Identifier,
    NonNegative,
    UTCTime,
    utc,
)
from erguoyuan_football.data.store import Store
from erguoyuan_football.markets.schemas import MarketGoalFeatures


class TrainingDataError(ValueError):
    """A material training-data or temporal-integrity error."""


class InsufficientData(ValueError):
    """No defensible model can be fitted for the requested scope."""


class TrainingXGObservation(Contract):
    """A dated xG observation with provider, model and content provenance."""

    match_id: Identifier
    xg_home: NonNegative
    xg_away: NonNegative
    xg_source_id: Identifier
    provider: Identifier
    model_name: Identifier
    model_version: Identifier
    retrieved_at: UTCTime
    as_of_time: UTCTime
    data_hash: Identifier

    @model_validator(mode="after")
    def authentic_provenance(self) -> TrainingXGObservation:
        values = (self.xg_source_id, self.provider, self.model_name, self.model_version, self.data_hash)
        if any(value.strip().upper() in {"UNKNOWN", "UNAVAILABLE", "NONE", "NULL"} for value in values):
            raise ValueError("xG provenance fields must identify a real source, provider, model and content hash")
        return self


class TeamHierarchyMembership(Contract):
    """PIT-versioned team placement across league, country and continent."""

    team_id: Identifier
    league_id: Identifier
    country_id: Identifier
    continent_id: Identifier
    valid_from: UTCTime
    valid_to: UTCTime | None = None
    source: Identifier
    retrieved_at: UTCTime
    as_of_time: UTCTime
    data_version: Identifier

    @model_validator(mode="after")
    def interval(self) -> TeamHierarchyMembership:
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("team hierarchy valid_to must follow valid_from")
        return self


class TrainingMatch(Contract):
    """One completed match, with result availability rather than kickoff alone."""

    match_id: Identifier
    competition_id: Identifier
    season: Identifier
    kickoff_time: UTCTime
    home_team_id: Identifier
    away_team_id: Identifier
    home_goals: int = Field(ge=0, strict=True)
    away_goals: int = Field(ge=0, strict=True)
    neutral_venue: bool
    source: Identifier
    completed_at: UTCTime
    as_of_time: UTCTime
    retrieved_at: UTCTime
    data_version: Identifier

    @property
    def available_at(self) -> datetime:
        """Earliest time the recorded result is usable by this system."""
        return max(self.completed_at, self.as_of_time, self.retrieved_at)

    @model_validator(mode="after")
    def integrity(self) -> TrainingMatch:
        """Reject impossible team identities and result publication times."""
        if self.home_team_id == self.away_team_id:
            raise ValueError("identical home/away team IDs")
        if self.completed_at <= self.kickoff_time or self.as_of_time < self.completed_at:
            raise ValueError("invalid match/result chronology")
        return self


class TrainingDataset(Contract):
    """Explicitly labelled data; synthetic fixtures can never silently become live data."""

    matches: tuple[TrainingMatch, ...]
    known_team_ids: frozenset[str]
    competition_hierarchy: dict[str, dict[str, str]] = Field(default_factory=dict)
    team_hierarchy: dict[str, dict[str, str]] = Field(default_factory=dict)
    team_hierarchy_timeline: tuple[TeamHierarchyMembership, ...] = ()
    xg_observations: tuple[TrainingXGObservation, ...] = ()
    market_goal_features: tuple[MarketGoalFeatures, ...] = ()
    dataset_kind: Literal["REAL", "SYNTHETIC_TEST"] = "REAL"
    temporal_mode: Literal["EXACT_UTC", "DATE_SAFE_BATCH"] = "EXACT_UTC"
    assumptions: tuple[str, ...] = ()

    @property
    def data_hash(self) -> str:
        """Content hash independent of incoming row order."""
        ordered = sorted(self.matches, key=lambda row: (row.kickoff_time, row.match_id))
        content = "\n".join(row.model_dump_json() for row in ordered)
        content += self.dataset_kind + self.temporal_mode + repr(self.assumptions)
        content += "|".join(sorted(self.known_team_ids))
        content += str(sorted((key, sorted(value.items())) for key, value in self.competition_hierarchy.items()))
        content += str(sorted((key, sorted(value.items())) for key, value in self.team_hierarchy.items()))
        content += str(tuple(item.model_dump(mode="json") for item in sorted(
            self.market_goal_features,
            key=lambda item: (item.match_id, item.prediction_horizon.value, item.prediction_time),
        )))
        content += "\n".join(item.model_dump_json() for item in self.team_hierarchy_timeline)
        content += "\n".join(item.model_dump_json() for item in self.xg_observations)
        return hashlib.sha256(content.encode()).hexdigest()

    def window(self, trained_until: datetime, days: int | None) -> TrainingDataset:
        """Apply a configured lookback only after validating the original cutoff."""
        rows = self.matches if days is None else tuple(
            row for row in self.matches if row.kickoff_time >= trained_until - timedelta(days=days)
        )
        return TrainingDataset(matches=rows, known_team_ids=self.known_team_ids,
                               competition_hierarchy=self.competition_hierarchy,
                               team_hierarchy=self.team_hierarchy,
                               team_hierarchy_timeline=self.team_hierarchy_timeline,
                               xg_observations=tuple(xg for xg in self.xg_observations
                                   if xg.match_id in {row.match_id for row in rows}),
                               market_goal_features=tuple(item for item in self.market_goal_features
                                   if item.match_id in {row.match_id for row in rows}
                                   and item.prediction_time <= trained_until
                                   and item.availability.value == "AVAILABLE"
                                   and item.as_of_time is not None and item.retrieved_at is not None
                                   and item.as_of_time <= trained_until and item.retrieved_at <= trained_until),
                               dataset_kind=self.dataset_kind, temporal_mode=self.temporal_mode,
                               assumptions=self.assumptions)


class TrainingDatasetValidator:
    """Reject duplicates, unknown IDs, extreme scores and all future availability."""

    def validate(self, data: TrainingDataset, trained_until: datetime, *, max_score: int = 30) -> None:
        """Raise rather than silently discard any material invalid row."""
        at = utc(trained_until)
        seen: set[str] = set()
        for row in data.matches:
            if row.match_id in seen:
                raise TrainingDataError(f"duplicate match: {row.match_id}")
            seen.add(row.match_id)
            if not {row.home_team_id, row.away_team_id} <= data.known_team_ids:
                raise TrainingDataError(f"unknown team ID: {row.match_id}")
            if max(row.home_goals, row.away_goals) > max_score:
                raise TrainingDataError(f"abnormal score: {row.match_id}")
            if data.temporal_mode == "DATE_SAFE_BATCH":
                if row.kickoff_time.date() >= at.date() or row.completed_at.date() >= at.date():
                    raise TrainingDataError(f"same-date or future result: {row.match_id}")
            elif row.kickoff_time >= at or row.completed_at > at or row.available_at > at:
                raise TrainingDataError(f"future or unavailable result: {row.match_id}")
            if data.dataset_kind == "REAL" and "SYNTHETIC" in row.source.upper():
                raise TrainingDataError("synthetic input cannot be labelled REAL")
        row_by_id = {row.match_id: row for row in data.matches}
        if data.temporal_mode == "DATE_SAFE_BATCH" and (data.xg_observations or data.market_goal_features
                or data.team_hierarchy_timeline):
            raise TrainingDataError("DATE_SAFE_BATCH_FORBIDS_SNAPSHOT_AND_MARKET_FEATURES")
        seen_xg: set[str] = set()
        for item in data.xg_observations:
            if item.match_id not in row_by_id or item.match_id in seen_xg:
                raise TrainingDataError(f"orphan or duplicate xG observation: {item.match_id}")
            seen_xg.add(item.match_id)
            if item.as_of_time > at or item.retrieved_at > at or row_by_id[item.match_id].kickoff_time >= at:
                raise TrainingDataError(f"future or unavailable xG observation: {item.match_id}")
        seen_market: set[tuple[str, str]] = set()
        for feature in data.market_goal_features:
            if feature.availability.value != "AVAILABLE":
                continue
            market_key = (feature.match_id, feature.prediction_horizon.value)
            if feature.match_id not in row_by_id or market_key in seen_market:
                raise TrainingDataError(f"orphan or duplicate market goal feature: {feature.match_id}")
            seen_market.add(market_key)
            row = row_by_id[feature.match_id]
            if (feature.prediction_time >= row.kickoff_time or feature.as_of_time is None
                    or feature.retrieved_at is None
                    or feature.prediction_time > at
                    or max(feature.as_of_time, feature.retrieved_at) > feature.prediction_time
                    or feature.as_of_time > at or feature.retrieved_at > at):
                raise TrainingDataError(f"future or unavailable market feature: {feature.match_id}")
        for membership in data.team_hierarchy_timeline:
            if membership.team_id not in data.known_team_ids:
                raise TrainingDataError(f"unknown hierarchy team: {membership.team_id}")
            if membership.as_of_time > at or membership.retrieved_at > at:
                raise TrainingDataError(f"future hierarchy membership: {membership.team_id}")


def dataset_from_store(store: Store, competition_id: str, trained_until: datetime) -> TrainingDataset:
    """Use complete competition history at the cutoff, not just the two target teams."""
    rows: list[TrainingMatch] = []
    for result in store.results_at(trained_until):
        fixture = store.fixture_at(result.match_id, trained_until)
        if fixture.competition_id != competition_id:
            continue
        if fixture.season is None or fixture.neutral_venue is None:
            raise InsufficientData("MISSING_SEASON_OR_NEUTRAL_VENUE")
        rows.append(TrainingMatch(
            match_id=fixture.match_id, competition_id=competition_id, season=fixture.season,
            kickoff_time=fixture.kickoff_time, home_team_id=fixture.home_team_id,
            away_team_id=fixture.away_team_id, home_goals=result.home_goals, away_goals=result.away_goals,
            neutral_venue=fixture.neutral_venue, source=result.source, completed_at=result.completed_at,
            as_of_time=max(result.as_of_time, fixture.as_of_time),
            retrieved_at=max(result.retrieved_at, fixture.retrieved_at),
            data_version=f"{fixture.data_version}:{result.data_version}",
        ))
    teams = frozenset(str(row[0]) for row in store.connection.execute("SELECT team_id FROM teams").fetchall())
    kind: Literal["REAL", "SYNTHETIC_TEST"] = "SYNTHETIC_TEST" if any("SYNTHETIC" in r.source for r in rows) else "REAL"
    market_features = tuple(store.market_goal_features_at(trained_until,
                                  match_ids=tuple(row.match_id for row in rows)))
    return TrainingDataset(matches=tuple(rows), known_team_ids=teams,
                           market_goal_features=market_features, dataset_kind=kind)
