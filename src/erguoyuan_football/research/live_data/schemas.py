"""Typed source-backed fixture and historical form records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class VerifiedFixture:
    """One source-verified fixture, with explicit source confidence and tier."""

    provider_match_id: str
    home_team_id: str
    away_team_id: str
    competition_id: str
    kickoff_at: datetime
    venue: str | None
    source_url: str
    provider_id: str
    verified_at: datetime
    fixture_confidence: str
    source_tier: int
    neutral_venue: bool | None = None


@dataclass(frozen=True)
class RecentFormRecord:
    """Counts computed from actual completed provider results before cutoff."""

    team_id: str
    matches: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    window: str
    observed_at: datetime
    derived_from: tuple[str, ...]


@dataclass(frozen=True)
class TeamMetric:
    """A numeric locally derived metric with explicit evidence ancestry."""

    team_id: str
    metric: str
    value: float
    window: str
    competition: str
    season: str
    observed_at: datetime
    provider_id: str
    source_url: str
    data_origin: str
    derived_from: tuple[str, ...]
