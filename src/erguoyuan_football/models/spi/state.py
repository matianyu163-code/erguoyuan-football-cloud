"""Versioned pre-match SPI team state and explicit performance estimators."""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum

from pydantic import Field

from erguoyuan_football.contracts.common import Contract, UTCTime, utc


class SPIFeatureMode(StrEnum):
    GOALS_ONLY = "GOALS_ONLY"
    GOALS_XG = "GOALS_XG"
    FULL_AVAILABLE = "FULL_AVAILABLE"


class SPIState(Contract):
    """Offense and goals-conceded defense; higher defense means more conceded goals."""

    team_id: str
    as_of_time: UTCTime
    competition_id: str
    offensive_rating: float
    defensive_rating: float
    overall_rating: float
    league_strength: float
    uncertainty: float = Field(ge=0)
    model_version: str
    feature_mode: SPIFeatureMode
    match_id: str
    phase: str = "PRE_MATCH"
    season: str


class SPIMatchPerformanceEstimator:
    """Convert real goals and the prior opponent state into log-rate performance."""

    def __init__(self, *, baseline_goals: float, smoothing: float, learning_rate: float) -> None:
        if min(baseline_goals, smoothing, learning_rate) <= 0:
            raise ValueError("SPI estimator parameters must be positive")
        self.baseline_goals = baseline_goals
        self.smoothing = smoothing
        self.learning_rate = learning_rate

    def estimate(self, scored: float, conceded: float, opponent_offense: float,
                 opponent_defense: float) -> tuple[float, float]:
        """Return performance updates; no xG or unobserved input is synthesized."""
        attack = math.log((scored + self.smoothing) / self.baseline_goals) - opponent_defense
        defense = math.log((conceded + self.smoothing) / self.baseline_goals) - opponent_offense
        return attack, defense


class SPIStateStore:
    """Chronological state journal exposing PRE_MATCH and POST_MATCH separately."""

    def __init__(self) -> None:
        self._journal: list[SPIState] = []
        self._current: dict[str, SPIState] = {}

    @property
    def journal(self) -> tuple[SPIState, ...]:
        return tuple(self._journal)

    def current(self, team_id: str) -> SPIState:
        return self._current[team_id]

    def before(self, team_id: str, as_of_time: datetime) -> SPIState | None:
        cutoff = utc(as_of_time)
        rows = [state for state in self._journal
                if state.team_id == team_id and state.phase == "PRE_MATCH" and state.as_of_time <= cutoff]
        return rows[-1] if rows else None

    def append(self, pre_state: SPIState, post_state: SPIState) -> None:
        if pre_state.phase != "PRE_MATCH" or post_state.phase != "POST_MATCH":
            raise ValueError("SPI state phase transition must be PRE_MATCH to POST_MATCH")
        self._journal.extend((pre_state, post_state))
        self._current[post_state.team_id] = post_state.model_copy(update={"phase": "PRE_MATCH"})
