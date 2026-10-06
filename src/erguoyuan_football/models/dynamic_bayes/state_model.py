"""Time-indexed latent team states and uncertainty propagation."""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum

from pydantic import Field

from erguoyuan_football.contracts.common import Contract, Identifier, UTCTime, utc
from erguoyuan_football.models.config import ModelConfig


class StateProcess(StrEnum):
    """Supported latent-state transition processes."""

    RANDOM_WALK = "RANDOM_WALK"
    AR1 = "AR1"


class TimeIndex(StrEnum):
    """Supported calendar indexing conventions."""

    MATCH_EVENT_TIME = "MATCH_EVENT_TIME"
    WEEKLY = "WEEKLY"


class PreMatchTeamState(Contract):
    """A state available immediately before a match or prediction timestamp."""

    team_id: Identifier
    as_of_time: UTCTime
    attack_mean: float
    attack_sd: float = Field(gt=0, allow_inf_nan=False)
    defence_mean: float
    defence_sd: float = Field(gt=0, allow_inf_nan=False)
    model_id: str = "DYNAMIC_BAYESIAN_POISSON_V1"
    model_version: str = "1.0.0"
    status: str = "PRE_MATCH"
    season: str | None = None


class DynamicStateStore:
    """Append-only state history; a query never returns a post-match state."""

    def __init__(self, *, model_id: str, model_version: str, league_attack_mean: float,
                 league_defence_mean: float, league_attack_sd: float, league_defence_sd: float) -> None:
        self.model_id = model_id
        self.model_version = model_version
        self.league_attack_mean = league_attack_mean
        self.league_defence_mean = league_defence_mean
        self.league_attack_sd = league_attack_sd
        self.league_defence_sd = league_defence_sd
        self._history: list[PreMatchTeamState] = []
        self._latest: dict[str, PreMatchTeamState] = {}
        self._latest_available_at: dict[str, datetime] = {}

    @property
    def history(self) -> tuple[PreMatchTeamState, ...]:
        """Return immutable chronological history, including repeated team states."""
        return tuple(self._history)

    def prior(self, team_id: str, as_of_time: datetime, *, season: str | None = None) -> PreMatchTeamState:
        """Create an explicitly uncertain league-level prior for an unseen team."""
        return PreMatchTeamState(
            team_id=team_id, as_of_time=utc(as_of_time),
            attack_mean=self.league_attack_mean, attack_sd=self.league_attack_sd,
            defence_mean=self.league_defence_mean, defence_sd=self.league_defence_sd,
            model_id=self.model_id, model_version=self.model_version, season=season,
        )

    def latest_before(self, team_id: str, as_of_time: datetime, *, season: str | None = None) -> PreMatchTeamState:
        """Return the latest state at or before a timestamp, or the league prior."""
        cutoff = utc(as_of_time)
        candidates = [item for item in self._history if item.team_id == team_id and item.as_of_time <= cutoff]
        current = self._latest.get(team_id)
        available_at = self._latest_available_at.get(team_id)
        if current is not None and available_at is not None and available_at <= cutoff:
            candidates.append(current)
        if not candidates:
            return self.prior(team_id, cutoff, season=season)
        return max(candidates, key=lambda item: item.as_of_time)

    def record(self, state: PreMatchTeamState) -> None:
        """Append a state and update the in-memory latest index without overwriting history."""
        if state.status != "PRE_MATCH":
            raise ValueError("only PRE_MATCH states can enter DynamicStateStore")
        if self._history and state.as_of_time < self._history[-1].as_of_time:
            raise ValueError("state history must be chronological")
        self._history.append(state)
        self._latest[state.team_id] = state
        self._latest_available_at[state.team_id] = state.as_of_time

    def set_latest(self, state: PreMatchTeamState) -> None:
        """Store an updated posterior state without exposing it as a pre-match observation."""
        if state.status != "PRE_MATCH":
            raise ValueError("only PRE_MATCH states can enter DynamicStateStore")
        previous = self._latest.get(state.team_id)
        if previous is not None and state.as_of_time < previous.as_of_time:
            raise ValueError("latest state cannot move backwards")
        self._latest[state.team_id] = state
        self._latest_available_at[state.team_id] = state.as_of_time

    def update_after_result(self, state: PreMatchTeamState, available_at: datetime) -> None:
        """Store posterior state with its actual result availability timestamp."""
        at = utc(available_at)
        if at < state.as_of_time:
            raise ValueError("posterior state cannot be available before its state time")
        self.set_latest(state)
        self._latest_available_at[state.team_id] = at

    def transition(self, state: PreMatchTeamState, target_time: datetime, config: ModelConfig,
                   *, season: str | None = None) -> PreMatchTeamState:
        """Propagate mean and uncertainty over elapsed calendar time."""
        target = utc(target_time)
        if target < state.as_of_time:
            raise ValueError("state transition cannot move backwards")
        days = max(0.0, (target - state.as_of_time).total_seconds() / 86400)
        index = TimeIndex(config.dynamic_time_index)
        elapsed_units = days / 7.0 if index == TimeIndex.WEEKLY else days
        process = StateProcess(config.dynamic_state_process)
        if process == StateProcess.RANDOM_WALK:
            attack_mean, defence_mean = state.attack_mean, state.defence_mean
            attack_var = state.attack_sd**2 + config.dynamic_sigma_attack**2 * elapsed_units
            defence_var = state.defence_sd**2 + config.dynamic_sigma_defence**2 * elapsed_units
        else:
            attack_decay = config.dynamic_phi_attack ** elapsed_units
            defence_decay = config.dynamic_phi_defence ** elapsed_units
            attack_mean = self.league_attack_mean + attack_decay * (state.attack_mean - self.league_attack_mean)
            defence_mean = self.league_defence_mean + defence_decay * (state.defence_mean - self.league_defence_mean)
            attack_var = attack_decay**2 * state.attack_sd**2 + config.dynamic_sigma_attack**2 * elapsed_units
            defence_var = defence_decay**2 * state.defence_sd**2 + config.dynamic_sigma_defence**2 * elapsed_units
        # The explicit growth term keeps an idle team's uncertainty from
        # remaining falsely constant when the configured process is very stable.
        growth_variance = config.dynamic_uncertainty_growth_per_day * days
        return PreMatchTeamState(
            team_id=state.team_id, as_of_time=target,
            attack_mean=attack_mean, attack_sd=math.sqrt(max(attack_var + growth_variance, 1e-12)),
            defence_mean=defence_mean, defence_sd=math.sqrt(max(defence_var + growth_variance, 1e-12)),
            model_id=self.model_id, model_version=self.model_version, season=season or state.season,
        )

    def apply_season_transition(self, state: PreMatchTeamState, target_time: datetime, season: str,
                                config: ModelConfig) -> PreMatchTeamState:
        """Shrink a previous-season state toward the league prior at season start."""
        if state.season is None or state.season == season:
            return state
        transitioned = self.transition(state, target_time, config, season=season)
        weight = config.dynamic_season_transition_weight
        return transitioned.model_copy(update={
            "attack_mean": weight * transitioned.attack_mean + (1 - weight) * self.league_attack_mean,
            "defence_mean": weight * transitioned.defence_mean + (1 - weight) * self.league_defence_mean,
            "attack_sd": math.sqrt(weight * transitioned.attack_sd**2 + (1 - weight) * self.league_attack_sd**2),
            "defence_sd": math.sqrt(weight * transitioned.defence_sd**2 + (1 - weight) * self.league_defence_sd**2),
            "season": season,
        })

    def state_at(self, team_id: str, target_time: datetime, config: ModelConfig,
                 *, season: str | None = None) -> PreMatchTeamState:
        """Query and transition one team's state without mutating historical rows."""
        base = self.latest_before(team_id, target_time, season=season)
        if season is not None and base.season not in (None, season):
            return self.apply_season_transition(base, target_time, season, config)
        return self.transition(base, target_time, config, season=season)

    def snapshot(self) -> tuple[PreMatchTeamState, ...]:
        """Return all persisted states for artifact serialization."""
        return self.history
