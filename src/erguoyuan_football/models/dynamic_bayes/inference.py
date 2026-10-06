"""Sequential Laplace posterior updates for time-indexed Poisson states.

PyMC is not available for this Python 3.11 environment. The implementation is
therefore an explicit Bayesian approximation: each match applies a Newton/Laplace
update to the four local latent states, while the random-walk transition carries
posterior means and variances across calendar time. It is a real likelihood update,
not a deterministic recent-form feature or a probability fallback.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np

from erguoyuan_football.contracts.common import utc
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.dynamic_bayes.likelihood import (
    posterior_predictive_checks,
    score_matrix_from_lambdas,
)
from erguoyuan_football.models.dynamic_bayes.priors import LeaguePrior
from erguoyuan_football.models.dynamic_bayes.state_model import (
    DynamicStateStore,
    PreMatchTeamState,
    TimeIndex,
)
from erguoyuan_football.models.point_in_time import validate_history_point_in_time
from erguoyuan_football.models.training import TrainingDataset, TrainingMatch


@dataclass(frozen=True)
class DynamicDiagnostics:
    """Auditable health and posterior-predictive diagnostics."""

    health: str
    r_hat: float | None
    ess: float | None
    divergences: int | None
    sampling_time_seconds: float
    posterior_draws: int
    posterior_predictive: dict[str, float]
    parameter_extremes: dict[str, float]
    reason: str | None = None


@dataclass(frozen=True)
class DynamicPrediction:
    """Primitive posterior predictive values consumed by ModelPrediction."""

    values: dict[str, object]
    home_state: PreMatchTeamState
    away_state: PreMatchTeamState


class DynamicBayesianInference:
    """Fit and query a dynamic state-space model with no future-state access."""

    def __init__(self, config: ModelConfig, *, model_id: str, model_version: str) -> None:
        self.config = config
        self.model_id = model_id
        self.model_version = model_version
        self.prior = LeaguePrior.from_config(config)
        self.state_store = DynamicStateStore(
            model_id=model_id, model_version=model_version,
            league_attack_mean=self.prior.attack_mean, league_defence_mean=self.prior.defence_mean,
            league_attack_sd=self.prior.attack_sd, league_defence_sd=self.prior.defence_sd,
        )
        self.mu = 0.0
        self.home_advantage = config.dynamic_home_advantage
        self.trained_until: datetime | None = None
        self.diagnostics: DynamicDiagnostics | None = None
        self._training_lambdas_home: list[float] = []
        self._training_lambdas_away: list[float] = []
        self._training_home_goals: list[int] = []
        self._training_away_goals: list[int] = []
        self._last_states: dict[str, PreMatchTeamState] = {}

    def _time_index(self) -> TimeIndex:
        return TimeIndex(self.config.dynamic_time_index)

    def _pre_time(self, kickoff: datetime) -> datetime:
        """Use an instant before kickoff so a state is never post-match."""
        return utc(kickoff) - timedelta(microseconds=1)

    def _fit_global_rates(self, rows: tuple[TrainingMatch, ...]) -> None:
        home = np.asarray([row.home_goals for row in rows], dtype=float)
        away = np.asarray([row.away_goals for row in rows], dtype=float)
        if not len(rows) or (home.mean() + away.mean()) <= 0:
            raise ValueError("DYNAMIC_GLOBAL_RATE_UNAVAILABLE")
        self.mu = float(math.log(max((home.mean() + away.mean()) / 2.0, 1e-6)))
        non_neutral = tuple(row for row in rows if not row.neutral_venue)
        if non_neutral:
            non_neutral_home = np.asarray([row.home_goals for row in non_neutral], dtype=float)
            non_neutral_away = np.asarray([row.away_goals for row in non_neutral], dtype=float)
            home_advantage_evidence = math.log(
                (non_neutral_home.mean() + 0.1) / (non_neutral_away.mean() + 0.1)
            )
        else:
            home_advantage_evidence = 0.0
        self.home_advantage = float(np.clip(
            self.config.dynamic_home_advantage + home_advantage_evidence, -0.6, 0.6,
        ))

    def _effective_home_advantage(self, neutral_venue: bool) -> float:
        """Neutral venues have no home-side intercept contribution."""
        return 0.0 if neutral_venue else self.home_advantage

    def _transition_states(self, row: TrainingMatch) -> tuple[PreMatchTeamState, PreMatchTeamState]:
        cutoff = self._pre_time(row.kickoff_time)
        home = self.state_store.state_at(row.home_team_id, cutoff, self.config, season=row.season)
        away = self.state_store.state_at(row.away_team_id, cutoff, self.config, season=row.season)
        return home, away

    def _training_exposures(self, row: TrainingMatch) -> tuple[PreMatchTeamState, PreMatchTeamState]:
        """Read current result-driven state only once its evidence was available."""
        return (
            self.state_store.state_at(row.home_team_id, row.available_at, self.config, season=row.season),
            self.state_store.state_at(row.away_team_id, row.available_at, self.config, season=row.season),
        )

    def _local_update(self, home: PreMatchTeamState, away: PreMatchTeamState,
                      row: TrainingMatch) -> tuple[PreMatchTeamState, PreMatchTeamState]:
        """Perform a damped four-dimensional Laplace update for one result."""
        prior = np.asarray([home.attack_mean, home.defence_mean, away.attack_mean, away.defence_mean], dtype=float)
        variances = np.asarray([home.attack_sd**2, home.defence_sd**2,
                                away.attack_sd**2, away.defence_sd**2], dtype=float)
        design = np.asarray([[1.0, 0.0, 0.0, -1.0], [0.0, -1.0, 1.0, 0.0]], dtype=float)
        offset = np.asarray([self.mu + self._effective_home_advantage(row.neutral_venue), self.mu], dtype=float)
        state = prior.copy()
        for _ in range(12):
            rates = np.exp(np.clip(offset + design @ state, -8, 5))
            gradient = -(state - prior) / variances + design.T @ (np.asarray([row.home_goals, row.away_goals]) - rates)
            precision = np.diag(1.0 / variances) + design.T @ np.diag(rates) @ design
            step = np.linalg.solve(precision, gradient)
            candidate = state + np.clip(step, -1.0, 1.0)
            if np.max(np.abs(candidate - state)) < 1e-5:
                state = candidate
                break
            state = candidate
        posterior_precision = np.diag(1.0 / variances) + design.T @ np.diag(np.exp(np.clip(offset + design @ state, -8, 5))) @ design
        posterior_variances = np.maximum(np.diag(np.linalg.inv(posterior_precision)), 1e-8)
        learning_rate = self.config.dynamic_learning_rate
        means = prior + learning_rate * (state - prior)
        variances_out = (1 - learning_rate) * variances + learning_rate * posterior_variances
        return (
            home.model_copy(update={"as_of_time": row.available_at, "attack_mean": float(means[0]),
                                    "attack_sd": float(math.sqrt(variances_out[0])),
                                    "defence_mean": float(means[1]), "defence_sd": float(math.sqrt(variances_out[1])),
                                    "season": row.season}),
            away.model_copy(update={"as_of_time": row.available_at, "attack_mean": float(means[2]),
                                    "attack_sd": float(math.sqrt(variances_out[2])),
                                    "defence_mean": float(means[3]), "defence_sd": float(math.sqrt(variances_out[3])),
                                    "season": row.season}),
        )

    def _center_identifiability(self, states: tuple[PreMatchTeamState, ...]) -> None:
        """Center attack and defence independently and compensate the intercept."""
        if not states:
            return
        attack_mean = float(np.mean([state.attack_mean for state in states]))
        defence_mean = float(np.mean([state.defence_mean for state in states]))
        self.mu += attack_mean - defence_mean
        for state in states:
            centered = state.model_copy(update={
                "attack_mean": state.attack_mean - attack_mean,
                "defence_mean": state.defence_mean - defence_mean,
            })
            self.state_store.set_latest(centered)
            self._last_states[state.team_id] = centered

    def fit(self, data: TrainingDataset, trained_until: datetime) -> DynamicDiagnostics:
        """Run chronological state updates using only rows available by cutoff."""
        started = time.perf_counter()
        cutoff = utc(trained_until)
        rows = tuple(sorted(data.matches, key=lambda row: (row.kickoff_time, row.match_id)))
        if not rows:
            raise ValueError("DYNAMIC_NO_TRAINING_ROWS")
        # Validate the complete received history before global rates or state updates.
        validate_history_point_in_time(rows, cutoff)
        if any(row.available_at > cutoff for row in rows):
            raise ValueError("DYNAMIC_FUTURE_TRAINING_ROW_AVAILABLE_AFTER_CUTOFF")
        if not rows:
            raise ValueError("DYNAMIC_NO_AVAILABLE_TRAINING_ROWS")
        self._fit_global_rates(rows)
        observed_states: list[PreMatchTeamState] = []
        for row in rows:
            home, away = self._transition_states(row)
            self.state_store.record(home)
            self.state_store.record(away)
            home_advantage = self._effective_home_advantage(row.neutral_venue)
            self._training_lambdas_home.append(float(math.exp(self.mu + home_advantage + home.attack_mean - away.defence_mean)))
            self._training_lambdas_away.append(float(math.exp(self.mu + away.attack_mean - home.defence_mean)))
            self._training_home_goals.append(row.home_goals)
            self._training_away_goals.append(row.away_goals)
            updated_home, updated_away = self._local_update(home, away, row)
            self.state_store.update_after_result(updated_home, row.available_at)
            self.state_store.update_after_result(updated_away, row.available_at)
            self._last_states[updated_home.team_id] = updated_home
            self._last_states[updated_away.team_id] = updated_away
            observed_states.extend((updated_home, updated_away))
            self._center_identifiability(tuple(self._last_states.values()))
        self.trained_until = cutoff
        posterior_home = np.asarray(self._training_lambdas_home, dtype=float)
        posterior_away = np.asarray(self._training_lambdas_away, dtype=float)
        checks = posterior_predictive_checks(np.asarray(self._training_home_goals), np.asarray(self._training_away_goals),
                                             posterior_home, posterior_away)
        max_state = max((abs(item.attack_mean) for item in observed_states), default=0.0)
        max_defence = max((abs(item.defence_mean) for item in observed_states), default=0.0)
        if checks["mean_abs_discrepancy"] > 2.0 or max(max_state, max_defence) > 5:
            health = "FAILED"
        elif checks["mean_abs_discrepancy"] > 0.75 or max(max_state, max_defence) > 3:
            health = "WARNING"
        else:
            health = "GOOD"
        self.diagnostics = DynamicDiagnostics(
            health=health, r_hat=None, ess=None, divergences=None,
            sampling_time_seconds=time.perf_counter() - started,
            posterior_draws=self.config.dynamic_posterior_draws,
            posterior_predictive=checks,
            parameter_extremes={"max_abs_attack": max_state, "max_abs_defence": max_defence},
            reason=("NOT_MCMC_LAPLACE_APPROXIMATION" if health == "GOOD"
                    else f"{health}_POSTERIOR_DIAGNOSTIC;NOT_MCMC_LAPLACE_APPROXIMATION"),
        )
        return self.diagnostics

    def state_pair(self, match: Fixture, prediction_time: datetime) -> tuple[PreMatchTeamState, PreMatchTeamState]:
        """Return two states available at prediction time, never mutating history."""
        if self.trained_until is None or utc(prediction_time) < self.trained_until:
            raise ValueError("DYNAMIC_PREDICTION_BEFORE_TRAINING_CUTOFF")
        home = self.state_store.state_at(match.home_team_id, prediction_time, self.config, season=match.season)
        away = self.state_store.state_at(match.away_team_id, prediction_time, self.config, season=match.season)
        return home, away

    def posterior_prediction(self, match: Fixture, prediction_time: datetime) -> DynamicPrediction:
        """Generate a posterior-predictive score matrix from state draws."""
        home, away = self.state_pair(match, prediction_time)
        rng = np.random.default_rng(self.config.seed)
        draws = self.config.dynamic_posterior_draws
        home_attack = rng.normal(home.attack_mean, home.attack_sd, draws)
        away_defence = rng.normal(away.defence_mean, away.defence_sd, draws)
        away_attack = rng.normal(away.attack_mean, away.attack_sd, draws)
        home_defence = rng.normal(home.defence_mean, home.defence_sd, draws)
        effective_home_advantage = self._effective_home_advantage(bool(match.neutral_venue))
        lambda_home = np.exp(np.clip(self.mu + effective_home_advantage + home_attack - away_defence, -8, 5))
        lambda_away = np.exp(np.clip(self.mu + away_attack - home_defence, -8, 5))
        matrix = score_matrix_from_lambdas(lambda_home, lambda_away, self.config.max_goals)
        outcome = matrix.outcome()
        lambda_home_ci = tuple(float(value) for value in np.quantile(lambda_home, (0.05, 0.95)))
        lambda_away_ci = tuple(float(value) for value in np.quantile(lambda_away, (0.05, 0.95)))
        uncertainty = float(np.mean([
            np.std(lambda_home) / max(np.mean(lambda_home), 1e-6),
            np.std(lambda_away) / max(np.mean(lambda_away), 1e-6),
        ]))
        values: dict[str, object] = {
            "p_home": outcome.p_home, "p_draw": outcome.p_draw, "p_away": outcome.p_away,
            "lambda_home": float(np.mean(lambda_home)), "lambda_away": float(np.mean(lambda_away)),
            "expected_home_goals": float(np.mean(lambda_home)), "expected_away_goals": float(np.mean(lambda_away)),
            "score_matrix": matrix.values,
            "metadata": {
                "home_attack_mean": home.attack_mean, "home_attack_sd": home.attack_sd,
                "home_defence_mean": home.defence_mean, "home_defence_sd": home.defence_sd,
                "away_attack_mean": away.attack_mean, "away_attack_sd": away.attack_sd,
                "away_defence_mean": away.defence_mean, "away_defence_sd": away.defence_sd,
                "home_advantage": effective_home_advantage,
                "global_home_advantage": self.home_advantage, "neutral_venue": bool(match.neutral_venue),
                "mu": self.mu,
                "state_process": self.config.dynamic_state_process,
                "state_time_index": self.config.dynamic_time_index,
                "posterior_draws": draws, "posterior_method": "SEQUENTIAL_LAPLACE_POISSON",
                "score_matrix_max_goals": matrix.max_goals,
                "score_matrix_retained_mass": matrix.retained_mass,
                "score_matrix_tail_mass": matrix.tail_mass,
                "score_matrix_tail_policy": matrix.tail_policy,
                "lambda_home_ci": lambda_home_ci, "lambda_away_ci": lambda_away_ci,
                "lambda_home_p05": lambda_home_ci[0], "lambda_home_p95": lambda_home_ci[1],
                "lambda_away_p05": lambda_away_ci[0], "lambda_away_p95": lambda_away_ci[1],
                "uncertainty_score": uncertainty,
                "state_change_signal": {
                    "attack_delta": home.attack_mean - self.prior.attack_mean,
                    "defence_delta": home.defence_mean - self.prior.defence_mean,
                    "attack_z_change": (home.attack_mean - self.prior.attack_mean) / math.sqrt(
                        home.attack_sd**2 + self.prior.attack_sd**2
                    ),
                    "defence_z_change": (home.defence_mean - self.prior.defence_mean) / math.sqrt(
                        home.defence_sd**2 + self.prior.defence_sd**2
                    ),
                },
                "diagnostics": asdict(self.diagnostics) if self.diagnostics else {},
            },
        }
        return DynamicPrediction(values=values, home_state=home, away_state=away)
