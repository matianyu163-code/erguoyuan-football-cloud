"""Adapters around penaltyblog's public models; no third-party objects leave this module."""

from __future__ import annotations

from typing import Any

import numpy as np

from erguoyuan_football.models.score_matrix import ScoreMatrix
from erguoyuan_football.models.training import TrainingDataset


class PenaltyblogAdapter:
    """Fit one independent penaltyblog goal model and convert its grid to primitives."""

    def __init__(self, model_class: type[Any]) -> None:
        self.model_class = model_class
        self.backend: Any | None = None

    def _records(self, data: TrainingDataset, config):
        from penaltyblog.models import dixon_coles_weights
        rows = sorted(data.matches, key=lambda row: (row.kickoff_time, row.match_id))
        # penaltyblog computes day deltas through datetime.timedelta.days; preserve aware datetimes.
        dates = [row.kickoff_time for row in rows]
        weights = dixon_coles_weights(dates, xi=config.time_decay) if config.time_decay else None
        return rows, weights

    def fit(self, data: TrainingDataset, config) -> None:
        """Fit only on the supplied, already point-in-time validated rows."""
        rows, weights = self._records(data, config)
        self.backend = self.model_class(
            goals_home=[row.home_goals for row in rows], goals_away=[row.away_goals for row in rows],
            teams_home=[row.home_team_id for row in rows], teams_away=[row.away_team_id for row in rows],
            weights=weights, neutral_venue=[row.neutral_venue for row in rows])
        self.backend.fit(minimizer_options={"maxiter": config.max_iterations})

    def fit_bayesian(self, data: TrainingDataset, config, *, n_cores: int = 1) -> None:
        """Fit the public hierarchical Bayesian API with configured independent chains."""
        rows, weights = self._records(data, config)
        self.backend = self.model_class(
            [row.home_goals for row in rows], [row.away_goals for row in rows],
            [row.home_team_id for row in rows], [row.away_team_id for row in rows], weights,
            neutral_venue=[row.neutral_venue for row in rows])
        self.backend.fit(n_samples=config.draws, burn=config.tune, n_chains=config.chains,
                         thin=config.thin, n_cores=n_cores)

    def fit_bayesian_map_internal(self, data: TrainingDataset, config
                                  ) -> tuple[tuple[float, ...] | None, str | None]:
        """Optimize penaltyblog's same hierarchical log posterior for internal checks only."""
        from penaltyblog.models.hierarchical_bayesian_goal_model import (
            hierarchical_log_prob_wrapper,
        )
        from scipy.optimize import minimize

        rows, weights = self._records(data, config)
        backend = self.model_class(
            [row.home_goals for row in rows], [row.away_goals for row in rows],
            [row.home_team_id for row in rows], [row.away_team_id for row in rows],
            weights, neutral_venue=[row.neutral_venue for row in rows])
        initial = np.concatenate([backend._get_initial_params(), [-0.1]])
        starting_points = backend._generate_hierarchical_starts(1, initial)
        data_dict = {
            "home_idx": backend.home_idx,
            "away_idx": backend.away_idx,
            "goals_home": backend.goals_home,
            "goals_away": backend.goals_away,
            "weights": backend.weights,
            "neutral_venue": backend.neutral_venue,
            "n_teams": backend.n_teams,
        }

        def objective(parameters: np.ndarray) -> float:
            log_probability = float(hierarchical_log_prob_wrapper(
                np.ascontiguousarray(parameters, dtype=np.float64), data_dict))
            return -log_probability if np.isfinite(log_probability) else 1e100

        result = minimize(objective, starting_points[0], method="L-BFGS-B",
                          options={"maxiter": config.max_iterations})
        parameters = np.asarray(result.x, dtype=np.float64)
        if not result.success or not np.all(np.isfinite(parameters)):
            return None, f"MAP_OPTIMIZATION_FAILED:{result.message}"
        self.backend = backend
        return tuple(float(value) for value in parameters), None

    def predict(self, home_team: str, away_team: str, *, max_goals: int, neutral_venue: bool) -> dict[str, Any]:
        """Return a normalized score matrix with explicit tail accounting."""
        if self.backend is None:
            raise RuntimeError("adapter is not fitted")
        original_rho = None
        effective_rho = None
        params = self.backend.get_params()
        if "rho" in params and hasattr(self.backend, "_params"):
            original_rho = float(params["rho"])
            effective_rho = self._safe_rho(home_team, away_team, neutral_venue, original_rho)
            if effective_rho != original_rho:
                # penaltyblog exposes rho as a fitted parameter but permits
                # combinations that make a low-score correction negative.
                # Bound only this numerical validity issue for this prediction;
                # the fitted parameter remains auditable in backend_parameters.
                self.backend._params[-1] = effective_rho
        try:
            # penaltyblog interprets ``max_goals`` as the matrix width, while
            # our contract names the inclusive highest score (0..max_goals).
            grid = self.backend.predict(home_team, away_team, max_goals=max_goals + 1, normalize=False,
                                        neutral_venue=neutral_venue)
        finally:
            if original_rho is not None:
                self.backend._params[-1] = original_rho
        score = ScoreMatrix.from_raw(np.asarray(grid.grid, dtype=np.float64))
        outcome = score.outcome()
        metadata = {"score_matrix": score.model_dump(mode="json"), "backend_parameters": _jsonable(params)}
        if original_rho is not None:
            metadata.update({"rho_fitted": original_rho, "rho_for_prediction": effective_rho,
                             "rho_adjusted_for_validity": effective_rho != original_rho})
        values: dict[str, Any] = {
            "p_home": outcome.p_home, "p_draw": outcome.p_draw, "p_away": outcome.p_away,
            "lambda_home": float(grid.home_goal_expectation), "lambda_away": float(grid.away_goal_expectation),
            "expected_home_goals": float(grid.home_goal_expectation), "expected_away_goals": float(grid.away_goal_expectation),
            "score_matrix": score.values, "metadata": metadata,
        }
        if "lambda3" in params:
            values["correlation_parameter"] = float(params["lambda3"])
            metadata["lambda3"] = float(params["lambda3"])
        return values

    def _safe_rho(self, home_team: str, away_team: str, neutral_venue: bool, rho: float) -> float:
        """Return the nearest rho with non-negative Dixon-Coles low-score terms."""
        backend = self.backend
        if backend is None or not hasattr(backend, "team_to_idx") or not hasattr(backend, "_params"):
            return rho
        home_index = backend.team_to_idx[home_team]
        away_index = backend.team_to_idx[away_team]
        n_teams = int(backend.n_teams)
        params = np.asarray(backend._params, dtype=float)
        lambda_home = float(np.exp(params[home_index] + params[n_teams + away_index]
                                   + (0.0 if neutral_venue else params[-2])))
        lambda_away = float(np.exp(params[away_index] + params[n_teams + home_index]
                                   + (0.0 if neutral_venue else params[-2])))
        lower = -0.999999 / max(lambda_home, lambda_away)
        upper = 0.999999 / max(lambda_home * lambda_away, 1e-12)
        return float(min(max(rho, lower), upper))

    def posterior_lambda_intervals(self, home_team: str, away_team: str, *, neutral_venue: bool,
                                   quantiles: tuple[float, float] = (0.025, 0.975)) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Return posterior lambda intervals when the Bayesian backend exposes draws."""
        backend = self.backend
        if backend is None or not hasattr(backend, "trace_dict") or not hasattr(backend, "team_to_idx"):
            return None
        home_index = backend.team_to_idx[home_team]
        away_index = backend.team_to_idx[away_team]
        trace_dict = backend.trace_dict
        try:
            home_attack = np.asarray(trace_dict[f"attack_team_{home_index}"], dtype=float)
            away_defence = np.asarray(trace_dict[f"defense_team_{away_index}"], dtype=float)
            away_attack = np.asarray(trace_dict[f"attack_team_{away_index}"], dtype=float)
            home_defence = np.asarray(trace_dict[f"defense_team_{home_index}"], dtype=float)
            params = np.asarray(backend._params, dtype=float)
        except (KeyError, AttributeError, IndexError):
            return None
        home_advantage = 0.0 if neutral_venue else float(params[-2])
        home_samples = np.exp(home_attack + away_defence + home_advantage)
        away_samples = np.exp(away_attack + home_defence)
        home_values = tuple(float(value) for value in np.quantile(home_samples, quantiles))
        away_values = tuple(float(value) for value in np.quantile(away_samples, quantiles))
        home_interval = (home_values[0], home_values[1])
        away_interval = (away_values[0], away_values[1])
        return home_interval, away_interval


def _jsonable(value: Any) -> Any:
    """Convert numpy arrays/scalars in backend parameter dictionaries to JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
