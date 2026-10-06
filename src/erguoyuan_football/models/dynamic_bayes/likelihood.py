"""Poisson likelihood and posterior-predictive score calculations."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from erguoyuan_football.models.score_matrix import ScoreMatrix


def poisson_log_likelihood(home_goals: int, away_goals: int, lambda_home: float,
                           lambda_away: float) -> float:
    """Return the independent two-score Poisson log likelihood."""
    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError("Poisson rates must be positive")
    return float(poisson.logpmf(home_goals, lambda_home) + poisson.logpmf(away_goals, lambda_away))


def score_matrix_from_lambdas(lambda_home: np.ndarray, lambda_away: np.ndarray,
                              max_goals: int) -> ScoreMatrix:
    """Average exact-score matrices over posterior lambda draws."""
    if lambda_home.shape != lambda_away.shape or lambda_home.ndim != 1 or not len(lambda_home):
        raise ValueError("posterior lambda draws must be matching non-empty vectors")
    goals = np.arange(max_goals + 1)
    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for home_rate, away_rate in zip(lambda_home, lambda_away, strict=True):
        matrix += np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))
    return ScoreMatrix.from_raw(matrix / len(lambda_home))


def posterior_predictive_checks(home_goals: np.ndarray, away_goals: np.ndarray,
                               lambda_home: np.ndarray, lambda_away: np.ndarray) -> dict[str, float]:
    """Compare observed moments with posterior-predictive Poisson moments."""
    checks = {
        "observed_mean_goals": float(np.mean(home_goals + away_goals)),
        "posterior_mean_goals": float(np.mean(lambda_home + lambda_away)),
        "observed_home_mean": float(np.mean(home_goals)),
        "posterior_home_mean": float(np.mean(lambda_home)),
        "observed_away_mean": float(np.mean(away_goals)),
        "posterior_away_mean": float(np.mean(lambda_away)),
        "observed_zero_goal_rate": float(np.mean((home_goals + away_goals) == 0)),
        "posterior_zero_goal_rate": float(np.mean(np.exp(-(lambda_home + lambda_away)))),
        "observed_high_score_rate": float(np.mean((home_goals + away_goals) >= 6)),
        "posterior_high_score_rate": float(np.mean(1 - poisson.cdf(5, lambda_home + lambda_away))),
    }
    checks["mean_abs_discrepancy"] = float(np.mean([
        abs(checks["observed_mean_goals"] - checks["posterior_mean_goals"]),
        abs(checks["observed_home_mean"] - checks["posterior_home_mean"]),
        abs(checks["observed_away_mean"] - checks["posterior_away_mean"]),
        abs(checks["observed_zero_goal_rate"] - checks["posterior_zero_goal_rate"]),
        abs(checks["observed_high_score_rate"] - checks["posterior_high_score_rate"]),
    ]))
    return checks
