"""Fit market-implied scoring rates to de-vigged 1X2 and totals prices."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from erguoyuan_football.contracts.common import Availability, utc
from erguoyuan_football.markets.schemas import (
    MarketConsensus,
    MarketGoalFeatures,
    MarketSnapshot,
    MarketType,
    Selection,
)
from erguoyuan_football.markets.settlement import TotalSettlementEngine
from erguoyuan_football.models.dynamic_bayes.likelihood import score_matrix_from_lambdas
from erguoyuan_football.models.score_matrix import ScoreMatrix


class MarketImpliedGoalEngine:
    """Numerically infer market lambdas; values remain market-implied estimates, not truth."""

    def __init__(self, *, max_goals: int = 12, maximum_fit_error: float = 0.12,
                 one_x_two_uncertainty_multiplier: float = 2.0) -> None:
        if max_goals < 4 or maximum_fit_error <= 0 or one_x_two_uncertainty_multiplier < 1:
            raise ValueError("INVALID_MARKET_GOAL_CONFIG")
        self.max_goals = max_goals
        self.maximum_fit_error = maximum_fit_error
        self.one_x_two_uncertainty_multiplier = one_x_two_uncertainty_multiplier

    def fit(self, snapshot: MarketSnapshot, consensuses: tuple[MarketConsensus, ...]) -> MarketGoalFeatures:
        """Fit one match's current consensus, preferring 1X2 plus the 2.5 total line."""
        current = [item for item in consensuses if item.market_snapshot_id == snapshot.market_snapshot_id
                   and item.match_id == snapshot.match_id and item.prediction_time == snapshot.prediction_time]
        one_x_two = next((item for item in current if item.market_type == MarketType.MATCH_1X2), None)
        total = next((item for item in current if item.market_type == MarketType.TOTALS
                      and item.line_quarters == 10), None)
        if one_x_two is None:
            return self._unavailable(snapshot, "MARKET_1X2_CONSENSUS_UNAVAILABLE")
        if one_x_two.quality_status.value in {"POOR", "UNAVAILABLE"}:
            return self._unavailable(snapshot, "MARKET_QUALITY_POOR")
        inference_mode = "ONE_X_TWO_AND_TOTALS_2_5" if total is not None else "ONE_X_TWO_ONLY"
        target = [one_x_two.probabilities[Selection.HOME.value],
                  one_x_two.probabilities[Selection.DRAW.value],
                  one_x_two.probabilities[Selection.AWAY.value]]
        if total is not None:
            target.append(total.probabilities[Selection.OVER.value])
        if total is not None and total.quality_status.value in {"POOR", "UNAVAILABLE"}:
            total = None
            target = target[:3]
            inference_mode = "ONE_X_TWO_ONLY"
        def residual(log_rates: np.ndarray) -> np.ndarray:
            matrix = self._matrix(float(math.exp(log_rates[0])), float(math.exp(log_rates[1])))
            outcome = matrix.outcome()
            predicted = [outcome.p_home, outcome.p_draw, outcome.p_away]
            if total is not None:
                predicted.append(_total_over_conditional_probability(matrix, 2.5))
            return np.asarray(predicted) - np.asarray(target)
        try:
            result = least_squares(residual, x0=np.log([1.35, 1.1]),
                                   bounds=(np.log([0.05, 0.05]), np.log([8.0, 8.0])),
                                   method="trf", max_nfev=2_000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            rates = np.exp(result.x)
            fit_residuals = residual(result.x)
            fit_error = float(np.sqrt(np.mean(fit_residuals**2)))
            matrix = self._matrix(float(rates[0]), float(rates[1]))
        except (ValueError, FloatingPointError, OverflowError) as error:
            return self._unavailable(snapshot, f"MARKET_GOAL_OPTIMIZER_FAILED:{type(error).__name__}")
        max_error = self.maximum_fit_error * (self.one_x_two_uncertainty_multiplier
                                                if inference_mode == "ONE_X_TWO_ONLY" else 1.0)
        if not result.success or not np.isfinite(rates).all() or np.any(rates <= 0):
            return self._unavailable(snapshot, "MARKET_GOAL_OPTIMIZER_FAILED")
        if fit_error > max_error:
            return self._unavailable(snapshot, "MARKET_GOAL_FIT_POOR", fit_error=fit_error,
                                     optimizer_success=True,
                                     residuals={str(index): float(value) for index, value in enumerate(fit_residuals)})
        quote_ids = tuple(sorted({quote_id for consensus in (one_x_two, total) if consensus is not None
                                  for quote_id in consensus.quote_ids}))
        quote_by_id = {quote.quote_id: quote for quote in snapshot.quotes}
        lineage_quotes = [quote_by_id[item] for item in quote_ids if item in quote_by_id]
        if len(lineage_quotes) != len(quote_ids) or not lineage_quotes:
            return self._unavailable(snapshot, "MARKET_GOAL_QUOTE_LINEAGE_MISSING")
        as_of = max(utc(item.as_of_time) for item in lineage_quotes if item.as_of_time is not None)
        retrieved = max(utc(item.retrieved_at) for item in lineage_quotes)
        prediction_time = utc(snapshot.prediction_time)
        if max(as_of, retrieved) > prediction_time:
            return self._unavailable(snapshot, "POINT_IN_TIME_GUARD_V1_MARKET_GOALS")
        dispersion = {key: value for consensus in (one_x_two, total) if consensus is not None
                      for key, value in consensus.dispersion.items()}
        return MarketGoalFeatures(
            availability=Availability.AVAILABLE, match_id=snapshot.match_id,
            market_snapshot_id=snapshot.market_snapshot_id, prediction_time=prediction_time,
            prediction_horizon=snapshot.prediction_horizon,
            as_of_time=as_of, retrieved_at=retrieved,
            market_implied_lambda_home=float(rates[0]), market_implied_lambda_away=float(rates[1]),
            market_p_home=one_x_two.probabilities[Selection.HOME.value],
            market_p_draw=one_x_two.probabilities[Selection.DRAW.value],
            market_p_away=one_x_two.probabilities[Selection.AWAY.value],
            inference_mode=inference_mode, fit_error=fit_error,
            residuals={str(index): float(value) for index, value in enumerate(fit_residuals)},
            optimizer_success=True, matrix_mass=matrix.retained_mass, score_matrix=matrix.values,
            score_matrix_tail_mass=matrix.tail_mass, score_matrix_max_goals=matrix.max_goals,
            source_count=len({item.provider_id for item in lineage_quotes}),
            bookmaker_count=len({item.bookmaker_id for item in lineage_quotes}), dispersion=dispersion,
            source_ids=tuple(sorted({item.provider_id for item in lineage_quotes})),
            dependency_ids=quote_ids, devig_policy_version=one_x_two.devig_policy_version,
        )

    def _matrix(self, lambda_home: float, lambda_away: float) -> ScoreMatrix:
        return score_matrix_from_lambdas(np.asarray([lambda_home]), np.asarray([lambda_away]), self.max_goals)

    @staticmethod
    def _unavailable(snapshot: MarketSnapshot, reason: str, *, fit_error: float | None = None,
                     optimizer_success: bool = False, residuals: dict[str, float] | None = None) -> MarketGoalFeatures:
        return MarketGoalFeatures(
            availability=Availability.UNAVAILABLE, reason=reason, match_id=snapshot.match_id,
            market_snapshot_id=snapshot.market_snapshot_id, prediction_time=snapshot.prediction_time,
            prediction_horizon=snapshot.prediction_horizon,
            fit_error=fit_error, optimizer_success=optimizer_success, residuals=residuals or {},
        )


def _total_over_conditional_probability(matrix: ScoreMatrix, line: float) -> float:
    probabilities = TotalSettlementEngine.probabilities(matrix, line, Selection.OVER)
    denominator = probabilities.expected_win_equivalent + probabilities.expected_loss_equivalent
    if denominator <= 0:
        raise ValueError("TOTALS_MARKET_HAS_NO_RESOLVED_OUTCOMES")
    return probabilities.expected_win_equivalent / denominator
