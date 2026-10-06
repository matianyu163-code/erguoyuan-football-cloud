"""Chronological market-model validation, direct market baselines and closing benchmarks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np

from erguoyuan_football.backtesting.base_model_backtest import (
    accuracy,
    brier_score,
    log_loss,
    ranked_probability_score,
)
from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.contracts.predictions import ModelPrediction, ProbabilityVector
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.markets.schemas import MarketGoalFeatures, PredictionHorizon
from erguoyuan_football.models.historical_market_bayes import (
    HistoricalMarketBayesConfig,
    HistoricalMarketBayesianPoissonModel,
)
from erguoyuan_football.models.training import (
    InsufficientData,
    TrainingDataset,
    TrainingMatch,
)


@dataclass(frozen=True)
class MarketModelBacktestResult:
    """OOS metrics for one fixed market horizon and one model ablation."""

    model_id: str
    model_version: str
    mode: str
    prediction_horizon: PredictionHorizon
    attempted: int
    sample_size: int
    unavailable: int
    failed: int
    competition: str
    season: str
    date_range: tuple[datetime, datetime] | None
    log_loss: float | None
    brier: float | None
    rps: float | None
    accuracy: float | None
    ece: float | None
    availability_buckets: dict[str, int]
    source_count_buckets: dict[str, int]
    predictions: tuple[ModelPrediction, ...]


@dataclass(frozen=True)
class MarketBaselineCase:
    """One frozen market probability vector; closing rows are explicitly post-hoc only."""

    match_id: str
    competition_id: str
    season: str
    market_phase: Literal["OPENING", "CURRENT", "CLOSING", "MARKET_IMPLIED_POISSON"]
    prediction_horizon: PredictionHorizon
    prediction_time: datetime
    kickoff_time: datetime
    as_of_time: datetime
    retrieved_at: datetime
    probability: ProbabilityVector
    outcome: int
    source_quote_ids: tuple[str, ...]
    bookmaker_count: int
    post_hoc_benchmark_only: bool = False

    def __post_init__(self) -> None:
        if self.outcome not in {0, 1, 2} or not self.source_quote_ids or self.bookmaker_count < 1:
            raise ValueError("INVALID_MARKET_BASELINE_EVIDENCE")
        if not self.prediction_time < self.kickoff_time:
            raise ValueError("MARKET_BASELINE_PREDICTION_AFTER_KICKOFF")
        if self.market_phase == "CLOSING":
            if not self.post_hoc_benchmark_only or not self.prediction_time < self.as_of_time < self.kickoff_time:
                raise ValueError("CLOSING_MARKET_MUST_BE_POST_HOC_AND_AFTER_PREDICTION")
        elif self.post_hoc_benchmark_only or max(self.as_of_time, self.retrieved_at) > self.prediction_time:
            raise ValueError("PREMATCH_MARKET_BASELINE_CONTAINS_FUTURE_INFORMATION")


@dataclass(frozen=True)
class MarketBaselineResult:
    """Baseline scores partitioned by phase and horizon; closing remains a benchmark."""

    market_phase: str
    prediction_horizon: PredictionHorizon
    sample_size: int
    competition_breakdown: dict[str, int]
    season_breakdown: dict[str, int]
    log_loss: float
    brier: float
    rps: float
    accuracy: float
    ece: float


def walk_forward_market_bayes(data: TrainingDataset, *,
                              prediction_horizon: PredictionHorizon,
                              mode: Literal["FUSION", "HISTORICAL_ONLY"],
                              config: HistoricalMarketBayesConfig | None = None,
                              initial_matches: int = 36,
                              max_predictions: int | None = None) -> MarketModelBacktestResult:
    """Fit one prior-only model per historical target and emit strictly OOS predictions."""
    if initial_matches < 12 or max_predictions is not None and max_predictions < 1:
        raise ValueError("INVALID_WALK_FORWARD_LIMITS")
    if not data.matches:
        raise ValueError("EMPTY_MARKET_BACKTEST_DATA")
    base_config = config or HistoricalMarketBayesConfig(
        min_matches=initial_matches,
        min_team_matches=1,
        training_window=None,
        allow_test_data=data.dataset_kind == "SYNTHETIC_TEST",
        market_min_training_matches=max(12, min(initial_matches, 30)),
        market_prediction_horizon=prediction_horizon,
        market_mode=mode,
    )
    if base_config.market_mode != mode or base_config.market_prediction_horizon != prediction_horizon:
        raise ValueError("BACKTEST_CONFIG_MODE_OR_HORIZON_MISMATCH")
    ordered = tuple(sorted(data.matches, key=lambda row: (row.kickoff_time, row.match_id)))
    feature_map = {
        row.match_id: next((feature for feature in data.market_goal_features
                            if feature.match_id == row.match_id
                            and feature.prediction_horizon == prediction_horizon
                            and feature.availability.value == "AVAILABLE"), None)
        for row in ordered
    }
    forecasts: list[ModelPrediction] = []
    targets: list[TrainingMatch] = []
    availability_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    candidate_count = 0
    for target in ordered[initial_matches:]:
        feature = feature_map[target.match_id]
        if feature is None:
            availability_counts["MARKET_UNAVAILABLE"] += 1
            continue
        feature_as_of, feature_retrieved = feature.as_of_time, feature.retrieved_at
        if feature_as_of is None or feature_retrieved is None:
            availability_counts["MARKET_UNAVAILABLE"] += 1
            continue
        candidate_count += 1
        prediction_at = feature.prediction_time
        training_rows = tuple(row for row in ordered
            if row.match_id != target.match_id and row.kickoff_time < prediction_at and row.available_at <= prediction_at)
        if len(training_rows) < base_config.min_matches:
            availability_counts["HISTORY_UNAVAILABLE"] += 1
            continue
        trained_until = max(row.available_at for row in training_rows)
        if not trained_until <= prediction_at < target.kickoff_time:
            raise ValueError("MARKET_BACKTEST_TEMPORAL_ORDER_VIOLATION")
        training_ids = {row.match_id for row in training_rows}
        market_features = tuple(item for item in data.market_goal_features
            if item.match_id in training_ids and item.prediction_horizon == prediction_horizon
            and item.availability.value == "AVAILABLE" and item.prediction_time <= trained_until
            and item.as_of_time is not None and item.retrieved_at is not None
            and max(item.as_of_time, item.retrieved_at) <= trained_until)
        training = TrainingDataset(matches=training_rows, known_team_ids=data.known_team_ids,
            competition_hierarchy=data.competition_hierarchy, team_hierarchy=data.team_hierarchy,
            team_hierarchy_timeline=tuple(item for item in data.team_hierarchy_timeline
                                          if item.as_of_time <= trained_until
                                          and item.retrieved_at <= trained_until),
            xg_observations=tuple(item for item in data.xg_observations
                                  if item.match_id in training_ids and item.as_of_time <= trained_until
                                  and item.retrieved_at <= trained_until),
            market_goal_features=market_features,
            dataset_kind=data.dataset_kind)
        fixture = Fixture(match_id=target.match_id, competition_id=target.competition_id,
            home_team_id=target.home_team_id, away_team_id=target.away_team_id,
            kickoff_time=target.kickoff_time, source=target.source,
            retrieved_at=feature_retrieved, as_of_time=feature_as_of,
            data_version=target.data_version, season=target.season, neutral_venue=target.neutral_venue)
        snapshot = PredictionSnapshot(match_id=target.match_id, prediction_time=prediction_at,
            match_data_snapshot=fixture,
            market_goal_features=feature if mode == "FUSION" else None)
        model = HistoricalMarketBayesianPoissonModel()
        try:
            model.fit(training, trained_until, base_config)
            prediction = model.predict(fixture, snapshot)
        except InsufficientData as error:
            prediction = model.prediction_record(fixture, snapshot, ExecutionStatus.UNAVAILABLE,
                                                 reason=str(error))
        except (ArithmeticError, RuntimeError, ValueError) as error:
            prediction = model.prediction_record(fixture, snapshot, ExecutionStatus.FAILED,
                                                 reason=f"{type(error).__name__}:{error}")
        prediction = prediction.model_copy(update={"is_oos": True})
        forecasts.append(prediction)
        targets.append(target)
        availability_counts["MARKET_AVAILABLE"] += 1
        source_counts[_source_bucket(feature.source_count)] += 1
        if max_predictions is not None and len(forecasts) >= max_predictions:
            break
    if not forecasts:
        raise ValueError("NO_WALK_FORWARD_MARKET_PREDICTIONS")
    for prediction, target in zip(forecasts, targets, strict=True):
        if (not prediction.is_oos or prediction.training_end_time is None
                or not prediction.training_end_time <= prediction.prediction_time < target.kickoff_time):
            raise ValueError("MARKET_BACKTEST_OOS_PROVENANCE_FAILED")
    successful = [(prediction, target) for prediction, target in zip(forecasts, targets, strict=True)
                  if prediction.execution_status == ExecutionStatus.SUCCESS]
    probs = [(float(row.p_home), float(row.p_draw), float(row.p_away)) for row, _ in successful
             if row.p_home is not None and row.p_draw is not None and row.p_away is not None]
    labels = [_outcome(target) for _, target in successful]
    dates = [target.kickoff_time for _, target in successful]
    competitions = sorted({row.competition_id for row in targets})
    seasons = sorted({row.season for row in targets})
    return MarketModelBacktestResult(
        model_id=HistoricalMarketBayesianPoissonModel.model_id,
        model_version=HistoricalMarketBayesianPoissonModel.model_version,
        mode=mode, prediction_horizon=prediction_horizon, attempted=len(forecasts), sample_size=len(successful),
        unavailable=sum(row.execution_status == ExecutionStatus.UNAVAILABLE for row in forecasts),
        failed=sum(row.execution_status == ExecutionStatus.FAILED for row in forecasts),
        competition=",".join(competitions), season=",".join(seasons),
        date_range=(min(dates), max(dates)) if dates else None,
        log_loss=log_loss(probs, labels) if successful else None,
        brier=brier_score(probs, labels) if successful else None,
        rps=ranked_probability_score(probs, labels) if successful else None,
        accuracy=accuracy(probs, labels) if successful else None,
        ece=_ece(probs, labels) if successful else None,
        availability_buckets=dict(availability_counts), source_count_buckets=dict(source_counts),
        predictions=tuple(forecasts),
    )


def market_only_probabilities(features: tuple[MarketGoalFeatures, ...], *,
                              prediction_horizon: PredictionHorizon,
                              use_implied_poisson: bool = False) -> dict[str, ProbabilityVector]:
    """Build market-only baselines from sourced consensus or fitted market-implied score matrices."""
    result: dict[str, ProbabilityVector] = {}
    for feature in features:
        if feature.availability.value != "AVAILABLE" or feature.prediction_horizon != prediction_horizon:
            continue
        if use_implied_poisson:
            if feature.score_matrix is None:
                continue
            matrix = np.asarray(feature.score_matrix, dtype=float)
            result[feature.match_id] = ProbabilityVector(
                p_home=float(np.tril(matrix, -1).sum()), p_draw=float(np.trace(matrix)),
                p_away=float(np.triu(matrix, 1).sum()))
        else:
            market_home, market_draw, market_away = (
                feature.market_p_home, feature.market_p_draw, feature.market_p_away
            )
            if market_home is None or market_draw is None or market_away is None:
                continue
            result[feature.match_id] = ProbabilityVector(
                p_home=float(market_home), p_draw=float(market_draw), p_away=float(market_away))
    return result


def evaluate_market_baselines(cases: tuple[MarketBaselineCase, ...]) -> tuple[MarketBaselineResult, ...]:
    """Evaluate opening/current/closing and implied-Poisson baselines in separate horizons."""
    groups: dict[tuple[str, PredictionHorizon], list[MarketBaselineCase]] = defaultdict(list)
    for case in cases:
        groups[(case.market_phase, case.prediction_horizon)].append(case)
    results = []
    for (phase, horizon), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1].value)):
        vectors = [(row.probability.p_home, row.probability.p_draw, row.probability.p_away) for row in rows]
        outcomes = [row.outcome for row in rows]
        results.append(MarketBaselineResult(
            market_phase=phase, prediction_horizon=horizon, sample_size=len(rows),
            competition_breakdown=_counts(row.competition_id for row in rows),
            season_breakdown=_counts(row.season for row in rows),
            log_loss=log_loss(vectors, outcomes), brier=brier_score(vectors, outcomes),
            rps=ranked_probability_score(vectors, outcomes), accuracy=accuracy(vectors, outcomes),
            ece=_ece(vectors, outcomes)))
    return tuple(results)


def _outcome(row: TrainingMatch) -> int:
    return 0 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 2


def _source_bucket(count: int) -> str:
    return "1" if count <= 1 else "2-3" if count <= 3 else "4+"


def _counts(values) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for value in values:
        result[value] += 1
    return dict(result)


def _ece(probabilities: list[tuple[float, float, float]], outcomes: list[int], bins: int = 10) -> float:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("ECE_REQUIRES_MATCHED_OOS_ROWS")
    confidence = np.asarray([max(row) for row in probabilities])
    correct = np.asarray([int(int(np.argmax(row)) == label)
                          for row, label in zip(probabilities, outcomes, strict=True)])
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = (confidence >= lower) & ((confidence < upper) | ((index == bins - 1) & (confidence <= upper)))
        if members.any():
            error += float(members.mean()) * abs(float(confidence[members].mean() - correct[members].mean()))
    return error
