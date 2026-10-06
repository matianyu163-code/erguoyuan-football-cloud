"""Small synthetic chronological tests of market features and Bayesian OOS execution."""

from datetime import timedelta

import numpy as np
import pytest

from erguoyuan_football.backtesting.market_engine import (
    MarketBaselineCase,
    evaluate_market_baselines,
    market_only_probabilities,
    walk_forward_market_bayes,
)
from erguoyuan_football.contracts.common import Availability
from erguoyuan_football.contracts.predictions import ProbabilityVector
from erguoyuan_football.markets.schemas import MarketGoalFeatures, PredictionHorizon
from erguoyuan_football.models.dynamic_bayes.likelihood import score_matrix_from_lambdas
from erguoyuan_football.models.historical_market_bayes import (
    HistoricalMarketBayesConfig,
)
from tests.unit.test_phase3_models import make_dataset

pytestmark = pytest.mark.integration


def _dataset_with_market_features(rows: int = 48):
    dataset = make_dataset(rows=rows, dataset_kind="SYNTHETIC_TEST")
    features = []
    for index, row in enumerate(dataset.matches):
        predicted_at = row.kickoff_time - timedelta(minutes=30)
        as_of = predicted_at - timedelta(minutes=10)
        retrieved = predicted_at - timedelta(minutes=5)
        lambda_home = 1.15 + (index % 5) * 0.09
        lambda_away = 0.85 + ((index + 2) % 5) * 0.08
        matrix = score_matrix_from_lambdas(np.asarray([lambda_home]), np.asarray([lambda_away]), max_goals=12)
        outcome = matrix.outcome()
        features.append(MarketGoalFeatures(
            availability=Availability.AVAILABLE, match_id=row.match_id,
            market_snapshot_id=f"synthetic-market-snapshot-{index}",
            prediction_time=predicted_at, prediction_horizon=PredictionHorizon.CUSTOM,
            as_of_time=as_of, retrieved_at=retrieved,
            market_implied_lambda_home=lambda_home, market_implied_lambda_away=lambda_away,
            market_p_home=outcome.p_home, market_p_draw=outcome.p_draw, market_p_away=outcome.p_away,
            inference_mode="SYNTHETIC_TEST_FIXTURE", fit_error=0.025, optimizer_success=True,
            matrix_mass=matrix.retained_mass, score_matrix=matrix.values,
            score_matrix_tail_mass=matrix.tail_mass, score_matrix_max_goals=matrix.max_goals,
            source_count=1, bookmaker_count=2, source_ids=("SYNTHETIC_TEST_PROVIDER",),
            dispersion={"HOME": 0.02, "DRAW": 0.015, "AWAY": 0.02},
            dependency_ids=(f"synthetic-quote-a-{index}", f"synthetic-quote-b-{index}"),
            devig_policy_version="SYNTHETIC_TEST_POLICY",
        ))
    return dataset.model_copy(update={"market_goal_features": tuple(features)})


def test_market_bayesian_fusion_and_historical_ablation_are_oos() -> None:
    dataset = _dataset_with_market_features()
    common = HistoricalMarketBayesConfig(min_matches=30, min_team_matches=1, training_window=None,
        allow_test_data=True, market_min_training_matches=30, market_posterior_draws=40,
        market_prediction_horizon=PredictionHorizon.CUSTOM)
    fusion = walk_forward_market_bayes(dataset, prediction_horizon=PredictionHorizon.CUSTOM,
        mode="FUSION", config=common, initial_matches=36, max_predictions=3)
    historical = walk_forward_market_bayes(dataset, prediction_horizon=PredictionHorizon.CUSTOM,
        mode="HISTORICAL_ONLY", config=common.model_copy(update={"market_mode": "HISTORICAL_ONLY"}),
        initial_matches=36, max_predictions=3)
    assert fusion.sample_size == 3
    assert historical.sample_size == 3
    assert fusion.log_loss is not None and fusion.brier is not None and fusion.rps is not None
    assert historical.log_loss is not None
    assert all(row.is_oos and row.execution_status.value == "SUCCESS" for row in fusion.predictions)
    assert all(row.training_end_time <= row.prediction_time for row in fusion.predictions)
    assert all(row.metadata["uses_market"] is True for row in fusion.predictions)
    assert all(row.metadata["uses_market"] is False for row in historical.predictions)
    assert all(not row.dependency_tags for row in historical.predictions)


def test_market_only_baseline_and_horizon_partition() -> None:
    dataset = _dataset_with_market_features()
    features = dataset.market_goal_features
    probabilities = market_only_probabilities(features, prediction_horizon=PredictionHorizon.CUSTOM)
    assert set(probabilities) == {row.match_id for row in dataset.matches}
    matrix_probabilities = market_only_probabilities(features, prediction_horizon=PredictionHorizon.CUSTOM,
                                                      use_implied_poisson=True)
    assert set(matrix_probabilities) == set(probabilities)
    first = features[0]
    probabilities_by_phase = [
        MarketBaselineCase(match_id=first.match_id, competition_id="synthetic-league", season="2024",
            market_phase="CURRENT", prediction_horizon=PredictionHorizon.CUSTOM,
            prediction_time=first.prediction_time, kickoff_time=dataset.matches[0].kickoff_time,
            as_of_time=first.as_of_time, retrieved_at=first.retrieved_at,
            probability=probabilities[first.match_id], outcome=0,
            source_quote_ids=first.dependency_ids, bookmaker_count=2),
        MarketBaselineCase(match_id="closing-match", competition_id="synthetic-league", season="2024",
            market_phase="CLOSING", prediction_horizon=PredictionHorizon.T_MINUS_60M,
            prediction_time=first.prediction_time, kickoff_time=dataset.matches[0].kickoff_time,
            as_of_time=dataset.matches[0].kickoff_time - timedelta(minutes=5),
            retrieved_at=dataset.matches[0].kickoff_time + timedelta(hours=1),
            probability=probabilities[first.match_id], outcome=0,
            source_quote_ids=("posthoc-close-quote",), bookmaker_count=2, post_hoc_benchmark_only=True),
    ]
    results = evaluate_market_baselines(tuple(probabilities_by_phase))
    assert {(item.market_phase, item.prediction_horizon) for item in results} == {
        ("CURRENT", PredictionHorizon.CUSTOM), ("CLOSING", PredictionHorizon.T_MINUS_60M)}
    assert next(item for item in results if item.market_phase == "CLOSING").sample_size == 1


def test_market_baseline_rejects_future_nonclosing_quote() -> None:
    at = _dataset_with_market_features().market_goal_features[0].prediction_time
    with pytest.raises(ValueError, match="PREMATCH_MARKET_BASELINE_CONTAINS_FUTURE_INFORMATION"):
        MarketBaselineCase(match_id="m1", competition_id="league", season="2024",
            market_phase="CURRENT", prediction_horizon=PredictionHorizon.CUSTOM,
            prediction_time=at, kickoff_time=at + timedelta(hours=1),
            as_of_time=at + timedelta(minutes=1), retrieved_at=at,
            probability=ProbabilityVector(p_home=0.4, p_draw=0.3, p_away=0.3),
            outcome=0, source_quote_ids=("q1",), bookmaker_count=1)
