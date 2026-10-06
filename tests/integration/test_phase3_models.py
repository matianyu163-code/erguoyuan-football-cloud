"""Small end-to-end base-model run with real model code and no probability mocks."""

from datetime import timedelta

import pytest

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.data.availability import (
    AvailabilityItem,
    DataAvailabilityReport,
)
from erguoyuan_football.models.comparison import comparison_report
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.runner import ModelRunner
from tests.unit.test_phase3_models import (
    available_report,
    fit_config,
    make_dataset,
    make_match,
    make_snapshot,
)

pytestmark = pytest.mark.integration


def test_base_models_execute_independently() -> None:
    data = make_dataset(rows=48)
    match = make_match()
    snapshot = make_snapshot(match, report=available_report(match.match_id))
    result = ModelRunner().run(
        match, snapshot, data, trained_until=data.matches[-1].kickoff_time + timedelta(days=1),
        config=fit_config(draws=1000, tune=1000, chains=4, min_ess=100, max_rhat=1.2),
    )
    by_model = result.by_model()
    assert set(by_model) == {
        "DIXON_COLES_V1", "BIVARIATE_POISSON_V1", "BAYESIAN_HIERARCHICAL_V1", "ELO_V1", "PI_RATING_V1",
        "DYNAMIC_BAYESIAN_POISSON_V1", "CORE_SPI_LIKE_V1", "CORE_OPTA_XG_ELO_LIKE_V1",
        "HISTORICAL_MARKET_BAYESIAN_POISSON_V1",
    }
    assert all(prediction.execution_status in {
        ExecutionStatus.SUCCESS, ExecutionStatus.UNAVAILABLE, ExecutionStatus.FAILED,
    } for prediction in result.predictions)
    successful = [prediction for prediction in result.predictions if prediction.execution_status == ExecutionStatus.SUCCESS]
    assert {prediction.model_id for prediction in successful} >= {
        "DIXON_COLES_V1", "BIVARIATE_POISSON_V1", "BAYESIAN_HIERARCHICAL_V1", "ELO_V1", "PI_RATING_V1",
        "DYNAMIC_BAYESIAN_POISSON_V1",
        "CORE_SPI_LIKE_V1", "CORE_OPTA_XG_ELO_LIKE_V1",
    }
    dynamic = by_model["DYNAMIC_BAYESIAN_POISSON_V1"]
    assert dynamic.execution_status == ExecutionStatus.SUCCESS
    assert dynamic.metadata["posterior_method"] == "SEQUENTIAL_LAPLACE_POISSON"
    market_model = by_model["HISTORICAL_MARKET_BAYESIAN_POISSON_V1"]
    assert market_model.execution_status == ExecutionStatus.UNAVAILABLE
    assert "market_odds" in (market_model.reason or "")
    bayes = by_model["BAYESIAN_HIERARCHICAL_V1"]
    assert bayes.metadata.get("lambda_home_ci") and bayes.metadata.get("lambda_away_ci")
    report = comparison_report(match, result)
    assert {item["model_id"] for item in report["models"]} == set(by_model)
    assert all(value["execution_status"] in {status.value for status in ExecutionStatus} for value in report["models"])


def test_runner_without_real_availability_fails_closed() -> None:
    data = make_dataset(rows=48)
    match = make_match()
    missing = DataAvailabilityReport(match_id=match.match_id, items={
        "historical_goals": AvailabilityItem(availability="UNAVAILABLE", reason="NO_DATA"),
        "historical_results": AvailabilityItem(availability="UNAVAILABLE", reason="NO_DATA"),
    })
    result = ModelRunner().run(match, make_snapshot(match, report=missing), data,
                               trained_until=data.matches[-1].kickoff_time + timedelta(days=1),
                               config=ModelConfig(min_matches=24, min_team_matches=2, training_window=None))
    assert all(prediction.execution_status == ExecutionStatus.UNAVAILABLE for prediction in result.predictions)
