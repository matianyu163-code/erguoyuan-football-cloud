"""Regression tests for the strict dynamic model historical-data boundary."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.models.dynamic_bayes.inference import DynamicBayesianInference
from erguoyuan_football.models.dynamic_bayes.model import (
    CoreDynamicBayesianPoissonModel,
)
from erguoyuan_football.models.point_in_time import (
    FAILURE_CODE_TRAINING_CUTOFF_AFTER_PREDICTION,
    FutureHistoryLeakageError,
    validate_history_point_in_time,
)
from tests.unit.test_phase3_models import fit_config, make_dataset, make_match


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _failed_prediction_fields(prediction_time: datetime, trained_until: datetime) -> dict:
    return {
        "match_id": "audit_match",
        "prediction_snapshot_id": "audit_snapshot",
        "model_id": "DYNAMIC_BAYESIAN_POISSON_V1",
        "model_version": "1.0.0",
        "implementation_type": "REAL_IMPLEMENTATION",
        "training_end_time": trained_until,
        "trained_until": trained_until,
        "prediction_time": prediction_time,
        "input_data_version": "test-data-v1",
        "data_source": ("TEST_SOURCE",),
        "data_status": "UNAVAILABLE",
        "execution_status": "FAILED",
        "failure_code": FAILURE_CODE_TRAINING_CUTOFF_AFTER_PREDICTION,
        "reason": "Future historical data detected.",
    }


def test_history_pit_guard_checks_every_observation() -> None:
    prediction_time = _utc(2026, 9, 20, 12)
    history = [
        {"match_time": _utc(2026, 9, 1)},
        {"match_time": _utc(2026, 9, 5)},
        {"match_time": _utc(2026, 9, 10)},
        {"match_time": _utc(2026, 9, 25)},
        {"match_time": _utc(2026, 9, 15)},
    ]
    with pytest.raises(FutureHistoryLeakageError, match="POINT_IN_TIME_GUARD_V1"):
        validate_history_point_in_time(history, prediction_time)


def test_history_equal_prediction_time_is_rejected() -> None:
    prediction_time = _utc(2026, 9, 20, 12)
    with pytest.raises(FutureHistoryLeakageError):
        validate_history_point_in_time(
            [{"match_time": _utc(2026, 9, 1)}, {"match_time": prediction_time}], prediction_time,
        )


def test_history_before_prediction_time_is_allowed() -> None:
    prediction_time = _utc(2026, 9, 20, 12)
    result = validate_history_point_in_time([
        {"match_time": _utc(2026, 9, 1)},
        {"match_time": _utc(2026, 9, 5)},
        {"match_time": _utc(2026, 9, 19, 23, 59)},
    ], prediction_time)
    assert len(result) == 3


def test_failed_prediction_can_preserve_future_cutoff_as_audit_evidence() -> None:
    prediction_time = _utc(2026, 9, 20, 12)
    trained_until = _utc(2026, 9, 25, 12)
    result = ModelPrediction(**_failed_prediction_fields(prediction_time, trained_until))
    assert result.execution_status == ExecutionStatus.FAILED
    assert result.trained_until > result.prediction_time
    assert result.failure_code == FAILURE_CODE_TRAINING_CUTOFF_AFTER_PREDICTION
    assert result.p_home is result.p_draw is result.p_away is None
    assert result.lambda_home is result.lambda_away is None
    assert result.score_matrix is None


def test_success_prediction_rejects_future_training_cutoff() -> None:
    prediction_time = _utc(2026, 9, 20, 12)
    trained_until = _utc(2026, 9, 21, 12)
    fields = _failed_prediction_fields(prediction_time, trained_until)
    fields.update({
        "data_status": "AVAILABLE", "execution_status": "SUCCESS", "failure_code": None,
        "reason": None, "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
        "lambda_home": 1.5, "lambda_away": 1.0,
        "score_matrix": ((0.2, 0.1), (0.3, 0.4)),
    })
    with pytest.raises(ValueError, match="POINT_IN_TIME_GUARD_V1"):
        ModelPrediction(**fields)


def test_failed_prediction_rejects_probabilities() -> None:
    fields = _failed_prediction_fields(_utc(2026, 9, 20, 12), _utc(2026, 9, 25, 12))
    fields.update({"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2})
    with pytest.raises(ValueError, match="null prediction values"):
        ModelPrediction(**fields)


def test_dynamic_predict_blocks_future_history_before_goal_rates_or_posterior(monkeypatch) -> None:
    data = make_dataset(rows=48)
    model = CoreDynamicBayesianPoissonModel().fit(
        data, data.matches[-1].kickoff_time + timedelta(days=1),
        fit_config(dynamic_posterior_draws=80),
    )
    prediction_time = _utc(2026, 9, 20, 12)
    kickoff = _utc(2026, 10, 1, 12)
    match = make_match(kickoff=kickoff).model_copy(update={
        "as_of_time": prediction_time,
        "retrieved_at": prediction_time,
    })
    snapshot = PredictionSnapshot(
        match_id=match.match_id, prediction_time=prediction_time, match_data_snapshot=match,
    )
    posterior = Mock(wraps=model.inference.posterior_prediction)
    monkeypatch.setattr(model.inference, "posterior_prediction", posterior)
    predict_goal_rates = Mock(wraps=model._predict_values)
    monkeypatch.setattr(model, "_predict_values", predict_goal_rates)
    history = [
        {"match_time": _utc(2026, 9, 1)},
        {"match_time": _utc(2026, 9, 5)},
        {"match_time": _utc(2026, 9, 25)},
        {"match_time": _utc(2026, 9, 15)},
    ]

    result = model.predict(match, snapshot, history=history)

    assert result.execution_status == ExecutionStatus.FAILED
    assert result.failure_code == FAILURE_CODE_TRAINING_CUTOFF_AFTER_PREDICTION
    assert result.trained_until == _utc(2026, 9, 25)
    assert result.p_home is result.p_draw is result.p_away is None
    assert result.lambda_home is result.lambda_away is None
    assert result.score_matrix is None
    predict_goal_rates.assert_not_called()
    posterior.assert_not_called()


def test_dynamic_inference_rejects_future_row_before_global_fit(monkeypatch) -> None:
    data = make_dataset(rows=24)
    cutoff = data.matches[-1].kickoff_time + timedelta(days=1)
    future_kickoff = cutoff + timedelta(days=1)
    future = data.matches[-1].model_copy(update={
        "match_id": "future_training_row",
        "kickoff_time": future_kickoff,
        "completed_at": future_kickoff + timedelta(hours=2),
        "as_of_time": future_kickoff + timedelta(hours=4),
        "retrieved_at": future_kickoff + timedelta(hours=4),
    })
    contaminated = data.model_copy(update={"matches": (*data.matches, future)})
    inference = DynamicBayesianInference(
        fit_config(dynamic_posterior_draws=20),
        model_id="DYNAMIC_BAYESIAN_POISSON_V1", model_version="1.0.0",
    )
    fit_rates = Mock(wraps=inference._fit_global_rates)
    transition = Mock(wraps=inference._transition_states)
    monkeypatch.setattr(inference, "_fit_global_rates", fit_rates)
    monkeypatch.setattr(inference, "_transition_states", transition)

    with pytest.raises(FutureHistoryLeakageError):
        inference.fit(contaminated, cutoff)

    fit_rates.assert_not_called()
    transition.assert_not_called()
