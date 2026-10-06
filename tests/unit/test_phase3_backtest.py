"""Temporal split and scoring tests for the Phase 3 evaluation harness."""

from datetime import timedelta

import pytest

from erguoyuan_football.backtesting.base_model_backtest import (
    accuracy,
    brier_score,
    evaluate_oos,
    independent_poisson_benchmark,
    log_loss,
    naive_league_frequency,
    ranked_probability_score,
    walk_forward_split,
)
from erguoyuan_football.contracts.predictions import ModelPrediction
from tests.unit.test_phase3_models import START, make_dataset


def test_walk_forward_split() -> None:
    data = make_dataset(rows=20)
    windows = walk_forward_split(data, initial_matches=8, horizon=3, step=2)
    assert windows
    assert all(window.training_end <= window.prediction_time for window in windows)
    assert all(window.test_rows[0].kickoff_time > window.prediction_time for window in windows)
    assert windows[1].test_rows[0].kickoff_time > windows[0].test_rows[0].kickoff_time


def test_walk_forward_skips_late_training_evidence() -> None:
    data = make_dataset(rows=12)
    late = data.matches[7].model_copy(update={"retrieved_at": data.matches[8].kickoff_time + timedelta(hours=1)})
    changed = data.model_copy(update={"matches": data.matches[:7] + (late,) + data.matches[8:]})
    windows = walk_forward_split(changed, initial_matches=8, horizon=1)
    assert all(window.training_end <= window.prediction_time for window in windows)


def test_oos_prediction_only() -> None:
    rows = make_dataset(rows=16).matches[8:10]
    predictions = tuple(ModelPrediction(
        match_id=row.match_id, prediction_snapshot_id=f"snapshot_{row.match_id}", model_id="ELO_V1",
        model_version="1.0.0", implementation_type="REAL_IMPLEMENTATION",
        training_end_time=START + timedelta(days=7), trained_until=START + timedelta(days=7),
        prediction_time=row.kickoff_time - timedelta(microseconds=1), input_data_version="test",
        p_home=0.4, p_draw=0.3, p_away=0.3, data_source=("REAL_TEST",), data_status="AVAILABLE",
        execution_status="SUCCESS", is_oos=True,
    ) for row in rows)
    result = evaluate_oos(predictions, rows, competition="league_1", model_version="1.0.0")
    assert result.sample_size == 2 and result.model_version == "1.0.0"
    with pytest.raises(ValueError, match="IN_SAMPLE"):
        evaluate_oos((predictions[0].model_copy(update={"is_oos": False}), predictions[1]), rows,
                     competition="league_1", model_version="1.0.0")


def test_log_loss() -> None:
    assert log_loss([(0.5, 0.3, 0.2)], [0]) == pytest.approx(-__import__("math").log(0.5))


def test_brier() -> None:
    assert brier_score([(1.0, 0.0, 0.0)], [0]) == pytest.approx(0.0)


def test_rps() -> None:
    assert ranked_probability_score([(1.0, 0.0, 0.0)], [0]) == pytest.approx(0.0)


def test_accuracy() -> None:
    assert accuracy([(0.1, 0.8, 0.1)], [1]) == 1.0


def test_benchmarks() -> None:
    data = make_dataset(rows=12)
    frequency = naive_league_frequency(data.matches)
    poisson, matrix = independent_poisson_benchmark(data.matches, max_goals=5)
    assert abs(frequency.p_home + frequency.p_draw + frequency.p_away - 1) < 1e-8
    assert abs(poisson.p_home + poisson.p_draw + poisson.p_away - 1) < 1e-8
    assert abs(sum(map(sum, matrix)) - 1) < 1e-8


def test_oos_time_violation_rejected() -> None:
    row = make_dataset(rows=16).matches[8]
    prediction = ModelPrediction(
        match_id=row.match_id, prediction_snapshot_id="snapshot", model_id="ELO_V1", model_version="1.0.0",
        implementation_type="REAL_IMPLEMENTATION", training_end_time=row.kickoff_time,
        trained_until=row.kickoff_time, prediction_time=row.kickoff_time, input_data_version="test",
        p_home=0.4, p_draw=0.3, p_away=0.3, data_source=("REAL_TEST",), data_status="AVAILABLE",
        execution_status="SUCCESS", is_oos=True,
    )
    with pytest.raises(ValueError, match="OOS_TIME_ORDER"):
        evaluate_oos((prediction,), (row,), competition="league_1", model_version="1.0.0")
