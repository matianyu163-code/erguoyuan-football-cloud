"""Phase 5 model calculations run through chronological OOS training and prediction."""

import pytest

from erguoyuan_football.backtesting.strength_models import (
    correlation_report,
    expected_calibration_error,
    walk_forward_strength_model,
)
from tests.unit.test_phase3_models import make_dataset

pytestmark = pytest.mark.integration


@pytest.mark.model
@pytest.mark.parametrize("model_id", ["CORE_SPI_LIKE_V1", "CORE_OPTA_XG_ELO_LIKE_V1"])
def test_strength_model_walk_forward_oos(model_id: str) -> None:
    data = make_dataset(rows=48, dataset_kind="SYNTHETIC_TEST")
    result = walk_forward_strength_model(model_id, data, initial_matches=24, horizon=4, step=4)
    assert result.sample_size > 0
    assert all(item.is_oos for item in result.predictions)
    assert all(item.training_end_time <= item.prediction_time for item in result.predictions)
    assert result.log_loss >= 0 and result.brier >= 0 and result.rps >= 0


def test_oos_ece_and_prediction_correlations() -> None:
    probabilities = [(0.7, 0.2, 0.1), (0.3, 0.5, 0.2), (0.1, 0.2, 0.7)]
    assert 0 <= expected_calibration_error(probabilities, [0, 1, 2]) <= 1
    report = correlation_report({"SPI": tuple(probabilities), "ELO": tuple(probabilities)})
    assert set(report["outcomes"]) == {"home", "draw", "away"}
    assert report["outcomes"]["home"][0][1] == pytest.approx(1)
