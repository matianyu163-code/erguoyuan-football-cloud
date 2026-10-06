"""Executed OOS model evaluation on explicitly synthetic test history."""

from __future__ import annotations

import numpy as np
import pytest

from erguoyuan_football.ml.backtest import (
    MLWalkForwardBacktester,
    expected_calibration_error,
)
from erguoyuan_football.ml.catboost_model import CoreCatBoostModel
from erguoyuan_football.ml.schemas import ModelFamily
from erguoyuan_football.ml.xgboost_model import CoreXGBoostModel

pytestmark = pytest.mark.model


@pytest.mark.parametrize(("family", "model_class"), [
    (ModelFamily.XGBOOST, CoreXGBoostModel),
    (ModelFamily.CATBOOST, CoreCatBoostModel),
])
def test_ml_walk_forward_true_execution(ml_dataset_factory, family, model_class) -> None:
    dataset, _, config = ml_dataset_factory(family=family)
    report = MLWalkForwardBacktester().run(dataset, model_class, config,
        initial_train=45, validation_size=10, test_size=10, step=10)
    assert report.performance_claim == "SYNTHETIC_TEST_ONLY"
    assert report.overall.sample_size == 20
    assert len(report.frozen_test_hashes) == 2
    assert report.overall.log_loss > 0
    assert 0 <= report.overall.ece <= 1
    assert {"competition", "season", "prediction_horizon", "market_availability",
            "feature_availability", "favorite_bucket"} <= set(report.strata)
    assert all(item.prediction.training_end_time <= item.prediction.prediction_time <
               item.row.vector.kickoff_time for item in report.observations)


def test_ece_known_example() -> None:
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
    labels = np.asarray([0, 2])
    assert expected_calibration_error(probabilities, labels, bins=2) == pytest.approx(0.25)


def test_ml_test_partition_never_used_in_fit(ml_dataset_factory) -> None:
    _, split, config = ml_dataset_factory()
    model = CoreXGBoostModel().fit(split.train, split.validation, config)
    assert {row.vector.match_id for row in split.test.rows}.isdisjoint(model.training_match_ids)
    assert model.trained_until < min(row.vector.prediction_time for row in split.test.rows)
