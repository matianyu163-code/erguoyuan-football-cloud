"""Ablations are new immutable datasets evaluated only out of sample."""

import pytest

from erguoyuan_football.ml.ablation import Ablation, MLFeatureAblation
from erguoyuan_football.ml.backtest import MLWalkForwardBacktester
from erguoyuan_football.ml.catboost_model import CoreCatBoostModel
from erguoyuan_football.ml.schemas import ModelFamily
from erguoyuan_football.ml.xgboost_model import CoreXGBoostModel

pytestmark = pytest.mark.model


@pytest.mark.parametrize("ablation", [Ablation.RATINGS_ONLY, Ablation.RATINGS_GOALS,
                                     Ablation.RATINGS_GOALS_FORM_XG, Ablation.FULL_NO_MARKET])
def test_ablation_dataset_preserves_time_and_excludes_market(ml_dataset_factory, ablation) -> None:
    original, _, config = ml_dataset_factory()
    transformed = MLFeatureAblation.transform(original, ablation)
    assert transformed.dataset.manifest.row_count == original.manifest.row_count
    assert all(row.vector.prediction_time < row.vector.kickoff_time for row in transformed.dataset.rows)
    assert not any("market_" in name for name in transformed.dataset.feature_schema.feature_names)
    if ablation != Ablation.FULL_NO_MARKET:
        assert transformed.dataset.manifest.data_hash != original.manifest.data_hash
    report = MLWalkForwardBacktester().run(transformed.dataset, CoreXGBoostModel, config,
        initial_train=45, validation_size=10, test_size=10, step=10)
    assert report.overall.sample_size == 20
    assert report.performance_claim == "SYNTHETIC_TEST_ONLY"


def test_catboost_ratings_ablation_executes(ml_dataset_factory) -> None:
    original, _, config = ml_dataset_factory(family=ModelFamily.CATBOOST)
    transformed = MLFeatureAblation.transform(original, Ablation.RATINGS_ONLY)
    assert transformed.dataset.feature_schema.categorical_features == ("competition_id",)
    report = MLWalkForwardBacktester().run(transformed.dataset, CoreCatBoostModel, config,
        initial_train=45, validation_size=10, test_size=10, step=10)
    assert report.overall.sample_size == 20
