"""Matched OOS benchmark interface never fills missing sources."""

import pytest

from erguoyuan_football.ml.backtest import MLWalkForwardBacktester
from erguoyuan_football.ml.benchmarks import compare_model_benchmarks
from erguoyuan_football.ml.catboost_model import CoreCatBoostModel
from erguoyuan_football.ml.schemas import ModelFamily
from erguoyuan_football.ml.xgboost_model import CoreXGBoostModel

pytestmark = pytest.mark.model


def test_matched_benchmarks_and_missing_market(ml_dataset_factory) -> None:
    dataset_xgb, _, config_xgb = ml_dataset_factory(family=ModelFamily.XGBOOST)
    dataset_cat, _, config_cat = ml_dataset_factory(family=ModelFamily.CATBOOST)
    evaluator = MLWalkForwardBacktester()
    xgb = evaluator.run(dataset_xgb, CoreXGBoostModel, config_xgb,
        initial_train=45, validation_size=10, test_size=10, step=10)
    cat = evaluator.run(dataset_cat, CoreCatBoostModel, config_cat,
        initial_train=45, validation_size=10, test_size=10, step=10)
    benchmarks = compare_model_benchmarks(xgb, {
        "CORE_CATBOOST_V1": tuple(item.prediction for item in cat.observations),
    })
    by_id = {item.benchmark_id: item for item in benchmarks}
    assert by_id["CORE_XGBOOST_V1"].status == "AVAILABLE"
    assert by_id["CORE_CATBOOST_V1"].status == "AVAILABLE"
    assert by_id["CORE_CATBOOST_V1"].metrics.sample_size == 20
    assert by_id["MARKET_CONSENSUS"].status == "UNAVAILABLE"
    assert by_id["NAIVE_LEAGUE_FREQUENCY"].status == "UNAVAILABLE"
    non_oos = tuple(item.prediction.model_copy(update={"is_oos": False})
                    for item in cat.observations)
    rejected = compare_model_benchmarks(xgb, {"CORE_CATBOOST_V1": non_oos})
    assert {item.benchmark_id: item.status for item in rejected}["CORE_CATBOOST_V1"] == "UNAVAILABLE"
