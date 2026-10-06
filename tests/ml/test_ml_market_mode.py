"""Synthetic quoted markets verify mode isolation and source-time rejection."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.markets.schemas import (
    ConsensusMethod,
    DeVigMethod,
    MarketConsensus,
    MarketQualityStatus,
    MarketSnapshot,
    MarketType,
    OddsQuote,
    PredictionHorizon,
    QuoteQuality,
    Selection,
    SourceType,
    TimestampQuality,
)
from erguoyuan_football.ml.ablation import Ablation, MLFeatureAblation
from erguoyuan_football.ml.backtest import MLWalkForwardBacktester
from erguoyuan_football.ml.catboost_model import CoreCatBoostModel
from erguoyuan_football.ml.config import load_ml_config
from erguoyuan_football.ml.dataset_builder import MLDatasetBuilder
from erguoyuan_football.ml.feature_builder import MLFeatureBuilder
from erguoyuan_football.ml.feature_contract import build_feature_schema
from erguoyuan_football.ml.schemas import FeatureMode, ModelFamily
from erguoyuan_football.ml.splits import MLWalkForwardSplit
from erguoyuan_football.ml.xgboost_model import CoreXGBoostModel

pytestmark = pytest.mark.model


def _with_synthetic_market(snapshot: PredictionSnapshot, *, future: bool = False) -> PredictionSnapshot:
    at = snapshot.prediction_time
    source_time = at + timedelta(minutes=1) if future else at - timedelta(minutes=10)
    prices = {Selection.HOME: Decimal("2.10"), Selection.DRAW: Decimal("3.30"),
              Selection.AWAY: Decimal("3.70")}
    quotes = tuple(OddsQuote(quote_id=f"{snapshot.match_id}:{selection.value}", match_id=snapshot.match_id,
        provider_id="SYNTHETIC_TEST", bookmaker_id="SYNTHETIC_BOOK", market_type=MarketType.MATCH_1X2,
        selection=selection, odds_decimal=price, source_type=SourceType.SYNTHETIC_TEST,
        source_time=source_time, as_of_time=source_time, retrieved_at=source_time,
        timestamp_quality=TimestampQuality.SOURCE_NATIVE, schema_version="test-v1",
        content_hash=f"test:{snapshot.match_id}:{selection.value}", quality_status=QuoteQuality.GOOD)
        for selection, price in prices.items())
    market = MarketSnapshot(match_id=snapshot.match_id, prediction_time=at,
        kickoff_time=snapshot.match_data_snapshot.kickoff_time,
        prediction_horizon=PredictionHorizon.T_MINUS_60M, horizon_seconds=3600,
        source_count=1, bookmaker_count=1, quote_count=3,
        markets_available=(MarketType.MATCH_1X2,), freshness_seconds=600,
        quality_status=MarketQualityStatus.GOOD,
        included_quote_ids=tuple(quote.quote_id for quote in quotes), quotes=quotes)
    consensus = MarketConsensus(match_id=snapshot.match_id,
        market_snapshot_id=market.market_snapshot_id, market_type=MarketType.MATCH_1X2,
        prediction_time=at, probabilities={"HOME": 0.45, "DRAW": 0.28, "AWAY": 0.27},
        bookmaker_count=1, provider_count=1, overround_mean=0.05, dispersion={},
        freshness_seconds=600, consensus_method=ConsensusMethod.MEDIAN,
        devig_method=DeVigMethod.MULTIPLICATIVE, devig_policy_version="SYNTHETIC_TEST_V1",
        quality_status=MarketQualityStatus.GOOD,
        quote_ids=tuple(quote.quote_id for quote in quotes))
    return PredictionSnapshot.model_validate({**snapshot.model_dump(),
        "canonical_market_snapshot": market, "canonical_market_consensus": (consensus,)})


def test_future_quote_cannot_enter_ml_feature(synthetic_history) -> None:
    snapshots, _ = synthetic_history
    with pytest.raises(ValueError, match="quote source time unavailable or after prediction time"):
        _with_synthetic_market(snapshots[0], future=True)


@pytest.mark.parametrize(("family", "model_class"), [
    (ModelFamily.XGBOOST, CoreXGBoostModel), (ModelFamily.CATBOOST, CoreCatBoostModel),
])
def test_with_market_train_predict_and_no_market_isolation(
        synthetic_history, project_root, family, model_class) -> None:
    snapshots, results = synthetic_history
    quoted = tuple(_with_synthetic_market(snapshot) for snapshot in snapshots)
    builder = MLFeatureBuilder()
    no_market = builder.build(quoted[65], (), mode=FeatureMode.NO_MARKET,
        family=family, for_training=True)
    assert not any(item.source_type == "MARKET" for item in no_market.feature_lineage)
    schema = build_feature_schema(FeatureMode.WITH_MARKET, family)
    vectors = builder.build_many(quoted, {}, mode=FeatureMode.WITH_MARKET,
        family=family, for_training=True)
    assert all(vector.features["market_available"] == 1 for vector in vectors)
    assert all(any(item.source_type == "MARKET" for item in vector.feature_lineage) for vector in vectors)
    dataset = MLDatasetBuilder().build(vectors, results, schema=schema,
        competition_ids={snapshot.match_id: snapshot.match_data_snapshot.competition_id for snapshot in quoted},
        seasons={snapshot.match_id: "2025" for snapshot in quoted},
        dataset_cutoff=max(result.retrieved_at for result in results.values()),
        dataset_kind="SYNTHETIC_TEST")
    assert MLFeatureAblation.transform(dataset, Ablation.FULL_WITH_MARKET).dataset == dataset
    split = MLWalkForwardSplit.rolling(dataset, initial_train=45, validation_size=10,
                                      test_size=10, step=10)[0]
    filename = "xgboost" if family == ModelFamily.XGBOOST else "catboost"
    config = load_ml_config(project_root / f"config/{filename}.yaml", family=family,
                            profile="development", allow_test_data=True)
    model = model_class().fit(split.train, split.validation, config)
    backtest = MLWalkForwardBacktester().run(dataset, model_class, config,
        initial_train=45, validation_size=10, test_size=10, step=10)
    assert backtest.overall.sample_size == 20
    assert backtest.performance_claim == "SYNTHETIC_TEST_ONLY"
    prediction = model.predict(split.test.rows[0].vector, oos=True)
    assert prediction.execution_status == ExecutionStatus.SUCCESS
    assert prediction.metadata["uses_market"] is True
    assert prediction.metadata["market_dependency_ids"]
    assert model.predict(no_market).execution_status == ExecutionStatus.FAILED
    assert model.predict(split.test.rows[0].vector, production=True,
        network_gate_status="PASS", market_gate_status="PASS").execution_status == ExecutionStatus.UNAVAILABLE
