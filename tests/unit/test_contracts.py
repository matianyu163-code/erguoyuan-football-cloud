from datetime import timedelta

import pytest

from erguoyuan_football.contracts.predictions import (
    CorePrediction,
    ModelPrediction,
)
from erguoyuan_football.data.market import devig_probabilities
from erguoyuan_football.data.schemas import OptaSnapshot
from erguoyuan_football.data.snapshots import SnapshotService
from erguoyuan_football.external.opta.provider import (
    PRODUCTS,
    ExternalData,
    UnavailableOptaProvider,
)
from erguoyuan_football.models.requirements import REQUIREMENTS, check_requirements
from erguoyuan_football.pipeline import Calibration, MetaStacking, ModelPipeline


@pytest.fixture
def prediction_fields(at):
    return {"match_id": "test_match_1", "prediction_snapshot_id": "test_snapshot", "model_id": "TEST_ONLY",
            "model_version": "test_v1", "implementation_type": "REAL_IMPLEMENTATION",
            "training_end_time": at - timedelta(days=1), "trained_until": at - timedelta(days=1),
            "prediction_time": at, "input_data_version": "SYNTHETIC_TEST",
            "data_source": ("SYNTHETIC_TEST",), "data_status": "AVAILABLE", "execution_status": "SUCCESS",
            "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}


@pytest.mark.parametrize("product", PRODUCTS)
def test_opta_unavailable(product, at):
    result = UnavailableOptaProvider().fetch(product, "test_match_1", at)
    assert result.availability == "UNAVAILABLE"
    assert result.payload is None and result.as_of_time is None and result.reason


def test_opta_cannot_claim_unsourced_data(at):
    with pytest.raises(ValueError):
        ExternalData(product="PREDICTION", source="NEWS", retrieved_at=at, as_of_time=at,
                     availability="AVAILABLE", payload={"p_home": 0.5})
    with pytest.raises(ValueError):
        OptaSnapshot(match_id="test_match_1", source="NEWS", retrieved_at=at, as_of_time=at,
                     data_version="test", product="PREDICTION", availability="UNAVAILABLE",
                     payload={"p_home": 0.5}, reason="NO_DATA")


@pytest.mark.parametrize("model_id", tuple(REQUIREMENTS))
def test_model_requirements(store, at, model_id):
    report = SnapshotService(store).create("test_match_1", at).data_completeness
    decision = check_requirements(model_id, report)
    assert decision.execution_status == "UNAVAILABLE"
    assert decision.missing_required
    assert len(REQUIREMENTS) == 14
    assert REQUIREMENTS["OPTA_SUPERCOMPUTER_LIKE"].eligible_for_meta is False


def test_model_prediction_schema(prediction_fields):
    prediction = ModelPrediction(**prediction_fields)
    assert prediction.p_home == 0.5
    assert prediction.trained_until == prediction.training_end_time
    assert "training_end_time" in prediction.model_dump()
    assert "input_data_version" in prediction.model_dump()


@pytest.mark.parametrize("changes", [
    {"p_home": float("nan")}, {"p_home": float("inf")}, {"p_home": -0.1}, {"p_home": 1.1},
    {"p_home": 0.7}, {"p_draw": None}, {"data_status": "UNAVAILABLE"}, {"data_source": ()},
    {"execution_status": "UNAVAILABLE"}, {"implementation_type": "UNAVAILABLE"},
    {"lambda_home": -1}, {"score_matrix": ((0.5, 0.5),)}, {"score_matrix": ()},
    {"score_matrix": ((0.2,), (0.3, 0.5))},
])
def test_model_prediction_invalid(prediction_fields, changes):
    with pytest.raises(ValueError):
        ModelPrediction(**{**prediction_fields, **changes})


def test_unavailable_prediction_nulls(prediction_fields):
    unavailable = ModelPrediction(**{**prediction_fields, "p_home": None, "p_draw": None, "p_away": None,
                                      "execution_status": "UNAVAILABLE", "data_status": "UNAVAILABLE", "reason": "MISSING_XG"})
    assert unavailable.p_home is None
    with pytest.raises(ValueError):
        ModelPrediction(**{**unavailable.model_dump(), "p_home": 0, "p_draw": 0, "p_away": 0})


def test_training_time_rejected(prediction_fields, at):
    with pytest.raises(ValueError):
        ModelPrediction(**{**prediction_fields, "training_end_time": at + timedelta(days=1),
                           "trained_until": at + timedelta(days=1)})
    with pytest.raises(ValueError):
        ModelPrediction(**{**prediction_fields, "trained_until": at})


def test_core_prediction_schema(at):
    value = CorePrediction(match_id="test_match_1", prediction_snapshot_id="test_snapshot", prediction_time=at)
    assert value.final_core_probability is None
    assert value.reason == "NOT_IMPLEMENTED"
    assert {"URS", "MDI", "rating", "expected_goals", "market_edge", "market_probabilities"} <= value.model_dump().keys()
    with pytest.raises(ValueError):
        CorePrediction(**{**value.model_dump(), "final_core_probability": {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}})


def test_core_rejects_mixed_lineage(prediction_fields, at):
    prediction = ModelPrediction(**prediction_fields)
    with pytest.raises(ValueError, match="lineage"):
        CorePrediction(match_id="test_match_2", prediction_snapshot_id="test_snapshot", prediction_time=at,
                       base_model_predictions=(prediction,))
    with pytest.raises(ValueError, match="simulation engine"):
        CorePrediction(match_id="test_match_1", prediction_snapshot_id="test_snapshot", prediction_time=at,
                       base_model_predictions=(ModelPrediction(**{**prediction_fields, "model_id": "OPTA_SUPERCOMPUTER_LIKE",
                                                                 "implementation_type": "LIKE_IMPLEMENTATION"}),))


@pytest.mark.parametrize("model_id", ["SPI_LIKE", "OPTA_XG_ELO_LIKE", "OPTA_SUPERCOMPUTER_LIKE"])
def test_like_identity_is_enforced(prediction_fields, model_id):
    with pytest.raises(ValueError, match="LIKE_IMPLEMENTATION"):
        ModelPrediction(**{**prediction_fields, "model_id": model_id})


def test_prediction_persistence_requires_snapshot(store, at, prediction_fields):
    snapshot = SnapshotService(store).create("test_match_1", at)
    value = ModelPrediction(**{**prediction_fields, "prediction_snapshot_id": snapshot.prediction_snapshot_id,
                              "input_data_version": snapshot.input_data_version, "is_oos": True})
    store.save_prediction(value, prediction_id="test_model")
    store.save_prediction(value, prediction_id="test_oos", table="oos_predictions")
    with pytest.raises(ValueError, match="input version"):
        store.save_prediction(ModelPrediction(**{**value.model_dump(), "input_data_version": "wrong"}), prediction_id="bad")
    with pytest.raises(ValueError, match="OOS"):
        store.save_prediction(ModelPrediction(**{**value.model_dump(), "is_oos": False}), prediction_id="bad_oos", table="oos_predictions")
    with pytest.raises(ValueError, match="frozen snapshot"):
        store.save_prediction(ModelPrediction(**{**value.model_dump(), "match_id": "wrong"}), prediction_id="bad_match")
    store.save_prediction(CorePrediction(match_id=value.match_id, prediction_snapshot_id=value.prediction_snapshot_id,
                                         prediction_time=at), prediction_id="test_core", table="final_predictions")


def test_not_implemented_boundaries():
    for cls in (ModelPipeline, MetaStacking, Calibration):
        assert cls().run(None).status == "NOT_IMPLEMENTED"
    with pytest.raises(NotImplementedError, match="NOT_IMPLEMENTED"):
        devig_probabilities([2, 3, 4])
