"""Official boosting libraries train and predict from real executed code."""

import json

import numpy as np
import pytest

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.ml.artifacts import current_code_version
from erguoyuan_football.ml.catboost_model import CoreCatBoostModel
from erguoyuan_football.ml.class_order import ProbabilityClassOrderGuard
from erguoyuan_football.ml.inference import MLModelRunner
from erguoyuan_football.ml.labels import CLASS_MAPPING
from erguoyuan_football.ml.schemas import ModelFamily
from erguoyuan_football.ml.xgboost_model import CoreXGBoostModel
from erguoyuan_football.models.registry import ModelRegistry

pytestmark = pytest.mark.model


@pytest.mark.parametrize(("family", "cls"), [
    (ModelFamily.XGBOOST, CoreXGBoostModel),
    (ModelFamily.CATBOOST, CoreCatBoostModel),
])
def test_real_ml_fit_predict_many_and_save_load(ml_dataset_factory, tmp_path, family, cls) -> None:
    _, split, config = ml_dataset_factory(family=family)
    model = cls().fit(split.train, split.validation, config)
    assert model.fitted
    assert model.training_metadata["calibration_status"] == "NOT_APPLIED"
    assert model.training_metadata["class_mapping"] == CLASS_MAPPING
    vectors = tuple(row.vector for row in split.test.rows[:3])
    predictions = model.predict_many(vectors, oos=True)
    assert len(predictions) == 3
    assert all(item.execution_status == ExecutionStatus.SUCCESS for item in predictions)
    assert all(abs(item.p_home + item.p_draw + item.p_away - 1) < 1e-6 for item in predictions)
    assert all(item.lambda_home is None and item.score_matrix is None and item.is_oos for item in predictions)
    artifact = model.save(tmp_path / family.value)
    loaded = cls.load(tmp_path / family.value,
                      expected_schema=split.train.feature_schema, expected_config_hash=config.config_hash)
    after = loaded.predict_many(vectors, oos=True)
    assert artifact.library_version != "NOT_INSTALLED"
    assert [(item.p_home, item.p_draw, item.p_away) for item in predictions] == pytest.approx(
        [(item.p_home, item.p_draw, item.p_away) for item in after], abs=1e-8)


def test_probability_class_order_guard() -> None:
    probabilities = np.asarray([[0.2, 0.5, 0.3]])
    ordered = ProbabilityClassOrderGuard.reorder(probabilities, (2, 0, 1))
    assert tuple(ordered[0]) == pytest.approx((0.5, 0.3, 0.2))
    with pytest.raises(ValueError, match="ML_CLASS_MAPPING_MISMATCH"):
        ProbabilityClassOrderGuard.reorder(probabilities, (0, 1, 3))


def test_feature_schema_mismatch_fails_closed(ml_dataset_factory) -> None:
    _, split, config = ml_dataset_factory(family=ModelFamily.XGBOOST)
    model = CoreXGBoostModel().fit(split.train, split.validation, config)
    vector = split.test.rows[0].vector.model_copy(update={"feature_schema_hash": "wrong"})
    result = model.predict(vector)
    assert result.execution_status == ExecutionStatus.FAILED
    assert result.failure_code == "FEATURE_SCHEMA_MISMATCH"


def test_early_stopping_never_sees_test(ml_dataset_factory) -> None:
    _, split, config = ml_dataset_factory()
    model = CoreXGBoostModel().fit(split.train, split.validation, config)
    assert not any(row.vector.match_id in model.training_match_ids for row in split.test.rows)
    assert model.trained_until < min(row.vector.prediction_time for row in split.test.rows)
    assert model.training_metadata["best_iteration"] is not None


def test_future_cutoff_returns_audited_failure(ml_dataset_factory) -> None:
    _, split, config = ml_dataset_factory()
    model = CoreXGBoostModel().fit(split.train, split.validation, config)
    result = model.predict(split.train.rows[0].vector)
    assert result.execution_status == ExecutionStatus.FAILED
    assert result.failure_code == "TRAINING_CUTOFF_AFTER_PREDICTION"
    assert result.p_home is None


def test_prediction_rechecks_feature_content_hash(ml_dataset_factory) -> None:
    _, split, config = ml_dataset_factory()
    model = CoreXGBoostModel().fit(split.train, split.validation, config)
    vector = split.test.rows[0].vector
    features = {**vector.features, "horizon_seconds": 9999.0}
    tampered = vector.model_copy(update={"features": features})
    result = model.predict(tampered)
    assert result.execution_status == ExecutionStatus.FAILED
    assert "FEATURE_DATA_HASH_MISMATCH" in result.reason


def test_artifact_integrity_guard(ml_dataset_factory, tmp_path) -> None:
    _, split, config = ml_dataset_factory()
    model = CoreXGBoostModel().fit(split.train, split.validation, config)
    directory = tmp_path / "model"
    model.save(directory)
    with (directory / "model.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="ML_ARTIFACT_CHECKSUM_MISMATCH"):
        CoreXGBoostModel.load(directory)
    model.save(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["code_version"] == current_code_version()
    invalid_code = {**manifest, "code_version": "WRONG_CODE"}
    (directory / "manifest.json").write_text(json.dumps(invalid_code), encoding="utf-8")
    with pytest.raises(ValueError, match="ML_ARTIFACT_VERSION_SCHEMA_OR_CONFIG_MISMATCH"):
        CoreXGBoostModel.load(directory)
    manifest["class_mapping"]["DRAW"] = 2
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="ML_ARTIFACT_SCHEMA_OR_CLASS_MISMATCH"):
        CoreXGBoostModel.load(directory)


def test_runner_isolates_unavailable_catboost(ml_dataset_factory) -> None:
    _, split, config = ml_dataset_factory()
    xgb = CoreXGBoostModel().fit(split.train, split.validation, config)
    vectors = {ModelFamily.XGBOOST: split.test.rows[0].vector}
    bundle = MLModelRunner().run(vectors, fitted_models={xgb.model_id: xgb})
    assert [row.execution_status for row in bundle.predictions] == [
        ExecutionStatus.SUCCESS, ExecutionStatus.UNAVAILABLE]


def test_synthetic_model_blocked_from_production(ml_dataset_factory) -> None:
    _, split, config = ml_dataset_factory()
    model = CoreXGBoostModel().fit(split.train, split.validation, config)
    result = model.predict(split.test.rows[0].vector, production=True,
        network_gate_status="PASS", market_gate_status="PASS")
    assert result.execution_status == ExecutionStatus.UNAVAILABLE
    assert result.reason == "SYNTHETIC_DATA_FORBIDDEN_IN_PRODUCTION"


def test_combined_registry_exposes_eleven_without_running_models() -> None:
    registry = ModelRegistry()
    assert len(registry.list_all_models()) == 11
    assert isinstance(registry.get_ml_model("CORE_XGBOOST_V1"), CoreXGBoostModel)
    assert isinstance(registry.get_ml_model("CORE_CATBOOST_V1"), CoreCatBoostModel)


def test_wrappers_refuse_non_probability_objectives(ml_dataset_factory) -> None:
    _, _, xgb_config = ml_dataset_factory(family=ModelFamily.XGBOOST)
    bad_xgb = xgb_config.model_copy(update={"params": {**xgb_config.params,
                                                        "objective": "reg:squarederror"}})
    with pytest.raises(ValueError, match="MULTICLASS_PROBABILITY_OBJECTIVE"):
        CoreXGBoostModel()._new_estimator(bad_xgb)
    _, _, cat_config = ml_dataset_factory(family=ModelFamily.CATBOOST)
    bad_cat = cat_config.model_copy(update={"params": {**cat_config.params,
                                                        "loss_function": "RMSE"}})
    with pytest.raises(ValueError, match="MULTICLASS_OBJECTIVE"):
        CoreCatBoostModel()._new_estimator(bad_cat)
