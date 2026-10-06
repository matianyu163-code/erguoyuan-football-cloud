"""Frozen snapshot -> executed base model -> OOS ML feature -> two real estimators."""

from __future__ import annotations

import pytest

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.ml.catboost_model import CoreCatBoostModel
from erguoyuan_football.ml.comparison import compare_ml_predictions
from erguoyuan_football.ml.feature_builder import MLFeatureBuilder
from erguoyuan_football.ml.feature_store import MLFeatureStore
from erguoyuan_football.ml.inference import MLModelRunner
from erguoyuan_football.ml.schemas import (
    BasePredictionEvidence,
    FeatureMode,
    ModelFamily,
)
from erguoyuan_football.ml.xgboost_model import CoreXGBoostModel
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.elo import CoreEloModel
from erguoyuan_football.models.training import TrainingDataset, TrainingMatch

pytestmark = pytest.mark.integration


def test_snapshot_oos_elo_feature_to_two_ml_models(ml_dataset_factory, synthetic_history, tmp_path) -> None:
    snapshots, results = synthetic_history
    target = snapshots[65]
    rows = []
    for snapshot in snapshots[:65]:
        fixture = snapshot.match_data_snapshot
        result = results[fixture.match_id]
        rows.append(TrainingMatch(match_id=fixture.match_id,
            competition_id=fixture.competition_id, season=fixture.season or "2025",
            kickoff_time=fixture.kickoff_time, home_team_id=fixture.home_team_id,
            away_team_id=fixture.away_team_id, home_goals=result.home_goals,
            away_goals=result.away_goals, neutral_venue=False, source="SYNTHETIC_TEST",
            completed_at=result.completed_at, as_of_time=result.as_of_time,
            retrieved_at=result.retrieved_at, data_version=result.data_version))
    training = TrainingDataset(matches=tuple(rows),
        known_team_ids=frozenset({team for row in rows for team in (row.home_team_id, row.away_team_id)}),
        dataset_kind="SYNTHETIC_TEST")
    cutoff = max(row.available_at for row in rows)
    elo = CoreEloModel().fit(training, cutoff,
        ModelConfig(min_matches=20, min_team_matches=2, allow_test_data=True))
    base_prediction = elo.predict(target.match_data_snapshot, target)
    assert base_prediction.execution_status == ExecutionStatus.SUCCESS
    assert target.match_id not in elo.training_ids
    assert cutoff < target.prediction_time
    evidence = BasePredictionEvidence(prediction=base_prediction.model_copy(update={"is_oos": True}),
        training_match_ids=tuple(sorted(elo.training_ids)), base_prediction_oos=True)
    builder = MLFeatureBuilder()
    with pytest.raises(ValueError, match="BASE_TRAINING_MATCH_IDS_HASH_MISMATCH"):
        builder.build(target, (evidence.model_copy(update={"training_match_ids": ("not_the_training_set",)}),),
            mode=FeatureMode.NO_MARKET, family=ModelFamily.XGBOOST, for_training=True)
    vectors = {family: builder.build(target, (evidence,), mode=FeatureMode.NO_MARKET,
               family=family, for_training=True) for family in ModelFamily}
    assert all(vector.features["elo_available"] == 1 for vector in vectors.values())
    with MLFeatureStore(tmp_path / "ml.duckdb") as store:
        store.append_many(tuple(vectors.values()))
        assert all(store.load_vector(vector.feature_data_hash) == vector for vector in vectors.values())
    _, xgb_split, xgb_config = ml_dataset_factory(family=ModelFamily.XGBOOST)
    _, cat_split, cat_config = ml_dataset_factory(family=ModelFamily.CATBOOST)
    models = {
        CoreXGBoostModel.model_id: CoreXGBoostModel().fit(xgb_split.train, xgb_split.validation, xgb_config),
        CoreCatBoostModel.model_id: CoreCatBoostModel().fit(cat_split.train, cat_split.validation, cat_config),
    }
    bundle = MLModelRunner().run(vectors, fitted_models=models, oos=True)
    assert len(bundle.predictions) == 2
    assert {row.model_id for row in bundle.predictions} == set(models)
    assert all(row.execution_status == ExecutionStatus.SUCCESS and row.is_oos for row in bundle.predictions)
    comparison = compare_ml_predictions(bundle.predictions[0], bundle.predictions[1],
                                        best_base=base_prediction)
    assert comparison.diagnostic_only
    assert comparison.xgb_cat_probability_distance is not None
    assert 0 <= comparison.xgb_cat_probability_distance <= 1


def test_in_sample_base_prediction_rejected(synthetic_history) -> None:
    snapshots, _ = synthetic_history
    target = snapshots[0]
    prediction = CoreEloModel().prediction_record(
        target.match_data_snapshot, target, ExecutionStatus.UNAVAILABLE, reason="TEST")
    evidence = BasePredictionEvidence(prediction=prediction,
        training_match_ids=(target.match_id,), base_prediction_oos=False)
    with pytest.raises(ValueError, match="IN_SAMPLE_BASE_PREDICTION_REJECTED"):
        MLFeatureBuilder().build(target, (evidence,), mode=FeatureMode.NO_MARKET,
            family=ModelFamily.XGBOOST, for_training=True)
