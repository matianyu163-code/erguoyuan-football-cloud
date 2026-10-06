"""Nested real ML windows, artifact reuse and compact ancestry checks."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from erguoyuan_football.backtesting.real_ml_oos import RealMLOOSRunner
from erguoyuan_football.ml.artifacts import MLArtifact
from erguoyuan_football.ml.feature_lineage import OOSFeatureValidator
from erguoyuan_football.ml.feature_store import MLFeatureStore
from erguoyuan_football.ml.schemas import ModelFamily
from erguoyuan_football.models.artifact_index import ArtifactIndex

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "football.duckdb"
FEATURE_DB = ROOT / "data" / "phase8_2_ml_features_compact.duckdb"
ARTIFACT_ROOT = ROOT / "artifacts" / "phase8_2"


@pytest.mark.parametrize("model_id,family", [
    ("CORE_XGBOOST_V1", ModelFamily.XGBOOST),
    ("CORE_CATBOOST_V1", ModelFamily.CATBOOST),
])
def test_real_ml_walk_forward_and_resume(tmp_path, model_id, family) -> None:
    temporary_db = tmp_path / "football.duckdb"
    shutil.copy2(DB, temporary_db)
    with MLFeatureStore(FEATURE_DB) as store:
        candidates = [row[0] for row in store.connection.execute(
            "SELECT data_hash FROM ml_datasets WHERE row_count=2070").fetchall()]
        dataset_hash = next(item for item in candidates if
            store.load_dataset(item).feature_schema.model_family == family)
    with duckdb.connect(str(DB), read_only=True) as connection:
        prediction_id = connection.execute("""SELECT prediction_id FROM real_oos_predictions_v2
            WHERE model_id=? AND match_date BETWEEN '2026-01-01' AND '2026-03-31'
            LIMIT 1""", [model_id]).fetchone()[0]
        artifact_id = connection.execute("""SELECT artifact_id FROM real_oos_lineage_v2
            WHERE prediction_id=?""", [prediction_id]).fetchone()[0]
    source_artifact = ARTIFACT_ROOT / model_id / artifact_id
    manifest = MLArtifact.model_validate_json((source_artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.trained_until < datetime(2026, 1, 1, tzinfo=UTC)
    (tmp_path / "artifacts").mkdir()
    with ArtifactIndex(tmp_path / "artifacts" / "artifact_index.duckdb") as index:
        index.register_ml_model(manifest)
    runner = RealMLOOSRunner(temporary_db, feature_store_path=FEATURE_DB,
        artifact_root=tmp_path / "artifacts", config_root=ROOT / "config")
    kwargs = {"model_id": model_id, "train_end": datetime(2025, 11, 1, tzinfo=UTC),
              "validation_end": datetime(2026, 1, 1, tzinfo=UTC),
              "test_end": datetime(2026, 4, 1, tzinfo=UTC)}
    first = runner.run(dataset_hash, **kwargs)
    second = runner.run(dataset_hash, **kwargs)
    assert first.persisted_oos_rows == second.persisted_oos_rows == 582
    with duckdb.connect(str(temporary_db), read_only=True) as connection:
        count = connection.execute("SELECT count(*) FROM real_oos_predictions_v2 "
                                   "WHERE model_id=?", [model_id]).fetchone()[0]
        test_ids = {row[0] for row in connection.execute("""SELECT match_id
            FROM real_oos_predictions_v2 WHERE model_id=?
            AND match_date BETWEEN '2026-01-01' AND '2026-03-31'""", [model_id]).fetchall()}
    assert count == 1472
    assert not test_ids.intersection(manifest.training_match_ids)


def test_compact_ancestry_hash_tamper_rejected() -> None:
    with MLFeatureStore(FEATURE_DB) as store:
        payload = store.connection.execute("""SELECT payload FROM ml_feature_vectors
            WHERE model_family='XGBOOST' LIMIT 1""").fetchone()[0]
    from erguoyuan_football.ml.schemas import MLFeatureVector

    vector = MLFeatureVector.model_validate_json(payload)
    evidence = vector.base_prediction_evidence[0].model_copy(
        update={"verified_training_ids_hash": "wrong"})
    tampered = vector.model_copy(update={"base_prediction_evidence": (
        evidence, *vector.base_prediction_evidence[1:])})
    with pytest.raises(ValueError, match="BASE_TRAINING_MATCH_IDS_HASH_MISMATCH"):
        OOSFeatureValidator().validate_training_vector(tampered)


def test_final_holdout_runner_rejected(tmp_path) -> None:
    runner = RealMLOOSRunner(DB, feature_store_path=FEATURE_DB,
        artifact_root=tmp_path, config_root=ROOT / "config")
    with MLFeatureStore(FEATURE_DB) as store:
        dataset_hash = store.connection.execute("SELECT data_hash FROM ml_datasets "
                                                "WHERE row_count=2070 LIMIT 1").fetchone()[0]
    with pytest.raises(ValueError, match="FINAL_HOLDOUT_LOCKED"):
        runner.run(dataset_hash, model_id="CORE_XGBOOST_V1",
            train_end=datetime(2026, 2, 1, tzinfo=UTC),
            validation_end=datetime(2026, 4, 1, tzinfo=UTC),
            test_end=datetime(2026, 9, 1, tzinfo=UTC))
