"""Real-only prediction matrix, nested ML and candidate-data audits."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest
import yaml

from erguoyuan_football.backtesting.phase8_2_audit import audit_phase8_2
from erguoyuan_football.backtesting.phase9_candidate import Phase9CandidateBuilder
from erguoyuan_football.backtesting.real_oos_matrix import RealOOSPredictionMatrix
from erguoyuan_football.ml.feature_lineage import OOSFeatureValidator
from erguoyuan_football.ml.feature_store import MLFeatureStore
from erguoyuan_football.ml.schemas import FeatureMode, ModelFamily

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "football.duckdb"
FEATURE_STORE = ROOT / "data" / "phase8_2_ml_features_compact.duckdb"


@pytest.fixture(scope="module")
def common_matrix():
    return RealOOSPredictionMatrix(DB).build(date(2025, 10, 1), date(2025, 12, 31))


@pytest.fixture(scope="module")
def ml_datasets():
    with MLFeatureStore(FEATURE_STORE) as store:
        hashes = [row[0] for row in store.connection.execute(
            "SELECT data_hash FROM ml_datasets WHERE row_count=2070").fetchall()]
        datasets = [store.load_dataset(item) for item in hashes]
    return {item.feature_schema.model_family: item for item in datasets}


def test_real_oos_matrix(common_matrix) -> None:
    assert len(common_matrix) == 523
    assert all(row.target in {0, 1, 2} and row.temporal_mode == "DATE_SAFE_BATCH"
               for row in common_matrix)


def test_matrix_only_real_and_oos(common_matrix) -> None:
    ids = [pid for row in common_matrix for pid in row.prediction_ids.values() if pid]
    with duckdb.connect(str(DB), read_only=True) as connection:
        actual, bad = connection.execute("""SELECT count(*),count(*) FILTER
            (WHERE data_origin<>'REAL' OR NOT is_oos) FROM real_oos_predictions_v2
            WHERE prediction_id IN (SELECT prediction_id FROM real_oos_predictions_v2
                WHERE match_date BETWEEN '2025-10-01' AND '2025-12-31')""").fetchone()
    assert ids and actual >= len(set(ids)) and bad == 0


def test_missing_not_uniform() -> None:
    rows = RealOOSPredictionMatrix(DB).build(date(2022, 8, 1), date(2022, 8, 31))
    missing = next(row for row in rows if row.availability["ELO_V1"] == 0)
    assert missing.probabilities["ELO_V1"] is None
    assert missing.feature_values()["elo_home"] is None
    assert missing.feature_values()["elo_available"] == 0


def test_availability_mask(common_matrix) -> None:
    row = common_matrix[0]
    assert row.availability["DIXON_COLES_V1"] == 1
    assert row.availability["CORE_XGBOOST_V1"] == 1
    assert row.availability["CORE_CATBOOST_V1"] == 1
    assert row.availability["BAYESIAN_HIERARCHICAL_V1"] == 0


def test_prediction_lineage(common_matrix) -> None:
    row = common_matrix[0]
    for model_id in ("DIXON_COLES_V1", "ELO_V1", "CORE_XGBOOST_V1"):
        assert row.prediction_ids[model_id] and row.lineage_hashes[model_id]
        with duckdb.connect(str(DB), read_only=True) as connection:
            match_id, stored_model = connection.execute("""SELECT match_id,model_id
                FROM real_oos_predictions_v2 WHERE prediction_id=?""",
                [row.prediction_ids[model_id]]).fetchone()
        assert (match_id, stored_model) == (row.match_id, model_id)


def test_common_sample_same_matches(common_matrix) -> None:
    models = ("DIXON_COLES_V1", "BIVARIATE_POISSON_V1", "ELO_V1",
              "PI_RATING_V1", "CORE_SPI_LIKE_V1")
    common = RealOOSPredictionMatrix.common_sample(common_matrix, models)
    assert len(common.match_ids) == 523
    assert len(set(common.match_ids)) == 523
    assert all(metric["sample_size"] == 523 for metric in common.metrics.values())


def test_common_sample_metrics(common_matrix) -> None:
    models = ("DIXON_COLES_V1", "BIVARIATE_POISSON_V1", "ELO_V1")
    common = RealOOSPredictionMatrix.common_sample(common_matrix, models)
    assert all(metric["log_loss"] > 0 and 0 <= metric["accuracy"] <= 1
               for metric in common.metrics.values())


def test_model_coverage_difference_reported() -> None:
    coverage = RealOOSPredictionMatrix(DB).coverage(date(2025, 10, 1),
        date(2026, 6, 30), model_ids=("DIXON_COLES_V1", "CORE_SPI_LIKE_V1"))
    assert coverage
    assert any(item.unavailable_rows > 0 for item in coverage if item.model_id == "CORE_SPI_LIKE_V1")
    assert all(item.eligible_matches == item.predicted_rows + item.unavailable_rows +
               item.failed_rows + item.not_run_rows for item in coverage)


def test_ml_real_oos_features_only(ml_datasets) -> None:
    assert set(ml_datasets) == {ModelFamily.XGBOOST, ModelFamily.CATBOOST}
    for dataset in ml_datasets.values():
        assert dataset.dataset_kind == "REAL" and dataset.manifest.row_count == 2070
        for row in dataset.rows[:20]:
            assert row.vector.base_prediction_evidence
            for evidence in row.vector.base_prediction_evidence:
                assert evidence.prediction.is_oos
                assert evidence.prediction.metadata["data_origin"] == "REAL"
                assert evidence.verification_method == "REAL_OOS_STORE_RECOMPUTED"
                assert evidence.verified_training_match_count > 0
            OOSFeatureValidator().validate_training_vector(row.vector)


def test_ml_no_market_purity(ml_datasets) -> None:
    for dataset in ml_datasets.values():
        assert dataset.feature_schema.mode == FeatureMode.NO_MARKET
        assert all("market" not in name.lower() and "target" not in name.lower()
                   for name in dataset.feature_schema.feature_names)
        assert all(item.source_type != "MARKET" for row in dataset.rows
                   for item in row.vector.feature_lineage)


def test_nested_base_oos(ml_datasets) -> None:
    for row in ml_datasets[ModelFamily.XGBOOST].rows[:100]:
        for evidence in row.vector.base_prediction_evidence:
            assert evidence.prediction.training_end_time <= row.vector.prediction_time
            assert evidence.verified_training_match_count > 0
            assert evidence.prediction.prediction_time.date() == row.vector.prediction_time.date()


def test_target_not_in_ml_training(ml_datasets) -> None:
    xgb = ml_datasets[ModelFamily.XGBOOST]
    train = {row.vector.match_id for row in xgb.rows if row.vector.prediction_time <
             datetime(2025, 8, 1, tzinfo=UTC)}
    validation = {row.vector.match_id for row in xgb.rows if
                  datetime(2025, 8, 1, tzinfo=UTC) <= row.vector.prediction_time <
                  datetime(2025, 10, 1, tzinfo=UTC)}
    test = {row.vector.match_id for row in xgb.rows if row.vector.prediction_time >=
            datetime(2025, 10, 1, tzinfo=UTC)}
    assert train and validation and test
    assert not train & test and not validation & test


@pytest.mark.parametrize("model_id", ["CORE_XGBOOST_V1", "CORE_CATBOOST_V1"])
def test_ml_real_oos_and_artifact_cutoff(model_id) -> None:
    with duckdb.connect(str(DB), read_only=True) as connection:
        n, bad = connection.execute("""SELECT count(*),count(*) FILTER
            (WHERE data_origin<>'REAL' OR NOT is_oos OR training_cutoff>prediction_time)
            FROM real_oos_predictions_v2 WHERE model_id=?""", [model_id]).fetchone()
        first = connection.execute("""SELECT prediction_id FROM real_oos_predictions_v2
            WHERE model_id=? LIMIT 1""", [model_id]).fetchone()
        artifact = connection.execute("""SELECT artifact_id FROM real_oos_lineage_v2
            WHERE prediction_id=?""", [first[0]]).fetchone()[0]
    assert n == 1472 and bad == 0 and artifact


def test_final_holdout_locked() -> None:
    config = yaml.safe_load((ROOT / "config" / "evaluation_windows.yaml").read_text(encoding="utf-8"))
    assert config["final_holdout"]["start"] == date(2026, 8, 1)
    assert config["final_holdout"]["status"] == "LOCKED_UNTOUCHED_BY_PHASE8_2"


def test_holdout_not_ml_train(ml_datasets) -> None:
    assert all(row.vector.prediction_time.date() < date(2026, 8, 1)
               for dataset in ml_datasets.values() for row in dataset.rows)


def test_holdout_not_tuning() -> None:
    with duckdb.connect(str(DB), read_only=True) as connection:
        count = connection.execute("""SELECT count(*) FROM real_oos_predictions_v2
            WHERE model_id IN ('CORE_XGBOOST_V1','CORE_CATBOOST_V1')
              AND match_date >= '2026-08-01'""").fetchone()[0]
    assert count == 0


def test_phase9_candidate_dataset_no_target_feature() -> None:
    with duckdb.connect(str(DB), read_only=True) as connection:
        location, total = connection.execute("""SELECT artifact_path,row_count
            FROM phase9_candidate_dataset_v2 ORDER BY created_at DESC LIMIT 1""").fetchone()
    folder = Path(location)
    assert total >= 6895
    with duckdb.connect(":memory:") as connection:
        feature_count = connection.execute("SELECT count(*) FROM read_parquet(?)",
                                           [str(folder / "features.parquet")]).fetchone()[0]
        columns = [item[0] for item in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)",
                                                 [str(folder / "features.parquet")]).fetchall()]
        label_columns = [item[0] for item in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)",
                                                 [str(folder / "labels.parquet")]).fetchall()]
    assert feature_count == total
    assert "target" not in columns and all("market" not in name.lower() for name in columns)
    assert label_columns == ["match_id", "target"]


def test_final_holdout_build_rejected(tmp_path) -> None:
    builder = Phase9CandidateBuilder(DB, output_root=tmp_path)
    with pytest.raises(ValueError, match="FINAL_HOLDOUT_LOCKED"):
        builder.build(date(2026, 8, 1), date(2026, 9, 1))


def test_phase8_2_real_lineage_self_review() -> None:
    report = audit_phase8_2(DB, artifact_root=ROOT / "artifacts" / "phase8_2")
    assert report.status == "PASS_WITH_LIMITATIONS"
    assert report.base_lineage_checked == 600
    assert report.nested_ml_lineage_checked == 200
    assert report.market_oos_rows == report.holdout_oos_rows == 0
