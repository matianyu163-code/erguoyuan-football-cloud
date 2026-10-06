"""Chronological XGBoost/CatBoost OOS from audited NO_MARKET real base OOS."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb

from erguoyuan_football.backtesting.date_safe_oos import _metrics
from erguoyuan_football.backtesting.phase8_2_store import prepare_phase8_2_migration
from erguoyuan_football.backtesting.real_oos_matrix import lineage_hash
from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.ml.config import load_ml_config
from erguoyuan_football.ml.feature_store import MLFeatureStore
from erguoyuan_football.ml.registry import MLModelRegistry
from erguoyuan_football.ml.schemas import FeatureMode, ModelFamily
from erguoyuan_football.ml.splits import MLWalkForwardSplit
from erguoyuan_football.models.artifact_index import ArtifactIndex


@dataclass(frozen=True)
class RealMLOOSReport:
    model_id: str
    dataset_hash: str
    test_start: date
    test_end: date
    eligible_rows: int
    persisted_oos_rows: int
    unavailable_rows: int
    failed_rows: int
    metrics: dict[str, float | int]
    artifact_id: str | None


class RealMLOOSRunner:
    """One frozen train/validation/test quarter; repeats are content-idempotent."""

    def __init__(self, db_path: str | Path, *, feature_store_path: str | Path,
                 artifact_root: str | Path, config_root: str | Path) -> None:
        self.db_path = Path(db_path)
        self.feature_store_path = Path(feature_store_path)
        self.artifact_root = Path(artifact_root)
        self.config_root = Path(config_root)

    def run(self, dataset_hash: str, *, model_id: str, train_end: datetime,
            validation_end: datetime, test_end: datetime) -> RealMLOOSReport:
        with MLFeatureStore(self.feature_store_path) as store:
            dataset = store.load_dataset(dataset_hash)
        if dataset.dataset_kind != "REAL" or dataset.feature_schema.mode != FeatureMode.NO_MARKET:
            raise ValueError("ML_OOS_REQUIRES_REAL_NO_MARKET_DATASET")
        if test_end.date() >= date(2026, 8, 1):
            raise ValueError("FINAL_HOLDOUT_LOCKED")
        family = ModelFamily.XGBOOST if model_id == "CORE_XGBOOST_V1" else (
            ModelFamily.CATBOOST if model_id == "CORE_CATBOOST_V1" else None)
        if family is None or dataset.feature_schema.model_family != family:
            raise ValueError("ML_FAMILY_OR_REGISTRY_MISMATCH")
        config = load_ml_config(self.config_root /
            ("xgboost.yaml" if family == ModelFamily.XGBOOST else "catboost.yaml"),
            family=family, profile="production")
        split = MLWalkForwardSplit.explicit(dataset, train_end=train_end,
            validation_end=validation_end, test_end=test_end)
        if split.test.manifest.start_time.date() >= date(2026, 8, 1):
            raise ValueError("FINAL_HOLDOUT_LOCKED")
        model = MLModelRegistry().get_model(model_id)
        model_type = type(model)
        trained_until = max(row.label_available_at for row in (*split.train.rows,
                                                                *split.validation.rows))
        training_hash = split.train.manifest.data_hash + ":" + split.validation.manifest.data_hash
        if trained_until > min(row.vector.prediction_time for row in split.test.rows):
            raise ValueError("NESTED_ML_TRAINING_AFTER_TEST")
        run_id = hashlib.sha256(f"{model_id}|{dataset_hash}|{training_hash}|{config.config_hash}|"
                                f"{train_end}|{validation_end}|{test_end}".encode()).hexdigest()[:32]
        prepare_phase8_2_migration(self.db_path)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        with ArtifactIndex(self.artifact_root / "artifact_index.duckdb") as index:
            cached_path = index.find_ml_model(model_id, model.model_version, trained_until,
                config.config_hash, training_hash, dataset.feature_schema.schema_hash)
            if cached_path is not None:
                model = model_type.load(cached_path,
                    expected_schema=dataset.feature_schema,
                    expected_config_hash=config.config_hash)
                artifact_path = cached_path
            else:
                model.fit(split.train, split.validation, config)
                artifact_id = hashlib.sha256(f"{model_id}|{trained_until}|{training_hash}|"
                    f"{config.config_hash}".encode()).hexdigest()[:32]
                artifact_path = str(self.artifact_root / model_id / artifact_id)
                artifact = model.save(artifact_path)
                index.register_ml_model(artifact)
        artifact_id = Path(artifact_path).name
        successes: list[tuple[float, float, float, int]] = []
        failed = unavailable = 0
        with duckdb.connect(str(self.db_path)) as connection:
            connection.execute("""INSERT INTO real_oos_run_manifest
                (run_id,model_ids,start_date,end_date,run_status,config_hash)
                VALUES (?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET run_status='RUNNING'""",
                [run_id, json.dumps([model_id]), split.test.manifest.start_time.date(),
                 split.test.manifest.end_time.date(), "RUNNING", config.config_hash])
            match_meta = {row[0]: row[1:] for row in connection.execute("""
                SELECT match_id,competition_id,season_id,match_date FROM real_canonical_matches
                WHERE match_id IN (SELECT match_id FROM real_canonical_matches
                    WHERE match_date BETWEEN ? AND ?)
            """, [split.test.manifest.start_time.date(),
                   split.test.manifest.end_time.date()]).fetchall()}
            for row in split.test.rows:
                vector = row.vector
                if vector.match_id in model.training_match_ids or model.trained_until is None or (
                        model.trained_until > vector.prediction_time):
                    raise ValueError("NESTED_OOS_TARGET_IN_TRAINING_OR_FUTURE_ARTIFACT")
                if any(item.source_type == "MARKET" for item in vector.feature_lineage):
                    raise ValueError("NO_MARKET_LINEAGE_VIOLATION")
                existing = connection.execute("""SELECT payload,config_hash FROM real_oos_predictions_v2
                    WHERE match_id=? AND model_id=?""", [vector.match_id, model_id]).fetchone()
                if existing is not None:
                    prior = ModelPrediction.model_validate_json(existing[0])
                    if (prior.metadata.get("feature_data_hash") != vector.feature_data_hash or
                            existing[1] != config.config_hash):
                        raise ValueError("CONFLICTING_ML_OOS_IDENTITY")
                    assert prior.p_home is not None and prior.p_draw is not None and prior.p_away is not None
                    successes.append((prior.p_home, prior.p_draw, prior.p_away, int(row.label)))
                    continue
                prediction = model.predict(vector, oos=True)
                if prediction.execution_status != ExecutionStatus.SUCCESS:
                    status = prediction.execution_status.value
                    if status == "UNAVAILABLE":
                        unavailable += 1
                    else:
                        failed += 1
                    connection.execute("""INSERT INTO real_oos_execution_status VALUES
                        (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(match_id,model_id) DO UPDATE SET
                        execution_status=excluded.execution_status,reason=excluded.reason,
                        updated_at=excluded.updated_at""", [vector.match_id, model_id,
                        vector.prediction_time.date(), "DATE_SAFE_BATCH", status,
                        prediction.reason, None, training_hash, config.config_hash,
                        run_id, datetime.now(UTC)])
                    continue
                pid = hashlib.sha256(f"{model_id}|{vector.match_id}|{vector.feature_data_hash}|"
                    f"{training_hash}|{config.config_hash}".encode()).hexdigest()[:32]
                digest = lineage_hash((pid, artifact_id, vector.feature_data_hash,
                    tuple(item.prediction.prediction_id for item in vector.base_prediction_evidence),
                    training_hash, config.config_hash))
                metadata = {**prediction.metadata, "data_origin": "REAL",
                    "prediction_temporal_mode": "DATE_SAFE_BATCH",
                    "match_date": vector.prediction_time.date().isoformat(),
                    "artifact_id": artifact_id, "lineage_hash": digest,
                    "feature_data_hash": vector.feature_data_hash,
                    "base_prediction_ids": tuple(item.prediction.prediction_id
                                                 for item in vector.base_prediction_evidence),
                    "training_match_count": len(model.training_match_ids),
                    "training_data_hash": training_hash}
                prediction = ModelPrediction.model_validate({**prediction.model_dump(),
                    "prediction_id": pid, "metadata": metadata})
                assert prediction.p_home is not None and prediction.p_draw is not None and prediction.p_away is not None
                comp, season, match_date = match_meta[vector.match_id]
                connection.execute("""INSERT INTO real_oos_predictions_v2
                    (prediction_id,match_id,competition_id,season_id,model_id,model_version,
                     training_cutoff,prediction_time,prediction_temporal_mode,kickoff_time_if_known,
                     match_date,p_home,p_draw,p_away,is_oos,data_origin,training_match_count,
                     training_data_hash,config_hash,prediction_snapshot_id,timestamp_precision,payload)
                    VALUES (?,?,?,?,?,?,?,?,?,NULL,?,?,?,?,TRUE,'REAL',?,?,?,?,?,?)""",
                    [pid, vector.match_id, comp, season, model_id, model.model_version,
                     model.trained_until, vector.prediction_time, "DATE_SAFE_BATCH", match_date,
                     prediction.p_home, prediction.p_draw, prediction.p_away,
                     len(model.training_match_ids), training_hash, config.config_hash,
                     vector.prediction_snapshot_id, "DATE_SAFE_BATCH",
                     prediction.model_dump_json()])
                connection.execute("""INSERT INTO real_oos_lineage_v2 VALUES
                    (?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
                    [pid, artifact_id, digest,
                     prediction.metadata["training_match_ids_hash"], "NO_MARKET",
                     json.dumps(metadata["base_prediction_ids"]), datetime.now(UTC)])
                connection.execute("""INSERT INTO real_oos_execution_status VALUES
                    (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(match_id,model_id) DO UPDATE SET
                    execution_status=excluded.execution_status,prediction_id=excluded.prediction_id,
                    updated_at=excluded.updated_at""", [vector.match_id, model_id,
                    match_date, "DATE_SAFE_BATCH", "SUCCESS", None, pid,
                    training_hash, config.config_hash, run_id, datetime.now(UTC)])
                successes.append((prediction.p_home, prediction.p_draw,
                                  prediction.p_away, int(row.label)))
            actual = connection.execute("""SELECT count(*) FILTER (WHERE execution_status='SUCCESS'),
                count(*) FILTER (WHERE execution_status='UNAVAILABLE'),
                count(*) FILTER (WHERE execution_status='FAILED')
                FROM real_oos_execution_status WHERE run_id=?""", [run_id]).fetchone()
            assert actual is not None
            connection.execute("""UPDATE real_oos_run_manifest SET
                current_date=?,completed_rows=?,unavailable_rows=?,failed_rows=?,
                artifact_count=1,last_checkpoint=?,run_status='COMPLETE' WHERE run_id=?""",
                [split.test.manifest.end_time.date(), *actual,
                 datetime.now(UTC), run_id])
        return RealMLOOSReport(model_id, dataset_hash,
            split.test.manifest.start_time.date(), split.test.manifest.end_time.date(),
            len(split.test.rows), len(successes), unavailable, failed,
            _metrics(successes), artifact_id)
