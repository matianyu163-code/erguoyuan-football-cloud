"""Phase 7 NO_MARKET feature contracts backed by audited real date-safe OOS."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import duckdb

from erguoyuan_football.backtesting.phase8_2_store import prepare_phase8_2_migration
from erguoyuan_football.backtesting.real_oos_matrix import BASE_MODEL_IDS
from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.ml.dataset_builder import MLDatasetBuilder
from erguoyuan_football.ml.feature_contract import MODEL_PREFIXES, build_feature_schema
from erguoyuan_football.ml.feature_lineage import OOSFeatureValidator
from erguoyuan_football.ml.feature_store import MLFeatureStore
from erguoyuan_football.ml.schemas import (
    BasePredictionEvidence,
    FeatureLineage,
    FeatureMode,
    FeatureSchema,
    MLDataset,
    MLFeatureVector,
    ModelFamily,
    stable_hash,
)

FEATURE_VERSION = "ML_FEATURE_V1_DATE_SAFE_BATCH"


def date_safe_schema(family: ModelFamily) -> FeatureSchema:
    """Retain Phase 7 columns; mark unknown exact kickoff horizon nullable."""
    original = build_feature_schema(FeatureMode.NO_MARKET, family)
    return FeatureSchema.model_validate({**original.model_dump(),
        "schema_version": "ML_FEATURE_SCHEMA_V1_DATE_SAFE_BATCH",
        "nullable_features": (*original.nullable_features, "horizon_seconds")})


class RealDateSafeMLDatasetBuilder:
    """Audit each OOS ancestor before using it as a tree-model feature."""

    def __init__(self, db_path: str | Path, *, feature_store_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.feature_store_path = str(feature_store_path)

    def build(self, start: date, end: date, *, family: ModelFamily,
              evidence_models: tuple[str, ...] = ("DIXON_COLES_V1", "ELO_V1"),
              max_rows: int | None = None) -> MLDataset:
        if end < start or end >= date(2026, 8, 1):
            raise ValueError("ML_DATASET_WINDOW_OR_HOLDOUT_INVALID")
        if not evidence_models or any(mid not in BASE_MODEL_IDS for mid in evidence_models):
            raise ValueError("ML_REQUIRES_REAL_BASE_MODEL_IDS")
        schema = date_safe_schema(family)
        with duckdb.connect(self.db_path, read_only=True) as connection:
            matches = connection.execute("""
                SELECT match_id,competition_id,season_id,match_date,
                       home_goals,away_goals,raw_hash
                FROM real_canonical_matches
                WHERE status='FINISHED' AND match_date BETWEEN ? AND ?
                ORDER BY match_date,match_id
            """, [start, end]).fetchall()
            selected = matches if max_rows is None else matches[:max_rows]
            if not selected:
                raise ValueError("NO_REAL_ML_MATCHES")
            prediction_rows = connection.execute("""
                SELECT match_id,model_id,payload,training_cutoff,
                       training_match_count,training_data_hash,competition_id
                FROM real_oos_predictions_v2
                WHERE match_date BETWEEN ? AND ? AND data_origin='REAL' AND is_oos
            """, [start, selected[-1][3]]).fetchall()
            predictions: dict[str, dict[str, tuple[Any, ...]]] = defaultdict(dict)
            wanted = {row[0] for row in selected}
            for match_id, model_id, *rest in prediction_rows:
                if match_id not in wanted or model_id not in evidence_models:
                    continue
                if model_id in predictions[match_id]:
                    raise ValueError("AMBIGUOUS_BASE_OOS_FEATURE")
                predictions[match_id][model_id] = tuple(rest)
            history_ids: dict[tuple[str, date], tuple[str, ...]] = {}
            vectors: list[MLFeatureVector] = []
            results: dict[str, tuple[int, int, date, str]] = {}
            comp_ids: dict[str, str] = {}
            seasons: dict[str, str] = {}
            validator = OOSFeatureValidator()
            for match_id, comp, season, day, home_goals, away_goals, raw_hash in selected:
                base_rows = predictions.get(match_id, {})
                if not base_rows:
                    continue
                evidence: list[BasePredictionEvidence] = []
                lineage: list[FeatureLineage] = []
                values: dict[str, float | str | None] = {name: None for name in schema.feature_names}
                values["horizon_seconds"] = None
                values["neutral_venue"] = None
                values["xg_available"] = 0.0
                values["form_available"] = 0.0
                snapshot_ids: set[str] = set()
                prediction_times: set[datetime] = set()
                versions: list[str] = []
                for model_id, prefix in MODEL_PREFIXES.items():
                    if prefix == "market_bayes":
                        continue
                    values[f"{prefix}_available"] = 0.0
                    if model_id not in base_rows:
                        continue
                    payload, cutoff, expected_count, data_hash, source_comp = base_rows[model_id]
                    prediction = ModelPrediction.model_validate_json(payload)
                    if (prediction.model_id != model_id or prediction.match_id != match_id or
                            prediction.execution_status != ExecutionStatus.SUCCESS or
                            not prediction.is_oos or source_comp != comp or
                            prediction.metadata.get("data_origin") != "REAL" or
                            prediction.metadata.get("prediction_temporal_mode") != "DATE_SAFE_BATCH" or
                            prediction.input_data_version != data_hash or
                            cutoff > prediction.prediction_time):
                        raise ValueError("BASE_OOS_FEATURE_LINEAGE_INVALID")
                    key = (comp, cutoff.date())
                    if key not in history_ids:
                        first_day = (cutoff - timedelta(days=1095)).date()
                        history_ids[key] = tuple(row[0] for row in connection.execute("""
                            SELECT match_id FROM real_canonical_matches
                            WHERE competition_id=? AND status='FINISHED'
                              AND match_date >= ? AND match_date < ? ORDER BY match_id
                        """, [comp, first_day, cutoff.date()]).fetchall())
                    ids = history_ids[key]
                    if len(ids) != expected_count or match_id in ids or not ids:
                        raise ValueError("BASE_OOS_TRAINING_ANCESTRY_INVALID")
                    full_evidence = BasePredictionEvidence(prediction=prediction,
                        training_match_ids=ids, base_prediction_oos=True)
                    if not validator._training_ids_hash_matches(prediction, full_evidence):
                        raise ValueError("BASE_OOS_TRAINING_IDS_HASH_MISMATCH")
                    item = BasePredictionEvidence(prediction=prediction,
                        training_match_ids=(), base_prediction_oos=True,
                        verified_training_ids_hash=prediction.metadata["training_match_ids_hash"],
                        verified_training_match_count=len(ids),
                        verification_method="REAL_OOS_STORE_RECOMPUTED")
                    snapshot_ids.add(prediction.prediction_snapshot_id)
                    prediction_times.add(prediction.prediction_time)
                    versions.append(prediction.input_data_version)
                    values[f"{prefix}_available"] = 1.0
                    values[f"{prefix}_p_home"] = prediction.p_home
                    values[f"{prefix}_p_draw"] = prediction.p_draw
                    values[f"{prefix}_p_away"] = prediction.p_away
                    if f"{prefix}_lambda_home" in values:
                        values[f"{prefix}_lambda_home"] = prediction.lambda_home
                        values[f"{prefix}_lambda_away"] = prediction.lambda_away
                    evidence.append(item)
                    lineage.append(FeatureLineage(
                        feature_names=tuple(name for name in schema.feature_names
                                            if name.startswith(prefix + "_")),
                        source_type="BASE_MODEL", source_id=model_id,
                        as_of_time=prediction.prediction_time,
                        retrieved_at=prediction.prediction_time,
                        data_version=f"{prediction.model_version}:{data_hash}",
                        dependency_ids=(prediction.prediction_id,)))
                if len(snapshot_ids) != 1 or len(prediction_times) != 1:
                    raise ValueError("BASE_OOS_BATCH_SNAPSHOT_MISMATCH")
                prediction_time = next(iter(prediction_times))
                if prediction_time != datetime.combine(day, time.min, UTC):
                    raise ValueError("BASE_OOS_NOT_DATE_START")
                if family == ModelFamily.CATBOOST:
                    values["competition_id"] = comp
                lineage.append(FeatureLineage(feature_names=("horizon_seconds", "neutral_venue"),
                    source_type="DATE_SAFE_BATCH_BOUNDARY", source_id="OPENFOOTBALL_DATE_ONLY",
                    as_of_time=prediction_time, retrieved_at=prediction_time,
                    data_version="UNKNOWN_EXACT_KICKOFF_AND_NEUTRAL_VENUE",
                    dependency_ids=(match_id,)))
                draft = {"match_id": match_id,
                    "prediction_snapshot_id": next(iter(snapshot_ids)),
                    "input_data_version": stable_hash(sorted(versions)),
                    "prediction_time": prediction_time,
                    "kickoff_time": datetime.combine(day + timedelta(days=1), time.min, UTC) -
                                    timedelta(microseconds=1),
                    "feature_version": FEATURE_VERSION,
                    "feature_schema_hash": schema.schema_hash,
                    "feature_mode": FeatureMode.NO_MARKET,
                    "model_family": family, "features": values,
                    "feature_lineage": tuple(lineage),
                    "base_prediction_evidence": tuple(evidence)}
                provisional = MLFeatureVector.model_construct(feature_data_hash="PENDING", **draft)
                vector = MLFeatureVector(feature_data_hash=stable_hash(
                    provisional.model_dump(mode="json", exclude={"feature_data_hash"})), **draft)
                validator.validate_training_vector(vector)
                vectors.append(vector)
                results[match_id] = (home_goals, away_goals, day,
                    hashlib.sha256(f"OPENFOOTBALL_RESULT|{match_id}|{raw_hash}".encode()).hexdigest()[:32])
                comp_ids[match_id] = comp
                seasons[match_id] = season
        dataset = MLDatasetBuilder().build_date_safe(vectors, results=results,
            schema=schema, competition_ids=comp_ids, seasons=seasons,
            dataset_cutoff=end + timedelta(days=1))
        with MLFeatureStore(self.feature_store_path) as store:
            store.append_many(tuple(vectors))
            store.save_dataset(dataset)
        prepare_phase8_2_migration(self.db_path)
        with duckdb.connect(self.db_path) as connection:
            connection.execute("""INSERT INTO real_ml_feature_datasets_v2 VALUES
                (?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
                [dataset.manifest.dataset_id, family.value, "NO_MARKET",
                 dataset.manifest.row_count, dataset.manifest.data_hash,
                 vectors[0].prediction_time.date(), vectors[-1].prediction_time.date(),
                 schema.schema_hash, str(Path(self.feature_store_path).resolve()),
                 datetime.now(UTC)])
        return dataset
