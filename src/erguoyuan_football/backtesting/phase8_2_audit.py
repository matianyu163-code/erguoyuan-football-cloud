"""Deterministic audit of real OOS ancestry, nested ML and holdout isolation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb

from erguoyuan_football.backtesting.real_oos_matrix import BASE_MODEL_IDS, ML_MODEL_IDS
from erguoyuan_football.ml.artifacts import MLArtifact
from erguoyuan_football.ml.schemas import stable_hash


@dataclass(frozen=True)
class Phase82Audit:
    status: str
    sampled_by_model: dict[str, int]
    base_lineage_checked: int
    nested_ml_lineage_checked: int
    market_oos_rows: int
    holdout_oos_rows: int
    candidate_rows: int
    reasons: tuple[str, ...]


def _sample(items: list[tuple], count: int = 100) -> list[tuple]:
    if len(items) < count:
        return items
    return [items[index * (len(items) - 1) // (count - 1)] for index in range(count)]


def audit_phase8_2(db_path: str | Path, *, artifact_root: str | Path) -> Phase82Audit:
    """Recompute 100 lineage proofs per ready model from canonical match dates."""
    sampled: dict[str, int] = {}
    base_checked = ml_checked = 0
    reasons: list[str] = []
    artifact_root = Path(artifact_root)
    with duckdb.connect(str(db_path), read_only=True) as connection:
        match_dates = dict(connection.execute("SELECT match_id,match_date "
            "FROM real_canonical_matches").fetchall())
        base_by_id = {row[0]: row[1:] for row in connection.execute("""
            SELECT prediction_id,match_id,training_cutoff,prediction_time,data_origin,is_oos
            FROM real_oos_predictions_v2 WHERE model_id IN
            ('DIXON_COLES_V1','BIVARIATE_POISSON_V1','BAYESIAN_HIERARCHICAL_V1',
             'ELO_V1','PI_RATING_V1','DYNAMIC_BAYESIAN_POISSON_V1',
             'CORE_SPI_LIKE_V1','CORE_OPTA_XG_ELO_LIKE_V1')""").fetchall()}
        history_cache: dict[tuple[str, date], tuple[str, ...]] = {}
        for model_id in (*BASE_MODEL_IDS, *ML_MODEL_IDS):
            rows = connection.execute("""SELECT prediction_id,match_id,competition_id,
                match_date,training_cutoff,prediction_time,training_match_count,
                training_data_hash,data_origin,is_oos,payload
                FROM real_oos_predictions_v2 WHERE model_id=?
                ORDER BY match_date,match_id""", [model_id]).fetchall()
            if not rows:
                continue
            chosen = _sample(rows)
            sampled[model_id] = len(chosen)
            if len(chosen) < 100:
                reasons.append(f"INSUFFICIENT_AUDIT_SAMPLE:{model_id}")
            for (pid, match_id, comp, day, cutoff, prediction_time,
                 count, data_hash, origin, is_oos, payload) in chosen:
                if origin != "REAL" or not is_oos or cutoff > prediction_time or (
                        prediction_time.date() != day or match_dates[match_id] != day):
                    raise ValueError(f"NON_REAL_OR_TEMPORALLY_INVALID_OOS:{pid}")
                record = json.loads(payload)
                metadata = record["metadata"]
                if (record["prediction_id"] != pid or metadata.get("data_origin") != "REAL" or
                        metadata.get("prediction_temporal_mode") != "DATE_SAFE_BATCH"):
                    raise ValueError(f"OOS_PAYLOAD_IDENTITY_INVALID:{pid}")
                if model_id in BASE_MODEL_IDS:
                    key = (comp, cutoff.date())
                    if key not in history_cache:
                        history_cache[key] = tuple(item[0] for item in connection.execute("""
                            SELECT match_id FROM real_canonical_matches
                            WHERE competition_id=? AND status='FINISHED'
                              AND match_date>=? AND match_date<? ORDER BY match_id""",
                            [comp, (cutoff - timedelta(days=1095)).date(),
                             cutoff.date()]).fetchall())
                    ids = history_cache[key]
                    legacy = hashlib.sha256("|".join(ids).encode()).hexdigest()
                    current = stable_hash(list(ids))
                    if (match_id in ids or len(ids) != count or
                            metadata.get("training_match_ids_hash") not in {legacy, current} or
                            metadata.get("training_data_hash") != data_hash or
                            any(match_dates[item] >= day for item in ids)):
                        raise ValueError(f"BASE_TRAINING_LINEAGE_INVALID:{pid}")
                    base_checked += 1
                else:
                    artifact_id = metadata.get("artifact_id")
                    manifest_path = artifact_root / model_id / str(artifact_id) / "manifest.json"
                    artifact = MLArtifact.model_validate_json(manifest_path.read_text(encoding="utf-8"))
                    ids = artifact.training_match_ids
                    if (artifact.trained_until > prediction_time or match_id in ids or
                            len(ids) != count or stable_hash(sorted(ids)) !=
                            metadata.get("training_match_ids_hash") or
                            metadata.get("uses_market") is not False or
                            any(match_dates[item] >= day for item in ids)):
                        raise ValueError(f"NESTED_ML_TRAINING_LINEAGE_INVALID:{pid}")
                    dependencies = metadata.get("base_prediction_ids") or []
                    if not dependencies:
                        raise ValueError(f"ML_BASE_ANCESTRY_MISSING:{pid}")
                    for base_pid in dependencies:
                        ancestor = base_by_id.get(base_pid)
                        if ancestor is None or (ancestor[0] != match_id or ancestor[1] > prediction_time or
                                                ancestor[2] != prediction_time or ancestor[3] != "REAL" or
                                                ancestor[4] is not True):
                            raise ValueError(f"ML_BASE_ANCESTRY_INVALID:{pid}:{base_pid}")
                    ml_checked += 1
        market_result = connection.execute("""SELECT count(*) FROM real_oos_predictions_v2
            WHERE model_id='HISTORICAL_MARKET_BAYESIAN_POISSON_V1'""").fetchone()
        holdout_result = connection.execute("""SELECT count(*) FROM real_oos_predictions_v2
            WHERE match_date>='2026-08-01'""").fetchone()
        assert market_result is not None and holdout_result is not None
        market_rows, holdout_rows = market_result[0], holdout_result[0]
        candidate = connection.execute("""SELECT row_count,artifact_path
            FROM phase9_candidate_dataset_v2 ORDER BY created_at DESC LIMIT 1""").fetchone()
        candidate_rows = 0 if candidate is None else candidate[0]
        if candidate is not None:
            with duckdb.connect(":memory:") as temporary:
                features = Path(candidate[1]) / "features.parquet"
                labels = Path(candidate[1]) / "labels.parquet"
                cols = [item[0] for item in temporary.execute("DESCRIBE SELECT * FROM read_parquet(?)",
                                                               [str(features)]).fetchall()]
                labels_result = temporary.execute("SELECT count(*) FROM read_parquet(?)",
                    [str(labels)]).fetchone()
                assert labels_result is not None
                labels_count = labels_result[0]
                if ("target" in cols or any("market" in col.lower() for col in cols)
                        or labels_count != candidate_rows):
                    raise ValueError("CANDIDATE_TARGET_OR_MARKET_LEAKAGE")
        if market_rows or holdout_rows or candidate_rows == 0:
            reasons.append("MARKET_HOLDOUT_OR_CANDIDATE_AUDIT_FAILED")
    status = "PASS_WITH_LIMITATIONS" if not reasons else "FAILED"
    return Phase82Audit(status, sampled, base_checked, ml_checked,
                        market_rows, holdout_rows, candidate_rows, tuple(reasons))
