"""Read-only, lineage-checked matrix of persisted real out-of-sample predictions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from erguoyuan_football.backtesting.date_safe_oos import _metrics
from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.ml.feature_contract import MODEL_PREFIXES

BASE_MODEL_IDS = (
    "DIXON_COLES_V1", "BIVARIATE_POISSON_V1", "BAYESIAN_HIERARCHICAL_V1",
    "ELO_V1", "PI_RATING_V1", "DYNAMIC_BAYESIAN_POISSON_V1",
    "CORE_SPI_LIKE_V1", "CORE_OPTA_XG_ELO_LIKE_V1",
)
ML_MODEL_IDS = ("CORE_XGBOOST_V1", "CORE_CATBOOST_V1")


def lineage_hash(values: tuple[Any, ...]) -> str:
    """Stable digest for a prediction cell and its training ancestry."""
    return hashlib.sha256(json.dumps(values, default=str, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class RealOOSMatrixRow:
    """A label is separate from features and may never enter an ML feature vector."""

    match_id: str
    competition_id: str
    season_id: str
    prediction_date: date
    temporal_mode: str
    target: int
    probabilities: dict[str, tuple[float, float, float] | None]
    availability: dict[str, int]
    prediction_ids: dict[str, str | None]
    lineage_hashes: dict[str, str | None]

    def feature_values(self) -> dict[str, float | None]:
        """Return numeric model features and masks, excluding the result label."""
        values: dict[str, float | None] = {}
        for model_id, probs in self.probabilities.items():
            prefix = MODEL_PREFIXES.get(model_id, {
                "CORE_XGBOOST_V1": "xgb", "CORE_CATBOOST_V1": "cat"}.get(model_id))
            if prefix is None:
                raise ValueError(f"UNKNOWN_MATRIX_MODEL:{model_id}")
            values[f"{prefix}_home"] = probs[0] if probs else None
            values[f"{prefix}_draw"] = probs[1] if probs else None
            values[f"{prefix}_away"] = probs[2] if probs else None
            values[f"{prefix}_available"] = float(self.availability[model_id])
        return values


@dataclass(frozen=True)
class CommonOOSComparison:
    model_ids: tuple[str, ...]
    match_ids: tuple[str, ...]
    metrics: dict[str, dict[str, float | int]]


@dataclass(frozen=True)
class RealOOSCoverage:
    model_id: str
    competition_id: str
    season_id: str
    temporal_mode: str
    eligible_matches: int
    predicted_rows: int
    unavailable_rows: int
    failed_rows: int
    not_run_rows: int
    coverage_pct: float


class RealOOSPredictionMatrix:
    """Materialize only successful REAL OOS records; missing cells remain null."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    def build(self, start: date, end: date, *,
              model_ids: tuple[str, ...] = BASE_MODEL_IDS + ML_MODEL_IDS) -> tuple[RealOOSMatrixRow, ...]:
        if end < start or len(set(model_ids)) != len(model_ids):
            raise ValueError("INVALID_MATRIX_WINDOW_OR_MODELS")
        with duckdb.connect(self.db_path, read_only=True) as connection:
            matches = connection.execute("""
                SELECT match_id, competition_id, season_id, match_date, home_goals, away_goals
                FROM real_canonical_matches
                WHERE status='FINISHED' AND match_date BETWEEN ? AND ?
                ORDER BY match_date, match_id
            """, [start, end]).fetchall()
            predictions = connection.execute("""
                SELECT prediction_id, match_id, model_id, training_cutoff,
                       prediction_time, prediction_temporal_mode, match_date,
                       p_home, p_draw, p_away, is_oos, data_origin,
                       training_data_hash, config_hash, prediction_snapshot_id, payload
                FROM real_oos_predictions_v2
                WHERE match_date BETWEEN ? AND ?
            """, [start, end]).fetchall()
            has_lineage = connection.execute("SELECT 1 FROM information_schema.tables "
                                             "WHERE table_name='real_oos_lineage_v2'").fetchone()
            stored_lineage = dict(connection.execute("SELECT prediction_id,lineage_hash "
                "FROM real_oos_lineage_v2").fetchall()) if has_lineage else {}
        by_match: dict[str, dict[str, tuple[tuple[float, float, float], str, str]]] = {}
        mode_by_match: dict[str, set[str]] = {}
        for item in predictions:
            (pid, match_id, model_id, cutoff, prediction_time, mode, match_date,
             home, draw, away, is_oos, origin, data_hash, config_hash, snapshot_id, payload) = item
            if model_id not in model_ids:
                continue
            if origin != "REAL" or is_oos is not True:
                raise ValueError("MATRIX_NON_REAL_OR_IN_SAMPLE_REJECTED")
            if cutoff > prediction_time or (mode == "DATE_SAFE_BATCH" and
                                            prediction_time.date() != match_date):
                raise ValueError("MATRIX_FUTURE_OR_INVALID_DATE_SAFE_CUTOFF")
            probs = (float(home), float(draw), float(away))
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in probs) or abs(sum(probs) - 1) >= 1e-6:
                raise ValueError("MATRIX_INVALID_PROBABILITY")
            record = ModelPrediction.model_validate_json(payload)
            if (record.prediction_id != pid or record.match_id != match_id or
                    record.model_id != model_id or record.execution_status != ExecutionStatus.SUCCESS or
                    not record.is_oos or record.prediction_snapshot_id != snapshot_id):
                raise ValueError("MATRIX_PAYLOAD_LINEAGE_MISMATCH")
            cells = by_match.setdefault(match_id, {})
            if model_id in cells:
                raise ValueError(f"AMBIGUOUS_OOS_PREDICTIONS:{match_id}:{model_id}")
            digest = lineage_hash((pid, match_id, model_id, cutoff, prediction_time,
                                   data_hash, config_hash, snapshot_id, mode))
            declared = record.metadata.get("lineage_hash")
            indexed = stored_lineage.get(pid)
            if declared is not None and indexed != declared:
                raise ValueError("MATRIX_STORED_LINEAGE_HASH_MISMATCH")
            if indexed is not None and declared is None:
                raise ValueError("MATRIX_LINEAGE_PAYLOAD_MISSING")
            digest = str(declared) if declared is not None else digest
            cells[model_id] = (probs, pid, digest)
            mode_by_match.setdefault(match_id, set()).add(mode)
        rows: list[RealOOSMatrixRow] = []
        for match_id, comp, season, match_date, home_goals, away_goals in matches:
            cells = by_match.get(match_id, {})
            modes = mode_by_match.get(match_id, set())
            if len(modes) > 1:
                raise ValueError(f"MIXED_TEMPORAL_MODE_FOR_MATCH:{match_id}")
            mode = next(iter(modes), "DATE_SAFE_BATCH")
            rows.append(RealOOSMatrixRow(
                match_id, comp, season, match_date, mode,
                0 if home_goals > away_goals else 1 if home_goals == away_goals else 2,
                {mid: cells[mid][0] if mid in cells else None for mid in model_ids},
                {mid: int(mid in cells) for mid in model_ids},
                {mid: cells[mid][1] if mid in cells else None for mid in model_ids},
                {mid: cells[mid][2] if mid in cells else None for mid in model_ids},
            ))
        return tuple(rows)

    @staticmethod
    def common_sample(rows: tuple[RealOOSMatrixRow, ...],
                      model_ids: tuple[str, ...]) -> CommonOOSComparison:
        if not model_ids or len(set(model_ids)) != len(model_ids):
            raise ValueError("COMMON_SAMPLE_REQUIRES_DISTINCT_MODELS")
        common = tuple(row for row in rows if all(row.availability.get(mid) == 1 for mid in model_ids))
        metrics: dict[str, dict[str, float | int]] = {}
        for mid in model_ids:
            values = []
            for row in common:
                probabilities = row.probabilities[mid]
                if probabilities is None:
                    raise ValueError("COMMON_SAMPLE_MISSING_MODEL_CELL")
                values.append((*probabilities, row.target))
            metrics[mid] = _metrics(values)
        return CommonOOSComparison(model_ids, tuple(row.match_id for row in common), metrics)

    def coverage(self, start: date, end: date, *,
                 model_ids: tuple[str, ...] = BASE_MODEL_IDS) -> tuple[RealOOSCoverage, ...]:
        rows = self.build(start, end, model_ids=model_ids)
        groups: dict[tuple[str, str, str], list[RealOOSMatrixRow]] = {}
        for row in rows:
            groups.setdefault((row.competition_id, row.season_id, row.temporal_mode), []).append(row)
        result: list[RealOOSCoverage] = []
        with duckdb.connect(self.db_path, read_only=True) as connection:
            has_status = connection.execute("SELECT 1 FROM information_schema.tables "
                                            "WHERE table_name='real_oos_execution_status'").fetchone()
            statuses = connection.execute("""
                SELECT match_id, model_id, execution_status FROM real_oos_execution_status
                WHERE prediction_date BETWEEN ? AND ?
            """, [start, end]).fetchall() if has_status else []
        status_by_key = {(match_id, model_id): status for match_id, model_id, status in statuses}
        for (comp, season, mode), group in sorted(groups.items()):
            for model_id in model_ids:
                predicted = sum(row.availability[model_id] for row in group)
                unavailable = sum(status_by_key.get((row.match_id, model_id)) == "UNAVAILABLE" for row in group)
                failed = sum(status_by_key.get((row.match_id, model_id)) == "FAILED" for row in group)
                result.append(RealOOSCoverage(model_id, comp, season, mode, len(group), predicted,
                    unavailable, failed, len(group) - predicted - unavailable - failed,
                    100 * predicted / len(group)))
        return tuple(result)
