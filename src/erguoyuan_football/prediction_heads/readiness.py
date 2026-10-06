"""Read real OOS score matrices and select a source using earlier development results."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path

import duckdb

from erguoyuan_football.ml.schemas import stable_hash
from erguoyuan_football.models.score_matrix import ScoreMatrix
from erguoyuan_football.output_contract.schemas import CanonicalPredictionResult
from erguoyuan_football.prediction_heads.reconcile import matrix_hash

SCORE_MODELS = ("DIXON_COLES_V1", "BIVARIATE_POISSON_V1",
                "BAYESIAN_HIERARCHICAL_V1", "DYNAMIC_BAYESIAN_POISSON_V1")
HOLDOUT_START = date(2026, 8, 1)


@dataclass(frozen=True)
class MatrixSource:
    matrix: ScoreMatrix
    model_id: str
    model_version: str
    artifact_id: str | None
    prediction_id: str
    prediction_snapshot_id: str
    score_matrix_hash: str
    training_cutoff: str
    training_match_count: int
    training_data_hash: str
    training_match_ids_hash: str
    config_hash: str
    lineage_status: str = "PASS"


@dataclass(frozen=True)
class MatrixSelection:
    model_id: str
    evaluated_until: date
    sample_count: int
    metrics: dict[str, dict[str, float | int]]
    final_holdout_rows_read: int = 0


def _matrix(record: dict) -> ScoreMatrix:
    encoded = record.get("metadata", {}).get("score_matrix")
    if not isinstance(encoded, dict) or not isinstance(record.get("score_matrix"), list):
        raise TypeError("SCORE_MATRIX_CONTRACT_MISSING")
    matrix = ScoreMatrix.model_validate(encoded)
    if matrix_hash(matrix) != matrix_hash(ScoreMatrix.model_validate({**encoded,
            "values": record["score_matrix"]})):
        raise ValueError("SCORE_MATRIX_PAYLOAD_MISMATCH")
    return matrix


class ScoreMatrixSourceSelector:
    """Choose one real source by exact score, total, result and coverage before target dates."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def select(self, *, evaluation_start: date, evaluation_end: date,
               target_start: date) -> MatrixSelection:
        if not evaluation_start < evaluation_end <= target_start < HOLDOUT_START:
            raise ValueError("MATRIX_SELECTION_TIME_LEAKAGE")
        with duckdb.connect(str(self.db_path), read_only=True) as db:
            rows = db.execute("""SELECT o.model_id,o.match_id,o.competition_id,o.prediction_time,
                o.training_cutoff,o.training_match_count,o.training_data_hash,o.config_hash,
                o.is_oos,o.data_origin,o.prediction_temporal_mode,o.payload,
                m.home_goals,m.away_goals,m.match_date
                FROM real_oos_predictions_v2 o JOIN real_canonical_matches m ON o.match_id=m.match_id
                WHERE o.match_date>=? AND o.match_date<? AND m.match_date>=? AND m.match_date<?
                AND o.model_id IN (?,?,?,?) AND m.status='FINISHED'""",
                [evaluation_start, evaluation_end, evaluation_start, evaluation_end, *SCORE_MODELS]).fetchall()
            history = db.execute("""SELECT match_id,competition_id,match_date
                FROM real_canonical_matches WHERE status='FINISHED' AND match_date>=?
                AND match_date<?""", [evaluation_start - timedelta(days=1095),
                                       evaluation_end]).fetchall()
        by_comp: dict[str, list[tuple[str, date]]] = {}
        for history_id, competition_id, history_date in history:
            by_comp.setdefault(competition_id, []).append((history_id, history_date))
        training_cache: dict[tuple[str, date], tuple[str, ...]] = {}
        by_model: dict[str, list[tuple[float, float, float, float, float, int]]] = {}
        for (model_id, match_id, comp, prediction_at, cutoff, count, data_hash, config_hash,
             is_oos, origin, mode, payload, home, away, day) in rows:
            if not (is_oos and origin == "REAL" and mode == "DATE_SAFE_BATCH"
                    and prediction_at.date() == day and cutoff <= prediction_at):
                continue
            record = json.loads(payload)
            if (record.get("match_id") != match_id or record.get("execution_status") != "SUCCESS"
                    or record.get("is_oos") is not True or record.get("data_status") != "AVAILABLE"):
                continue
            meta = record.get("metadata") or {}
            key = (comp, cutoff.date())
            if key not in training_cache:
                training_cache[key] = tuple(sorted(history_id for history_id, history_day
                    in by_comp.get(comp, ()) if cutoff.date() - timedelta(days=1095)
                    <= history_day < cutoff.date()))
            training_ids = training_cache[key]
            valid_hashes = {stable_hash(list(training_ids)),
                sha256("|".join(training_ids).encode()).hexdigest()}
            if (match_id in training_ids or len(training_ids) != count or
                meta.get("training_match_ids_hash") not in valid_hashes or
                meta.get("training_data_hash") != data_hash or
                meta.get("config_hash", config_hash) != config_hash):
                continue
            try:
                matrix = _matrix(record)
            except (TypeError, ValueError):
                continue
            if home > matrix.max_goals or away > matrix.max_goals:
                coverage = 0.0
                exact_loss = -math.log(1e-15)
            else:
                coverage = 1.0
                exact_loss = -math.log(max(1e-15, matrix.values[home][away]))
            total_probability = sum(float(p) for h, line in enumerate(matrix.values)
                                    for a, p in enumerate(line)
                                    if (min(7, h + a)) == (min(7, home + away)))
            total_loss = -math.log(max(1e-15, total_probability))
            result = "p_home" if home > away else "p_draw" if home == away else "p_away"
            outcome = matrix.outcome()
            result_loss = -math.log(max(1e-15, getattr(outcome, result)))
            result_probs = (outcome.p_home, outcome.p_draw, outcome.p_away)
            chosen_result = max(range(3), key=lambda index: result_probs[index])
            actual_result = 0 if home > away else 1 if home == away else 2
            by_model.setdefault(model_id, []).append((exact_loss, total_loss, result_loss,
                coverage, result_probs[chosen_result], int(chosen_result == actual_result)))
        metrics: dict[str, dict[str, float | int]] = {}
        for model_id, samples in by_model.items():
            n = len(samples)
            metrics[model_id] = {"sample_count": n,
                "exact_score_log_loss": math.fsum(s[0] for s in samples) / n,
                "goal_total_log_loss": math.fsum(s[1] for s in samples) / n,
                "result_log_loss": math.fsum(s[2] for s in samples) / n,
                "result_calibration_ece": sum(
                    len(group) / n * abs(math.fsum(s[4] for s in group) / len(group) -
                                         math.fsum(s[5] for s in group) / len(group))
                    for bin_index in range(10)
                    if (group := [s for s in samples
                        if min(9, int(s[4] * 10)) == bin_index])),
                "score_support_coverage": math.fsum(s[3] for s in samples) / n}
        if not metrics:
            raise ValueError("NO_LEGAL_DEVELOPMENT_SCORE_MATRIX_SOURCE")
        # Compare only sources with the same observed sample coverage.
        maximum = max(int(row["sample_count"]) for row in metrics.values())
        candidates = [key for key, value in metrics.items() if value["sample_count"] == maximum]
        chosen = min(candidates, key=lambda key: (
            float(metrics[key]["exact_score_log_loss"]),
            float(metrics[key]["goal_total_log_loss"]),
            float(metrics[key]["result_calibration_ece"]),
            -float(metrics[key]["score_support_coverage"]), key))
        return MatrixSelection(chosen, evaluation_end, maximum, metrics)


class ScoreMatrixReadinessAudit:
    """Verify the target's exact OOS prediction and reconstructed pre-target training IDs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def load_many(self, canonicals: tuple[CanonicalPredictionResult, ...],
                  selection: MatrixSelection) -> dict[str, MatrixSource]:
        if any(row.match_date is None or row.match_date >= HOLDOUT_START or
               row.match_date < selection.evaluated_until for row in canonicals):
            raise ValueError("FINAL_HOLDOUT_OR_MATRIX_SELECTION_LEAKAGE")
        ids = [row.data_lineage.get("source_prediction_ids", {}).get(selection.model_id)
               for row in canonicals]
        valid_ids = [value for value in ids if isinstance(value, str)]
        if not valid_ids:
            return {}
        with duckdb.connect(str(self.db_path), read_only=True) as db:
            rows = db.execute("""SELECT prediction_id,match_id,competition_id,model_id,model_version,
                training_cutoff,prediction_time,match_date,training_match_count,training_data_hash,
                config_hash,prediction_snapshot_id,prediction_temporal_mode,data_origin,is_oos,payload
                FROM real_oos_predictions_v2 WHERE prediction_id IN (SELECT unnest(?))
                AND match_date < DATE '2026-08-01'""", [valid_ids]).fetchall()
            earliest = min(row.match_date for row in canonicals if row.match_date is not None)
            history = db.execute("""SELECT match_id,competition_id,match_date FROM real_canonical_matches
                WHERE status='FINISHED' AND match_date >= ? AND match_date < ?""",
                [earliest - timedelta(days=1095), max(row.match_date for row in canonicals
                                                        if row.match_date is not None)]).fetchall()
        by_id = {row[0]: row for row in rows}
        by_comp: dict[str, list[tuple[str, date]]] = {}
        for match_id, comp, day in history:
            by_comp.setdefault(comp, []).append((match_id, day))
        result: dict[str, MatrixSource] = {}
        for canonical, prediction_id in zip(canonicals, ids, strict=True):
            if prediction_id is None:
                continue
            row = by_id.get(prediction_id)
            if row is None:
                continue
            (pid, match_id, comp, model_id, version, cutoff, prediction_at, day,
             count, data_hash, config_hash, snapshot_id, mode, origin, is_oos, payload) = row
            record = json.loads(payload)
            meta = record.get("metadata") or {}
            if (match_id != canonical.match_id or comp != canonical.competition_id or
                model_id != selection.model_id or day != canonical.match_date or
                origin != "REAL" or is_oos is not True or mode != "DATE_SAFE_BATCH" or
                prediction_at.date() != day or cutoff > prediction_at or
                record.get("prediction_id") != pid or record.get("match_id") != match_id or
                record.get("model_version") != version or record.get("execution_status") != "SUCCESS" or
                record.get("data_status") != "AVAILABLE" or record.get("is_oos") is not True or
                meta.get("data_origin") != "REAL" or meta.get("training_data_hash") != data_hash or
                meta.get("config_hash", config_hash) != config_hash):
                continue
            train_ids = tuple(sorted(item for item, match_day in by_comp.get(comp, ())
                if cutoff.date() - timedelta(days=1095) <= match_day < cutoff.date()))
            hashes = {stable_hash(list(train_ids)), sha256("|".join(train_ids).encode()).hexdigest()}
            if (match_id in train_ids or len(train_ids) != count or
                    meta.get("training_match_ids_hash") not in hashes):
                continue
            try:
                matrix = _matrix(record)
            except (TypeError, ValueError):
                continue
            result[match_id] = MatrixSource(matrix, model_id, version,
                meta.get("artifact_id"), pid, snapshot_id, matrix_hash(matrix),
                cutoff.isoformat(), count, data_hash,
                meta["training_match_ids_hash"], config_hash)
        return result
