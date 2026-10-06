"""Prepare inspectable NO_MARKET candidate features without fitting META."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from erguoyuan_football.backtesting.phase8_2_store import prepare_phase8_2_migration
from erguoyuan_football.backtesting.real_oos_matrix import (
    BASE_MODEL_IDS,
    ML_MODEL_IDS,
    RealOOSPredictionMatrix,
)
from erguoyuan_football.ml.schemas import stable_hash


@dataclass(frozen=True)
class Phase9CandidateDataset:
    dataset_id: str
    mode: str
    row_count: int
    first_prediction_date: date
    last_prediction_date: date
    data_hash: str
    artifact_path: str


@dataclass(frozen=True)
class SplitFeasibilityReport:
    meta_train_rows: int
    calibration_candidate_rows: int
    final_holdout_rows: int
    status: str
    reason: str


class Phase9CandidateBuilder:
    """Keep label, features and lineage in separate files to prevent label use."""

    def __init__(self, db_path: str | Path, *, output_root: str | Path) -> None:
        self.db_path = Path(db_path)
        self.output_root = Path(output_root)

    def build(self, start: date, end: date) -> Phase9CandidateDataset:
        if end >= date(2026, 8, 1) or end < start:
            raise ValueError("FINAL_HOLDOUT_LOCKED_OR_INVALID_WINDOW")
        matrix = RealOOSPredictionMatrix(self.db_path).build(start, end)
        rows = [row for row in matrix if any(row.availability[mid] for mid in BASE_MODEL_IDS)]
        if not rows:
            raise ValueError("NO_REAL_BASE_OOS_FOR_PHASE9_CANDIDATE")
        identifiers = [row.match_id for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("DUPLICATE_CANDIDATE_MATCH")
        feature_rows = []
        label_rows = []
        lineage_rows = []
        for row in rows:
            features = {"match_id": row.match_id,
                "prediction_date": row.prediction_date.isoformat(),
                "temporal_mode": row.temporal_mode,
                "competition_id": row.competition_id,
                "season_id": row.season_id,
                **row.feature_values()}
            if any("market" in key.lower() or "target" in key.lower() for key in features):
                raise ValueError("CANDIDATE_MARKET_OR_TARGET_FEATURE_REJECTED")
            feature_rows.append(features)
            label_rows.append({"match_id": row.match_id, "target": row.target})
            lineage_rows.append({"match_id": row.match_id,
                "prediction_ids": json.dumps(row.prediction_ids, sort_keys=True),
                "lineage_hashes": json.dumps(row.lineage_hashes, sort_keys=True),
                "dependency_tags": json.dumps({mid: ["REAL_OOS", "DATE_SAFE_BATCH",
                    "NESTED_BASE_OOS" if mid in ML_MODEL_IDS else "BASE_OOS"]
                    for mid, mask in row.availability.items() if mask}, sort_keys=True)})
        digest = stable_hash({"features": feature_rows, "labels": label_rows,
                              "lineage": lineage_rows})
        dataset_id = digest[:32]
        directory = self.output_root / dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(":memory:") as connection:
            for name, values in (("features", feature_rows),
                                 ("labels", label_rows), ("lineage", lineage_rows)):
                connection.register("source_rows", pd.DataFrame(values))
                destination = (directory / f"{name}.parquet").as_posix()
                connection.execute("COPY source_rows TO ? (FORMAT PARQUET)", [destination])
                connection.unregister("source_rows")
        record = Phase9CandidateDataset(dataset_id, "NO_MARKET", len(rows),
            rows[0].prediction_date, rows[-1].prediction_date, digest, str(directory.resolve()))
        prepare_phase8_2_migration(self.db_path)
        with duckdb.connect(str(self.db_path)) as connection:
            connection.execute("""INSERT INTO phase9_candidate_dataset_v2 VALUES
                (?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
                [record.dataset_id, record.mode, record.row_count,
                 record.first_prediction_date, record.last_prediction_date,
                 record.data_hash, record.artifact_path, datetime.now(UTC)])
        return record

    @staticmethod
    def split_feasibility(candidate: Phase9CandidateDataset, *,
                          db_path: str | Path) -> SplitFeasibilityReport:
        with duckdb.connect(str(db_path), read_only=True) as connection:
            counts = []
            for start, end in ((date(2022, 8, 1), date(2025, 6, 30)),
                               (date(2025, 8, 1), date(2026, 6, 30))):
                result = connection.execute("""SELECT count(DISTINCT match_id)
                    FROM real_oos_predictions_v2 WHERE data_origin='REAL' AND is_oos
                    AND match_date BETWEEN ? AND ?""", [start, end]).fetchone()
                assert result is not None
                counts.append(result[0])
        # Holdout remains deliberately unevaluated even if source results exist.
        status = "PARTIAL_FINAL_HOLDOUT_UNTOUCHED" if counts[0] >= 100 and counts[1] >= 100 else "INSUFFICIENT_SPLIT"
        return SplitFeasibilityReport(counts[0], counts[1], 0, status,
            "The 2026-27 final holdout is locked and excluded from Phase 8.2 evaluation")
