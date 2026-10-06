"""Read and verify the immutable Phase 8.2 candidate without rebuilding it."""

from __future__ import annotations

import json
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from erguoyuan_football.data.connection import connect_project_duckdb
from erguoyuan_football.ml.schemas import stable_hash

MODEL_PREFIXES = ("dc", "bivariate", "elo", "pi", "spi", "opta_like", "xgb", "cat")
DEFERRED_PREFIXES = ("hier_bayes", "dynamic_bayes")
FAMILY = {
    "dc": "GOAL_FAMILY", "bivariate": "GOAL_FAMILY", "spi": "GOAL_FAMILY",
    "elo": "RESULT_RATING_FAMILY", "pi": "RESULT_RATING_FAMILY",
    "opta_like": "RESULT_RATING_FAMILY", "xgb": "ML_FAMILY", "cat": "ML_FAMILY",
}
DEPENDENCY_TAGS = {
    "dc": ("GOALS", "RESULTS"), "bivariate": ("GOALS", "RESULTS"),
    "spi": ("GOALS", "RESULTS", "RATINGS"),
    "elo": ("RESULTS", "RATINGS"), "pi": ("GOALS", "RESULTS", "RATINGS"),
    "opta_like": ("RESULTS", "RATINGS"),
    "xgb": ("GOALS", "RESULTS", "RATINGS", "ML"),
    "cat": ("GOALS", "RESULTS", "RATINGS", "ML"),
}


@dataclass(frozen=True)
class VerifiedCandidate:
    """Loaded candidate with separate label and lineage tables."""

    dataset_id: str
    data_hash: str
    part_hashes: dict[str, str]
    features: pd.DataFrame
    labels: pd.DataFrame
    lineage: pd.DataFrame

    def copy_for_caller(self) -> VerifiedCandidate:
        """Return independent frames so a caller cannot mutate the shared cache."""
        return VerifiedCandidate(self.dataset_id, self.data_hash, dict(self.part_hashes),
                                self.features.copy(deep=True), self.labels.copy(deep=True),
                                self.lineage.copy(deep=True))

    def availability_report(self) -> dict[str, Any]:
        """Report candidate-only eligibility and availability/time confounding."""
        masks = self.features[[f"{prefix}_available" for prefix in MODEL_PREFIXES]]
        counts = masks.sum(axis=1).astype(int)
        patterns: Counter[str] = Counter()
        by_year: dict[str, Counter[str]] = {}
        for index, row in masks.iterrows():
            active = tuple(prefix for prefix in MODEL_PREFIXES if row[f"{prefix}_available"] == 1)
            key = "+".join(active)
            patterns[key] += 1
            year = str(self.features.at[index, "prediction_date"])[:4]
            by_year.setdefault(year, Counter())[key] += 1
        dates = self.features["prediction_date"].astype(str)
        coverage = {}
        for prefix in MODEL_PREFIXES:
            present = masks[f"{prefix}_available"] == 1
            covered_dates = dates[present]
            coverage[prefix] = {
                "rows": int(present.sum()), "coverage_pct": 100 * float(present.mean()),
                "first_prediction_date": str(covered_dates.min()) if len(covered_dates) else None,
                "last_prediction_date": str(covered_dates.max()) if len(covered_dates) else None,
            }
        return {
            "candidate_total_rows": len(self.features),
            "candidate_meta_train_rows": int(((dates >= "2022-08-01") &
                                              (dates <= "2025-06-30")).sum()),
            "candidate_calibration_rows": int(((dates >= "2025-08-01") &
                                               (dates <= "2026-06-30")).sum()),
            "candidate_holdout_rows": int(((dates >= "2026-08-01") &
                                           (dates <= "2027-06-30")).sum()),
            "eligible_meta_rows_min3": int(((dates <= "2025-06-30") & (counts >= 3)).sum()),
            "eligible_calibration_rows_min3": int(((dates >= "2025-08-01") &
                                                  (dates <= "2026-06-30") & (counts >= 3)).sum()),
            "models_per_row_distribution": dict(sorted(Counter(counts).items())),
            "availability_pattern_distribution": dict(sorted(patterns.items())),
            "availability_pattern_by_year": {year: dict(sorted(value.items()))
                                             for year, value in sorted(by_year.items())},
            "model_dependency_families": FAMILY,
            "model_dependency_tags": DEPENDENCY_TAGS,
            "model_coverage": coverage,
            "deferred_models": ["BAYESIAN_HIERARCHICAL_V1",
                                "DYNAMIC_BAYESIAN_POISSON_V1",
                                "HISTORICAL_MARKET_BAYESIAN_POISSON_V1"],
            "market_dependency_count": 0,
        }


def _content_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Restore Parquet nulls before reproducing Phase 8.2's canonical hash."""
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


class CandidateDatasetCache:
    """Small process-local cache of verified immutable Phase 9 candidate data."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[tuple[Any, ...], tuple[VerifiedCandidate, tuple[tuple[str, int, int], ...]]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _signature(paths: tuple[Path, ...]) -> tuple[tuple[str, int, int], ...] | None:
        try:
            return tuple((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                         for path in paths)
        except OSError:
            return None

    def get(self, key: tuple[Any, ...]) -> VerifiedCandidate | None:
        """Hit only if all three parquet files retain their verified size and mtime."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            candidate, old_signature = entry
            paths = tuple(Path(item[0]) for item in old_signature)
            if self._signature(paths) != old_signature:
                self._entries.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return candidate.copy_for_caller()

    def put(self, key: tuple[Any, ...], candidate: VerifiedCandidate,
            paths: tuple[Path, ...]) -> None:
        """Store a private immutable copy with source-file signatures."""
        signature = self._signature(paths)
        if signature is None:
            return
        with self._lock:
            self._entries[key] = (candidate.copy_for_caller(), signature)

    def clear(self) -> None:
        """Clear cached rows and counters, mainly for deterministic tests."""
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0


candidate_dataset_cache = CandidateDatasetCache()


def load_verified_candidate(db_path: str | Path, *, dataset_id: str,
                            expected_hash: str,
                            expected_part_hashes: dict[str, str] | None = None) -> VerifiedCandidate:
    """Fail closed on changed files, mismatched labels, or non-OOS ancestry."""
    if dataset_id != expected_hash[:32]:
        raise ValueError("CANDIDATE_ID_HASH_MISMATCH")
    resolved_db = Path(db_path).resolve()
    part_key = tuple(sorted((expected_part_hashes or {}).items()))
    cache_key = (str(resolved_db), dataset_id, expected_hash, part_key)
    cached = candidate_dataset_cache.get(cache_key)
    if cached is not None:
        return cached
    parquet_paths: list[Path] = []
    with connect_project_duckdb(db_path, read_only=True) as connection:
        record = connection.execute("""SELECT mode,row_count,data_hash,artifact_path
            FROM phase9_candidate_dataset_v2 WHERE dataset_id=?""", [dataset_id]).fetchone()
        if record is None or record[0] != "NO_MARKET" or record[2] != expected_hash:
            raise ValueError("CANDIDATE_REGISTRY_MISMATCH")
        path = Path(record[3])
        frames: dict[str, pd.DataFrame] = {}
        for name in ("features", "labels", "lineage"):
            file = path / f"{name}.parquet"
            if not file.is_file():
                raise ValueError(f"CANDIDATE_PART_MISSING:{name}")
            frames[name] = connection.execute("SELECT * FROM read_parquet(?)",
                                              [file.as_posix()]).fetchdf()
            parquet_paths.append(file)
        part_rows = {name: _content_rows(frame) for name, frame in frames.items()}
        part_hashes = {name: stable_hash(rows) for name, rows in part_rows.items()}
        combined = stable_hash(part_rows)
        if combined != expected_hash or any(len(frame) != record[1] for frame in frames.values()):
            raise ValueError("CANDIDATE_CONTENT_HASH_MISMATCH")
        if expected_part_hashes is not None and part_hashes != expected_part_hashes:
            raise ValueError("CANDIDATE_PART_HASH_MISMATCH")
        feature_ids = frames["features"]["match_id"].tolist()
        if len(set(feature_ids)) != len(feature_ids) or any(
                frames[name]["match_id"].tolist() != feature_ids for name in ("labels", "lineage")):
            raise ValueError("CANDIDATE_ID_OR_ORDER_MISMATCH")
        if not set(frames["labels"]["target"]).issubset({0, 1, 2}):
            raise ValueError("INVALID_CANDIDATE_LABEL")
        if any("market" in col.lower() or "target" in col.lower()
               for col in frames["features"].columns):
            raise ValueError("MARKET_OR_TARGET_FEATURE_REJECTED")
        if any(date.fromisoformat(str(value)) >= date(2026, 8, 1)
               for value in frames["features"]["prediction_date"]):
            raise ValueError("FINAL_HOLDOUT_LOCK_VIOLATED")
        for prefix in DEFERRED_PREFIXES:
            if frames["features"][f"{prefix}_available"].sum() != 0:
                raise ValueError("DEFERRED_MODEL_UNEXPECTEDLY_AVAILABLE")
        # The join is intentionally by every declared prediction ID, never by team string.
        source_rows = connection.execute("""SELECT prediction_id,match_id,is_oos,data_origin,
            training_cutoff,prediction_time FROM real_oos_predictions_v2
            WHERE match_date < DATE '2026-08-01'""").fetchall()
        source_by_id = {row[0]: row[1:] for row in source_rows}
        for match_id, encoded, dependencies in zip(feature_ids,
                frames["lineage"]["prediction_ids"],
                frames["lineage"]["dependency_tags"], strict=True):
            identifiers = json.loads(encoded)
            tags = json.loads(dependencies)
            for model_id, prediction_id in identifiers.items():
                if prediction_id is None:
                    continue
                entry = source_by_id.get(prediction_id)
                if entry is None or entry[0] != match_id or entry[1] is not True or entry[2] != "REAL":
                    raise ValueError("CANDIDATE_NON_REAL_OOS_LINEAGE")
                if entry[3] > entry[4]:
                    raise ValueError("CANDIDATE_FUTURE_TRAINING_LINEAGE")
                if model_id not in tags or "REAL_OOS" not in tags[model_id] or any(
                        "MARKET" in str(tag).upper() for tag in tags[model_id]):
                    raise ValueError("CANDIDATE_MARKET_OR_MISSING_DEPENDENCY_TAG")
    candidate = VerifiedCandidate(dataset_id, expected_hash, part_hashes,
        frames["features"], frames["labels"], frames["lineage"])
    candidate_dataset_cache.put(cache_key, candidate, tuple(parquet_paths))
    return candidate
