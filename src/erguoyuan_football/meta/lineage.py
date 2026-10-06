"""Recursive, date-safe ancestry checks for Phase 9 OOS candidates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb

from erguoyuan_football.backtesting.real_oos_matrix import (
    BASE_MODEL_IDS,
    ML_MODEL_IDS,
    lineage_hash,
)
from erguoyuan_football.meta.candidate import VerifiedCandidate
from erguoyuan_football.ml.artifacts import MLArtifact, file_sha256
from erguoyuan_football.ml.schemas import stable_hash


def assert_acyclic(graph: dict[str, tuple[str, ...]]) -> None:
    """Reject a prediction dependency cycle before any META fit."""
    active: set[str] = set()
    done: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            raise ValueError("LINEAGE_CYCLE")
        if node in done:
            return
        active.add(node)
        for parent in graph.get(node, ()):
            visit(parent)
        active.remove(node)
        done.add(node)

    for node in graph:
        visit(node)


@dataclass(frozen=True)
class LineageValidationReport:
    """Counts of verified source, nested and training-history links."""

    status: str
    candidate_rows: int
    base_predictions_checked: int
    ml_predictions_checked: int
    ml_artifacts_checked: int
    immediate_base_edges_checked: int
    training_match_ids_checked: int
    nested_training_vectors_checked: int
    model_versions: dict[str, str]


class PredictionLineageValidator:
    """Trace every candidate input to source OOS, artifact and earlier match IDs."""

    def __init__(self, db_path: str | Path, *, artifact_root: str | Path) -> None:
        self.db_path = Path(db_path)
        self.artifact_root = Path(artifact_root)

    def validate(self, candidate: VerifiedCandidate) -> LineageValidationReport:
        """Validate the frozen candidate without accessing final-holdout results."""
        with duckdb.connect(str(self.db_path), read_only=True) as connection:
            rows = connection.execute("""SELECT prediction_id,match_id,competition_id,model_id,
                training_cutoff,prediction_time,match_date,training_match_count,training_data_hash,
                config_hash,prediction_snapshot_id,prediction_temporal_mode,data_origin,is_oos,payload
                FROM real_oos_predictions_v2 WHERE match_date < DATE '2026-08-01'""").fetchall()
            indexed = dict(connection.execute("SELECT prediction_id,lineage_hash "
                "FROM real_oos_lineage_v2").fetchall())
            history = connection.execute("""SELECT match_id,competition_id,match_date
                FROM real_canonical_matches WHERE status='FINISHED'
                AND match_date < DATE '2026-08-01'""").fetchall()
        by_id = {row[0]: row for row in rows}
        model_versions: dict[str, str] = {}
        match_dates = {row[0]: row[2] for row in history}
        by_comp: dict[str, list[tuple[str, date]]] = {}
        for match_id, comp, day in history:
            by_comp.setdefault(comp, []).append((match_id, day))
        training_cache: dict[tuple[str, date], tuple[str, ...]] = {}
        artifacts: dict[str, MLArtifact] = {}
        visited: set[str] = set()
        graph: dict[str, tuple[str, ...]] = {}
        base_count = ml_count = edges = training_ids_checked = 0

        def visit(prediction_id: str, target_match_id: str) -> None:
            nonlocal base_count, ml_count, edges, training_ids_checked
            if prediction_id in visited:
                return
            row = by_id.get(prediction_id)
            if row is None:
                raise ValueError("LINEAGE_PREDICTION_NOT_FOUND")
            (pid, match_id, comp, model_id, cutoff, prediction_time, day,
             training_count, data_hash, config_hash, _snapshot_id, mode, origin, is_oos, payload) = row
            if (match_id != target_match_id or origin != "REAL" or is_oos is not True or
                    mode != "DATE_SAFE_BATCH" or day != prediction_time.date() or
                    cutoff > prediction_time or match_dates.get(match_id) != day):
                raise ValueError("LINEAGE_NON_REAL_OR_FUTURE_PREDICTION")
            record = json.loads(payload)
            metadata = record["metadata"]
            prior_version = model_versions.setdefault(model_id, record["model_version"])
            if prior_version != record["model_version"]:
                raise ValueError("LINEAGE_MIXED_BASE_MODEL_VERSIONS")
            if (record["prediction_id"] != pid or metadata.get("data_origin") != "REAL" or
                    metadata.get("training_data_hash") != data_hash or
                    metadata.get("config_hash", config_hash) != config_hash):
                raise ValueError("LINEAGE_PAYLOAD_OR_DATA_HASH_MISMATCH")
            declared_hash = metadata.get("lineage_hash")
            if declared_hash is not None and indexed.get(pid) != declared_hash:
                raise ValueError("LINEAGE_INDEX_HASH_MISMATCH")
            if model_id in BASE_MODEL_IDS:
                key = (comp, cutoff.date())
                if key not in training_cache:
                    training_cache[key] = tuple(sorted(item for item, match_day in by_comp[comp]
                        if cutoff.date() - timedelta(days=1095) <= match_day < cutoff.date()))
                training_ids = training_cache[key]
                valid_hashes = {stable_hash(list(training_ids)),
                    hashlib.sha256("|".join(training_ids).encode()).hexdigest()}
                if (target_match_id in training_ids or len(training_ids) != training_count or
                        metadata.get("training_match_ids_hash") not in valid_hashes or
                        any(match_dates[item] >= day for item in training_ids)):
                    raise ValueError("LINEAGE_BASE_TRAINING_IDS_INVALID")
                training_ids_checked += len(training_ids)
                base_count += 1
                graph[pid] = ()
            elif model_id in ML_MODEL_IDS:
                artifact_id = metadata.get("artifact_id")
                if not isinstance(artifact_id, str):
                    raise ValueError("LINEAGE_ML_ARTIFACT_MISSING")
                if artifact_id not in artifacts:
                    path = self.artifact_root / model_id / artifact_id / "manifest.json"
                    artifact = MLArtifact.model_validate_json(path.read_text(encoding="utf-8"))
                    if (artifact.model_id != model_id or artifact.feature_schema.mode != "NO_MARKET" or
                            artifact.training_metadata.get("uses_market") is not False or
                            artifact.dataset_hash != metadata.get("dataset_hash") or
                            artifact.config_hash != metadata.get("model_config_hash") or
                            file_sha256(path.parent / artifact.model_file) != artifact.payload_sha256):
                        raise ValueError("LINEAGE_ML_ARTIFACT_INVALID")
                    artifacts[artifact_id] = artifact
                artifact = artifacts[artifact_id]
                ids = artifact.training_match_ids
                if (target_match_id in ids or len(ids) != training_count or
                        stable_hash(sorted(ids)) != metadata.get("training_match_ids_hash") or
                        artifact.trained_until > prediction_time or
                        any(match_dates.get(item, day) >= day for item in ids)):
                    raise ValueError("LINEAGE_NESTED_ML_TRAINING_INVALID")
                dependencies = tuple(metadata.get("base_prediction_ids") or ())
                if not dependencies or metadata.get("uses_market") is not False:
                    raise ValueError("LINEAGE_ML_BASE_ANCESTRY_MISSING")
                graph[pid] = dependencies
                for parent in dependencies:
                    if parent == pid:
                        raise ValueError("LINEAGE_CYCLE")
                    visit(parent, match_id)
                    edges += 1
                training_ids_checked += len(ids)
                ml_count += 1
            else:
                raise ValueError("LINEAGE_UNKNOWN_MODEL")
            visited.add(pid)

        for match_id, encoded_ids, encoded_hashes in zip(
                candidate.features["match_id"], candidate.lineage["prediction_ids"],
                candidate.lineage["lineage_hashes"], strict=True):
            declared = json.loads(encoded_hashes)
            for model_id, pid in json.loads(encoded_ids).items():
                if pid is None:
                    continue
                row = by_id.get(pid)
                if row is None or row[3] != model_id:
                    raise ValueError("LINEAGE_CANDIDATE_MODEL_ID_MISMATCH")
                expected_hash = json.loads(row[14])["metadata"].get("lineage_hash") or lineage_hash(
                    (pid, row[1], row[3], row[4], row[5], row[8], row[9], row[10], row[11]))
                if declared[model_id] != expected_hash:
                    raise ValueError("LINEAGE_CANDIDATE_HASH_MISMATCH")
                visit(pid, match_id)
        assert_acyclic(graph)
        nested_vectors = 0
        # Check every actual ML training vector and recurse through its base-OOS evidence.
        # Both store paths are explicitly known Phase 8.2 inputs; no artifact-directory scan.
        stores = (self.db_path.parent / "phase8_2_ml_features.duckdb",
                  self.db_path.parent / "phase8_2_ml_features_compact.duckdb")
        for artifact in artifacts.values():
            ids = artifact.training_match_ids
            training_ids = tuple(ids)
            family = "XGBOOST" if artifact.model_id == "CORE_XGBOOST_V1" else "CATBOOST"
            vectors: dict[str, str] = {}
            for offset in range(0, len(training_ids), 500):
                batch = training_ids[offset:offset + 500]
                for store in stores:
                    if not store.is_file():
                        continue
                    with duckdb.connect(str(store), read_only=True) as feature_db:
                        for match_id, payload in feature_db.execute("""SELECT match_id,payload
                            FROM ml_feature_vectors WHERE model_family=?
                            AND match_id IN (SELECT unnest(?))""", [family, batch]).fetchall():
                            vectors.setdefault(match_id, payload)
            for training_match_id in training_ids:
                payload = vectors.get(training_match_id)
                if payload is None:
                    raise ValueError("LINEAGE_ML_TRAINING_FEATURE_NOT_FOUND")
                vector = json.loads(payload)
                if stable_hash({key: value for key, value in vector.items()
                                if key != "feature_data_hash"}) != vector.get("feature_data_hash"):
                    raise ValueError("LINEAGE_ML_FEATURE_CONTENT_HASH_MISMATCH")
                if (vector["match_id"] != training_match_id or vector["feature_mode"] != "NO_MARKET" or
                        vector["feature_schema_hash"] != artifact.feature_schema_hash or
                        date.fromisoformat(vector["prediction_time"][:10]) != match_dates[training_match_id] or
                        not vector["base_prediction_evidence"]):
                    raise ValueError("LINEAGE_ML_TRAINING_FEATURE_INVALID")
                for evidence in vector["base_prediction_evidence"]:
                    parent = evidence["prediction"]["prediction_id"]
                    if (evidence["prediction"]["match_id"] != training_match_id or
                            evidence.get("base_prediction_oos") is not True or
                            by_id.get(parent) is None or
                            json.loads(by_id[parent][14]) != evidence["prediction"]):
                        raise ValueError("LINEAGE_NESTED_BASE_EVIDENCE_MISMATCH")
                    visit(parent, training_match_id)
                    if training_match_id in evidence["training_match_ids"]:
                        raise ValueError("LINEAGE_NESTED_TARGET_IN_BASE_TRAINING")
                nested_vectors += 1
        return LineageValidationReport("PASS", len(candidate.features), base_count, ml_count,
            len(artifacts), edges, training_ids_checked, nested_vectors, model_versions)
