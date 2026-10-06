"""Development-only CALIBRATED CORE P and CORE_OUTPUT_V2 integration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date
from pathlib import Path

import numpy as np

from erguoyuan_football.contracts.common import ExecutionStatus, now
from erguoyuan_football.contracts.predictions import CorePrediction, ProbabilityVector
from erguoyuan_football.data.connection import connect_project_duckdb
from erguoyuan_football.meta.artifact import MetaArtifact
from erguoyuan_football.meta.candidate import (
    DEPENDENCY_TAGS,
    MODEL_PREFIXES,
    VerifiedCandidate,
)
from erguoyuan_football.meta.contracts import (
    CalibratedCoreProbability,
    MetaRawPrediction,
    ModelDisagreementReport,
)
from erguoyuan_football.meta.diagnostics import measure_disagreement
from erguoyuan_football.meta.model import MetaRows
from erguoyuan_football.ml.schemas import stable_hash
from erguoyuan_football.output_contract.builder import CanonicalPredictionResultBuilder
from erguoyuan_football.output_contract.schemas import CanonicalPredictionResult


@dataclass(frozen=True)
class DevelopmentCoreOutput:
    """Separate CORE and canonical records with the same fitted probabilities."""

    core: CorePrediction
    meta_raw: MetaRawPrediction
    calibrated_core: CalibratedCoreProbability
    disagreement: ModelDisagreementReport
    canonical: CanonicalPredictionResult


def build_development_core(db_path: str | Path, *, candidate: VerifiedCandidate,
                           artifact: MetaArtifact, match_id: str,
                           evaluation_start: date, evaluation_end: date,
                           _source_records: dict[str, tuple[object, ...]] | None = None,
                           _probability_pair: tuple[np.ndarray, np.ndarray] | None = None
                           ) -> DevelopmentCoreOutput:
    """Build only on a later development-evaluation match, never on fit rows."""
    found = candidate.features.index[candidate.features["match_id"] == match_id].tolist()
    if len(found) != 1:
        raise ValueError("CORE_MATCH_NOT_UNIQUE_IN_CANDIDATE")
    index = int(found[0])
    feature = candidate.features.iloc[index]
    prediction_date = date.fromisoformat(str(feature["prediction_date"]))
    if not evaluation_start <= prediction_date <= evaluation_end or prediction_date >= date(2026, 8, 1):
        raise ValueError("CORE_DEVELOPMENT_EVAL_WINDOW_REQUIRED")
    if (artifact.manifest["candidate_data_hash"] != candidate.data_hash or
            artifact.manifest["candidate_part_hashes"] != candidate.part_hashes or
            artifact.manifest["candidate_dataset_schema_hash"] != stable_hash([
                (name, str(dtype)) for name, dtype in candidate.features.dtypes.items()])):
        raise ValueError("CORE_ARTIFACT_CANDIDATE_MISMATCH")
    rows = MetaRows(candidate.features.iloc[[index]].copy(),
                    np.asarray([candidate.labels.iloc[index]["target"]], dtype=int),
                    np.asarray([index]))
    if _probability_pair is None:
        raw, calibrated = artifact.predict_many(rows)
    else:
        raw = _probability_pair[0].reshape(1, 3)
        calibrated = _probability_pair[1].reshape(1, 3)
    raw_vector = ProbabilityVector(p_home=float(raw[0, 0]), p_draw=float(raw[0, 1]),
                                   p_away=float(raw[0, 2]))
    final_vector = ProbabilityVector(p_home=float(calibrated[0, 0]),
                                     p_draw=float(calibrated[0, 1]),
                                     p_away=float(calibrated[0, 2]))
    ids = json.loads(candidate.lineage.iloc[index]["prediction_ids"])
    used = tuple(model_id for model_id, prediction_id in ids.items() if prediction_id is not None)
    unavailable = tuple(model_id for model_id, prediction_id in ids.items() if prediction_id is None)
    source_ids = [pid for pid in ids.values() if pid is not None]
    if _source_records is None:
        with connect_project_duckdb(db_path, read_only=True) as connection:
            source = connection.execute("""SELECT prediction_id,prediction_snapshot_id,prediction_time,
                match_id FROM real_oos_predictions_v2 WHERE prediction_id IN (SELECT unnest(?))""",
                [source_ids]).fetchall()
    else:
        source = [_source_records[prediction_id] for prediction_id in source_ids
                  if prediction_id in _source_records]
    if len(source) != len(source_ids) or any(row[3] != match_id for row in source):
        raise ValueError("CORE_SOURCE_PREDICTION_LINEAGE_MISSING")
    prediction_times = {row[2].astimezone(UTC) for row in source}
    if len(prediction_times) != 1:
        raise ValueError("CORE_SOURCE_PREDICTION_TIMES_DIFFER")
    prediction_time = next(iter(prediction_times))
    if prediction_time.date() != prediction_date:
        raise ValueError("CORE_DATE_SAFE_TIME_MISMATCH")
    source_snapshots = tuple(sorted({row[1] for row in source}))
    composite_snapshot = stable_hash((match_id, candidate.dataset_id, source_snapshots))[:32]
    prediction_id = stable_hash((match_id, artifact.artifact_id, composite_snapshot))[:32]
    dependency_summary = {
        "market_dependency_count": 0,
        "tags_by_model": {prefix: DEPENDENCY_TAGS[prefix] for prefix in MODEL_PREFIXES
                          if int(feature[f"{prefix}_available"]) == 1},
    }
    availability_pattern = "+".join(prefix for prefix in MODEL_PREFIXES
                                    if int(feature[f"{prefix}_available"]) == 1)
    created = now()
    meta_raw = MetaRawPrediction(
        match_id=match_id, prediction_snapshot_id=composite_snapshot,
        meta_model_id="META_NO_MARKET_V1", meta_model_version=artifact.manifest["model_version"],
        p_home=raw_vector.p_home, p_draw=raw_vector.p_draw, p_away=raw_vector.p_away,
        models_used=used, models_missing=unavailable, availability_pattern=availability_pattern,
        dependency_summary=dependency_summary, temporal_mode="DATE_SAFE_BATCH",
        validation_status="DEVELOPMENT_VALIDATED_DATE_SAFE",
    )
    calibrated_core = CalibratedCoreProbability(
        prediction_id=prediction_id, prediction_snapshot_id=composite_snapshot,
        match_id=match_id, meta_model_id="META_NO_MARKET_V1",
        meta_model_version=artifact.manifest["model_version"],
        meta_raw_p_home=raw_vector.p_home, meta_raw_p_draw=raw_vector.p_draw,
        meta_raw_p_away=raw_vector.p_away, calibrated_p_home=final_vector.p_home,
        calibrated_p_draw=final_vector.p_draw, calibrated_p_away=final_vector.p_away,
        calibrator_id=artifact.manifest["calibrator_id"], calibrator_version="1.0.0",
        models_used=used, models_missing=unavailable, dependency_summary=dependency_summary,
        temporal_mode="DATE_SAFE_BATCH", validation_status="DEVELOPMENT_VALIDATED_DATE_SAFE",
        created_at=created,
    )
    probabilities = [(float(feature[f"{prefix}_home"]), float(feature[f"{prefix}_draw"]),
                      float(feature[f"{prefix}_away"]))
                     for prefix in MODEL_PREFIXES if int(feature[f"{prefix}_available"]) == 1]
    disagreement = measure_disagreement(match_id, probabilities)
    core = CorePrediction(
        match_id=match_id, prediction_snapshot_id=composite_snapshot,
        prediction_time=prediction_time,
        validation_status="DEVELOPMENT_VALIDATED_DATE_SAFE",
        source_prediction_ids=tuple(sorted(source_ids)),
        source_snapshot_ids=source_snapshots,
        meta_artifact_id=artifact.artifact_id,
        meta_raw_probability=raw_vector, calibrated_probability=final_vector,
        final_core_probability=final_vector,
        execution_status=ExecutionStatus.SUCCESS, reason=None,
    )
    canonical = CanonicalPredictionResultBuilder().from_calibrated_core_probability(
        calibrated_core, competition_id=str(feature["competition_id"]),
        match_date=prediction_date, prediction_time=prediction_time,
        data_lineage={
            "candidate_dataset_id": candidate.dataset_id,
            "candidate_data_hash": candidate.data_hash,
            "candidate_part_hashes": candidate.part_hashes,
            "source_prediction_ids": ids,
            "source_snapshot_ids": source_snapshots,
            "composite_snapshot_policy": "SHA256_OF_SOURCE_SNAPSHOT_IDS",
            "temporal_mode": "DATE_SAFE_BATCH",
            "market_dependency_count": 0,
            "validation_status": "DEVELOPMENT_ONLY",
        },
        model_lineage={
            "model_id": "META_NO_MARKET_V1",
            "model_version": artifact.manifest["model_version"],
            "artifact_id": artifact.artifact_id,
            "trained_until": artifact.manifest["meta_trained_until"],
            "calibration_fit_end": artifact.manifest["calibration_fit_end"],
            "calibration_method": artifact.manifest["selected_calibration_method"],
            "raw_meta_probability": raw_vector.model_dump(),
            "validation_status": "DEVELOPMENT_VALIDATED_DATE_SAFE",
            "production_compatibility": "RESEARCH_ONLY_NOT_LIVE_VALIDATED",
        },
    )
    return DevelopmentCoreOutput(core, meta_raw, calibrated_core, disagreement, canonical)


def build_development_core_many(db_path: str | Path, *, candidate: VerifiedCandidate,
                                artifact: MetaArtifact, match_ids: list[str],
                                evaluation_start: date, evaluation_end: date
                                ) -> list[DevelopmentCoreOutput]:
    """Build development outputs with one META batch and one batched lineage query."""
    if not match_ids or len(match_ids) != len(set(match_ids)):
        raise ValueError("CORE_BATCH_MATCH_IDS_EMPTY_OR_DUPLICATE")
    positions: list[int] = []
    for match_id in match_ids:
        found = candidate.features.index[candidate.features["match_id"] == match_id].tolist()
        if len(found) != 1:
            raise ValueError("CORE_MATCH_NOT_UNIQUE_IN_CANDIDATE")
        positions.append(int(found[0]))
    batch = MetaRows(candidate.features.iloc[positions].copy(),
                     candidate.labels.iloc[positions]["target"].to_numpy(dtype=int),
                     np.asarray(positions))
    raw, calibrated = artifact.predict_many(batch)
    identifiers = [json.loads(candidate.lineage.iloc[position]["prediction_ids"])
                   for position in positions]
    source_ids = [prediction_id for row_ids in identifiers
                  for prediction_id in row_ids.values() if prediction_id is not None]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("CORE_BATCH_SOURCE_PREDICTION_DUPLICATE")
    with connect_project_duckdb(db_path, read_only=True) as connection:
        rows = connection.execute("""SELECT prediction_id,prediction_snapshot_id,prediction_time,
            match_id FROM real_oos_predictions_v2
            WHERE prediction_id IN (SELECT unnest(?))""", [source_ids]).fetchall()
    source_records = {row[0]: row for row in rows}
    if len(source_records) != len(source_ids):
        raise ValueError("CORE_SOURCE_PREDICTION_LINEAGE_MISSING")
    outputs = []
    for index, match_id in enumerate(match_ids):
        outputs.append(build_development_core(
            db_path, candidate=candidate, artifact=artifact, match_id=match_id,
            evaluation_start=evaluation_start, evaluation_end=evaluation_end,
            _source_records=source_records,
            _probability_pair=(raw[index], calibrated[index])))
    return outputs
