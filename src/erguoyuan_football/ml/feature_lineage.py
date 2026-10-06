"""Reject in-sample or future base-model outputs before feature construction."""

from __future__ import annotations

import hashlib

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.ml.schemas import (
    BasePredictionEvidence,
    MLFeatureVector,
    stable_hash,
)


class OOSFeatureValidator:
    """Positive proof that each successful training base feature was predicted OOS."""

    def validate(self, evidence: BasePredictionEvidence, snapshot: PredictionSnapshot, *,
                 for_training: bool) -> None:
        prediction = evidence.prediction
        if (prediction.match_id, prediction.prediction_snapshot_id, prediction.prediction_time,
                prediction.input_data_version) != (snapshot.match_id, snapshot.prediction_snapshot_id,
                                                   snapshot.prediction_time, snapshot.input_data_version):
            raise ValueError("BASE_FEATURE_SNAPSHOT_IDENTITY_MISMATCH")
        if prediction.prediction_time >= snapshot.match_data_snapshot.kickoff_time:
            raise ValueError("FUTURE_MODEL_PREDICTION_REJECTED")
        if prediction.training_end_time is not None and prediction.training_end_time > prediction.prediction_time:
            raise ValueError("FUTURE_BASE_MODEL_TRAINING_REJECTED")
        if snapshot.match_id in evidence.training_match_ids:
            raise ValueError("IN_SAMPLE_BASE_PREDICTION_REJECTED")
        if prediction.execution_status != ExecutionStatus.SUCCESS:
            return
        if not self._training_ids_hash_matches(prediction, evidence):
            raise ValueError("BASE_TRAINING_MATCH_IDS_HASH_MISMATCH")
        if for_training and (not prediction.is_oos or not evidence.base_prediction_oos or
                             prediction.training_end_time is None or not self._has_training_evidence(evidence)):
            raise ValueError("BASE_MODEL_FEATURE_MUST_BE_OOS")

    def validate_training_vector(self, vector: MLFeatureVector) -> None:
        """Recheck persisted evidence at fit time; builder checks alone are insufficient."""
        for evidence in vector.base_prediction_evidence:
            prediction = evidence.prediction
            date_safe = (vector.feature_version == "ML_FEATURE_V1_DATE_SAFE_BATCH" and
                         prediction.metadata.get("prediction_temporal_mode") == "DATE_SAFE_BATCH" and
                         prediction.metadata.get("data_origin") == "REAL")
            if (prediction.match_id, prediction.prediction_snapshot_id, prediction.prediction_time) != (
                    vector.match_id, vector.prediction_snapshot_id, vector.prediction_time) or (
                    not date_safe and prediction.input_data_version != vector.input_data_version):
                raise ValueError("BASE_FEATURE_SNAPSHOT_IDENTITY_MISMATCH")
            if (vector.match_id in evidence.training_match_ids or
                    prediction.training_end_time is not None and
                    prediction.training_end_time > vector.prediction_time):
                raise ValueError("IN_SAMPLE_OR_FUTURE_BASE_FEATURE_REJECTED")
            if prediction.execution_status != ExecutionStatus.SUCCESS:
                continue
            if not self._training_ids_hash_matches(prediction, evidence):
                raise ValueError("BASE_TRAINING_MATCH_IDS_HASH_MISMATCH")
            if (not prediction.is_oos or not evidence.base_prediction_oos or
                    prediction.training_end_time is None or
                    not self._has_training_evidence(evidence)):
                raise ValueError("IN_SAMPLE_OR_FUTURE_BASE_FEATURE_REJECTED")

    @staticmethod
    def _has_training_evidence(evidence: BasePredictionEvidence) -> bool:
        return bool(evidence.training_match_ids) or (
            evidence.verification_method == "REAL_OOS_STORE_RECOMPUTED" and
            evidence.verified_training_match_count is not None and
            evidence.verified_training_ids_hash is not None)

    @staticmethod
    def _training_ids_hash_matches(prediction, evidence: BasePredictionEvidence) -> bool:
        """Accept the documented Phase 8.1 pipe digest only for date-safe legacy rows."""
        claimed = prediction.metadata.get("training_match_ids_hash")
        if not evidence.training_match_ids:
            return (evidence.verification_method == "REAL_OOS_STORE_RECOMPUTED" and
                prediction.metadata.get("prediction_temporal_mode") == "DATE_SAFE_BATCH" and
                prediction.metadata.get("data_origin") == "REAL" and
                evidence.verified_training_match_count ==
                    prediction.metadata.get("training_match_count") and
                evidence.verified_training_ids_hash == claimed)
        ordered = sorted(evidence.training_match_ids)
        if claimed == stable_hash(ordered):
            return True
        if prediction.metadata.get("prediction_temporal_mode") != "DATE_SAFE_BATCH":
            return False
        legacy = hashlib.sha256("|".join(ordered).encode()).hexdigest()
        return claimed == legacy
