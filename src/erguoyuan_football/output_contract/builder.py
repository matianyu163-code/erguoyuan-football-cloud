"""Explicit Phase 8 builder; no aggregation or invented final probabilities."""

from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

from erguoyuan_football.contracts.common import now
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.meta.contracts import CalibratedCoreProbability
from erguoyuan_football.models.runner import BaseModelPredictionBundle
from erguoyuan_football.output_contract.adapters import BaseOutputAdapter
from erguoyuan_football.output_contract.schemas import (
    AvailabilityStatus,
    CanonicalPredictionResult,
    ProbabilityStage,
)


class CanonicalPredictionResultBuilder:
    """Select one named model as an intermediate view, or return UNAVAILABLE."""

    def from_calibrated_core_probability(self, core: CalibratedCoreProbability, *,
                                         competition_id: str, match_date: date,
                                         prediction_time: datetime,
                                         data_lineage: dict[str, object],
                                         model_lineage: dict[str, object]) -> CanonicalPredictionResult:
        """Transfer a verified development CORE probability to V2 without recalculation."""
        if prediction_time.date() != match_date or core.production_status != "NOT_PROMOTED":
            raise ValueError("CALIBRATED_CORE_DATE_OR_PROMOTION_INVALID")
        if data_lineage.get("market_dependency_count") != 0:
            raise ValueError("CALIBRATED_CORE_MARKET_DEPENDENCY")
        return CanonicalPredictionResult(
            prediction_id=core.prediction_id,
            prediction_snapshot_id=core.prediction_snapshot_id,
            match_id=core.match_id, competition_id=competition_id,
            kickoff_time=None, match_date=match_date,
            prediction_temporal_mode="DATE_SAFE_BATCH", prediction_time=prediction_time,
            prediction_horizon="DATE_SAFE_BATCH", data_origin="REAL",
            status=AvailabilityStatus.AVAILABLE, data_quality_status="DATE_SAFE_BATCH",
            pit_status="PASS", probability_stage=ProbabilityStage.FINAL_CORE_CALIBRATED,
            validation_status="DEVELOPMENT_ONLY", production_status="NOT_PROMOTED",
            p_home=core.calibrated_p_home, p_draw=core.calibrated_p_draw,
            p_away=core.calibrated_p_away,
            calibration_status=AvailabilityStatus.AVAILABLE,
            models_used=(*core.models_used, core.meta_model_id),
            models_unavailable=core.models_missing,
            data_lineage=data_lineage, model_lineage=model_lineage,
            created_at=core.created_at,
        )

    def build(self, fixture: Fixture, bundle: BaseModelPredictionBundle, *,
              prediction_snapshot_id: str, prediction_time: datetime,
              selected_model_id: str | None = None) -> CanonicalPredictionResult:
        for prediction in bundle.predictions:
            if (prediction.match_id != fixture.match_id or
                prediction.prediction_snapshot_id != prediction_snapshot_id or
                    prediction.prediction_time != prediction_time):
                raise ValueError("MIXED_PREDICTION_SNAPSHOT_LINEAGE")
        if selected_model_id is not None:
            selected = bundle.by_model().get(selected_model_id)
            if selected is None:
                raise ValueError("SELECTED_MODEL_NOT_IN_BUNDLE")
            result = BaseOutputAdapter().convert(fixture, selected)
            return result.model_copy(update={
                "models_unavailable": tuple(p.model_id for p in bundle.predictions
                                            if p.execution_status.value == "UNAVAILABLE"),
                "models_failed": tuple(p.model_id for p in bundle.predictions
                                       if p.execution_status.value == "FAILED"),
            })
        return CanonicalPredictionResult(
            prediction_id=str(uuid4()), prediction_snapshot_id=prediction_snapshot_id,
            match_id=fixture.match_id, competition_id=fixture.competition_id,
            kickoff_time=fixture.kickoff_time, prediction_time=prediction_time,
            prediction_horizon="UNSPECIFIED", data_origin="UNKNOWN",
            status=AvailabilityStatus.UNAVAILABLE, data_quality_status="UNVERIFIED",
            pit_status="UNVERIFIED", models_unavailable=tuple(p.model_id for p in bundle.predictions
                if p.execution_status.value == "UNAVAILABLE"),
            models_failed=tuple(p.model_id for p in bundle.predictions
                if p.execution_status.value == "FAILED"),
            data_lineage={"reason": "NO_EXPLICIT_INTERMEDIATE_MODEL_SELECTED"}, created_at=now(),
        )

    def from_date_safe_prediction(self, prediction: ModelPrediction, *, competition_id: str,
                                  match_date: date) -> CanonicalPredictionResult:
        """Build a PRE_META result without inventing an exact kickoff timestamp."""
        if prediction.metadata.get("prediction_temporal_mode") != "DATE_SAFE_BATCH":
            raise ValueError("DATE_SAFE_MODE_REQUIRED")
        if not prediction.is_oos or prediction.metadata.get("data_origin") != "REAL":
            raise ValueError("REAL_OOS_PREDICTION_REQUIRED")
        if prediction.prediction_time.date() != match_date:
            raise ValueError("DATE_SAFE_BOUNDARY_MISMATCH")
        return CanonicalPredictionResult(
            prediction_id=prediction.prediction_id,
            prediction_snapshot_id=prediction.prediction_snapshot_id,
            match_id=prediction.match_id, competition_id=competition_id,
            kickoff_time=None, match_date=match_date,
            prediction_temporal_mode="DATE_SAFE_BATCH",
            prediction_time=prediction.prediction_time, prediction_horizon="DATE_SAFE_BATCH",
            data_origin="REAL", status=AvailabilityStatus.AVAILABLE,
            data_quality_status="DATE_SAFE_BATCH", pit_status="PASS",
            probability_stage=ProbabilityStage.PRE_META,
            p_home=prediction.p_home, p_draw=prediction.p_draw, p_away=prediction.p_away,
            score_matrix_ref=prediction.prediction_id if prediction.score_matrix is not None else None,
            score_matrix_status=AvailabilityStatus.AVAILABLE if prediction.score_matrix is not None else
                AvailabilityStatus.UNAVAILABLE,
            models_used=(prediction.model_id,),
            data_lineage={"data_source": prediction.data_source,
                          "input_data_version": prediction.input_data_version,
                          "training_match_count": prediction.metadata.get("training_match_count"),
                          "training_ids_hash": prediction.metadata.get("training_match_ids_hash"),
                          "neutral_venue_policy": prediction.metadata.get("neutral_venue_policy")},
            model_lineage={"model_id": prediction.model_id, "model_version": prediction.model_version,
                           "trained_until": prediction.trained_until,
                           "prediction_temporal_mode": "DATE_SAFE_BATCH"},
            created_at=now(),
        )

    def from_exact_oos_prediction(self, prediction: ModelPrediction, *, competition_id: str,
                                  match_date: date, kickoff_time: datetime) -> CanonicalPredictionResult:
        """Build a PRE_META result using a verified UTC kickoff, without final-stage inference."""
        if prediction.metadata.get("prediction_temporal_mode") != "EXACT_UTC":
            raise ValueError("EXACT_UTC_MODE_REQUIRED")
        if not prediction.is_oos or prediction.metadata.get("data_origin") != "REAL":
            raise ValueError("REAL_OOS_PREDICTION_REQUIRED")
        if not prediction.prediction_time < kickoff_time:
            raise ValueError("PREDICTION_MUST_PRECEDE_KICKOFF")
        return CanonicalPredictionResult(
            prediction_id=prediction.prediction_id,
            prediction_snapshot_id=prediction.prediction_snapshot_id,
            match_id=prediction.match_id, competition_id=competition_id,
            kickoff_time=kickoff_time, prediction_time=prediction.prediction_time,
            prediction_horizon="EXACT_UTC_OOS", data_origin="REAL",
            status=AvailabilityStatus.AVAILABLE, data_quality_status="EXACT_UTC_OOS",
            pit_status="PASS", probability_stage=ProbabilityStage.PRE_META,
            p_home=prediction.p_home, p_draw=prediction.p_draw, p_away=prediction.p_away,
            score_matrix_ref=prediction.prediction_id if prediction.score_matrix is not None else None,
            score_matrix_status=AvailabilityStatus.AVAILABLE if prediction.score_matrix is not None else
                AvailabilityStatus.UNAVAILABLE,
            models_used=(prediction.model_id,),
            data_lineage={"data_source": prediction.data_source,
                          "input_data_version": prediction.input_data_version,
                          "training_match_count": prediction.metadata.get("training_match_count"),
                          "training_ids_hash": prediction.metadata.get("training_match_ids_hash"),
                          "neutral_venue_policy": prediction.metadata.get("neutral_venue_policy")},
            model_lineage={"model_id": prediction.model_id, "model_version": prediction.model_version,
                           "trained_until": prediction.trained_until,
                           "prediction_temporal_mode": "EXACT_UTC"},
            created_at=now(),
        )
