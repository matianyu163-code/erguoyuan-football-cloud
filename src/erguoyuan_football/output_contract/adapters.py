"""Structured Phase 8 adapters. Neither adapter selects or recommends wagers."""

from __future__ import annotations

from typing import Literal, cast

from erguoyuan_football.contracts.common import ExecutionStatus, now
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.output_contract.schemas import (
    AvailabilityStatus,
    CanonicalPredictionResult,
    ProbabilityStage,
)


class BaseOutputAdapter:
    """Promote one actual model result, keeping its intermediate-stage label."""

    def convert(self, fixture: Fixture, prediction: ModelPrediction) -> CanonicalPredictionResult:
        if fixture.match_id != prediction.match_id:
            raise ValueError("MODEL_MATCH_MISMATCH")
        success = prediction.execution_status == ExecutionStatus.SUCCESS
        origin = str(prediction.metadata.get("dataset_kind", "UNKNOWN"))
        if origin == "SYNTHETIC_TEST":
            origin = "SYNTHETIC"
        if origin not in {"REAL", "SYNTHETIC", "TEST_FIXTURE"}:
            origin = "UNKNOWN"
        typed_origin = cast(Literal["REAL", "SYNTHETIC", "TEST_FIXTURE", "UNKNOWN"], origin)
        return CanonicalPredictionResult(
            prediction_id=prediction.prediction_id,
            prediction_snapshot_id=prediction.prediction_snapshot_id,
            match_id=fixture.match_id,
            competition_id=fixture.competition_id,
            kickoff_time=fixture.kickoff_time,
            prediction_time=prediction.prediction_time,
            prediction_horizon="UNSPECIFIED",
            data_origin=typed_origin,
            status=AvailabilityStatus.AVAILABLE if success else AvailabilityStatus.UNAVAILABLE,
            data_quality_status=prediction.data_status.value,
            pit_status="PASS" if success else "UNVERIFIED",
            probability_stage=ProbabilityStage.ML_MODEL if success and prediction.model_id in {"XGBOOST", "CATBOOST"} else
                ProbabilityStage.BASE_MODEL if success else None,
            p_home=prediction.p_home,
            p_draw=prediction.p_draw,
            p_away=prediction.p_away,
            score_matrix_ref=prediction.prediction_id if prediction.score_matrix is not None else None,
            score_matrix_status=AvailabilityStatus.AVAILABLE if prediction.score_matrix is not None else
                AvailabilityStatus.UNAVAILABLE,
            models_used=(prediction.model_id,) if success else (),
            models_unavailable=(prediction.model_id,) if prediction.execution_status == ExecutionStatus.UNAVAILABLE else (),
            models_failed=(prediction.model_id,) if prediction.execution_status == ExecutionStatus.FAILED else (),
            data_lineage={"source": prediction.data_source, "input_data_version": prediction.input_data_version},
            model_lineage={"model_id": prediction.model_id, "model_version": prediction.model_version,
                           "trained_until": prediction.trained_until, "is_oos": prediction.is_oos},
            created_at=now(),
        )


class CoreReportV2PreviewAdapter:
    """Machine-readable preview only; the Phase 11 Chinese report is absent."""

    def render(self, result: CanonicalPredictionResult) -> dict[str, object]:
        return result.model_dump(mode="json")
