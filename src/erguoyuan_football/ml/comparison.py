"""Internal probability comparison; it never blends or calibrates models."""

from __future__ import annotations

from dataclasses import dataclass

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.contracts.predictions import ModelPrediction, ProbabilityVector


@dataclass(frozen=True)
class MLComparisonReport:
    match_id: str
    prediction_snapshot_id: str
    probabilities: dict[str, tuple[float, float, float] | None]
    xgb_cat_probability_distance: float | None
    diagnostic_only: bool = True


def compare_ml_predictions(xgboost: ModelPrediction, catboost: ModelPrediction, *,
                           best_base: ModelPrediction | None = None,
                           market_consensus: ProbabilityVector | None = None) -> MLComparisonReport:
    records = (xgboost, catboost) + ((best_base,) if best_base is not None else ())
    identity = (xgboost.match_id, xgboost.prediction_snapshot_id, xgboost.prediction_time)
    if any((record.match_id, record.prediction_snapshot_id, record.prediction_time) != identity
           for record in records):
        raise ValueError("ML_COMPARISON_IDENTITY_MISMATCH")
    if xgboost.model_id != "CORE_XGBOOST_V1" or catboost.model_id != "CORE_CATBOOST_V1":
        raise ValueError("ML_COMPARISON_MODEL_ID_MISMATCH")

    def values(prediction: ModelPrediction) -> tuple[float, float, float] | None:
        if prediction.execution_status != ExecutionStatus.SUCCESS:
            return None
        if prediction.p_home is None or prediction.p_draw is None or prediction.p_away is None:
            raise ValueError("SUCCESS_WITHOUT_PROBABILITIES")
        return prediction.p_home, prediction.p_draw, prediction.p_away

    xgb, cat = values(xgboost), values(catboost)
    probabilities = {xgboost.model_id: xgb, catboost.model_id: cat,
                     "BEST_EXISTING_BASE": values(best_base) if best_base is not None else None,
                     "MARKET_CONSENSUS": (market_consensus.p_home, market_consensus.p_draw,
                                          market_consensus.p_away) if market_consensus is not None else None}
    distance = sum(abs(a - b) for a, b in zip(xgb, cat, strict=True)) / 2 if xgb and cat else None
    return MLComparisonReport(match_id=identity[0], prediction_snapshot_id=identity[1],
                              probabilities=probabilities, xgb_cat_probability_distance=distance)
