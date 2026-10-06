"""Internal comparison report; presentation keeps nulls explicit."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.data.schemas import Fixture


def comparison_report(match_label: str | Fixture, predictions: Iterable[ModelPrediction] | Any) -> dict[str, Any]:
    """Build a transparent primitive report without filling unavailable results."""
    if hasattr(predictions, "predictions"):
        predictions = predictions.predictions
    if isinstance(match_label, Fixture):
        label = f"{match_label.home_team_id} vs {match_label.away_team_id}"
    else:
        label = match_label
    return {"match": label, "models": [
        {"model_id": prediction.model_id, "execution_status": prediction.execution_status.value,
         "p_home": prediction.p_home, "p_draw": prediction.p_draw, "p_away": prediction.p_away,
         "lambda_home": prediction.lambda_home, "lambda_away": prediction.lambda_away,
         "metadata": prediction.metadata, "reason": prediction.reason}
        for prediction in predictions
    ]}
