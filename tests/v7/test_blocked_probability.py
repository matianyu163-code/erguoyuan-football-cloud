"""Blocked or failed model records can never carry an invented 1X2 vector."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from erguoyuan_football.contracts.v7 import V7PredictionContract


def test_blocked_model_requires_null_probability() -> None:
    payload = {
        "match": "M-BLOCKED",
        "competition": "C-1",
        "teams": {"home_id": "T-H", "home_name": "Home",
                  "away_id": "T-A", "away_name": "Away"},
        "probabilities": {"home": 0.33, "draw": 0.33, "away": 0.34},
        "model_status": {"ELO_V1": "BLOCKED"},
        "confidence": "UNAVAILABLE",
        "timestamp": datetime.now(UTC),
    }
    with pytest.raises(ValidationError, match="V7_BLOCKED_OR_FAILED_MODELS_CANNOT_HAVE_PROBABILITIES"):
        V7PredictionContract(**payload)
