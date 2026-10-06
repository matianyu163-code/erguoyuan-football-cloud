"""V7 evidence and output status are validated at the contract boundary."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from erguoyuan_football.contracts.v7 import V7PredictionContract


def _base(now: datetime) -> dict:
    return {
        "match": "M-1",
        "competition": "C-1",
        "teams": {"home_id": "T-H", "home_name": "Home",
                  "away_id": "T-A", "away_name": "Away"},
        "probabilities": {"home": 0.5, "draw": 0.25, "away": 0.25},
        "model_status": {"ELO_V1": "EXECUTED"},
        "confidence": "MEDIUM",
        "evidence": [{"provider": "P1", "source": "RESULTS",
                      "observed_at": now - timedelta(minutes=1),
                      "evidence_id": "E1"}],
        "timestamp": now,
    }


def test_v7_contract_accepts_audited_execution() -> None:
    now = datetime.now(UTC)
    output = V7PredictionContract(**_base(now))
    assert output.model_status["ELO_V1"] == "EXECUTED"


def test_v7_contract_rejects_future_evidence() -> None:
    now = datetime.now(UTC)
    payload = _base(now)
    payload["evidence"][0]["observed_at"] = now + timedelta(seconds=1)
    with pytest.raises(ValidationError, match="V7_EVIDENCE_AFTER_OUTPUT_TIMESTAMP"):
        V7PredictionContract(**payload)
