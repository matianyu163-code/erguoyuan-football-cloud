"""V7 required fields and status vocabulary remain stable."""

from erguoyuan_football.contracts.v7 import (
    V7ModelStatus,
    V7PredictionContract,
)


def test_v7_schema_contains_required_ordered_fields() -> None:
    assert tuple(V7PredictionContract.model_fields) == (
        "match", "competition", "teams", "probabilities", "model_status",
        "confidence", "evidence", "timestamp",
    )


def test_v7_model_status_vocabulary_is_fixed() -> None:
    assert {status.value for status in V7ModelStatus} == {
        "EXECUTED", "BLOCKED", "DEGRADED", "FAILED",
    }
