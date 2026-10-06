"""Shared fail-closed checks for historical observations at model boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from erguoyuan_football.contracts.common import utc

FAILURE_CODE_TRAINING_CUTOFF_AFTER_PREDICTION = "TRAINING_CUTOFF_AFTER_PREDICTION"
_TIME_FIELDS = ("event_time", "kickoff_time", "match_time", "timestamp")


class FutureHistoryLeakageError(ValueError):
    """Historical observations contain information unavailable at prediction time."""

    def __init__(self, message: str, *, trained_until: datetime) -> None:
        super().__init__(message)
        self.trained_until = utc(trained_until)


def history_event_time(row: Any) -> datetime:
    """Extract the canonical event timestamp from a mapping or typed row."""
    for field in _TIME_FIELDS:
        value = row.get(field) if isinstance(row, dict) else getattr(row, field, None)
        if value is not None:
            if not isinstance(value, datetime):
                raise ValueError(f"historical {field} must be a datetime")
            return utc(value)
    raise ValueError("Historical observation does not expose a supported event time field.")


def validate_history_point_in_time(history: Iterable[Any], prediction_time: datetime) -> list[Any]:
    """Require every event time to be strictly earlier than prediction time.
    """
    # NEVER silently repair point-in-time leakage.
    # Future information at the model boundary breaks the temporal data contract.
    # Fail closed: no probabilities, lambdas, or score matrix.
    # Retain audit timestamps only to explain why execution was blocked.
    rows = list(history)
    at = utc(prediction_time)
    timed_rows = [(index, row, history_event_time(row)) for index, row in enumerate(rows)]
    offending = [(index, event_time) for index, _, event_time in timed_rows if event_time >= at]
    if offending:
        cutoff = max(event_time for _, event_time in offending)
        details = ", ".join(
            f"index={index}, event_time={event_time.isoformat()}" for index, event_time in offending[:20]
        )
        raise FutureHistoryLeakageError(
            "POINT_IN_TIME_GUARD_V1 blocked future information. "
            f"prediction_time={at.isoformat()}, violation_count={len(offending)}, violations=[{details}]",
            trained_until=cutoff,
        )
    return rows
