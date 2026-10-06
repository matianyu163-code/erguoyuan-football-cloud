"""Strict timestamp guards for the Phase 15 replay boundary."""

from datetime import UTC, datetime, timedelta

import pytest

from erguoyuan_football.backtesting.golden_selector import validate_oos_timestamps


def test_pit_accepts_only_inputs_available_by_prediction_time() -> None:
    kickoff = datetime(2026, 8, 10, 18, tzinfo=UTC)
    prediction = kickoff - timedelta(hours=12)
    validate_oos_timestamps(
        training_end_time=prediction - timedelta(days=1),
        prediction_time=prediction, kickoff_time=kickoff,
        input_timestamps=(prediction - timedelta(minutes=1), prediction),
    )


def test_pit_rejects_future_input() -> None:
    kickoff = datetime(2026, 8, 10, 18, tzinfo=UTC)
    prediction = kickoff - timedelta(hours=12)
    with pytest.raises(ValueError, match="FUTURE_DATA_DETECTED"):
        validate_oos_timestamps(
            training_end_time=prediction, prediction_time=prediction,
            kickoff_time=kickoff, input_timestamps=(prediction + timedelta(seconds=1),),
        )


def test_pit_rejects_prediction_at_kickoff() -> None:
    kickoff = datetime(2026, 8, 10, 18, tzinfo=UTC)
    with pytest.raises(ValueError, match="PREDICTION_NOT_BEFORE_KICKOFF"):
        validate_oos_timestamps(
            training_end_time=kickoff - timedelta(days=1),
            prediction_time=kickoff, kickoff_time=kickoff,
        )
