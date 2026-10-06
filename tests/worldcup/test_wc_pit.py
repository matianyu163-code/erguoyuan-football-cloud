"""PIT tests use synthetic timestamps and evidence only."""

from datetime import UTC, datetime, timedelta

import pytest

from erguoyuan_football.backtesting.worldcup_dataset import (
    WorldCupPrediction,
    WorldCupSourceEvidence,
    validate_worldcup_prediction_pit,
)


def _evidence(*, as_of: datetime, retrieved: datetime) -> WorldCupSourceEvidence:
    return WorldCupSourceEvidence(
        evidence_id="SYNTHETIC_TEST_INPUT",
        source="SYNTHETIC_TEST",
        source_url="https://example.invalid/synthetic",
        as_of_time=as_of,
        retrieved_at=retrieved,
    )


def test_wc_pit_allows_only_pre_match_inputs() -> None:
    prediction_time = datetime(2026, 6, 10, tzinfo=UTC)
    kickoff = prediction_time + timedelta(hours=2)
    evidence = _evidence(as_of=prediction_time - timedelta(minutes=1), retrieved=prediction_time)
    validate_worldcup_prediction_pit(
        prediction_time=prediction_time,
        kickoff_time=kickoff,
        training_end_time=prediction_time - timedelta(days=1),
        input_evidence=(evidence,),
    )
    prediction = WorldCupPrediction(
        match_id="SYNTHETIC_TEST_WC_1",
        model_id="SYNTHETIC_TEST_MODEL",
        model_version="test",
        prediction_time=prediction_time,
        kickoff_time=kickoff,
        training_end_time=prediction_time - timedelta(days=1),
        input_data_version="SYNTHETIC_TEST",
        input_evidence=(evidence,),
        status="SUCCESS",
        p_home=0.5,
        p_draw=0.25,
        p_away=0.25,
    )
    assert prediction.status == "SUCCESS"
    assert "result" not in prediction.model_dump()


@pytest.mark.parametrize("which", ["as_of", "retrieved"])
def test_wc_pit_rejects_future_input_evidence(which: str) -> None:
    prediction_time = datetime(2026, 6, 10, tzinfo=UTC)
    kickoff = prediction_time + timedelta(hours=2)
    later = prediction_time + timedelta(seconds=1)
    evidence = _evidence(
        as_of=later if which == "as_of" else prediction_time,
        retrieved=later if which == "retrieved" else prediction_time,
    )
    with pytest.raises(ValueError, match="FUTURE_DATA_DETECTED"):
        validate_worldcup_prediction_pit(
            prediction_time=prediction_time,
            kickoff_time=kickoff,
            training_end_time=prediction_time - timedelta(days=1),
            input_evidence=(evidence,),
        )


def test_wc_prediction_rejects_after_kickoff_training() -> None:
    kickoff = datetime(2026, 6, 10, tzinfo=UTC)
    with pytest.raises(ValueError, match="PREDICTION_MUST_PRECEDE_KICKOFF"):
        WorldCupPrediction(
            match_id="SYNTHETIC_TEST_WC_1",
            model_id="SYNTHETIC_TEST_MODEL",
            model_version="test",
            prediction_time=kickoff,
            kickoff_time=kickoff,
            training_end_time=kickoff - timedelta(days=1),
            input_data_version="SYNTHETIC_TEST",
            input_evidence=(_evidence(as_of=kickoff, retrieved=kickoff),),
            status="SUCCESS",
            p_home=0.5,
            p_draw=0.25,
            p_away=0.25,
        )
