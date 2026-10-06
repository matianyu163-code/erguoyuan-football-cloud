"""Historical time eligibility is specific to the information's data class."""

from datetime import UTC, date, datetime

from erguoyuan_football.data.temporal import (
    PredictionTemporalMode,
    TemporalDataClass,
    TemporalEligibilityPolicy,
    TemporalEvidence,
    TemporalQuality,
)


def evidence(data_class: TemporalDataClass, **values) -> TemporalEvidence:
    fields = {"data_id": "event-1", "data_class": data_class,
              "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC), "source_id": "SOURCE",
              "temporal_quality": TemporalQuality.RETRIEVAL_ONLY}
    fields.update(values)
    return TemporalEvidence(**fields)


def test_event_immutable_late_retrieval_allowed() -> None:
    item = evidence(TemporalDataClass.EVENT_IMMUTABLE,
                    event_time=datetime(2022, 1, 1, tzinfo=UTC))
    result = TemporalEligibilityPolicy().evaluate(item,
        prediction_time=datetime(2023, 1, 1, tzinfo=UTC))
    assert result.eligible_for_training and result.eligible_for_prediction


def test_market_late_retrieval_not_historical_snapshot() -> None:
    item = evidence(TemporalDataClass.MARKET_TIME_SERIES,
        as_of_time=datetime(2022, 1, 1, tzinfo=UTC))
    result = TemporalEligibilityPolicy().evaluate(item,
        prediction_time=datetime(2023, 1, 1, tzinfo=UTC))
    assert not result.eligible_for_prediction
    assert "HISTORICAL_SNAPSHOT_EVIDENCE_MISSING" in result.reason_codes


def test_external_forecast_requires_snapshot() -> None:
    item = evidence(TemporalDataClass.EXTERNAL_FORECAST)
    result = TemporalEligibilityPolicy().evaluate(item,
        prediction_time=datetime(2023, 1, 1, tzinfo=UTC))
    assert not result.eligible_for_training


def test_post_match_target_data_rejected() -> None:
    item = evidence(TemporalDataClass.POST_MATCH_ONLY)
    result = TemporalEligibilityPolicy().evaluate(item,
        prediction_time=datetime(2023, 1, 1, tzinfo=UTC), target_match_id="event-1")
    assert not result.eligible_for_prediction
    assert result.reason_codes == ("TARGET_POST_MATCH_DATA",)


def test_reconstructed_state_allowed() -> None:
    item = evidence(TemporalDataClass.EVENT_TIME_SERIES,
        event_time=datetime(2022, 1, 1, tzinfo=UTC),
        reconstruction_method="ORDERED_IMMUTABLE_RESULTS_V1",
        temporal_quality=TemporalQuality.RECONSTRUCTED_POINT_IN_TIME)
    result = TemporalEligibilityPolicy().evaluate(item,
        prediction_time=datetime(2023, 1, 1, tzinfo=UTC))
    assert result.eligible_for_training


def test_date_only_event_uses_strictly_prior_date_only_in_batch_mode() -> None:
    item = evidence(TemporalDataClass.EVENT_IMMUTABLE, event_date=date(2025, 9, 30))
    result = TemporalEligibilityPolicy().evaluate(item,
        prediction_time=datetime(2025, 10, 1, tzinfo=UTC),
        target_event_date=date(2025, 10, 1),
        prediction_mode=PredictionTemporalMode.DATE_SAFE_BATCH)
    assert result.eligible_for_prediction


def test_same_date_event_is_not_prior_in_batch_mode() -> None:
    item = evidence(TemporalDataClass.EVENT_IMMUTABLE, event_date=date(2025, 10, 1))
    result = TemporalEligibilityPolicy().evaluate(item,
        prediction_time=datetime(2025, 10, 1, tzinfo=UTC),
        target_event_date=date(2025, 10, 1),
        prediction_mode=PredictionTemporalMode.DATE_SAFE_BATCH)
    assert not result.eligible_for_prediction


def test_future_immutable_event_is_rejected_even_if_retrieved_early() -> None:
    item = evidence(TemporalDataClass.EVENT_IMMUTABLE,
        event_time=datetime(2025, 10, 2, tzinfo=UTC),
        retrieved_at=datetime(2025, 10, 1, tzinfo=UTC))
    result = TemporalEligibilityPolicy().evaluate(item,
        prediction_time=datetime(2025, 10, 1, tzinfo=UTC))
    assert not result.eligible_for_prediction
