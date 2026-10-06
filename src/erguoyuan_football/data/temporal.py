"""Data-class-specific historical eligibility for DATE_SAFE and exact-time forecasts."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from erguoyuan_football.contracts.common import Contract, Identifier, UTCTime, utc


class TemporalDataClass(StrEnum):
    EVENT_IMMUTABLE = "EVENT_IMMUTABLE"
    EVENT_TIME_SERIES = "EVENT_TIME_SERIES"
    SNAPSHOT_TIME_SERIES = "SNAPSHOT_TIME_SERIES"
    EXTERNAL_FORECAST = "EXTERNAL_FORECAST"
    MARKET_TIME_SERIES = "MARKET_TIME_SERIES"
    POST_MATCH_ONLY = "POST_MATCH_ONLY"


class TemporalQuality(StrEnum):
    VERIFIED_EVENT_TIME = "VERIFIED_EVENT_TIME"
    DATE_ONLY = "DATE_ONLY"
    UNVERIFIED_TIMEZONE = "UNVERIFIED_TIMEZONE"
    VERIFIED_HISTORICAL_SNAPSHOT = "VERIFIED_HISTORICAL_SNAPSHOT"
    RECONSTRUCTED_POINT_IN_TIME = "RECONSTRUCTED_POINT_IN_TIME"
    RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
    UNKNOWN = "UNKNOWN"


class PredictionTemporalMode(StrEnum):
    EXACT_UTC = "EXACT_UTC"
    DATE_SAFE_BATCH = "DATE_SAFE_BATCH"


class TemporalEvidence(Contract):
    """Evidence dates preserve event, as-of and retrieval time separately."""

    data_id: Identifier
    data_class: TemporalDataClass
    event_time: UTCTime | None = None
    event_date: date | None = None
    as_of_time: UTCTime | None = None
    retrieved_at: UTCTime
    source_id: Identifier
    reconstruction_method: str | None = None
    temporal_quality: TemporalQuality = TemporalQuality.UNKNOWN
    eligible_for_training: bool = False
    eligible_for_prediction: bool = False
    reason_codes: tuple[str, ...] = ()


class TemporalEligibilityPolicy:
    """Historical labels are event facts; snapshots and forecasts need PIT capture."""

    def evaluate(self, evidence: TemporalEvidence, *, prediction_time: datetime,
                 target_match_id: str | None = None,
                 target_event_date: date | None = None,
                 prediction_mode: PredictionTemporalMode = PredictionTemporalMode.EXACT_UTC) -> TemporalEvidence:
        at = utc(prediction_time)
        reasons: list[str] = []
        if evidence.data_class == TemporalDataClass.POST_MATCH_ONLY and evidence.data_id == target_match_id:
            reasons.append("TARGET_POST_MATCH_DATA")
        elif evidence.data_class == TemporalDataClass.EVENT_IMMUTABLE:
            exact_event_is_prior = evidence.event_time is not None and utc(evidence.event_time) < at
            date_only_event_is_prior = (
                prediction_mode == PredictionTemporalMode.DATE_SAFE_BATCH
                and evidence.event_date is not None
                and target_event_date is not None
                and evidence.event_date < target_event_date
            )
            if not (exact_event_is_prior or date_only_event_is_prior):
                reasons.append("EVENT_NOT_BEFORE_PREDICTION")
        elif evidence.data_class == TemporalDataClass.EVENT_TIME_SERIES:
            if evidence.event_time is None or utc(evidence.event_time) >= at:
                reasons.append("STATE_EVENT_NOT_BEFORE_PREDICTION")
            elif not evidence.reconstruction_method:
                reasons.append("STATE_RECONSTRUCTION_MISSING")
        else:
            if (evidence.as_of_time is None or utc(evidence.as_of_time) > at
                    or utc(evidence.retrieved_at) > at):
                reasons.append("HISTORICAL_SNAPSHOT_EVIDENCE_MISSING")
            elif evidence.data_class == TemporalDataClass.EXTERNAL_FORECAST and evidence.as_of_time is None:
                reasons.append("EXTERNAL_FORECAST_SNAPSHOT_MISSING")
        eligible = not reasons
        return evidence.model_copy(update={
            "eligible_for_training": eligible,
            "eligible_for_prediction": eligible,
            "reason_codes": tuple(reasons),
        })


class TemporalEligibilityValidator:
    """Fail closed at training and feature boundaries."""

    def __init__(self, policy: TemporalEligibilityPolicy | None = None):
        self.policy = policy or TemporalEligibilityPolicy()

    def require(self, evidence: TemporalEvidence, **context) -> TemporalEvidence:
        result = self.policy.evaluate(evidence, **context)
        if not result.eligible_for_training:
            raise ValueError("TEMPORAL_DATA_INELIGIBLE:" + ",".join(result.reason_codes))
        return result
