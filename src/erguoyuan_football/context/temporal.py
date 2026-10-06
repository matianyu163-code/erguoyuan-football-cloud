"""Fail-closed point-in-time eligibility for reconstructed and snapshot context."""

from __future__ import annotations

from datetime import date, datetime

from erguoyuan_football.context.schemas import (
    ContextDataClass,
    HistoricalMatchEvent,
    InjuryEvidence,
    LineupEvidenceSnapshot,
)
from erguoyuan_football.contracts.common import utc


class ContextTemporalPolicy:
    """Apply strict prior-date rules to DATE_SAFE and strict timestamps to live data."""

    @staticmethod
    def require_event(event: HistoricalMatchEvent, *, target_match_id: str,
                      target_date: date, prediction_time: datetime,
                      date_safe: bool = True) -> None:
        if event.match_id == target_match_id:
            raise ValueError("TARGET_MATCH_RESULT_REJECTED")
        if date_safe:
            if event.match_date >= target_date:
                raise ValueError("CONTEXT_EVENT_NOT_BEFORE_TARGET_DATE")
        elif event.as_of_time > utc(prediction_time) or event.retrieved_at > utc(prediction_time):
            raise ValueError("CONTEXT_EVENT_NOT_AVAILABLE_AT_PREDICTION_TIME")

    @staticmethod
    def require_lineup(evidence: LineupEvidenceSnapshot, *, target_match_id: str,
                       prediction_time: datetime, kickoff_time: datetime | None) -> None:
        if evidence.match_id != target_match_id:
            raise ValueError("LINEUP_MATCH_ID_MISMATCH")
        if evidence.data_class in {ContextDataClass.POST_MATCH_CONTEXT,
                                   ContextDataClass.UNKNOWN_CONTEXT}:
            raise ValueError("POST_MATCH_LINEUP_REJECTED")
        if evidence.retrieved_at > utc(prediction_time):
            raise ValueError("LINEUP_RETRIEVED_AFTER_PREDICTION")
        if evidence.source_time >= (utc(kickoff_time) if kickoff_time else utc(prediction_time)):
            raise ValueError("POST_KICKOFF_LINEUP_REJECTED")

    @staticmethod
    def require_injury(evidence: InjuryEvidence, *, target_match_id: str,
                       prediction_time: datetime) -> None:
        if evidence.match_id != target_match_id:
            raise ValueError("INJURY_MATCH_ID_MISMATCH")
        if evidence.published_at > utc(prediction_time) or evidence.retrieved_at > utc(prediction_time):
            raise ValueError("FUTURE_INJURY_EVIDENCE_REJECTED")

