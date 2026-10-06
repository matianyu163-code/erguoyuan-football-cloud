"""PIT provenance required before official handicap or independent HTFT heads."""

from __future__ import annotations

from pydantic import model_validator

from erguoyuan_football.contracts.common import Contract, UTCTime
from erguoyuan_football.prediction_heads.heads import htft_top2


class OfficialHandicapEvidence(Contract):
    match_id: str
    home_handicap: int
    quote_id: str
    source: str
    source_time: UTCTime
    retrieved_at: UTCTime
    prediction_time: UTCTime
    kickoff_time: UTCTime

    @model_validator(mode="after")
    def verified(self) -> OfficialHandicapEvidence:
        if (self.source not in {"SPORTTERY_OFFICIAL_PUBLIC", "AUTHORIZED_PROVIDER"} or
            not self.quote_id or not self.source_time <= self.retrieved_at <= self.prediction_time
            < self.kickoff_time):
            raise ValueError("OFFICIAL_HANDICAP_PIT_EVIDENCE_INVALID")
        return self


class HalfFullTimeEvidence(Contract):
    match_id: str
    model_id: str
    model_version: str
    prediction_id: str
    prediction_snapshot_id: str
    trained_until: UTCTime
    prediction_time: UTCTime
    kickoff_time: UTCTime
    historical_half_time_source: str
    training_match_ids_hash: str
    target_excluded_from_training: bool
    is_oos: bool
    data_origin: str
    distribution: dict[str, float]

    @model_validator(mode="after")
    def independently_trained(self) -> HalfFullTimeEvidence:
        if (not self.model_id or not self.model_version or not self.prediction_id or
            not self.historical_half_time_source or not self.training_match_ids_hash or
            not self.target_excluded_from_training or not self.is_oos or
            self.data_origin != "REAL" or not self.trained_until <= self.prediction_time < self.kickoff_time):
            raise ValueError("HTFT_INDEPENDENT_OOS_LINEAGE_INVALID")
        htft_top2(self.distribution, independent_oos_model=True)
        return self
