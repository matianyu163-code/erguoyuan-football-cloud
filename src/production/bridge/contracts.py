"""Wire contracts for externally researched facts and CORE execution results."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from production.bridge import BRIDGE_VERSION


class BridgeContract(BaseModel):
    """Reject undeclared protocol fields instead of silently dropping them."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceEvidence(BridgeContract):
    """Source citation with acquisition and publication chronology."""

    source: str
    source_url: str
    source_tier: Literal["A", "B", "C"]
    fetched_at: datetime
    published_at: datetime | None = None
    evidence_type: str = "FIXTURE"


class BridgeMatch(BridgeContract):
    """Fixture metadata asserted by the external research process."""

    jc_match_code: str | None = None
    competition: str
    home: str
    away: str
    source_home_name: str | None = None
    source_away_name: str | None = None
    kickoff_original: str
    kickoff_timezone: str
    kickoff_utc: datetime
    source_type: Literal["USER_JC_CONFIRMED", "AUTO_DISCOVERY", "RESEARCH_TEST"]
    neutral_venue: bool | None = None


class BridgeEntities(BridgeContract):
    """Expected canonical IDs; CORE resolves names and compares them again."""

    home_entity: str
    away_entity: str


class FixtureEvidence(BridgeContract):
    """At least one cited fixture source is required for a real snapshot."""

    sources: tuple[SourceEvidence, ...] = Field(min_length=1)


class HistoryMatch(BridgeContract):
    """One completed result with exact source and temporal evidence."""

    date: str
    kickoff: datetime | None = None
    competition: str
    home: str
    away: str
    source_home_name: str | None = None
    source_away_name: str | None = None
    home_entity: str | None = None
    away_entity: str | None = None
    match_id: str | None = None
    home_goals: int = Field(ge=0, le=30, strict=True)
    away_goals: int = Field(ge=0, le=30, strict=True)
    neutral_venue: bool
    source: str
    source_url: str
    source_tier: Literal["A", "B", "C"]
    fetched_at: datetime
    published_at: datetime | None = None


class BridgeHistory(BridgeContract):
    """Categories are provenance hints; CORE reclassifies canonical samples."""

    home_matches: tuple[HistoryMatch, ...] = ()
    away_matches: tuple[HistoryMatch, ...] = ()
    direct_matches: tuple[HistoryMatch, ...] = ()
    competition_matches: tuple[HistoryMatch, ...] = ()

    def all_matches(self) -> tuple[HistoryMatch, ...]:
        """Return every imported observation before canonical deduplication."""
        return (self.home_matches + self.away_matches + self.direct_matches
                + self.competition_matches)


class OptionalResearch(BridgeContract):
    """Untrusted optional observations retained for audit, never auto-promoted."""

    odds: tuple[dict[str, Any], ...] = ()
    xg: tuple[dict[str, Any], ...] = ()
    lineup: tuple[dict[str, Any], ...] = ()
    injuries: tuple[dict[str, Any], ...] = ()
    rankings: tuple[dict[str, Any], ...] = ()
    news: tuple[dict[str, Any], ...] = ()


class CoreDataPacketV1(BridgeContract):
    """Only source-backed observations may cross the bridge boundary."""

    schema_version: Literal["CORE_DATA_PACKET_V1"]
    request_id: str
    created_at: datetime
    research_as_of: datetime
    match: BridgeMatch
    entities: BridgeEntities
    fixture_evidence: FixtureEvidence
    history: BridgeHistory
    optional: OptionalResearch = Field(default_factory=OptionalResearch)

    @model_validator(mode="after")
    def chronology(self) -> CoreDataPacketV1:
        """Require aware timestamps before deeper source and PIT validation."""
        stamps = (self.created_at, self.research_as_of, self.match.kickoff_utc)
        if any(stamp.tzinfo is None or stamp.utcoffset() is None for stamp in stamps):
            raise ValueError("BRIDGE_TIMESTAMP_TIMEZONE_REQUIRED")
        if not all((self.request_id.strip(), self.match.home.strip(), self.match.away.strip(),
                    self.match.competition.strip(), self.entities.home_entity.strip(),
                    self.entities.away_entity.strip())):
            raise ValueError("BRIDGE_INPUT_INVALID")
        return self


class CoreResultPacketV1(BridgeContract):
    """Machine-readable outcome; null probability means no CORE promotion."""

    schema_version: Literal["CORE_RESULT_PACKET_V1"] = "CORE_RESULT_PACKET_V1"
    bridge_version: str = BRIDGE_VERSION
    build_id: str
    request_id: str
    prediction_id: str | None = None
    status: str
    primary_blocker: str | None = None
    snapshot_id: str | None = None
    models_planned: tuple[str, ...] = ()
    models_executed: tuple[str, ...] = ()
    models_failed: tuple[str, ...] = ()
    model_versions: dict[str, str] = Field(default_factory=dict)
    raw_probabilities: dict[str, dict[str, float]] = Field(default_factory=dict)
    core_probability: dict[str, float] | None = None
    v7_output: str
    warnings: tuple[str, ...] = ()
    packet_hash: str | None = None
