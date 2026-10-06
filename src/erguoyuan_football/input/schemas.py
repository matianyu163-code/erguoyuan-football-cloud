"""User input and identified fixture contracts; unresolved IDs stay null."""

from enum import StrEnum
from uuid import uuid4

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import (
    Contract,
    Identifier,
    Probability,
    UTCTime,
    now,
)
from erguoyuan_football.match_source.match_source import MatchSourceType


class InputType(StrEnum):
    TEXT = "TEXT"
    SCREENSHOT = "SCREENSHOT"
    MANUAL = "MANUAL"
    BATCH = "BATCH"


class ResolutionStatus(StrEnum):
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"
    INVALID = "INVALID"


class MatchRequest(Contract):
    request_id: Identifier = Field(default_factory=lambda: str(uuid4()))
    created_at: UTCTime = Field(default_factory=now)
    input_type: InputType
    raw_text: str | None = None
    image_path: str | None = None
    source: Identifier
    match_id: str | None = None
    lottery_match_no: str | None = None
    competition_id: str | None = None
    competition_name: str | None = None
    home_team_id: str | None = None
    home_team_name: str | None = None
    away_team_id: str | None = None
    away_team_name: str | None = None
    kickoff_time: UTCTime | None = None
    input_confidence: Probability | None = None
    resolution_status: ResolutionStatus = ResolutionStatus.NOT_FOUND
    reason: str | None = "NOT_RESOLVED"
    candidates: tuple[str, ...] = ()
    # Preserve unparsed names: never split an English team name speculatively.
    team_query: str | None = None
    match_source_type: MatchSourceType = MatchSourceType.AUTO_DISCOVERY
    jc_confirmed: bool = False

    @model_validator(mode="after")
    def resolved_fields(self):
        if self.resolution_status == ResolutionStatus.RESOLVED:
            if not all((self.match_id, self.competition_id, self.competition_name,
                        self.home_team_id, self.home_team_name, self.away_team_id,
                        self.away_team_name, self.kickoff_time)):
                raise ValueError("RESOLVED requires complete canonical fixture")
            if self.home_team_id == self.away_team_id:
                raise ValueError("home and away must differ")
        elif not self.reason:
            raise ValueError("unresolved request requires reason")
        return self
