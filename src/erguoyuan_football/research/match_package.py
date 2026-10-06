"""Intermediate match research package; deliberately contains no probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from erguoyuan_football.research.live_data.readiness import LiveDataReadinessReport
    from erguoyuan_football.research.live_data.schemas import VerifiedFixture

from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.research.research_status import ResearchStatus
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.policies.conflict_policy import ConflictRecord
from erguoyuan_football.web_research.search.search_task import SearchTask
from erguoyuan_football.web_research.sources.source_schema import SourceRecord


@dataclass
class MatchResearchPackage:
    """One as-of research view, not a fixture confirmation or model prediction."""

    match_id: str | None
    home_team: TeamIdentity | None
    away_team: TeamIdentity | None
    competition: CompetitionIdentity | None
    research_tasks: list[SearchTask] = field(default_factory=list)
    available_data: dict[str, list[EvidenceRecord]] = field(default_factory=dict)
    missing_data: list[str] = field(default_factory=list)
    sources: list[SourceRecord] = field(default_factory=list)
    quality_score: float = 0.0
    status: ResearchStatus = ResearchStatus.MISSING
    as_of_time: str = ""
    error_code: str | None = None
    research_session_id: str | None = None
    network_status: str = "NOT_REQUESTED"
    live_fetch_attempted: bool = False
    live_fetch_completed: bool = False
    cache_status: str = "NOT_CHECKED"
    conflicts: list[ConflictRecord] = field(default_factory=list)
    prediction_executed: bool = False
    verified_fixture: VerifiedFixture | None = None
    live_data_status: str | None = None
    readiness_report: LiveDataReadinessReport | None = None
