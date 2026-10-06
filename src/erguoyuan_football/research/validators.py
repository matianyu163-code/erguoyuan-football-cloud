"""Fail-closed identity, provenance and package invariants."""

from __future__ import annotations

from collections.abc import Iterable

from erguoyuan_football.app.input.match_input import MatchRequest
from erguoyuan_football.knowledge.match_identity import (
    GlobalKnowledgeResolver,
    KnowledgeMatchIdentity,
)
from erguoyuan_football.research.match_package import MatchResearchPackage
from erguoyuan_football.research.research_status import ResearchStatus
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.sources.source_schema import SourceRecord
from erguoyuan_football.web_research.time_utils import parse_utc


def validate_identity(
    request: MatchRequest,
    resolver: GlobalKnowledgeResolver,
) -> KnowledgeMatchIdentity:
    """Resolve only explicit team/competition names; preserve failures."""
    if (request.validation_status not in {"VALID", "RESOLVED"}
            or request.home_team is None or request.away_team is None):
        return KnowledgeMatchIdentity(None, None, None,
                                      request.error_code or "MATCH_SYNTAX_INVALID")
    if request.validation_status == "RESOLVED" and not (request.match_id or "").strip():
        return KnowledgeMatchIdentity(None, None, None, "RESOLVED_MATCH_ID_REQUIRED")
    return resolver.resolve_names(request.home_team, request.away_team,
                                  request.competition)


def validate_sources(
    evidence: Iterable[EvidenceRecord],
    registry: SourceRegistry,
) -> list[SourceRecord]:
    """Check every evidence URL against the enabled transport allowlist."""
    found: dict[str, SourceRecord] = {}
    for item in evidence:
        found[item.source_id] = registry.validate_result_url(item.source_id,
                                                               item.source_url)
    return [found[key] for key in sorted(found)]


def validate_package(package: MatchResearchPackage) -> None:
    """Reject contradictory identity, evidence, status and time fields."""
    as_of = parse_utc(package.as_of_time)
    if not 0 <= package.quality_score <= 1:
        raise ValueError("QUALITY_OUT_OF_RANGE")
    if package.status == ResearchStatus.FAILED:
        if not package.error_code or package.research_tasks or package.available_data:
            raise ValueError("FAILED_PACKAGE_INCONSISTENT")
        return
    if package.home_team is None or package.away_team is None:
        raise ValueError("PACKAGE_TEAM_IDENTITY_REQUIRED")
    if package.home_team.team_id == package.away_team.team_id:
        raise ValueError("SAME_TEAM_NOT_A_MATCH")
    if set(package.available_data) & set(package.missing_data):
        raise ValueError("AVAILABLE_AND_MISSING_OVERLAP")
    if package.available_data and package.match_id is None:
        raise ValueError("EVIDENCE_REQUIRES_VERIFIED_MATCH_ID")
    source_ids = {source.source_id for source in package.sources}
    for records in package.available_data.values():
        for record in records:
            if record.source_id not in source_ids:
                raise ValueError("PACKAGE_SOURCE_NOT_LISTED")
            evidence_times = [parse_utc(record.fetched_time), parse_utc(record.as_of_time)]
            if record.published_time is not None:
                evidence_times.append(parse_utc(record.published_time))
            if record.observed_time is not None:
                evidence_times.append(parse_utc(record.observed_time))
            if max(evidence_times) > as_of:
                raise ValueError("PACKAGE_FUTURE_EVIDENCE")
    if package.status == ResearchStatus.FOUND and package.missing_data:
        raise ValueError("FOUND_PACKAGE_HAS_MISSING_DATA")
    if package.status == ResearchStatus.PARTIAL and (
        not package.available_data or not package.missing_data
    ):
        raise ValueError("PARTIAL_PACKAGE_INCONSISTENT")
    if package.status == ResearchStatus.MISSING and package.available_data:
        raise ValueError("MISSING_PACKAGE_HAS_EVIDENCE")
