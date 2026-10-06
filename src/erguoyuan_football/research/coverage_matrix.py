"""Readable competition-by-capability matrix derived only from profile evidence."""

from __future__ import annotations

from dataclasses import dataclass

from erguoyuan_football.research.coverage_resolver import CompetitionDataProfile
from erguoyuan_football.research.global_provider_registry import GlobalProviderRegistry
from erguoyuan_football.research.provider_coverage_profile import CoverageStatus


@dataclass(frozen=True)
class CompetitionCoverage:
    """Provider evidence status for one competition and capability."""

    competition_id: str
    capability: str
    status: CoverageStatus
    providers: tuple[str, ...]


def build_coverage_matrix(registry: GlobalProviderRegistry,
                          competitions: tuple[CompetitionDataProfile, ...],
                          capabilities: tuple[str, ...]) -> tuple[CompetitionCoverage, ...]:
    """Aggregate verified, partial and missing source evidence without guessing."""
    from erguoyuan_football.research.coverage_resolver import CoverageResolver

    resolver = CoverageResolver(registry)
    result = []
    for competition in competitions:
        for capability in capabilities:
            candidates = resolver.resolve(competition, capability, include_unverified=True)
            statuses = [row.profile.status_for(capability) for row in candidates]
            if CoverageStatus.VERIFIED in statuses:
                status = CoverageStatus.VERIFIED
            elif CoverageStatus.PARTIAL in statuses:
                status = CoverageStatus.PARTIAL
            elif CoverageStatus.DEGRADED in statuses:
                status = CoverageStatus.DEGRADED
            else:
                status = (CoverageStatus.UNVERIFIED if candidates
                          else CoverageStatus.UNSUPPORTED)
            result.append(CompetitionCoverage(competition.competition_id, capability,
                                               status, tuple(row.profile.provider_id
                                                             for row in candidates)))
    return tuple(result)
