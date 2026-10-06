"""Select providers by verified scope, capability, health and source policy."""

from __future__ import annotations

from dataclasses import dataclass

from erguoyuan_football.research.global_provider_registry import (
    GlobalProviderRegistry,
    RegisteredProvider,
)
from erguoyuan_football.research.provider_coverage_profile import CoverageStatus


@dataclass(frozen=True)
class CompetitionDataProfile:
    """Known competition dimensions used to filter provider coverage."""

    competition_id: str
    country: str = "UNKNOWN"
    federation: str = "UNKNOWN"
    gender: str = "UNKNOWN"
    age_group: str = "UNKNOWN"
    entity_type: str = "UNKNOWN"
    squad_level: str = "UNKNOWN"
    competition_type: str = "UNKNOWN"
    region: str = "UNKNOWN"
    league_level: int | None = None


def parse_competition_data_profile(competition_id: str, hint: str) -> CompetitionDataProfile:
    """Use the existing conservative hint parser and preserve unknown dimensions."""
    from erguoyuan_football.knowledge.entities.competition_hint_parser import (
        CompetitionHintParser,
    )

    parsed = CompetitionHintParser().parse(hint)
    return CompetitionDataProfile(
        competition_id, country=parsed.country or "UNKNOWN",
        federation=parsed.federation or "UNKNOWN",
        gender=parsed.gender or "UNKNOWN",
        age_group=parsed.age_group or "UNKNOWN",
        entity_type=parsed.entity_type or "UNKNOWN",
        competition_type=parsed.competition_type or "UNKNOWN",
        league_level=parsed.league_level,
    )


class CoverageResolver:
    """Rank only configured providers that declare matching dimensions."""

    _DIMENSIONS = ("countries", "federations", "genders", "age_groups",
                   "entity_types", "squad_levels", "competition_types", "regions")

    def __init__(self, registry: GlobalProviderRegistry) -> None:
        self.registry = registry

    @staticmethod
    def _matches(values: frozenset[str], target: str) -> bool:
        return target == "UNKNOWN" or "*" in values or target in values

    def resolve(self, competition: CompetitionDataProfile, capability: str,
                *, minimum_tier: int = 2, allow_degraded: bool = True,
                include_unverified: bool = True
                ) -> tuple[RegisteredProvider, ...]:
        """Return candidates ordered by evidence, health, tier and scope specificity."""
        rows: list[tuple[tuple[int, int, int, int, str], RegisteredProvider]] = []
        expected = {
            "countries": competition.country, "federations": competition.federation,
            "genders": competition.gender, "age_groups": competition.age_group,
            "entity_types": competition.entity_type, "squad_levels": competition.squad_level,
            "competition_types": competition.competition_type, "regions": competition.region,
        }
        for row in self.registry.list():
            profile = row.profile
            if not row.configured or profile.source_tier < minimum_tier:
                continue
            status = profile.status_for(capability)
            if status == CoverageStatus.UNSUPPORTED or (
                    status == CoverageStatus.UNVERIFIED and not include_unverified):
                continue
            if status == CoverageStatus.DEGRADED and not allow_degraded:
                continue
            if capability not in profile.capabilities:
                continue
            scopes = [getattr(profile, field) for field in self._DIMENSIONS]
            if any(not self._matches(scope, expected[field])
                   for field, scope in zip(self._DIMENSIONS, scopes, strict=True)):
                continue
            exactness = sum(expected[field] != "UNKNOWN" and expected[field] in scope
                            for field, scope in zip(self._DIMENSIONS, scopes, strict=True))
            status_rank = {CoverageStatus.VERIFIED: 0, CoverageStatus.PARTIAL: 1,
                           CoverageStatus.DEGRADED: 2, CoverageStatus.UNVERIFIED: 3}[status]
            healthy = row.health == "HEALTHY"
            freshness = profile.last_verified_at.timestamp() if profile.last_verified_at else 0.0
            rows.append(((status_rank, -int(healthy), -profile.source_tier,
                          -exactness, f"{-freshness:.4f}:{profile.provider_id}"), row))
        return tuple(row for _, row in sorted(rows, key=lambda item: item[0]))
