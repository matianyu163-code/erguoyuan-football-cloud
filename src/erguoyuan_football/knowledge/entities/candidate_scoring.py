"""Explain candidate quality only after hard constraints have passed."""

from __future__ import annotations

from functools import lru_cache

from erguoyuan_football.knowledge.entities.competition_hint_parser import (
    ParsedCompetitionHint,
)
from erguoyuan_football.knowledge.entities.country_alias_registry import (
    CountryAliasRegistry,
)
from erguoyuan_football.knowledge.entities.discovery import TeamDiscoveryCandidate
from erguoyuan_football.knowledge.entities.entity_constraints import hard_conflicts
from erguoyuan_football.knowledge.entities.team_entity_parser import (
    ParsedTeamEntityHint,
    UniversalTeamNameParser,
)
from erguoyuan_football.knowledge.teams.alias_matcher import (
    normalize_alias,
    normalize_club_alias,
)


@lru_cache(maxsize=1)
def _country_registry() -> CountryAliasRegistry:
    """Load the static ISO registry once for candidate compatibility checks."""
    return UniversalTeamNameParser().countries


def _same_country(hint_country: str | None, candidate_country: str | None) -> bool:
    """Accept canonical ISO-3 and established football codes for one country."""
    if not hint_country or not candidate_country:
        return False
    if hint_country.upper() == candidate_country.upper():
        return True
    registry = _country_registry()
    hinted = registry.resolve(hint_country)
    candidate = registry.resolve(candidate_country)
    return hinted is not None and candidate is not None and hinted.iso3 == candidate.iso3


def score_candidate(hint: ParsedTeamEntityHint, candidate: TeamDiscoveryCandidate,
                    competition: ParsedCompetitionHint | None = None) -> float | None:
    """Return 0–1 explanatory score; conflicts return None and are never ranked."""
    country = candidate.country
    if _same_country(hint.country_hint, candidate.country):
        country = hint.country_hint
    if hard_conflicts(hint, country=country, gender=candidate.gender,
                      age_group=candidate.age_group,
                      entity_type=candidate.entity_type,
                      squad_level=candidate.squad_level):
        return None
    parsed = UniversalTeamNameParser().parse(candidate.official_name)
    name = (normalize_alias(hint.raw_name) == normalize_alias(candidate.official_name)
            or normalize_alias(hint.base_name) == normalize_alias(parsed.base_name)
            or (hint.entity_type_hint == candidate.entity_type == "CLUB"
                and normalize_club_alias(hint.base_name)
                == normalize_club_alias(parsed.base_name)))
    if not name:
        return None
    checks = (
        (hint.country_hint is not None
         and _same_country(hint.country_hint, candidate.country), 0.12),
        (hint.gender_hint != "UNKNOWN" and hint.gender_hint == candidate.gender, 0.12),
        (hint.age_group_hint == candidate.age_group, 0.08),
        (hint.entity_type_hint == candidate.entity_type, 0.08),
        (hint.squad_level_hint == candidate.squad_level, 0.08),
        (competition is not None and competition.federation == candidate.federation, 0.07),
        (candidate.source_tier == 3, 0.05),
        (candidate.confidence == "HIGH", 0.05),
    )
    return round(min(1.0, 0.35 + sum(weight for matched, weight in checks if matched)), 6)
