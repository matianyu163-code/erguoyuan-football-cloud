"""Input Parser V2 → competition hint → independent team identities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from erguoyuan_football.app.input.match_input import MatchInputParserV2, MatchRequest
from erguoyuan_football.knowledge.entities.competition_hint_parser import (
    ParsedCompetitionHint,
    split_competition_hint,
)
from erguoyuan_football.knowledge.entities.universal_team_resolver import (
    EntityResolution,
    UniversalTeamResolver,
)


@dataclass(frozen=True)
class UniversalMatchResolution:
    """Team resolution is not fixture verification or a prediction."""

    request: MatchRequest
    competition_hint: ParsedCompetitionHint | None
    home: EntityResolution | None
    away: EntityResolution | None
    status: str
    fixture_verified: bool = False
    prediction_executed: bool = False


def resolve_match_text(text: str, resolver: UniversalTeamResolver, *,
                       allow_discovery: bool = False,
                       as_of: datetime | None = None) -> UniversalMatchResolution:
    """Preserve parser syntax and fail closed on one/both unresolved teams."""
    match_text, competition = split_competition_hint(text)
    request = MatchInputParserV2().parse(match_text)
    if request.validation_status != "VALID" or not request.home_team or not request.away_team:
        return UniversalMatchResolution(request, competition, None, None,
                                        "MATCH_SYNTAX_INVALID")
    home = resolver.resolve(request.home_team, competition=competition,
                            allow_discovery=allow_discovery, as_of=as_of)
    away = resolver.resolve(request.away_team, competition=competition,
                            allow_discovery=allow_discovery, as_of=as_of)
    if home.identity is not None and away.identity is not None and (
        home.identity.team_id == away.identity.team_id):
        status = "TEAM_IDENTITY_CONFLICT"
    elif home.status == away.status == "VERIFIED":
        status = "IDENTITIES_RESOLVED_NO_FIXTURE"
    elif "TEAM_DISCOVERY_AMBIGUOUS" in {home.status, away.status}:
        status = "TEAM_DISCOVERY_AMBIGUOUS"
    elif "PROVIDER_COVERAGE_MISSING" in {home.status, away.status}:
        status = "PROVIDER_COVERAGE_MISSING"
    else:
        status = "TEAM_NOT_FOUND"
    return UniversalMatchResolution(request, competition, home, away, status)
