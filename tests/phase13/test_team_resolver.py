"""Phase 13.1 offline team identities; custom fixtures are SYNTHETIC_TEST."""

from erguoyuan_football.app.input.match_input import MatchInputParserV2
from erguoyuan_football.knowledge.match_identity import GlobalKnowledgeResolver
from erguoyuan_football.knowledge.teams.alias_matcher import AliasMatcher
from erguoyuan_football.knowledge.teams.team_database import TeamDatabase
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity


def test_phase13_seed_team_identities() -> None:
    """All four user-specified IDs resolve without a network or fixture lookup."""
    resolver = TeamResolver()
    cases = {
        "库克群岛": ("OFC_COK", "OFC"),
        "Cook Islands": ("OFC_COK", "OFC"),
        "Tahiti": ("OFC_TAH", "OFC"),
        "塔西提": ("OFC_TAH", "OFC"),
        "Arsenal": ("ENG_ARS", "UEFA"),
        "利物浦": ("ENG_LIV", "UEFA"),
    }
    for name, expected in cases.items():
        team = resolver.resolve(name)
        assert team is not None
        assert (team.team_id, team.federation) == expected


def test_phase13_alias_canonicalization_is_exact() -> None:
    """Case, spaces, and dash variants are safe canonical aliases, not fuzzy matches."""
    resolver = TeamResolver()
    for name in ("ARSENAL FC", "arsenal-fc", " Arsenal  FC "):
        team = resolver.resolve(name)
        assert team is not None and team.team_id == "ENG_ARS"
    assert resolver.resolve("Arsenall") is None
    assert resolver.resolve("未知球队名称") is None
    assert resolver.resolve("") is None


def test_phase13_ambiguous_alias_is_not_selected() -> None:
    """SYNTHETIC_TEST: a canonical collision returns None, never the first team."""
    teams = (
        TeamIdentity("TEST_A", "First", "Test", "TEST", ["Shared Alias"]),
        TeamIdentity("TEST_B", "Second", "Test", "TEST", ["Shared-Alias"]),
    )
    matcher = AliasMatcher(teams)
    assert matcher.candidate_ids("sharedalias") == ("TEST_A", "TEST_B")
    assert TeamResolver(TeamDatabase(teams)).resolve("Shared Alias") is None


def test_phase13_catalog_returns_defensive_alias_copies() -> None:
    """Callers cannot alter the process-local seed through a returned list."""
    resolver = TeamResolver()
    team = resolver.resolve("Arsenal")
    assert team is not None
    team.aliases.append("Injected")
    assert resolver.resolve("Injected") is None
    stored = TeamDatabase().get("ENG_ARS")
    assert stored is not None and "Injected" not in stored.aliases


def test_phase13_parser_to_identity_without_fixture() -> None:
    """Known teams do not imply a verified scheduled match or a prediction."""
    request = MatchInputParserV2().parse("库克群岛VS塔西提")
    assert request.validation_status == "VALID"
    assert request.home_team is not None and request.away_team is not None
    identity = GlobalKnowledgeResolver().resolve_request(request)
    assert identity.home_team is not None and identity.home_team.team_id == "OFC_COK"
    assert identity.away_team is not None and identity.away_team.team_id == "OFC_TAH"
    assert identity.competition is None
    assert identity.status == "IDENTITIES_RESOLVED_NO_FIXTURE"
    with_competition = GlobalKnowledgeResolver().resolve_names(
        request.home_team, request.away_team, "OFC Nations Cup"
    )
    assert with_competition.competition is not None
    assert GlobalKnowledgeResolver().resolve_text("ABC").status == "MATCH_SYNTAX_INVALID"


def test_phase13_unknown_or_same_team_cannot_become_match() -> None:
    """SYNTHETIC_TEST: identity failures remain explicit and non-predictive."""
    resolver = GlobalKnowledgeResolver()
    assert resolver.resolve_names("Unknown", "Tahiti").status == "TEAM_NOT_FOUND"
    assert resolver.resolve_names("Tahiti", "TAH").status == "INVALID_MATCH"
    assert resolver.resolve_names("Cook Islands", "Tahiti", "Unknown Cup").status == (
        "COMPETITION_NOT_FOUND"
    )
