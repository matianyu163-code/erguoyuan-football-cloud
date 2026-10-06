"""SYNTHETIC_TEST research package boundaries; no predictor is invoked."""

from dataclasses import asdict, replace

from erguoyuan_football.app.input.match_input import MatchInputParserV2
from erguoyuan_football.research.orchestrator import MatchResearchOrchestrator
from erguoyuan_football.research.research_status import ResearchStatus


def test_match_package_has_identities_and_no_fake_fixture() -> None:
    """Text-only teams produce a package, never a fabricated match ID."""
    request = MatchInputParserV2().parse("Arsenal VS Liverpool")
    package = MatchResearchOrchestrator().build(request)
    assert package.home_team is not None and package.home_team.team_id == "ENG_ARS"
    assert package.away_team is not None and package.away_team.team_id == "ENG_LIV"
    assert package.match_id is None
    assert package.status == ResearchStatus.MISSING
    assert len(package.research_tasks) == 6
    assert package.available_data == {}
    assert "xg" in package.missing_data and "competition" in package.missing_data
    assert package.quality_score == 0.2
    assert not {"p_home", "p_draw", "p_away"} & asdict(package).keys()


def test_known_competition_does_not_create_evidence() -> None:
    """A resolved competition increases identity coverage but not data availability."""
    request = replace(MatchInputParserV2().parse("Arsenal VS Liverpool"),
                      competition="UEFA Champions League")
    package = MatchResearchOrchestrator().build(request)
    assert package.competition is not None and package.competition.competition_id == "UEFA_CL"
    assert package.status == ResearchStatus.MISSING
    assert package.quality_score == 0.3
    assert "competition" not in package.missing_data
