"""Evidence-driven JC resource scheduling tests."""

from erguoyuan_football.jc_verification.jc_resource_scheduler import (
    CompetitionProfile,
    JCResourceScheduler,
    ResourcePriority,
)
from erguoyuan_football.jc_verification.jc_schema import JCStatus
from erguoyuan_football.research.match_universe import MatchUniverse


def test_unknown_uses_basic_mode_and_does_not_block() -> None:
    plan = JCResourceScheduler().plan(JCStatus.UNKNOWN)
    assert plan.mode == "BASIC"
    assert plan.universe == MatchUniverse.GLOBAL_RESEARCH
    assert plan.ordered_models


def test_confirmed_uses_configured_competition_profile() -> None:
    profile = CompetitionProfile("premier_league", ResourcePriority.S)
    plan = JCResourceScheduler().plan(JCStatus.CONFIRMED, profile)
    assert plan.mode == "FULL"
    assert plan.universe == MatchUniverse.JC_PRODUCTION
    assert "odds" in plan.requested_data


def test_tier_a_source_is_likely_until_official_confirmation() -> None:
    plan = JCResourceScheduler().plan(JCStatus.LIKELY,
        CompetitionProfile("ucl", ResourcePriority.A))
    assert plan.mode == "STANDARD"
    assert plan.universe == MatchUniverse.JC_PRODUCTION
    assert "lineup" not in plan.requested_data


def test_user_confirmed_uses_jc_basic_mode_without_profile() -> None:
    plan = JCResourceScheduler().plan(JCStatus.USER_CONFIRMED)
    assert plan.mode == "BASIC"
    assert plan.universe == MatchUniverse.JC_PRODUCTION


def test_not_jc_is_limited_research() -> None:
    plan = JCResourceScheduler().plan(JCStatus.NOT_JC)
    assert plan.mode == "LIMITED"
    assert plan.universe == MatchUniverse.GLOBAL_RESEARCH
