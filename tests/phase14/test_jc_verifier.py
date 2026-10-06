"""Verifier fail-closed behavior; fixture lists are explicitly SYNTHETIC_TEST."""

from datetime import UTC, date, datetime

from erguoyuan_football.jc_verification.jc_match_matcher import JCMatchMatcher
from erguoyuan_football.jc_verification.jc_match_verifier import JCMatchVerifier
from erguoyuan_football.jc_verification.jc_provider import (
    JCProviderRegistry,
    ProviderRegistration,
    ProviderTier,
)
from erguoyuan_football.jc_verification.jc_schema import (
    JCMatch,
    JCMatchQuery,
    JCStatus,
)
from erguoyuan_football.knowledge.teams.team_database import TeamDatabase
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity


class _Provider:
    provider_id = "synthetic-test-provider"
    tier = ProviderTier.OFFICIAL

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def get_daily_matches(self, match_date: date) -> list[JCMatch]:
        self.calls += 1
        if self.fail:
            raise OSError("synthetic provider failure")
        return [JCMatch("match-1", "Arsenal", "Liverpool", "UEFA Champions League",
            datetime(2026, 10, 4, 15, tzinfo=UTC), ("1X2",), self.provider_id,
            "synthetic-evidence", competition_id="UEFA_CL")]


def _query() -> JCMatchQuery:
    resolver = TeamResolver()
    home, away = resolver.resolve("Arsenal"), resolver.resolve("Liverpool")
    assert home and away
    return JCMatchQuery(home.team_id, "Arsenal", away.team_id, "Liverpool",
        "UEFA_CL", "UEFA Champions League", datetime(2026, 10, 4, 15, tzinfo=UTC))


def test_missing_provider_returns_unknown_source_unavailable() -> None:
    result = JCMatchVerifier().verify(_query(), now=datetime(2026, 10, 2, tzinfo=UTC))
    assert result.status == JCStatus.UNKNOWN
    assert result.reason == "SOURCE_UNAVAILABLE"
    assert result.confidence_score == 0


def test_provider_failure_never_becomes_not_jc() -> None:
    registration = ProviderRegistration(_Provider(fail=True))
    verifier = JCMatchVerifier(JCProviderRegistry((registration,)))
    result = verifier.verify(_query(), now=datetime(2026, 10, 2, tzinfo=UTC))
    assert result.status == JCStatus.UNKNOWN
    assert result.reason == "SOURCE_UNAVAILABLE"


def test_exact_official_evidence_confirms_fixture() -> None:
    registration = ProviderRegistration(_Provider())
    verifier = JCMatchVerifier(JCProviderRegistry((registration,)))
    result = verifier.verify(_query(), now=datetime(2026, 10, 2, tzinfo=UTC))
    assert result.status == JCStatus.CONFIRMED
    assert result.source == "synthetic-test-provider"
    assert result.evidence_id == "synthetic-evidence"


def test_official_candidate_within_window_but_time_shifted_is_likely() -> None:
    class ShiftedProvider(_Provider):
        def get_daily_matches(self, match_date: date) -> list[JCMatch]:
            self.calls += 1
            return [JCMatch("match-1", "Arsenal", "Liverpool", "UEFA Champions League",
                datetime(2026, 10, 4, 15, 15, tzinfo=UTC), ("1X2",), self.provider_id,
                "synthetic-evidence", competition_id="UEFA_CL")]

    verifier = JCMatchVerifier(JCProviderRegistry((ProviderRegistration(ShiftedProvider()),)))
    result = verifier.verify(_query(), now=datetime(2026, 10, 2, tzinfo=UTC))
    assert result.status == JCStatus.LIKELY
    assert result.confidence_score == 0.90


def test_missing_kickoff_cannot_be_confirmed() -> None:
    from dataclasses import replace

    result = JCMatchVerifier().verify(replace(_query(), kickoff_time=None),
                                      now=datetime(2026, 10, 2, tzinfo=UTC))
    assert result.status == JCStatus.UNKNOWN
    assert result.reason == "MATCH_DATE_OR_KICKOFF_REQUIRED"


def test_complete_provider_and_canonical_youth_identity_can_mark_not_jc() -> None:
    youth_teams = TeamResolver(TeamDatabase((
        TeamIdentity("GER_U16", "Germany U16", "Germany", "UEFA", ["Germany U16"],
                     age_group="U16"),
        TeamIdentity("GRE_U16", "Greece U16", "Greece", "UEFA", ["Greece U16"],
                     age_group="U16"),
    )))

    class CompleteEmptyProvider(_Provider):
        provider_id = "complete-synthetic-test-provider"

        def get_daily_matches(self, match_date: date) -> list[JCMatch]:
            return []

    home, away = youth_teams.resolve("Germany U16"), youth_teams.resolve("Greece U16")
    assert home and away
    query = JCMatchQuery(home.team_id, home.official_name, away.team_id,
        away.official_name, None, None, datetime(2026, 10, 4, 15, tzinfo=UTC))
    verifier = JCMatchVerifier(
        JCProviderRegistry((ProviderRegistration(CompleteEmptyProvider(), True),)),
        matcher=JCMatchMatcher(youth_teams))
    result = verifier.verify(query, now=datetime(2026, 10, 2, tzinfo=UTC))
    assert result.status == JCStatus.NOT_JC
    assert result.reason == "EXPLICIT_YOUTH_TEAMS_OUTSIDE_JC_SCOPE"


def test_historical_miss_never_queries_live_provider() -> None:
    provider = _Provider()
    verifier = JCMatchVerifier(JCProviderRegistry((ProviderRegistration(provider),)))
    result = verifier.verify(_query(), now=datetime(2026, 10, 5, tzinfo=UTC))
    assert result.status == JCStatus.UNKNOWN
    assert result.reason == "HISTORICAL_CACHE_MISS_READ_ONLY"
    assert provider.calls == 0
