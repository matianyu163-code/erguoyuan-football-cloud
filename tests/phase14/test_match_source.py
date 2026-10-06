"""Match source classification tests; all teams and requests are SYNTHETIC_TEST."""

from datetime import UTC, datetime

from erguoyuan_football.app.input.match_input import MatchInputParserV2
from erguoyuan_football.input.schemas import (
    InputType,
)
from erguoyuan_football.input.schemas import (
    MatchRequest as CoreMatchRequest,
)
from erguoyuan_football.jc_verification.jc_match_verifier import JCMatchVerifier
from erguoyuan_football.jc_verification.jc_schema import (
    JCMatchQuery,
    JCStatus,
    JCVerificationResult,
)
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from erguoyuan_football.match_source.match_source import MatchSourceType
from erguoyuan_football.match_source.match_source_audit import MatchSourceAuditStore
from erguoyuan_football.match_source.match_source_classifier import (
    MatchSourceClassifier,
)
from erguoyuan_football.research.production_match_pipeline import _jc_result_for_source


def test_china_sporttery_id_classifies_user_confirmed_and_cleans_prefix() -> None:
    classifier = MatchSourceClassifier()
    raw = "中国竞彩 周001 阿森纳VS利物浦"
    audit = classifier.classify(raw)
    parsed = MatchInputParserV2().parse(classifier.clean_match_text(raw))
    assert audit.source_type == MatchSourceType.USER_JC_CONFIRMED
    assert audit.confidence == 1.0
    assert parsed.home_team == "阿森纳"
    assert parsed.away_team == "利物浦"
    assert "SPORTTERY_WEEK_MATCH_NUMBER" in audit.evidence


def test_attached_ticket_number_is_recognized_without_whitespace() -> None:
    classifier = MatchSourceClassifier()
    raw = "周001阿森纳VS利物浦"
    assert classifier.classify(raw).source_type == MatchSourceType.USER_JC_CONFIRMED
    parsed = MatchInputParserV2().parse(classifier.clean_match_text(raw))
    assert parsed.home_team == "阿森纳"


def test_plain_match_is_automatic_discovery() -> None:
    audit = MatchSourceClassifier().classify("阿森纳VS利物浦")
    assert audit.source_type == MatchSourceType.AUTO_DISCOVERY


def test_explicit_sporttery_phrase_is_user_confirmed_without_number() -> None:
    audit = MatchSourceClassifier().classify("中国体育彩票竞彩足球 阿森纳 VS 利物浦")
    assert audit.source_type == MatchSourceType.USER_JC_CONFIRMED
    assert "EXPLICIT_CHINA_SPORTTERY_MARKER" in audit.evidence


def test_youth_match_is_research_test() -> None:
    audit = MatchSourceClassifier().classify("德国U16VS希腊U16")
    assert audit.source_type == MatchSourceType.RESEARCH_TEST
    assert "EXPLICIT_YOUTH_MARKER:U16" in audit.evidence


def test_match_request_fields_are_backward_compatible() -> None:
    request = MatchInputParserV2().parse("Arsenal VS Liverpool")
    assert request.match_source_type == MatchSourceType.AUTO_DISCOVERY
    assert request.jc_confirmed is False
    contract_request = CoreMatchRequest(input_type=InputType.TEXT, source="USER_TEXT")
    assert contract_request.match_source_type == MatchSourceType.AUTO_DISCOVERY
    assert contract_request.jc_confirmed is False


def test_explicit_user_jc_confirmation_skips_provider_query() -> None:
    class NoCallVerifier(JCMatchVerifier):
        def verify(self, query: JCMatchQuery, *, now=None) -> JCVerificationResult:
            raise AssertionError("official JC lookup must be skipped")

    teams = TeamResolver()
    home, away = teams.resolve("Arsenal"), teams.resolve("Liverpool")
    assert home and away
    query = JCMatchQuery(home.team_id, home.official_name, away.team_id,
        away.official_name, "UEFA_CL", "UEFA Champions League",
        datetime(2026, 10, 4, 15, tzinfo=UTC))
    audit = MatchSourceClassifier().classify(
        "Arsenal vs Liverpool", jc_confirmed=True,
        created_time=datetime(2026, 10, 2, tzinfo=UTC))
    result = _jc_result_for_source(audit, query, NoCallVerifier())
    assert result.status == JCStatus.USER_CONFIRMED
    assert result.source == "USER_INPUT"
    assert result.confidence_score == 1.0
    assert result.evidence_id == audit.audit_id


def test_source_audit_is_append_only_and_readable(tmp_path) -> None:
    store = MatchSourceAuditStore(tmp_path / "source-audit.sqlite")
    try:
        first = MatchSourceClassifier().classify("阿森纳 VS 利物浦")
        second = MatchSourceClassifier().classify("中国竞彩 周001 阿森纳VS利物浦")
        store.append(first)
        store.append(second)
        saved = store.list()
        assert len(saved) == 2
        assert saved[0].source_type == MatchSourceType.AUTO_DISCOVERY
        assert saved[1].source_type == MatchSourceType.USER_JC_CONFIRMED
    finally:
        store.close()
