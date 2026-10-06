"""INPUT_PARSER_V2 syntax and strict desktop resolution tests.

The names below are SYNTHETIC_TEST input strings, never prediction data.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from erguoyuan_football.app.config import AppConfig
from erguoyuan_football.app.input.match_input import (
    INPUT_PARSER_VERSION,
    MatchInputParserV2,
    MatchRequest,
    normalize_team_name,
)
from erguoyuan_football.app.input.parser import MatchRequest as LegacyMatchRequest
from erguoyuan_football.app.pipeline import ApplicationPipeline


@pytest.mark.parametrize(("raw", "home", "away"), [
    ("库克群岛 VS 塔西提", "库克群岛", "塔西提"),
    ("库克群岛VS塔西提", "库克群岛", "塔西提"),
    ("库克群岛 vs 塔西提", "库克群岛", "塔西提"),
    ("库克群岛vs塔西提", "库克群岛", "塔西提"),
    ("库克群岛-塔西提", "库克群岛", "塔西提"),
    ("库克群岛 对 塔西提", "库克群岛", "塔西提"),
    ("Cook Islands VS Tahiti", "Cook Islands", "Tahiti"),
    ("Cook Islands-Tahiti", "Cook Islands", "Tahiti"),
    ("Real Madrid vs Barcelona", "Real Madrid", "Barcelona"),
    ("Manchester City-Liverpool", "Manchester City", "Liverpool"),
    ("ArsenalVChelsea", "Arsenal", "Chelsea"),
    ("Arsenal V Chelsea", "Arsenal", "Chelsea"),
    ("Arsenal v Chelsea", "Arsenal", "Chelsea"),
    ("Arsenal vs. Chelsea", "Arsenal", "Chelsea"),
    ("阿森纳 对阵 切尔西", "阿森纳", "切尔西"),
    ("阿森纳ＶＳ切尔西", "阿森纳", "切尔西"),
])
def test_syntax_formats(raw: str, home: str, away: str) -> None:
    parsed = MatchInputParserV2().parse(raw)
    assert parsed.validation_status == "VALID"
    assert parsed.home_team == home
    assert parsed.away_team == away
    assert parsed.input_source == "USER_TEXT"
    assert parsed.match_id is None


@pytest.mark.parametrize("raw", ["ABC", "VS塔西提", "库克群岛VS", "  "])
def test_invalid_syntax(raw: str) -> None:
    parsed = MatchInputParserV2().parse(raw)
    assert parsed.validation_status == "INVALID"
    assert parsed.error_code == "MATCH_SYNTAX_INVALID"
    assert parsed.home_team is None
    assert parsed.away_team is None


def test_clean_and_team_normalization() -> None:
    parser = MatchInputParserV2()
    assert parser.version == INPUT_PARSER_VERSION == "INPUT_PARSER_V2"
    assert parser.clean_text(" 库克群岛   VS   塔西提 ") == "库克群岛VS塔西提"
    assert normalize_team_name(" 皇家 马德里 ") == "皇家马德里"
    assert normalize_team_name("  Cook   Islands  ") == "Cook Islands"


def test_legacy_match_request_fields_unchanged() -> None:
    assert MatchRequest is LegacyMatchRequest
    names = tuple(field.name for field in fields(MatchRequest))
    assert names[:9] == (
        "match_id", "home_team", "away_team", "competition", "date",
        "input_source", "validation_status", "reason", "raw_text")
    assert names[9:] == ("match_source_type", "jc_confirmed")
    legacy = MatchRequest(None, "Home", "Away", None, None,
                          "USER_TEXT", "VALID", None, "Home VS Away")
    assert legacy.home_team == "Home"
    assert legacy.match_source_type.value == "AUTO_DISCOVERY"
    assert legacy.jc_confirmed is False


def test_desktop_pipeline_reports_fixture_not_found_after_team_resolution() -> None:
    root = Path(__file__).resolve().parents[2]
    config = AppConfig.load(root / "config/application.yaml")
    result = ApplicationPipeline(config).run_text("库克群岛VS塔西提")
    assert result.status == "UNAVAILABLE"
    assert result.report is None
    assert result.report_hash is None
    assert result.requests[0].validation_status == "NOT_FOUND"
    assert result.requests[0].error_code == "FIXTURE_NOT_FOUND"
    assert "FIXTURE_NOT_FOUND" in result.report_text
    assert "TEAM_NOT_FOUND" not in result.report_text
    assert "MATCH_SYNTAX_INVALID" not in result.report_text
