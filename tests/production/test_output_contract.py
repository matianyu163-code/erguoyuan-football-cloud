"""V7 trial output remains explicit about unavailable values and disabled actions."""

from datetime import UTC, date, datetime

from erguoyuan_football.app.input.match_input import MatchRequest
from erguoyuan_football.match_source.match_source import MatchSourceType
from production.v7_renderer import V7TrialRenderer


def test_v7_trial_layout_has_all_sections_without_fabricated_values() -> None:
    request = MatchRequest(
        match_id=None,
        home_team="SYNTHETIC_TEST_HOME",
        away_team="SYNTHETIC_TEST_AWAY",
        competition="SYNTHETIC_TEST_LEAGUE",
        date=date(2026, 10, 2),
        input_source="SYNTHETIC_TEST",
        validation_status="VALID",
    )
    text = V7TrialRenderer().render(
        request,
        source_type=MatchSourceType.USER_JC_CONFIRMED,
        competition="SYNTHETIC_TEST_LEAGUE",
        blocked_reasons=("SYNTHETIC_TEST_BLOCKED_INPUT",),
        simulation=True,
    )
    for section in ("比赛信息", "数据质量", "模型状态", "概率输出", "玩法输出",
                    "组合方案", "风险说明"):
        assert section in text
    assert "Elo: BLOCKED" in text
    assert "JC状态: USER_CONFIRMED" in text
    assert "胜平负概率: UNAVAILABLE" in text
    assert "AUTO_BET=false; AUTO_PUBLISH=false" in text
    assert "RUN MODE: PROCESS_SIMULATION" in text
    assert "SYNTHETIC_TEST_BLOCKED_INPUT" in text


def test_v7_real_trial_shows_verified_kickoff_and_only_supplied_probabilities() -> None:
    request = MatchRequest(None, "Borussia Dortmund", "SV Werder Bremen", "Bundesliga",
                           None, "USER_TEXT", "VALID")
    text = V7TrialRenderer().render(
        request, source_type=MatchSourceType.RESEARCH_TEST, competition="Bundesliga",
        blocked_reasons=("PHASE9_NOT_PROMOTED",),
        model_predictions=({"model_id": "ELO_V1", "execution_status": "EXECUTED",
            "probability": {"p_home": 0.5, "p_draw": 0.25, "p_away": 0.25}},),
        fixture_verified=True, snapshot_id="real-snapshot",
        kickoff_time=datetime(2026, 10, 9, 18, 30, tzinfo=UTC))
    assert "RUN MODE: REAL_PRODUCTION_TRIAL" in text
    assert "时间: 2026-10-09T18:30:00+00:00" in text
    assert "ELO_V1: EXECUTED; H/D/A=0.500000/0.250000/0.250000" in text
    assert "CORE概率: UNAVAILABLE (Phase 9尚未晋级)" in text
