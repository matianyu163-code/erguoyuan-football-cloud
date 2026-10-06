"""Fail-closed V7 trial layout; unavailable values remain explicit."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from erguoyuan_football.app.input.match_input import MatchRequest
from erguoyuan_football.match_source.match_source import MatchSourceType


class V7TrialRenderer:
    """Render the trial headings without computing predictions or stakes."""

    def render(self, request: MatchRequest, *, source_type: MatchSourceType,
               competition: str | None, blocked_reasons: tuple[str, ...],
               simulation: bool = False,
               model_predictions: tuple[Mapping[str, object], ...] = (),
               fixture_verified: bool = False, snapshot_id: str | None = None,
               kickoff_time: datetime | None = None,
               primary_blocker: str | None = None,
               secondary_consequences: tuple[str, ...] = (),
               pipeline_diagnostics: tuple[str, ...] = (),
               run_mode: str | None = None,
               fixture_origin: str | None = None) -> str:
        """Return a structured V7 diagnostic whose missing facts are UNAVAILABLE."""
        teams = f"{request.home_team or 'UNAVAILABLE'} VS {request.away_team or 'UNAVAILABLE'}"
        jc_status = ("USER_CONFIRMED" if source_type == MatchSourceType.USER_JC_CONFIRMED
                     else "AUTO_RESEARCH")
        displayed_mode = (run_mode or ("PROCESS_SIMULATION" if simulation else
                    "REAL_PRODUCTION_TRIAL" if fixture_verified else "TRIAL_PREFLIGHT"))
        displayed_kickoff = (kickoff_time.isoformat() if kickoff_time else
                             request.date.isoformat() if request.date else "UNAVAILABLE")
        lines = [
            *( ("本次预测未完成:", primary_blocker)
               if run_mode == "JC_PRODUCTION" and primary_blocker else ()),
            f"PRIMARY BLOCKER: {primary_blocker or 'NONE'}",
            "CORE Football Prediction System V1.0 TRIAL — V7 DIAGNOSTIC",
            f"RUN MODE: {displayed_mode}",
            "PIPELINE:",
            *(pipeline_diagnostics or ("NOT_AVAILABLE",)),
            "================================",
            "比赛信息",
            "================================",
            f"比赛: {teams}",
            f"赛事: {competition or 'UNAVAILABLE'}",
            f"时间: {displayed_kickoff}",
            f"JC状态: {jc_status}",
            *( (f"Fixture Origin: {fixture_origin}",) if fixture_origin else ()),
            "",
            "================================",
            "数据质量",
            "================================",
            f"数据来源: {'OPENLIGADB (ODbL-1.0)' if fixture_verified else 'USER_INPUT' if fixture_origin else 'UNAVAILABLE'}",
            f"覆盖程度: {'1 verified fixture snapshot' if fixture_verified else 'user-confirmed fixture metadata; no provider-verified snapshot' if fixture_origin else '0 verified snapshots'}",
            f"Snapshot ID: {snapshot_id or 'UNAVAILABLE'}",
            "风险等级: HIGH",
            "",
            "================================",
            "模型状态",
            "================================",
        ]
        if model_predictions:
            for result in model_predictions:
                model = str(result.get("model_id", "UNKNOWN_MODEL"))
                status = str(result.get("execution_status", "UNAVAILABLE"))
                probability = result.get("probability")
                if isinstance(probability, Mapping):
                    values = (probability.get("p_home"), probability.get("p_draw"),
                              probability.get("p_away"))
                    lines.append(f"{model}: {status}; H/D/A=" + "/".join(
                        f"{float(value):.6f}" if isinstance(value, (int, float)) else "UNAVAILABLE"
                        for value in values))
                else:
                    reason = result.get("reason") or "NO_VALID_MODEL_OUTPUT"
                    lines.append(f"{model}: {status} ({reason})")
        else:
            for model in ("Elo", "Dixon-Coles", "Pi Rating", "SPI-like",
                          "Dynamic Bayesian", "Bayesian"):
                lines.append(f"{model}: BLOCKED (UNAVAILABLE: REQUIRED_PRODUCTION_INPUTS_MISSING)")
        lines.extend((
            "",
            "================================",
            "概率输出",
            "================================",
            ("胜平负概率: AVAILABLE (独立底模概率见上；未生成CORE/Meta/校准概率)"
             if any(isinstance(item.get("probability"), Mapping)
                    for item in model_predictions) else "胜平负概率: UNAVAILABLE"),
            "",
            "================================",
            "玩法输出",
            "================================",
            "半全场TOP2: UNAVAILABLE",
            "比分TOP2: UNAVAILABLE",
            "总进球TOP2: UNAVAILABLE",
            "",
            "================================",
            "组合方案",
            "================================",
            "二串一: NO_BET / UNAVAILABLE",
            "让球: UNAVAILABLE",
            "总进球: UNAVAILABLE",
            "半全场: UNAVAILABLE",
            "400元方案: NOT_GENERATED",
            "100元方案: NOT_GENERATED",
            "20元方案: NOT_GENERATED",
            "",
            "================================",
            "风险说明",
            "================================",
            *(f"- {reason}" for reason in blocked_reasons),
            *( ("SECONDARY CONSEQUENCES: " + ";".join(secondary_consequences),)
               if secondary_consequences else ()),
            "AUTO_BET=false; AUTO_PUBLISH=false",
            ("CORE概率: UNAVAILABLE (Phase 9尚未晋级)" if model_predictions
             else "CORE概率: UNAVAILABLE"),
            ("FINAL_V7_STATUS: BASE_MODELS_EXECUTED_WITH_WARNINGS"
             if any(isinstance(item.get("probability"), Mapping)
                    for item in model_predictions) else "FINAL_V7_STATUS: NOT_READY"),
        ))
        return "\n".join(lines) + "\n"
