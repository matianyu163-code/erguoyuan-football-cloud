"""Read-only production trial status derived from the append-only audit database."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

NO_PREDICTION_MESSAGE = "尚无实时预测记录。"
SUCCESS_MESSAGE = "当前为生产试运行模式，预测结果已按实战流程生成并留档。"
READY_WITH_WARNINGS = "PRODUCTION_TRIAL_READY_WITH_WARNINGS"
READY_WITH_WARNINGS_MESSAGE = "生产试运行已启用（存在已知告警）"


@dataclass(frozen=True)
class ProductionStatusView:
    """User-facing live status and the latest eligible production record."""

    global_status: str
    global_message: str
    prediction_message: str
    prediction_id: str | None = None


def read_production_status(database_path: Path) -> ProductionStatusView:
    """Derive UI status from real, non-simulation records without mutating SQLite."""
    if not database_path.is_file():
        return _no_prediction()
    try:
        uri = database_path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            rows = connection.execute(
                "SELECT prediction_id, source_type, match_name, input_snapshot, prediction_output "
                "FROM trial_prediction_records ORDER BY rowid DESC"
            ).fetchall()
    except sqlite3.Error:
        return ProductionStatusView("STATE_UNAVAILABLE", "生产状态暂不可读取。",
                                    "生产记录暂不可读取。")

    eligible: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    accepted_sources = {"AUTO_DISCOVERY", "USER_JC_CONFIRMED", "RESEARCH_TEST"}
    for prediction_id, source_type, match_name, raw_snapshot, raw_output in rows:
        try:
            snapshot = json.loads(raw_snapshot)
            output = json.loads(raw_output)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(snapshot, dict) or not isinstance(output, dict):
            continue
        if (source_type not in accepted_sources or snapshot.get("simulation_only") is True
                or snapshot.get("synthetic_data") is True):
            continue
        if "SYNTHETIC_TEST" in str(match_name).upper():
            continue
        if not (snapshot.get("simulation_only") is False or (
                snapshot.get("fixture_verified") is True
                and snapshot.get("real_snapshot_created") is True)):
            continue
        eligible.append((str(prediction_id), snapshot, output))

    if not eligible:
        return _no_prediction()

    prediction_id, _, latest_output = eligible[0]
    latest_success = next((record for record in eligible
        if _has_real_probability(record[2]) and record[1].get("fixture_verified") is True
        and record[1].get("real_snapshot_created") is True), None)
    if latest_success is None:
        return ProductionStatusView("PRODUCTION_BLOCKED", "生产预测当前受阻。",
            f"本次预测未完成：{_blocking_reason(latest_output)}。", prediction_id)

    _, _, successful_output = latest_success
    warnings = successful_output.get("warning_codes")
    has_warnings = (bool(warnings) or
                    successful_output.get("status") == "BASE_MODELS_EXECUTED_WITH_WARNINGS")
    if has_warnings:
        global_status = READY_WITH_WARNINGS
        global_message = READY_WITH_WARNINGS_MESSAGE
    else:
        global_status = "PRODUCTION_TRIAL_READY"
        global_message = "生产试运行已启用。"

    if (_has_real_probability(latest_output) and latest_success is eligible[0]):
        prediction_message = SUCCESS_MESSAGE
    else:
        prediction_message = f"本次预测未完成：{_blocking_reason(latest_output)}。"
    return ProductionStatusView(global_status, global_message, prediction_message,
                                prediction_id)


def _no_prediction() -> ProductionStatusView:
    return ProductionStatusView("PRODUCTION_NOT_RUN", NO_PREDICTION_MESSAGE,
                                NO_PREDICTION_MESSAGE)


def _has_real_probability(output: dict[str, Any]) -> bool:
    models = output.get("models")
    if not isinstance(models, list):
        return False
    for model in models:
        if not isinstance(model, dict):
            continue
        if model.get("execution_status") not in {"EXECUTED", "DEGRADED_EXECUTED"}:
            continue
        probability = model.get("probability")
        if not isinstance(probability, dict):
            continue
        values = [probability.get(key) for key in ("p_home", "p_draw", "p_away")]
        if not all(isinstance(value, (int, float)) for value in values):
            continue
        numeric_values = cast(list[float], values)
        if (all(math.isfinite(value) for value in numeric_values)
                and all(0 <= value <= 1 for value in numeric_values)
                and abs(sum(numeric_values) - 1) < 1e-6):
            return True
    return False


def _blocking_reason(output: dict[str, Any]) -> str:
    explicit = output.get("blocking_reason") or output.get("reason")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    steps = output.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict) or step.get("status") not in {"WARNING", "FAILED"}:
                continue
            detail = step.get("detail")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()
    models = output.get("models")
    if isinstance(models, list):
        model_errors = [
            f"{model.get('model_id', 'UNKNOWN_MODEL')}: {model['reason'].strip()}"
            for model in models if isinstance(model, dict)
            and isinstance(model.get("reason"), str) and model["reason"].strip()
        ]
        if model_errors:
            return "; ".join(model_errors)
    status = output.get("status")
    if isinstance(status, str) and status.strip():
        return f"生产记录状态为 {status.strip()}"
    return "生产记录未包含可执行模型概率或明确阻断详情"
