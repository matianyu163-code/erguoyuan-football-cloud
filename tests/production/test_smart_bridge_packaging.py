"""Desktop mode routing and build provenance; no model values are mocked."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from erguoyuan_football.app.installer.version import (
    DEFAULT_DESKTOP_MODE,
    SMART_BRIDGE_VERSION,
    source_fingerprints,
)
from erguoyuan_football.app.ui.window import run_desktop_text
from production.bridge import BRIDGE_VERSION


def test_packaged_exe_default_is_smart_bridge(production_config) -> None:
    assert DEFAULT_DESKTOP_MODE == "SMART_BRIDGE"
    result = run_desktop_text("塞浦路斯 VS 拉脱维亚", production_config)
    assert result.report_text.startswith("RUN MODE: SMART_BRIDGE\n")
    root = Path(__file__).resolve().parents[2]
    exe = (root / ".workspace/dist/COREFootballEngine_BRIDGE_V1_2"
           / "COREFootballEngine_BRIDGE_V1_2.exe")
    if not exe.is_file():
        pytest.skip("PACKAGED_EXE_NOT_BUILT")
    completed = subprocess.run([str(exe), "--check"], check=True,
                               capture_output=True, text=True, encoding="utf-8",
                               timeout=60)
    packaged = json.loads(completed.stdout)
    assert packaged["default_mode"] == "SMART_BRIDGE"
    assert packaged["build_info"]["smart_bridge_version"] == SMART_BRIDGE_VERSION


def test_team_only_input_creates_research_request(production_config) -> None:
    result = run_desktop_text("法国 VS 比利时", production_config)
    assert result.status == "WAITING_FOR_BRIDGE_RESEARCH"
    assert result.bridge_request_id is not None
    root = production_config.record_database.parents[1]
    path = root / "data/bridge/inbox" / f"{result.bridge_request_id}.json"
    request = json.loads(path.read_text(encoding="utf-8"))
    assert (request["home_raw"], request["away_raw"]) == ("法国", "比利时")


def test_cyprus_latvia_does_not_require_jc_metadata(production_config) -> None:
    result = run_desktop_text("塞浦路斯 VS 拉脱维亚", production_config)
    for text in ("RESEARCH_REQUEST      CREATED", "SNAPSHOT              NOT_CREATED",
                 "MODELS                NOT_EXECUTED", "WAITING_FOR_RESEARCH"):
        assert text in result.report_text
    assert "JC_FIXTURE_METADATA_INCOMPLETE" not in result.report_text


def test_complete_fixture_routes_to_jc_production(production_config) -> None:
    result = run_desktop_text(
        "欧国联 法国 VS 比利时 2099-10-05 20:45", production_config)
    assert "RUN MODE: JC_PRODUCTION" in result.report_text
    assert result.bridge_request_id is None


def test_explicit_jc_mode_still_requires_metadata(production_config) -> None:
    result = run_desktop_text("塞浦路斯 VS 拉脱维亚", production_config,
                              mode="JC_PRODUCTION")
    assert "JC_FIXTURE_METADATA_INCOMPLETE" in result.report_text


def test_previous_mode_persistence_does_not_break_default(production_config) -> None:
    explicit = run_desktop_text("塞浦路斯 VS 拉脱维亚", production_config,
                                mode="JC_PRODUCTION")
    assert "JC_FIXTURE_METADATA_INCOMPLETE" in explicit.report_text
    following = run_desktop_text("塞浦路斯 VS 拉脱维亚", production_config)
    assert following.status == "WAITING_FOR_BRIDGE_RESEARCH"


def test_packaged_build_contains_smart_bridge_router() -> None:
    root = Path(__file__).resolve().parents[2]
    fingerprints = source_fingerprints(root)
    assert fingerprints["desktop_dispatch"] != "MISSING"
    assert fingerprints["bridge_intake"] != "MISSING"
    assert fingerprints["bridge_version"] != "MISSING"
    assert SMART_BRIDGE_VERSION == "SMART_BRIDGE_V1_2"
    assert BRIDGE_VERSION == "CORE_BRIDGE_V1"


def test_auto_research_regression(production_config) -> None:
    result = run_desktop_text("德国U19 VS 捷克U19", production_config,
                              mode="AUTO_RESEARCH")
    assert "RUN MODE: AUTO_RESEARCH" in result.report_text
    assert "JC_FIXTURE_METADATA_INCOMPLETE" not in result.report_text
