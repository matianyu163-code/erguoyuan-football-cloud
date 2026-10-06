"""Smart intake tests use clearly synthetic fixtures, never prediction samples."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from erguoyuan_football.app.ui.window import run_desktop_text
from production.bridge.contracts import CoreDataPacketV1, CoreResultPacketV1
from production.bridge.intake import (
    AmbiguousFixtureResponse,
    SmartBridgeIntake,
    classify_smart_input,
)


def _intake(production_config) -> SmartBridgeIntake:
    return SmartBridgeIntake(production_config,
                             production_config.record_database.parents[1])


def _synthetic_packet(request_id: str) -> CoreDataPacketV1:
    """Build a test-only packet used to validate file handoff, not fixture truth."""
    now = datetime.now(UTC).replace(microsecond=0)
    kickoff = now.replace(year=now.year + 1)
    return CoreDataPacketV1.model_validate({
        "schema_version": "CORE_DATA_PACKET_V1", "request_id": request_id,
        "created_at": now.isoformat(), "research_as_of": now.isoformat(),
        "match": {"competition": "SYNTHETIC_TEST LEAGUE",
            "home": "SYNTHETIC_TEST HOME", "away": "SYNTHETIC_TEST AWAY",
            "kickoff_original": kickoff.isoformat(), "kickoff_timezone": "UTC",
            "kickoff_utc": kickoff.isoformat(), "source_type": "RESEARCH_TEST",
            "neutral_venue": False},
        "entities": {"home_entity": "SYNTHETIC_TEST_HOME",
                     "away_entity": "SYNTHETIC_TEST_AWAY"},
        "fixture_evidence": {"sources": [{"source": "SYNTHETIC_TEST",
            "source_url": "https://www.uefa.com/synthetic-test/fixtures/",
            "source_tier": "A", "fetched_at": now.isoformat()}]},
        "history": {},
    })


def test_smart_bridge_teams_only_creates_request_and_database_event(
    production_config,
) -> None:
    decision = classify_smart_input("法国VS比利时")
    assert decision.route == "BRIDGE_RESEARCH"
    assert (decision.home, decision.away) == ("法国", "比利时")

    intake = _intake(production_config)
    request = intake.create_request("法国VS比利时", decision)
    path = intake.inbox / f"{request.request_id}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == "CORE_RESEARCH_REQUEST_V1"
    assert saved["requested_output"] == "CORE_DATA_PACKET_V1"
    assert saved["home_raw"] == "法国"
    assert saved["away_raw"] == "比利时"
    with sqlite3.connect(production_config.record_database) as connection:
        row = connection.execute(
            "SELECT status FROM bridge_research_events WHERE request_id=?",
            (request.request_id,),
        ).fetchone()
    assert row == ("BRIDGE_RESEARCH_REQUESTED",)


def test_smart_desktop_defaults_to_research_request(production_config) -> None:
    result = run_desktop_text("法国VS比利时", production_config,
                              mode="SMART_BRIDGE")
    assert result.status == "WAITING_FOR_BRIDGE_RESEARCH"
    assert result.bridge_request_id is not None
    assert "JC_FIXTURE_METADATA_INCOMPLETE" not in result.report_text
    assert "BRIDGE_AGENT_NOT_CONNECTED" in result.report_text
    assert "WAITING_FOR_RESEARCH" in result.report_text


def test_complete_jc_input_routes_directly(production_config) -> None:
    decision = classify_smart_input(
        "周日001 欧国联 法国VS比利时 2099-10-04 20:00")
    assert decision.route == "JC_PRODUCTION"
    result = run_desktop_text(
        "周日001 欧国联 法国VS比利时 2099-10-04 20:00",
        production_config, mode="SMART_BRIDGE")
    assert result.status == "PRODUCTION_TRIAL"
    assert result.bridge_request_id is None
    assert "RUN MODE: JC_PRODUCTION" in result.report_text
    assert "JC_FIXTURE_METADATA_INCOMPLETE" not in result.report_text
    project_root = production_config.record_database.parents[1]
    assert not (project_root / "data" / "bridge" / "inbox").exists()


def test_request_poll_reports_agent_disconnected(production_config) -> None:
    intake = _intake(production_config)
    request = intake.create_request("法国VS比利时",
                                    classify_smart_input("法国VS比利时"))
    polled = intake.poll(request.request_id)
    assert polled.status == "WAITING_FOR_RESEARCH"
    assert "BRIDGE_AGENT_NOT_CONNECTED" in polled.message


def test_ambiguous_fixture_requires_explicit_candidate_choice(
    production_config,
) -> None:
    intake = _intake(production_config)
    request = intake.create_request("法国VS比利时",
                                    classify_smart_input("法国VS比利时"))
    intake.ready.mkdir(parents=True, exist_ok=True)
    ambiguous = AmbiguousFixtureResponse.model_validate({
        "schema_version": "BRIDGE_FIXTURE_AMBIGUOUS_V1",
        "request_id": request.request_id,
        "candidates": [
            {"candidate_id": "SYNTHETIC_TEST_A", "competition": "SYNTHETIC_TEST A",
             "kickoff_original": "2099-01-01T12:00:00", "kickoff_timezone": "UTC",
             "kickoff_utc": "2099-01-01T12:00:00Z",
             "source_url": "https://www.uefa.com/synthetic-test/a"},
            {"candidate_id": "SYNTHETIC_TEST_B", "competition": "SYNTHETIC_TEST B",
             "kickoff_original": "2099-02-01T12:00:00", "kickoff_timezone": "UTC",
             "kickoff_utc": "2099-02-01T12:00:00Z",
             "source_url": "https://www.uefa.com/synthetic-test/b"},
        ],
    })
    (intake.ready / f"{request.request_id}.ambiguous.json").write_text(
        ambiguous.model_dump_json(indent=2), encoding="utf-8")

    polled = intake.poll(request.request_id)
    assert polled.status == "BRIDGE_FIXTURE_AMBIGUOUS"
    assert "未自动选择" in polled.message
    selection = intake.record_user_selection(request.request_id, "SYNTHETIC_TEST_B")
    assert json.loads(selection.read_text(encoding="utf-8"))["candidate_id"] == "SYNTHETIC_TEST_B"
    waiting = intake.poll(request.request_id)
    assert waiting.status == "WAITING_FOR_RESEARCH"
    assert "SYNTHETIC_TEST_B" in waiting.message


def test_ready_packet_ingestion_and_request_id_result_polling(
    production_config, monkeypatch,
) -> None:
    intake = _intake(production_config)
    request = intake.create_request("SYNTHETIC_TEST HOME VS SYNTHETIC_TEST AWAY",
        classify_smart_input("SYNTHETIC_TEST HOME VS SYNTHETIC_TEST AWAY"))
    intake.ready.mkdir(parents=True, exist_ok=True)
    packet = _synthetic_packet(request.request_id)
    (intake.ready / f"{request.request_id}.packet.json").write_text(
        packet.model_dump_json(indent=2), encoding="utf-8")
    calls: list[str] = []

    class _Runner:
        def __init__(self, _config, *, project_root: Path) -> None:
            assert project_root == intake.root

        def run(self, received: CoreDataPacketV1) -> CoreResultPacketV1:
            calls.append(received.request_id)
            return CoreResultPacketV1(
                build_id="SYNTHETIC_TEST_BUILD", request_id=received.request_id,
                status="BLOCKED", primary_blocker="HISTORY_INSUFFICIENT",
                snapshot_id=None, v7_output="HISTORY_INSUFFICIENT",
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr("production.bridge.intake.BridgeRunner", _Runner)
    processed = intake.poll(request.request_id)
    assert calls == [request.request_id]
    assert processed.status == "BLOCKED"
    assert processed.result is not None
    assert (intake.results / f"{request.request_id}.result.json").is_file()
    reread = intake.poll(request.request_id)
    assert reread.result is not None
    assert calls == [request.request_id]
