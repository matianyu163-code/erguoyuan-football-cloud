"""JC desktop provenance and PIT gates; fixtures here are SYNTHETIC_TEST inputs."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from erguoyuan_football.app.ui.window import run_desktop_text
from production.jc_metadata import JCMetadataParser
from production.runner import ProductionRunner


def test_explicit_desktop_mode_is_jc_production(production_config) -> None:
    result = run_desktop_text("马耳他VS安道尔", production_config,
                              mode="JC_PRODUCTION")
    assert "RUN MODE: JC_PRODUCTION" in result.report_text
    assert result.requests[0].jc_confirmed is True


def test_jc_input_does_not_fallback_to_auto_research(production_config) -> None:
    result = run_desktop_text("马耳他VS安道尔", production_config,
                              mode="JC_PRODUCTION")
    assert "JC_FIXTURE_METADATA_INCOMPLETE" in result.report_text
    assert "competition,kickoff_local" in result.report_text
    assert "AUTO_RESEARCH" not in result.report_text


def test_jc_compact_metadata_and_timezone() -> None:
    metadata = JCMetadataParser().parse(
        "周日001 欧国联 马耳他VS安道尔 2026-10-04 23:00"
    )
    assert metadata.jc_match_code == "周日001"
    assert metadata.competition_original == "欧国联"
    assert metadata.competition_canonical == "UEFA Nations League"
    assert metadata.kickoff_utc == datetime(2026, 10, 4, 15, tzinfo=UTC)
    assert metadata.missing_fields == ()


def test_jc_kickoff_timezone_conversion() -> None:
    metadata = JCMetadataParser().parse(
        "马耳他VS安道尔", competition="欧国联",
        kickoff_local="2026-10-04 17:00", timezone="Europe/Berlin",
    )
    assert metadata.kickoff_utc == datetime(2026, 10, 4, 15, tzinfo=UTC)
    assert metadata.source_timezone == "Europe/Berlin"


def test_jc_complete_metadata_bypasses_fixture_discovery(
    production_config, monkeypatch,
) -> None:
    runner = ProductionRunner(production_config)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("OFFICIAL_FIXTURE_RESEARCH_MUST_NOT_BE_REQUIRED")

    monkeypatch.setattr(runner, "_discover_fixture", forbidden)
    try:
        result = runner.run(
            "周日001 欧国联 马耳他VS安道尔 2099-10-04 23:00",
            mode="JC_PRODUCTION",
        )
    finally:
        runner.close()
    assert result.source_type.value == "USER_JC_CONFIRMED"
    assert "PRIMARY BLOCKER: HISTORY_PROVIDER_UNAVAILABLE" in result.rendered_output
    assert "Fixture Origin: USER_CONFIRMED_CHINA_SPORTTERY" in result.rendered_output
    assert "OFFICIAL_RESEARCH    NOT_REQUIRED" in result.rendered_output
    assert "COMPETITION          USER_CONFIRMED:UEFA Nations League" in result.rendered_output
    assert result.prediction_id is not None
    with sqlite3.connect(production_config.record_database) as connection:
        row = connection.execute(
            "SELECT input_snapshot,prediction_output FROM trial_prediction_records "
            "WHERE prediction_id=?", (result.prediction_id,),
        ).fetchone()
    assert row is not None
    snapshot, output = (json.loads(value) for value in row)
    assert snapshot["mode"] == "JC_PRODUCTION"
    assert snapshot["fixture_verified"] is False
    assert snapshot["user_confirmed_fixture"] is True
    assert snapshot["kickoff_utc"] == "2099-10-04T15:00:00+00:00"
    assert output["blocking_reason"] == "HISTORY_PROVIDER_UNAVAILABLE"


def test_jc_missing_metadata_blocks_without_research(production_config) -> None:
    runner = ProductionRunner(production_config)
    try:
        result = runner.run("马耳他VS安道尔", mode="JC_PRODUCTION")
    finally:
        runner.close()
    assert result.stages[2].detail.startswith("JC_FIXTURE_METADATA_INCOMPLETE")
    assert all(stage.stage != "FIXTURE_DISCOVERY" for stage in result.stages)


def test_jc_pit_block(production_config) -> None:
    runner = ProductionRunner(production_config)
    try:
        result = runner.run("欧国联 马耳他VS安道尔 2001-10-04 23:00",
                            mode="JC_PRODUCTION")
    finally:
        runner.close()
    assert "PRIMARY BLOCKER: PIT_LIVE_PREDICTION_BLOCKED" in result.rendered_output
    assert "HISTORY_PROVIDER_UNAVAILABLE" not in result.rendered_output


def test_official_enrichment_failure_is_warning(production_config, monkeypatch) -> None:
    runner = ProductionRunner(production_config)

    def unavailable(*_args, **_kwargs):
        raise OSError("SYNTHETIC_TEST_NETWORK_UNAVAILABLE")

    monkeypatch.setattr(runner, "_discover_fixture", unavailable)
    try:
        result = runner.run(
            "欧国联 马耳他VS安道尔 2099-10-04 23:00",
            mode="JC_PRODUCTION", jc_fields={"official_enrichment": "true"},
        )
    finally:
        runner.close()
    official = next(stage for stage in result.stages
                    if stage.stage == "OFFICIAL_RESEARCH")
    assert official.status == "WARNING"
    assert official.detail == "OFFICIAL_ENRICHMENT_UNAVAILABLE"
    assert "PRIMARY BLOCKER: HISTORY_PROVIDER_UNAVAILABLE" in result.rendered_output


def test_v7_primary_blocker(production_config) -> None:
    result = run_desktop_text(
        "欧国联 科索沃VS奥地利 2099-10-04 23:00", production_config,
        mode="JC_PRODUCTION",
    )
    assert result.report_text.startswith(
        "本次预测未完成:\nHISTORY_PROVIDER_UNAVAILABLE\n"
        "PRIMARY BLOCKER: HISTORY_PROVIDER_UNAVAILABLE"
    )


def test_auto_research_unchanged(production_config) -> None:
    result = run_desktop_text("德国U19VS捷克U19", production_config,
                              mode="AUTO_RESEARCH")
    assert "RUN MODE: AUTO_RESEARCH" in result.report_text
    assert "JC_FIXTURE_METADATA_INCOMPLETE" not in result.report_text
    declared = run_desktop_text("周日001 德国U19VS捷克U19", production_config,
                                mode="AUTO_RESEARCH")
    assert declared.requests[0].match_source_type.value == "AUTO_DISCOVERY"


def test_jc_batch_independent_missing_fields(production_config) -> None:
    result = run_desktop_text(
        "周日001 欧国联 马耳他VS安道尔 2099-10-04 23:00\n"
        "周日002 欧国联 科索沃VS奥地利", production_config,
        mode="JC_PRODUCTION",
    )
    assert len(result.requests) == 2
    assert all(request.jc_confirmed for request in result.requests)
    assert "HISTORY_PROVIDER_UNAVAILABLE" in result.report_text
    assert "JC_FIXTURE_METADATA_INCOMPLETE" in result.report_text
