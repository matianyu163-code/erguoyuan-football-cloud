"""Phase 12 contract and real local development-flow verification."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from erguoyuan_football.app.config import AppConfig
from erguoyuan_football.app.export.reports import export_html, export_json, export_pdf
from erguoyuan_football.app.history.repository import HistoryRepository
from erguoyuan_football.app.input.parser import MatchInputResolver, _CatalogRow
from erguoyuan_football.app.installer.version import version_check
from erguoyuan_football.app.pipeline import ApplicationPipeline
from erguoyuan_football.app.runtime import check_readiness, configure_workspace

ROOT = Path(__file__).resolve().parents[2]
FIRST_ID = "700fedf625427e26697df271"


@pytest.fixture(scope="module")
def app_config() -> AppConfig:
    """Use the repository's real local development artifacts, never synthetic P."""
    return AppConfig.load(ROOT / "config/application.yaml")


def test_startup_checks_and_production_gate(app_config: AppConfig) -> None:
    configure_workspace(app_config)
    ready = check_readiness(app_config)
    assert ready.status == "DEVELOPMENT_REVIEW_READY"
    assert ready.live_production_ready is False
    assert all(status == "OK" for status in ready.checks.values())


def test_missing_artifact_blocks_startup(app_config: AppConfig, tmp_path: Path) -> None:
    missing = replace(app_config, artifact_path=tmp_path / "missing")
    ready = check_readiness(missing)
    assert ready.status == "SYSTEM_NOT_READY"
    assert ready.checks["model_artifact"] == "MISSING"


def test_exact_input_and_ambiguity(app_config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = MatchInputResolver(app_config)
    row = _CatalogRow("1" * 24, "EPL", __import__("datetime").date(2026, 5, 1),
                      "Arsenal FC", "Chelsea FC")
    duplicate = _CatalogRow("2" * 24, "EPL", __import__("datetime").date(2026, 5, 9),
                            "Arsenal FC", "Chelsea FC")
    monkeypatch.setattr(resolver, "_catalog", lambda: (row, duplicate))
    assert resolver.parse_text("Arsenal FC VS Chelsea FC")[0].validation_status == "AMBIGUOUS"
    selected = resolver.parse_text("英超：\n2026-05-01 | Arsenal FC VS Chelsea FC")[0]
    assert selected.match_id == row.match_id
    assert resolver.parse_text("未知队 VS Chelsea FC")[0].validation_status == "NOT_FOUND"
    assert resolver.parse_text("123 nonsense")[0].validation_status == "INVALID"


def test_screenshot_requires_review(app_config: AppConfig, tmp_path: Path) -> None:
    image = tmp_path / "fixture.png"
    image.write_bytes(b"not image data")
    request = MatchInputResolver.parse_image(image)[0]
    assert request.validation_status == "INPUT_REVIEW_REQUIRED"
    assert request.match_id is None


@pytest.fixture(scope="module")
def real_result(app_config: AppConfig):
    """Run a Phase 9 real OOS row through the unmodified Phase 11 chain."""
    return ApplicationPipeline(app_config).run_text(FIRST_ID)


def test_real_report_and_unavailable_visible(real_result) -> None:
    assert real_result.status == "DEVELOPMENT_REPORT"
    assert real_result.report is not None
    assert len(real_result.report.all_matches) == 1
    assert real_result.report.model_data_status["live_production_ready"] is False
    assert "UNAVAILABLE" in real_result.report_text
    assert "六、模型与数据状态" in real_result.report_text


def test_twenty_real_matches_form_fixed_report(app_config: AppConfig) -> None:
    with app_config.development_predictions.open(encoding="utf-8") as handle:
        ids = [json.loads(line)["match_id"] for _, line in zip(range(20), handle)]
    result = ApplicationPipeline(app_config).run_text("\n".join(ids))
    assert result.report is not None
    assert len(result.report.all_matches) == 20
    assert result.report.section_order == (
        "all_matches", "highest_hit", "high_hit_400", "value_100",
        "longshot_20", "model_data_status")
    assert result.report.model_data_status["final_holdout_rows_read"] == 0


def test_repeat_report_hash(app_config: AppConfig, real_result) -> None:
    repeat = ApplicationPipeline(app_config).run_text(FIRST_ID)
    assert repeat.report_hash == real_result.report_hash


def test_unknown_match_never_gets_probability(app_config: AppConfig) -> None:
    result = ApplicationPipeline(app_config).run_text("虚构队 VS 不存在队")
    assert result.report is None
    assert result.status == "UNAVAILABLE"
    assert result.report_hash is None


def test_database_failure_is_visible_partial_report(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duckdb

    from erguoyuan_football.app import pipeline as app_pipeline

    def fail(**_kwargs: object) -> None:
        raise duckdb.InvalidInputException("TEST_DYNAMIC_IMPORT_UNAVAILABLE")

    monkeypatch.setattr(app_pipeline, "run_development_batch", fail)
    result = ApplicationPipeline(app_config).run_text(FIRST_ID)
    assert result.status == "PARTIAL_REPORT"
    assert result.report is None
    assert "InvalidInputException" in result.report_text


def test_history_append_recovery_and_null_settlement(tmp_path: Path, real_result) -> None:
    repository = HistoryRepository(tmp_path / "history.duckdb")
    session = repository.start_session("1.0.0", "hash")
    interrupted = repository.begin_request(session, FIRST_ID)
    assert (interrupted, FIRST_ID, "USER_TEXT") in repository.recoverable_requests()
    repository.record(session, FIRST_ID, real_result, run_id=interrupted)
    assert repository.recoverable_requests() == ()
    second = repository.begin_request(session, FIRST_ID)
    repository.record(session, FIRST_ID, real_result, run_id=second)
    assert len(repository.recent()) == 4  # STARTED and completed rows are never overwritten.
    record = repository.add_settlement(repository.recent()[0][0], match_result="HOME",
        candidate_result=None, actual_result="HOME")
    import duckdb
    with duckdb.connect(str(repository.path), read_only=True) as db:
        assert db.execute("SELECT stake,return_amount,roi FROM settlement_records "
                          "WHERE record_id=?", [record]).fetchone() == (None, None, None)
        assert db.execute("SELECT COUNT(*) FROM prediction_reports").fetchone()[0] == 2


def test_exports_and_offline_update(tmp_path: Path, real_result) -> None:
    machine = export_json(real_result, tmp_path / "report.json")
    browser = export_html(real_result, tmp_path / "report.html")
    assert json.loads(machine.read_text(encoding="utf-8"))["report_hash"] == real_result.report_hash
    assert "UNAVAILABLE" in browser.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_json(real_result, machine)
    pdf = export_pdf(real_result, tmp_path / "report.pdf")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert version_check()["status"] == "NOT_IMPLEMENTED_OFFLINE"
