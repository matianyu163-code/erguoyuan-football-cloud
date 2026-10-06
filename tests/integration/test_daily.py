import subprocess
import sys

import pytest

from erguoyuan_football.daily import DailyPredictionRequest, DailyService
from erguoyuan_football.data.store import Store


@pytest.mark.integration
@pytest.mark.parametrize("kind,value", [("TEXT", "阿森纳"), ("MANUAL", "周二001 阿森纳VS切尔西"),
                                        ("BATCH", "周二001 阿森纳VS切尔西\n周二002 皇马VS巴萨")])
def test_daily_input(store, at, kind, value):
    result = DailyService(store).run(DailyPredictionRequest(input_type=kind, raw_text=value, prediction_time=at))
    assert all(r.resolution_status == "RESOLVED" for r in result.requests)
    assert len(result.snapshots) == (2 if kind == "BATCH" else 1)
    assert all(s.status == "NOT_IMPLEMENTED" for s in result.stages)
    assert all(p.final_core_probability is None for p in result.core_predictions)
    assert result.output.two_leg.conclusion == "无合格组合 / NO-BET"
    assert store.connection.execute("SELECT count(*) FROM match_requests").fetchone()[0] == len(result.requests)


def test_daily_unknown_does_not_predict(store, at):
    result = DailyService(store).run(DailyPredictionRequest(input_type="TEXT", raw_text="不存在的球队", prediction_time=at))
    assert result.requests[0].resolution_status == "NOT_FOUND"
    assert not result.snapshots and not result.core_predictions


def test_daily_batch_list_deduplicates(store, at):
    result = DailyService(store).run(DailyPredictionRequest(input_type="BATCH", matches=("阿森纳", "阿森纳 VS 切尔西"), prediction_time=at))
    assert len(result.requests) == 2 and len(result.snapshots) == 1


def test_daily_screenshot_without_provider(store, at, tmp_path):
    image = tmp_path / "synthetic.png"
    image.touch()
    result = DailyService(store).run(DailyPredictionRequest(input_type="SCREENSHOT", image_path=str(image), prediction_time=at))
    assert result.requests[0].reason == "VISION_PROVIDER_UNAVAILABLE"
    assert not result.snapshots


def test_database_reopen(tmp_path):
    database = tmp_path / "reopen.duckdb"
    with Store(database) as first:
        tables = first.tables
    with Store(database) as second:
        assert second.tables == tables


def test_cli_empty_catalog(tmp_path, at, project_root):
    result = subprocess.run([sys.executable, "-m", "erguoyuan_football", "--db", str(tmp_path / "cli.duckdb"),
                             "--at", at.isoformat(), "--text", "unknown"], cwd=project_root,
                             capture_output=True, text=True, encoding="utf-8", check=False,
                             env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert result.returncode == 0, result.stderr
    assert "NOT_FOUND" in result.stdout and "NOT_IMPLEMENTED" in result.stdout
