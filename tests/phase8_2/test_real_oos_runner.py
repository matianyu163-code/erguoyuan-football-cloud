"""Phase 8.2 real base OOS execution and date-boundary regressions."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from erguoyuan_football.backtesting.real_oos_runner import (
    CAPABILITIES,
    UnifiedRealOOSRunner,
)
from erguoyuan_football.models.registry import ModelRegistry

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
REAL_DB = ROOT / "data" / "football.duckdb"


@pytest.fixture(scope="module")
def run_on_copy(tmp_path_factory):
    if not REAL_DB.exists():
        pytest.skip("pinned real OpenFootball warehouse is not installed")
    directory = tmp_path_factory.mktemp("phase8-2-runner")
    database = directory / "football.duckdb"
    shutil.copy2(REAL_DB, database)
    with duckdb.connect(str(database)) as connection:
        ids = [row[0] for row in connection.execute("""SELECT match_id
            FROM real_canonical_matches WHERE competition_id='EPL'
            AND match_date='2025-10-03' AND status='FINISHED'""").fetchall()]
        assert ids
        connection.execute("DELETE FROM real_oos_predictions_v2 WHERE match_id IN "
            "(SELECT match_id FROM real_canonical_matches WHERE competition_id='EPL' "
            "AND match_date='2025-10-03') AND model_id IN "
            "('BIVARIATE_POISSON_V1','CORE_SPI_LIKE_V1')")
    runner = UnifiedRealOOSRunner(database, artifact_root=directory / "artifacts")
    first = runner.run(date(2025, 10, 3), date(2025, 10, 3),
        model_ids=("BIVARIATE_POISSON_V1", "CORE_SPI_LIKE_V1"), competitions=("EPL",))
    return database, runner, first, ids


def test_model_registry_execution() -> None:
    registered = set(ModelRegistry().list_models())
    assert {cap.model_id for cap in CAPABILITIES} <= registered
    assert all(not cap.requires_market for cap in CAPABILITIES)


def test_multi_model_real_oos_runner(run_on_copy) -> None:
    database, _runner, report, ids = run_on_copy
    assert report.completed_rows == len(ids) * 2
    with duckdb.connect(str(database), read_only=True) as connection:
        rows = connection.execute("""SELECT model_id,p_home,p_draw,p_away,data_origin,is_oos
            FROM real_oos_predictions_v2 WHERE match_id=? AND model_id IN
            ('BIVARIATE_POISSON_V1','CORE_SPI_LIKE_V1')""", [ids[0]]).fetchall()
    assert len(rows) == 2
    assert all(row[4:] == ("REAL", True) and abs(sum(row[1:4]) - 1) < 1e-6 for row in rows)


def test_bivariate_real_oos(run_on_copy) -> None:
    database, *_ = run_on_copy
    with duckdb.connect(str(database), read_only=True) as connection:
        payload = connection.execute("""SELECT payload FROM real_oos_predictions_v2
            WHERE model_id='BIVARIATE_POISSON_V1' AND match_date='2025-10-03'
            ORDER BY match_id LIMIT 1""").fetchone()[0]
    record = json.loads(payload)
    assert record["lambda_home"] > 0 and record["lambda_away"] > 0
    assert record["metadata"]["lambda3"] >= 0


def test_spi_goals_only_real_oos(run_on_copy) -> None:
    database, *_ = run_on_copy
    with duckdb.connect(str(database), read_only=True) as connection:
        payload = connection.execute("""SELECT payload FROM real_oos_predictions_v2
            WHERE model_id='CORE_SPI_LIKE_V1' AND match_date='2025-10-03'
            ORDER BY match_id LIMIT 1""").fetchone()[0]
    record = json.loads(payload)
    assert record["implementation_type"] == "LIKE_IMPLEMENTATION"
    assert record["metadata"]["feature_mode"] == "GOALS_ONLY"


def test_runner_resume(run_on_copy) -> None:
    database, runner, first, ids = run_on_copy
    again = runner.run(date(2025, 10, 3), date(2025, 10, 3),
        model_ids=("BIVARIATE_POISSON_V1", "CORE_SPI_LIKE_V1"), competitions=("EPL",))
    with duckdb.connect(str(database), read_only=True) as connection:
        count = connection.execute("""SELECT count(*) FROM real_oos_predictions_v2
            WHERE match_id=? AND model_id IN
            ('BIVARIATE_POISSON_V1','CORE_SPI_LIKE_V1')""", [ids[0]]).fetchone()[0]
    assert count == 2 and again.completed_rows == first.completed_rows


def test_runner_checkpoint_resumes_next_date(tmp_path) -> None:
    database = tmp_path / "football.duckdb"
    shutil.copy2(REAL_DB, database)
    with duckdb.connect(str(database)) as connection:
        target_count = connection.execute("""SELECT count(*) FROM real_canonical_matches
            WHERE competition_id='EPL' AND match_date BETWEEN '2025-10-03' AND '2025-10-04'
              AND status='FINISHED'""").fetchone()[0]
        assert target_count > 1
        connection.execute("""DELETE FROM real_oos_predictions_v2 WHERE
            model_id='BIVARIATE_POISSON_V1' AND match_id IN
            (SELECT match_id FROM real_canonical_matches WHERE competition_id='EPL'
             AND match_date BETWEEN '2025-10-03' AND '2025-10-04')""")
    runner = UnifiedRealOOSRunner(database, artifact_root=tmp_path / "artifacts")
    partial = runner.run(date(2025, 10, 3), date(2025, 10, 4),
        model_ids=("BIVARIATE_POISSON_V1",), competitions=("EPL",), max_dates=1)
    assert partial.status == "PARTIAL" and 0 < partial.completed_rows < target_count
    resumed = runner.run(date(2025, 10, 3), date(2025, 10, 4),
        model_ids=("BIVARIATE_POISSON_V1",), competitions=("EPL",))
    assert resumed.status == "COMPLETE" and resumed.completed_rows == target_count


def test_runner_idempotence(run_on_copy) -> None:
    _database, runner, first, _ids = run_on_copy
    again = runner.run(date(2025, 10, 3), date(2025, 10, 3),
        model_ids=("BIVARIATE_POISSON_V1", "CORE_SPI_LIKE_V1"), competitions=("EPL",))
    assert again.artifact_count == first.artifact_count == 2


def test_same_day_leakage_guard(run_on_copy) -> None:
    database, _runner, _report, ids = run_on_copy
    with duckdb.connect(str(database), read_only=True) as connection:
        cutoff, target_date, count, payload = connection.execute("""SELECT
            training_cutoff,match_date,training_match_count,payload
            FROM real_oos_predictions_v2 WHERE match_id=?
              AND model_id='BIVARIATE_POISSON_V1'""", [ids[0]]).fetchone()
        training = [row[0] for row in connection.execute("""SELECT match_id
            FROM real_canonical_matches WHERE competition_id='EPL' AND status='FINISHED'
              AND match_date >= ? AND match_date < ? ORDER BY match_id""",
            [(cutoff - timedelta(days=1095)).date(), cutoff.date()]).fetchall()]
    assert target_date > cutoff.date() and ids[0] not in training
    assert len(training) == count
    assert hashlib.sha256(json.dumps(training, separators=(",", ":")).encode()).hexdigest() == (
        json.loads(payload)["metadata"]["training_match_ids_hash"])


def test_model_unavailable_recorded(run_on_copy) -> None:
    database, runner, _report, ids = run_on_copy
    result = runner.run(date(2025, 10, 3), date(2025, 10, 3),
        model_ids=("DYNAMIC_BAYESIAN_POISSON_V1", "BAYESIAN_HIERARCHICAL_V1"),
        competitions=("EPL",))
    assert result.unavailable_rows == len(ids) * 2 and result.failed_rows == 0
    with duckdb.connect(str(database), read_only=True) as connection:
        statuses = connection.execute("""SELECT model_id,execution_status,reason
            FROM real_oos_execution_status WHERE match_id=? AND model_id IN
            ('DYNAMIC_BAYESIAN_POISSON_V1','BAYESIAN_HIERARCHICAL_V1')""",
            [ids[0]]).fetchall()
    assert {row[0] for row in statuses} == {
        "DYNAMIC_BAYESIAN_POISSON_V1", "BAYESIAN_HIERARCHICAL_V1"}
    assert all(row[1] == "UNAVAILABLE" and row[2] for row in statuses)


def test_opta_like_result_real_oos() -> None:
    with duckdb.connect(str(REAL_DB), read_only=True) as connection:
        payload = connection.execute("""SELECT payload FROM real_oos_predictions_v2
            WHERE model_id='CORE_OPTA_XG_ELO_LIKE_V1' LIMIT 1""").fetchone()[0]
    record = json.loads(payload)
    assert record["implementation_type"] == "LIKE_IMPLEMENTATION"
    assert record["metadata"]["xg_mode"] == "WITHOUT_XG_RESULT_ONLY"
