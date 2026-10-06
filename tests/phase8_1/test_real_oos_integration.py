"""Real-data Phase 8.1 OOS and CORE V2 pipeline regression tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import duckdb
import pytest

from erguoyuan_football.backtesting.date_safe_oos import (
    _persist_metrics,
    run_real_date_safe_oos,
)
from erguoyuan_football.output_contract.schemas import (
    AvailabilityStatus,
    CanonicalPredictionResult,
    ProbabilityStage,
)

pytestmark = pytest.mark.integration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_DATABASE = PROJECT_ROOT / "data" / "football.duckdb"


@pytest.fixture(scope="module")
def real_oos_run(tmp_path_factory):
    if not REAL_DATABASE.is_file():
        pytest.skip("pinned OpenFootball Phase 8 DuckDB is not present")
    directory = tmp_path_factory.mktemp("phase8-1-real")
    database = directory / "football.duckdb"
    shutil.copy2(REAL_DATABASE, database)
    report = run_real_date_safe_oos(database, evaluation_start=date(2025, 10, 1),
        evaluation_end=date(2026, 1, 31), minimum_history_matches=80,
        minimum_metric_sample=100, competitions=("EPL",))
    return database, report


def test_real_elo_oos(real_oos_run) -> None:
    _database, report = real_oos_run
    assert report.real_oos_rows_by_model.get("ELO_V1", 0) > 0


def test_real_pi_oos(real_oos_run) -> None:
    _database, report = real_oos_run
    assert report.real_oos_rows_by_model.get("PI_RATING_V1", 0) > 0


def test_real_poisson_oos(real_oos_run) -> None:
    _database, report = real_oos_run
    assert report.real_oos_rows_by_model.get("SIMPLE_INDEPENDENT_POISSON_V1", 0) > 0


def test_real_dixon_coles_oos(real_oos_run) -> None:
    _database, report = real_oos_run
    assert report.real_oos_rows_by_model.get("DIXON_COLES_V1", 0) > 0


def test_target_match_excluded_and_future_match_excluded(real_oos_run) -> None:
    database, _report = real_oos_run
    with duckdb.connect(str(database)) as connection:
        rows = connection.execute("""SELECT p.match_id,p.competition_id,p.match_date,p.training_cutoff,
            p.training_data_hash,p.training_match_count,p.payload FROM real_oos_predictions_v2 p
            WHERE p.model_id IN ('ELO_V1','PI_RATING_V1','DIXON_COLES_V1')
            ORDER BY p.model_id,p.match_date LIMIT 30""").fetchall()
        for target_id, competition_id, target_date, cutoff, data_hash, expected_count, payload in rows:
            cutoff_date = cutoff.date()
            history = connection.execute("""SELECT match_id FROM real_canonical_matches
                WHERE competition_id=? AND status='FINISHED' AND match_date < ?
                  AND match_date >= ? ORDER BY match_id""",
                [competition_id, cutoff_date,
                 (cutoff - __import__("datetime").timedelta(days=1095)).date()]).fetchall()
            ids = [row[0] for row in history]
            decoded = json.loads(payload)
            expected_hash = hashlib.sha256("|".join(ids).encode()).hexdigest()
            assert target_id not in ids
            assert all(connection.execute("SELECT match_date FROM real_canonical_matches WHERE match_id=?",
                                          [item]).fetchone()[0] < target_date for item in ids)
            assert expected_hash == decoded["metadata"]["training_match_ids_hash"]
            assert len(ids) == expected_count
            assert data_hash == decoded["input_data_version"]


def test_real_oos_persisted_and_data_origin(real_oos_run) -> None:
    database, _report = real_oos_run
    with duckdb.connect(str(database), read_only=True) as connection:
        total, bad = connection.execute("""SELECT count(*), count(*) FILTER
            (WHERE NOT is_oos OR data_origin<>'REAL' OR p_home<0 OR p_draw<0 OR p_away<0
             OR abs(p_home+p_draw+p_away-1)>1e-6)
            FROM real_oos_predictions_v2 WHERE match_date BETWEEN '2025-10-01' AND '2026-01-31'""").fetchone()
        assert total > 0 and bad == 0


def test_real_metrics_only_and_temporal_modes_separate(real_oos_run) -> None:
    database, report = real_oos_run
    assert report.metrics
    assert all(metric["sample_size"] > 0 for metric in report.metrics)
    with duckdb.connect(str(database)) as connection:
        modes = {row[0] for row in connection.execute(
            "SELECT DISTINCT temporal_mode FROM real_oos_metrics").fetchall()}
        assert modes == {"DATE_SAFE_BATCH"}
        assert connection.execute("SELECT count(*) FROM real_oos_metrics "
                                  "WHERE temporal_mode='DATE_SAFE_BATCH'").fetchone()[0] > 0
        with pytest.raises(duckdb.ConstraintException):
            connection.execute("UPDATE real_oos_predictions_v2 SET data_origin='SYNTHETIC_TEST'")


def test_insufficient_sample_warning(real_oos_run) -> None:
    database, _report = real_oos_run
    with duckdb.connect(str(database)) as connection:
        rows = _persist_metrics(connection, date(2025, 10, 1), date(2026, 1, 31), 100_000)
        assert rows and all(row["status"] == "INSUFFICIENT_SAMPLE" for row in rows)


def test_real_oos_to_canonical_pre_meta_no_fake_final(real_oos_run) -> None:
    database, report = real_oos_run
    assert report.canonical_result_count > 0 and report.canonical_preview is not None
    with duckdb.connect(str(database), read_only=True) as connection:
        payload = connection.execute("SELECT payload FROM real_canonical_predictions_v2 "
            "WHERE data_origin='REAL' ORDER BY created_at LIMIT 1").fetchone()[0]
    canonical = CanonicalPredictionResult.model_validate(json.loads(payload))
    assert canonical.status == AvailabilityStatus.AVAILABLE
    assert canonical.probability_stage == ProbabilityStage.PRE_META
    assert canonical.pit_status == "PASS" and canonical.data_origin == "REAL"
    assert canonical.market_status == AvailabilityStatus.UNAVAILABLE
    assert canonical.calibration_status == AvailabilityStatus.NOT_IMPLEMENTED
