"""Additive Phase 8.2 run, failure, artifact and dataset lineage storage."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import duckdb


def prepare_phase8_2_migration(db_path: str | Path) -> Path | None:
    """Back up an existing warehouse once, then add only new Phase 8.2 tables."""
    path = Path(db_path)
    backup: Path | None = None
    if str(path) != ":memory:" and path.exists():
        with duckdb.connect(str(path), read_only=True) as connection:
            applied = connection.execute("SELECT 1 FROM information_schema.tables "
                "WHERE table_name='phase8_2_migrations'").fetchone()
        if applied is None:
            backup = path.with_name(
                f"{path.stem}.phase8_2-pre-migration-{datetime.now(UTC):%Y%m%dT%H%M%SZ}{path.suffix}")
            shutil.copy2(path, backup)
    with duckdb.connect(str(path)) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS phase8_2_migrations (
                migration_id VARCHAR PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL);
            CREATE TABLE IF NOT EXISTS real_oos_run_manifest (
                run_id VARCHAR PRIMARY KEY, model_ids JSON NOT NULL,
                start_date DATE NOT NULL, end_date DATE NOT NULL,
                current_date DATE, completed_rows INTEGER NOT NULL DEFAULT 0,
                failed_rows INTEGER NOT NULL DEFAULT 0,
                unavailable_rows INTEGER NOT NULL DEFAULT 0,
                artifact_count INTEGER NOT NULL DEFAULT 0,
                last_checkpoint TIMESTAMPTZ, run_status VARCHAR NOT NULL,
                config_hash VARCHAR NOT NULL);
            CREATE TABLE IF NOT EXISTS real_oos_execution_status (
                match_id VARCHAR NOT NULL, model_id VARCHAR NOT NULL,
                prediction_date DATE NOT NULL, temporal_mode VARCHAR NOT NULL,
                execution_status VARCHAR NOT NULL,
                reason VARCHAR, prediction_id VARCHAR,
                training_data_hash VARCHAR, config_hash VARCHAR,
                run_id VARCHAR, updated_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY(match_id,model_id));
            CREATE TABLE IF NOT EXISTS real_oos_lineage_v2 (
                prediction_id VARCHAR PRIMARY KEY, artifact_id VARCHAR,
                lineage_hash VARCHAR NOT NULL, training_match_ids_hash VARCHAR NOT NULL,
                model_mode VARCHAR, dependency_tags JSON NOT NULL,
                created_at TIMESTAMPTZ NOT NULL);
            CREATE TABLE IF NOT EXISTS real_ml_feature_datasets_v2 (
                dataset_id VARCHAR PRIMARY KEY, family VARCHAR NOT NULL,
                mode VARCHAR NOT NULL CHECK(mode='NO_MARKET'),
                row_count INTEGER NOT NULL, data_hash VARCHAR NOT NULL,
                first_prediction_date DATE NOT NULL, last_prediction_date DATE NOT NULL,
                schema_hash VARCHAR NOT NULL, artifact_path VARCHAR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL);
            CREATE TABLE IF NOT EXISTS phase9_candidate_dataset_v2 (
                dataset_id VARCHAR PRIMARY KEY, mode VARCHAR NOT NULL CHECK(mode='NO_MARKET'),
                row_count INTEGER NOT NULL, first_prediction_date DATE NOT NULL,
                last_prediction_date DATE NOT NULL, data_hash VARCHAR NOT NULL,
                artifact_path VARCHAR NOT NULL, created_at TIMESTAMPTZ NOT NULL);
        """)
        connection.execute("INSERT INTO phase8_2_migrations VALUES ('PHASE8_2_V1', ?) "
                           "ON CONFLICT DO NOTHING", [datetime.now(UTC)])
        # Phase 8.1 counted missing model predictions but did not persist each
        # unavailable target. A successful baseline proves that the target-day
        # batch was evaluated, so this records the legacy absence explicitly.
        connection.execute("""INSERT INTO real_oos_execution_status
            (match_id,model_id,prediction_date,temporal_mode,execution_status,
             reason,prediction_id,training_data_hash,config_hash,run_id,updated_at)
            SELECT b.match_id,models.model_id,b.match_date,b.prediction_temporal_mode,
                   'UNAVAILABLE','PHASE8_1_NO_SUCCESS_RECORD',NULL,NULL,
                   b.config_hash,'PHASE8_1_LEGACY',?
            FROM real_oos_predictions_v2 b
            CROSS JOIN (SELECT UNNEST(['DIXON_COLES_V1','ELO_V1','PI_RATING_V1']) model_id) models
            LEFT JOIN real_oos_predictions_v2 p ON p.match_id=b.match_id
                  AND p.model_id=models.model_id
            WHERE b.model_id='NAIVE_LEAGUE_FREQUENCY_V1' AND p.prediction_id IS NULL
            ON CONFLICT DO NOTHING""", [datetime.now(UTC)])
    return backup
