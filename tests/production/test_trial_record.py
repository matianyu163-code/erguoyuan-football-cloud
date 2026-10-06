"""Trial record tests verify append-only persistence and empty-result handling."""

import sqlite3
from datetime import UTC, datetime

import pytest

from production.trial_record import TrialPredictionRecord, TrialRecordStore


def _record(prediction_id: str) -> TrialPredictionRecord:
    return TrialPredictionRecord(
        prediction_id=prediction_id,
        created_time=datetime.now(UTC),
        match_name="SYNTHETIC_TEST_HOME VS SYNTHETIC_TEST_AWAY",
        competition="SYNTHETIC_TEST_LEAGUE",
        source_type="USER_JC_CONFIRMED",
        models_used=(),
        input_snapshot={"data": "SYNTHETIC_TEST"},
        prediction_output={"status": "UNAVAILABLE"},
    )


def test_trial_record_persists_required_fields(tmp_path) -> None:
    store = TrialRecordStore(tmp_path / "trial_prediction_records.sqlite")
    try:
        store.append(_record("SYNTHETIC_TEST_ID"))
        row = store.list_records()[0]
    finally:
        store.close()
    assert row["prediction_id"] == "SYNTHETIC_TEST_ID"
    assert row["final_result"] is None
    assert row["actual_result"] is None
    assert row["data_snapshot"]["data"] == "SYNTHETIC_TEST"
    assert row["review_status"] == "NOT_REVIEWED"
    assert row["models_used"] == []


def test_trial_record_does_not_overwrite_existing_id(tmp_path) -> None:
    store = TrialRecordStore(tmp_path / "trial_prediction_records.sqlite")
    try:
        store.append(_record("SYNTHETIC_TEST_ID"))
        with pytest.raises(sqlite3.IntegrityError):
            store.append(_record("SYNTHETIC_TEST_ID"))
        assert len(store.list_records()) == 1
    finally:
        store.close()


def test_trial_record_correction_is_append_only(tmp_path) -> None:
    store = TrialRecordStore(tmp_path / "trial_prediction_records.sqlite")
    try:
        store.append(_record("SYNTHETIC_TEST_ID"))
        annotation_id = store.append_annotation(
            prediction_id="SYNTHETIC_TEST_ID",
            note="SYNTHETIC_TEST correction",
            operator="SYNTHETIC_TEST",
        )
        assert annotation_id
        assert len(store.list_records()) == 1
        with pytest.raises(ValueError, match="TRIAL_ANNOTATION_TARGET_NOT_FOUND"):
            store.append_annotation(
                prediction_id="UNKNOWN", note="SYNTHETIC_TEST", operator="SYNTHETIC_TEST")
    finally:
        store.close()


def test_trial_record_database_rejects_direct_update_and_delete(tmp_path) -> None:
    store = TrialRecordStore(tmp_path / "trial_prediction_records.sqlite")
    try:
        store.append(_record("SYNTHETIC_TEST_ID"))
        with pytest.raises(sqlite3.IntegrityError, match="TRIAL_RECORDS_APPEND_ONLY"):
            store._connection.execute(
                "UPDATE trial_prediction_records SET match_name='changed' "
                "WHERE prediction_id='SYNTHETIC_TEST_ID'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="TRIAL_RECORDS_APPEND_ONLY"):
            store._connection.execute(
                "DELETE FROM trial_prediction_records WHERE prediction_id='SYNTHETIC_TEST_ID'"
            )
        assert len(store.list_records()) == 1
    finally:
        store.close()
