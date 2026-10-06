"""Production UI status is derived from genuine append-only trial records."""

from production.status import read_production_status
from production.trial_record import TrialPredictionRecord, TrialRecordStore


def _record(*, prediction_output: dict, synthetic: bool = False,
            simulation: bool = False) -> TrialPredictionRecord:
    return TrialPredictionRecord.create(
        match_name="Borussia Dortmund VS SV Werder Bremen",
        competition="Bundesliga",
        source_type="RESEARCH_TEST",
        models_used=("DIXON_COLES_V1",),
        input_snapshot={
            "fixture_verified": True,
            "real_snapshot_created": True,
            "synthetic_data": synthetic,
            "simulation_only": simulation,
        },
        prediction_output=prediction_output,
    )


def _success_output(*, warnings: bool = False) -> dict:
    return {
        "status": "BASE_MODELS_EXECUTED_WITH_WARNINGS" if warnings else "READY",
        "warning_codes": ["BAYESIAN_DIAGNOSTICS_WARNING"] if warnings else [],
        "models": [{
            "model_id": "DIXON_COLES_V1",
            "execution_status": "DEGRADED_EXECUTED",
            "probability": {"p_home": 0.5, "p_draw": 0.25, "p_away": 0.25},
        }],
    }


def test_no_prediction_record_shows_no_live_prediction(tmp_path) -> None:
    state = read_production_status(tmp_path / "missing.sqlite")

    assert state.prediction_message == "尚无实时预测记录。"
    assert state.global_message == "尚无实时预测记录。"


def test_real_blocked_attempt_without_snapshot_shows_its_pipeline_reason(tmp_path) -> None:
    database = tmp_path / "trial.sqlite"
    store = TrialRecordStore(database)
    try:
        store.append(TrialPredictionRecord.create(
            match_name="Arsenal VS Chelsea",
            competition="Premier League",
            source_type="AUTO_DISCOVERY",
            models_used=(),
            input_snapshot={"home_team": "Arsenal", "away_team": "Chelsea",
                            "simulation_only": False},
            prediction_output={"status": "UNAVAILABLE", "steps": [{
                "stage": "DATA_PIPELINE", "status": "WARNING",
                "detail": "VERIFIED_FIXTURE_AND_PIT_SNAPSHOT_UNAVAILABLE",
            }]},
        ))
    finally:
        store.close()

    state = read_production_status(database)

    assert state.prediction_message == (
        "本次预测未完成：VERIFIED_FIXTURE_AND_PIT_SNAPSHOT_UNAVAILABLE。")


def test_model_failure_reason_is_shown_for_blocked_real_run(tmp_path) -> None:
    database = tmp_path / "trial.sqlite"
    store = TrialRecordStore(database)
    try:
        store.append(_record(prediction_output={
            "status": "UNAVAILABLE",
            "models": [{"model_id": "DIXON_COLES_V1", "execution_status": "FAILED",
                        "reason": "TRAINING_HISTORY_INSUFFICIENT"}],
        }))
    finally:
        store.close()

    state = read_production_status(database)

    assert state.prediction_message == (
        "本次预测未完成：DIXON_COLES_V1: TRAINING_HISTORY_INSUFFICIENT。")


def test_real_prediction_record_does_not_show_not_run(tmp_path) -> None:
    database = tmp_path / "trial.sqlite"
    store = TrialRecordStore(database)
    try:
        store.append(_record(prediction_output=_success_output()))
    finally:
        store.close()

    state = read_production_status(database)

    assert "尚未运行" not in state.prediction_message
    assert state.prediction_message == "当前为生产试运行模式，预测结果已按实战流程生成并留档。"
    assert state.global_status == "PRODUCTION_TRIAL_READY"


def test_blocked_record_shows_actual_block_reason(tmp_path) -> None:
    database = tmp_path / "trial.sqlite"
    store = TrialRecordStore(database)
    try:
        store.append(_record(prediction_output={
            "status": "UNAVAILABLE",
            "steps": [{"stage": "FIXTURE_VERIFICATION", "status": "WARNING",
                       "detail": "FIXTURE_NOT_VERIFIED: provider returned no exact fixture"}],
            "models": [],
        }))
    finally:
        store.close()

    state = read_production_status(database)

    assert state.prediction_message == (
        "本次预测未完成：FIXTURE_NOT_VERIFIED: provider returned no exact fixture。")
    assert state.global_status == "PRODUCTION_BLOCKED"


def test_ready_with_warnings_shows_global_trial_status(tmp_path) -> None:
    database = tmp_path / "trial.sqlite"
    store = TrialRecordStore(database)
    try:
        store.append(_record(prediction_output=_success_output(warnings=True)))
    finally:
        store.close()

    state = read_production_status(database)

    assert state.global_status == "PRODUCTION_TRIAL_READY_WITH_WARNINGS"
    assert state.global_message == "生产试运行已启用（存在已知告警）"


def test_blocked_latest_attempt_preserves_real_global_trial_state(tmp_path) -> None:
    database = tmp_path / "trial.sqlite"
    store = TrialRecordStore(database)
    try:
        store.append(_record(prediction_output=_success_output(warnings=True)))
        store.append(_record(prediction_output={
            "status": "UNAVAILABLE",
            "steps": [{"stage": "DATA_PIPELINE", "status": "WARNING",
                       "detail": "HISTORICAL_SNAPSHOT_UNAVAILABLE"}],
            "models": [],
        }))
    finally:
        store.close()

    state = read_production_status(database)

    assert state.global_message == "生产试运行已启用（存在已知告警）"
    assert state.prediction_message == "本次预测未完成：HISTORICAL_SNAPSHOT_UNAVAILABLE。"


def test_synthetic_or_simulation_records_do_not_count_as_real(tmp_path) -> None:
    database = tmp_path / "trial.sqlite"
    store = TrialRecordStore(database)
    try:
        store.append(_record(prediction_output=_success_output(), synthetic=True))
        store.append(_record(prediction_output=_success_output(), simulation=True))
    finally:
        store.close()

    assert read_production_status(database).prediction_message == "尚无实时预测记录。"
