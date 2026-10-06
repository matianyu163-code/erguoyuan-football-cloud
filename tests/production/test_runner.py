"""Runner tests prove each stage is explicit and fail-closed."""

from erguoyuan_football.match_source.match_source import MatchSourceType
from production.runner import ProductionRunner
from production.trial_record import TrialRecordStore


def test_runner_user_confirmed_request_audits_all_stages(
    production_config,
) -> None:
    result = ProductionRunner(production_config).run(
        "SYNTHETIC_TEST_HOME VS SYNTHETIC_TEST_AWAY",
        competition="SYNTHETIC_TEST_LEAGUE",
        jc_confirmed=True,
        simulation=True,
    )
    assert result.source_type == MatchSourceType.USER_JC_CONFIRMED
    assert result.status == "WARNING"
    assert [row.stage for row in result.stages] == [
        "INPUT", "SOURCE_CLASSIFIER", "ENTITY_RESOLVER", "COMPETITION_RESOLVER",
        "JC_VERIFICATION", "DATA_PIPELINE", "MODEL_PLANNER", "MODEL_EXECUTION",
        "ENSEMBLE", "V7_RENDER", "SAVE_RECORD",
    ]
    assert next(row for row in result.stages if row.stage == "JC_VERIFICATION").status == "READY"
    assert next(row for row in result.stages if row.stage == "MODEL_EXECUTION").status == "WARNING"
    assert result.prediction_id is not None
    store = TrialRecordStore(production_config.record_database)
    try:
        records = store.list_records()
    finally:
        store.close()
    assert len(records) == 1
    assert records[0]["models_used"] == []
    assert records[0]["final_result"] is None
    assert records[0]["input_snapshot"]["simulation_only"] is True


def test_runner_invalid_syntax_does_not_guess_teams(production_config) -> None:
    result = ProductionRunner(production_config).run("SYNTHETIC_TEST_NO_SEPARATOR")
    assert result.request.validation_status == "INVALID"
    assert result.request.home_team is None
    assert result.request.away_team is None
    assert next(row for row in result.stages if row.stage == "INPUT").status == "FAILED"
