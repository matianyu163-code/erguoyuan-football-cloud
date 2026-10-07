"""Synthetic verification of the separate official blind-test publish level."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from erguoyuan_football.blind_test_r3 import market_p2a, settlement
from erguoyuan_football.blind_test_r3.capability import execute_available
from erguoyuan_football.blind_test_r3.finalize import finalize
from erguoyuan_football.blind_test_r3.market_evaluation import (
    aggregate_market_evaluations,
    append_market_evaluation,
)
from erguoyuan_football.blind_test_r3.market_lock import append_market_lock
from erguoyuan_football.blind_test_r3.official import (
    blind_test_gate,
    oos_eligibility,
    publish_locked,
)
from erguoyuan_football.blind_test_r3.production_gate import evaluate_production_gate
from erguoyuan_football.blind_test_r3.store import (
    R3Store,
    canonical_bytes,
    sha256,
    write_once,
)


def _synthetic_case(tmp_path):
    store = R3Store(tmp_path / "r3")
    store.initialize()
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"SYNTHETIC_TEST_FROZEN_MODEL")
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text(json.dumps({"model_id": "SYNTHETIC_TEST_DC",
        "model_version": "1.0.0", "payload_sha256": sha256(artifact.read_bytes()),
        "trained_until": "2026-01-01T00:00:00Z", "training_data_hash": "synthetic",
        "config_hash": "synthetic"}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("synthetic: true", encoding="utf-8")
    source = tmp_path / "source.py"
    source.write_text("# SYNTHETIC_TEST", encoding="utf-8")
    spec = {"model_id": "SYNTHETIC_TEST_DC", "model_version": "1.0.0",
            "feature_version": "SYNTHETIC_TEST", "data_version": "synthetic",
            "score_matrix_method": "SYNTHETIC_TEST",
            "derivation_method": "SYNTHETIC_TEST",
            "calibrator_version": "UNAVAILABLE", "code_hash": "SYNTHETIC_TEST_HASH",
            "code_commit": "UNVERSIONED_SOURCE_TREE"}
    files = {"model_artifact": artifact, "artifact_manifest": manifest,
             "model_config": config, "model_source": source,
             "score_matrix_source": source, "derivation_source": source,
             "runner_source": source}
    snapshot_id = store.freeze_snapshot(spec, files)
    frozen = store.verify_snapshot(snapshot_id)
    now = datetime.now(UTC)
    run_time = now - timedelta(minutes=1)
    kickoff = now + timedelta(days=1)
    fixture_retrieved = now - timedelta(minutes=5)
    fixture = {"fixture_id": "SYNTHETIC_TEST_FIXTURE", "jc_code": "SYNTHETIC_TEST_001",
               "source_url": "https://example.test/fixture",
               "kickoff_utc": kickoff.isoformat()}
    slate_match = {"home": "SYNTHETIC_HOME", "away": "SYNTHETIC_AWAY",
                   "handicap": -1}
    payload = {"prediction_date": now.date().isoformat(),
               "competition": "SYNTHETIC_TEST", "fixture_id": fixture["fixture_id"],
               "home_team": slate_match["home"], "away_team": slate_match["away"],
               "kickoff_time": kickoff.isoformat(), "handicap": -1,
               "model_snapshot_id": snapshot_id,
               "data_as_of": fixture_retrieved.isoformat(),
               "run_time": run_time.isoformat(),
               "raw_model_output": {"execution_status": "SUCCESS",
                   "model_id": spec["model_id"], "model_version": spec["model_version"]},
               "standardized_output": {
                   "one_x_two": {"probabilities": {
                       "HOME": 0.5, "DRAW": 0.3, "AWAY": 0.2}},
                   "total_goals": {"top2": [{"selection": "2", "probability": 0.3},
                                           {"selection": "3", "probability": 0.2}]},
                   "exact_score": {"top2": [{"selection": "1-0", "probability": 0.1},
                                          {"selection": "1-1", "probability": 0.08}]},
                   "half_full_time": {"status": "UNAVAILABLE",
                                      "reason": "SYNTHETIC_TEST_NO_HTFT"},
                   "daily_combinations": {"status": "UNAVAILABLE",
                                          "reason": "SYNTHETIC_TEST_OLD_RULE"},
                   "reference_plans_400_100_20": {"status": "UNAVAILABLE",
                                                  "reason": "SYNTHETIC_TEST_OLD_RULE"}},
               "model_execution_record_id": "SYNTHETIC_TEST_EXECUTION",
               "prediction_snapshot_id": "SYNTHETIC_TEST_INPUT_SNAPSHOT",
               "fixture_evidence_retrieved_at": fixture_retrieved.isoformat(),
               "market_verified": False}
    release = {"release_version": "SYNTHETIC_TEST_RELEASE",
               "publisher_source_sha256": "SYNTHETIC_TEST_CODE"}
    return store, frozen, fixture, slate_match, payload, release


def test_production_false_market_and_htft_missing_still_publish(tmp_path) -> None:
    store, frozen, fixture, slate_match, payload, release = _synthetic_case(tmp_path)
    assert not evaluate_production_gate({})["production_eligible"]
    first_id = store.append_prediction(payload)
    store.lock_prediction(first_id)
    before = (store.root / "predictions" / f"{first_id}.json").read_bytes()
    official, rich_lock, record = publish_locked(
        store, first_id, frozen, fixture, slate_match, release)
    assert record["publish_level"] == "FROZEN_BLIND_TEST"
    assert record["result_status"] == "OFFICIAL_BLIND_TEST"
    assert record["r3_official_blind_test"] is True
    assert record["r3_production_signal"] is False
    assert record["model_probabilities"] == {"HOME": 0.5, "DRAW": 0.3, "AWAY": 0.2}
    assert record["derived_outputs"]["total_goals"]["top2"]
    assert record["derived_outputs"]["exact_score"]["top2"]
    assert record["unavailable_modules"]["half_full_time"] == "SYNTHETIC_TEST_NO_HTFT"
    assert record["unavailable_modules"]["market"] == "MARKET_VERIFIED_FALSE"
    assert official.exists() and rich_lock.exists()
    assert (store.root / "predictions" / f"{first_id}.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        publish_locked(store, first_id, frozen, fixture, slate_match, release)
    second_id = store.append_prediction(payload)
    store.lock_prediction(second_id)
    assert first_id != second_id
    assert store.verify_lock(first_id) and store.verify_lock(second_id)


def test_late_prediction_is_rejected(tmp_path) -> None:
    store, frozen, fixture, slate_match, payload, _ = _synthetic_case(tmp_path)
    payload["run_time"] = payload["kickoff_time"]
    payload["prediction_id"] = "SYNTHETIC_TEST_ID"
    payload["mode"] = "BLIND_TEST_R3"
    with pytest.raises(ValueError, match="NOT_PREMATCH"):
        blind_test_gate(payload, frozen, fixture, slate_match)
    assert not list((store.root / "official").glob("*.json")) if (
        store.root / "official").exists() else True


def test_one_model_failure_does_not_block_other_frozen_outputs() -> None:
    def failed():
        raise RuntimeError("SYNTHETIC_TEST_MODEL_FAILURE")

    good = lambda: {"HOME": 0.5, "DRAW": 0.3, "AWAY": 0.2}
    runners = {f"SYNTHETIC_TEST_MODEL_{i}": good for i in range(5)}
    runners["SYNTHETIC_TEST_MODEL_5"] = failed
    result = execute_available(runners)
    assert result["available_model_count"] == 5
    assert result["registered_model_count"] == 6
    assert len(result["unavailable_models"]) == 1
    assert result["available_model_ensemble"] == good()


def test_only_locked_official_result_can_enter_r3_oos() -> None:
    publication = {"publish_level": "DEVELOPMENT",
                   "result_status": "DEVELOPMENT_ONLY", "lock_status": "LOCKED",
                   "input_snapshot_sha256": "SYNTHETIC_TEST_HASH"}
    assert not oos_eligibility(publication, result_verified=True)["r3_oos_eligible"]
    publication.update(publish_level="FROZEN_BLIND_TEST",
                       result_status="OFFICIAL_BLIND_TEST",
                       r3_official_blind_test=True,
                       model_probabilities={"HOME": 0.5, "DRAW": 0.3, "AWAY": 0.2})
    assert not oos_eligibility(publication, result_verified=False)["r3_oos_eligible"]
    result = oos_eligibility(publication, result_verified=True)
    assert result["r3_oos_eligible"]
    assert not result["phase9_golden_holdout_promoted"]
    assert not evaluate_production_gate({"golden_oos": True})["production_eligible"]


def test_model_failure_not_counted_in_brier() -> None:
    failure = {"record_type": "MODEL_EXECUTION_FAILURE_AUDIT",
               "result_status": "MODEL_EXECUTION_FAILED",
               "r3_official_blind_test": False, "model_probabilities": None,
               "publish_level": "FROZEN_BLIND_TEST", "lock_status": "LOCKED",
               "input_snapshot_sha256": "SYNTHETIC_TEST_HASH"}
    eligibility = oos_eligibility(failure, result_verified=True)
    assert eligibility["r3_oos_eligible"] is False
    assert eligibility["reason"] == "MODEL_EXECUTION_FAILURE_NOT_SCOREABLE"


def test_verified_result_and_r3_oos_are_append_only_after_kickoff(
        tmp_path, monkeypatch) -> None:
    store, frozen, fixture, slate_match, payload, release = _synthetic_case(tmp_path)
    prediction_id = store.append_prediction(payload)
    store.lock_prediction(prediction_id)
    publish_locked(store, prediction_id, frozen, fixture, slate_match, release)
    kickoff = datetime.fromisoformat(payload["kickoff_time"])
    monkeypatch.setattr(settlement, "_now", lambda: kickoff + timedelta(hours=3))
    evidence = {"fixture_id": fixture["fixture_id"], "home_score": 2,
                "away_score": 1, "source": "SYNTHETIC_TEST_OFFICIAL",
                "source_url": "https://example.test/result",
                "source_timestamp": (kickoff + timedelta(hours=2)).isoformat(),
                "retrieved_at": (kickoff + timedelta(hours=2, minutes=5)).isoformat(),
                "verification_status": "VERIFIED"}
    result_path = settlement.append_verified_result(store, prediction_id, evidence)
    evaluation_path = settlement.append_r3_oos_evaluation(store, prediction_id)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["r3_oos_eligible"] is True
    assert evaluation["phase9_golden_holdout_promoted"] is False
    assert evaluation["log_loss"] == pytest.approx(0.6931471805599453)
    assert evaluation["multiclass_brier_score"] == pytest.approx(0.38)
    assert result_path.exists()
    with pytest.raises(FileExistsError):
        settlement.append_verified_result(store, prediction_id, evidence)
    with pytest.raises(FileExistsError):
        settlement.append_r3_oos_evaluation(store, prediction_id)


def test_result_cannot_be_recorded_before_kickoff(tmp_path) -> None:
    store, frozen, fixture, slate_match, payload, release = _synthetic_case(tmp_path)
    prediction_id = store.append_prediction(payload)
    store.lock_prediction(prediction_id)
    publish_locked(store, prediction_id, frozen, fixture, slate_match, release)
    evidence = {"fixture_id": fixture["fixture_id"]}
    with pytest.raises(ValueError, match="BEFORE_KICKOFF"):
        settlement.append_verified_result(store, prediction_id, evidence)


def test_final_batch_appends_audit_correction_without_overwrite(tmp_path) -> None:
    store, frozen, fixture, slate_match, payload, release = _synthetic_case(tmp_path)
    ids = []
    for _ in range(2):
        prediction_id = store.append_prediction(payload)
        store.lock_prediction(prediction_id)
        publish_locked(store, prediction_id, frozen, fixture, slate_match, release)
        ids.append(prediction_id)
    selection_id = "SYNTHETIC_TEST_SELECTION"
    selection = {"event": "DERIVED_SELECTION", "selection_id": selection_id,
                 "model_snapshot_id": frozen["snapshot_id"],
                 "prediction_date": payload["prediction_date"],
                 "identity": {"prediction_ids": ids,
                     "lock_sha256": {prediction_id: sha256((store.root / "locks" /
                         f"{prediction_id}.json").read_bytes()) for prediction_id in ids}},
                 "results": {key: {"status": "AVAILABLE"} for key in (
                     "one_x_two_pair", "handicap_pair", "total_goals_duplex",
                     "score_pair_duplex", "standard_plan_400", "free_plan_100")}}
    selection["results"]["half_full_time_duplex"] = {"status": "UNAVAILABLE"}
    selection["results"]["high_odds_20"] = {"status": "UNAVAILABLE"}
    selection_path = store.root / "selections" / f"{selection_id}.json"
    write_once(selection_path, canonical_bytes(selection))
    original = (store.root / "official" / f"{ids[0]}.json").read_bytes()
    batch_path, batch = finalize(store, selection_id)
    assert len(batch["post_lock_error_paths"]) == 2
    assert batch["final_module_statuses"]["standard_plan_400"] == "AVAILABLE"
    assert (store.root / "official" / f"{ids[0]}.json").read_bytes() == original
    assert sha256(batch_path.read_bytes()) == sha256(finalize(store, selection_id)[0].read_bytes())


def test_market_supplement_lock_is_model_only_and_never_changes_old_lock(
        tmp_path, monkeypatch) -> None:
    store, frozen, fixture, slate_match, payload, release = _synthetic_case(tmp_path)
    prediction_id = store.append_prediction(payload)
    store.lock_prediction(prediction_id)
    publish_locked(store, prediction_id, frozen, fixture, slate_match, release)
    old_path = store.root / "official_locks" / prediction_id / "prediction_lock.json"
    old_sha = sha256(old_path.read_bytes())
    new_path, market_lock = append_market_lock(store, prediction_id,
        at=datetime.now(UTC), manual_directory=tmp_path / "no_manual")
    assert market_lock["market"]["status"] == "UNAVAILABLE"
    assert market_lock["market"]["no_vig"] is None
    assert market_lock["market"]["fusion"]["mode"] == "MODEL_ONLY"
    assert market_lock["final_fusion_probability"] == market_lock["model_probability"]
    assert market_lock["market"]["provider_status"]["JC_OFFICIAL_PROVIDER"][
        "status"] == "NOT_CONFIGURED"
    assert new_path.exists() and sha256(old_path.read_bytes()) == old_sha
    kickoff = datetime.fromisoformat(payload["kickoff_time"])
    monkeypatch.setattr(settlement, "_now", lambda: kickoff + timedelta(hours=3))
    evidence = {"fixture_id": fixture["fixture_id"], "home_score": 1,
                "away_score": 0, "source": "SYNTHETIC_TEST_OFFICIAL",
                "source_url": "https://example.test/result",
                "source_timestamp": (kickoff + timedelta(hours=2)).isoformat(),
                "retrieved_at": (kickoff + timedelta(hours=2, minutes=5)).isoformat(),
                "verification_status": "VERIFIED"}
    settlement.append_verified_result(store, prediction_id, evidence)
    evaluation_path = append_market_evaluation(store, prediction_id, new_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["model"]["status"] == "AVAILABLE"
    assert evaluation["market_no_vig"]["status"] == "UNAVAILABLE"
    assert evaluation["final_fusion"]["log_loss"] == evaluation["model"]["log_loss"]
    aggregate = aggregate_market_evaluations([evaluation_path])
    assert aggregate["model"]["match_count"] == 1
    assert aggregate["market_no_vig"]["status"] == "UNAVAILABLE"
    assert aggregate["model"]["top1_ece_10_bins"] >= 0
    with pytest.raises(FileExistsError):
        append_market_evaluation(store, prediction_id, new_path)


def test_p2a_real_shape_http_appends_model_market_supplement(
        tmp_path, monkeypatch) -> None:
    store, frozen, fixture, slate_match, payload, release = _synthetic_case(tmp_path)
    prediction_id = store.append_prediction(payload)
    store.lock_prediction(prediction_id)
    publish_locked(store, prediction_id, frozen, fixture, slate_match, release)
    base_path = store.root / "official_locks" / prediction_id / "prediction_lock.json"
    base_hash = sha256(base_path.read_bytes())
    monkeypatch.setenv("YYCORE_THE_ODDS_API_KEY", "SYNTHETIC_TEST_SECRET")
    monkeypatch.setattr(market_p2a, "verify_p2a_release", lambda: {
        "version": "R3_THE_ODDS_API_P2A_V1", "model_snapshot_id": frozen["snapshot_id"]})
    synthetic_release = tmp_path / "p2a_release.json"
    synthetic_release.write_text('{"version":"R3_THE_ODDS_API_P2A_V1"}',
                                 encoding="utf-8")
    monkeypatch.setattr(market_p2a, "RELEASE_PATH", synthetic_release)
    monkeypatch.setattr(market_p2a, "load_sport_map", lambda _path: {
        "version": "THE_ODDS_API_SPORT_MAP_V1", "candidates": [],
        "mappings": {"SYNTHETIC_TEST": {"sport_key": "soccer_synthetic",
            "verified": True, "source": "THE_ODDS_API_SPORTS_ENDPOINT"}}})
    now = datetime.now(UTC)
    class Response:
        status_code = 200

        def __init__(self, body):
            self.body = body
            self.headers = {"x-requests-remaining": "17", "x-requests-used": "3",
                            "x-requests-last": "2"}

        def json(self):
            return self.body

    class Http:
        calls = 0

        def get(self, url, *, params, headers):
            self.calls += 1
            if url.endswith("/sports/"):
                return Response([{"key": "soccer_synthetic",
                    "title": "SYNTHETIC_TEST", "active": True}])
            return Response([{"id": "SYNTHETIC_EVENT", "sport_key": "soccer_synthetic",
                "commence_time": payload["kickoff_time"],
                "home_team": "SYNTHETIC_HOME", "away_team": "SYNTHETIC_AWAY",
                "bookmakers": [{"key": "book_a", "title": "SYNTHETIC_BOOK",
                    "markets": [{"key": "h2h", "last_update": (
                        now - timedelta(minutes=1)).isoformat(),
                        "outcomes": [{"name": "SYNTHETIC_HOME", "price": 1.9},
                                     {"name": "Draw", "price": 3.5},
                                     {"name": "SYNTHETIC_AWAY", "price": 4.5}]}]}]}])
    http = Http()
    lock_path, lock = market_p2a.append_p2a_market_lock(
        store, prediction_id, http_client=http,
        manual_directory=tmp_path / "no_manual")
    assert http.calls == 2
    assert lock_path.exists()
    assert lock["market"]["status"] == "AVAILABLE"
    assert lock["market"]["source_quality"] == "TRUSTED_API"
    assert lock["market"]["bookmaker_count"] == 1
    assert lock["market"]["fusion"]["mode"] == "MODEL_MARKET"
    assert lock["market"]["r5_market_only"]["mode"] == "MARKET_ONLY"
    assert lock["market"]["r5_market_only"]["probabilities"] == lock["market_probability"]
    assert lock["market"]["model_market_edge"] is not None
    assert lock["market"]["movement"] == "INSUFFICIENT_HISTORY"
    assert lock["base_official_lock_sha256"] == sha256(base_path.read_bytes()) == base_hash
    assert "SYNTHETIC_TEST_SECRET" not in lock_path.read_text(encoding="utf-8")
    import sqlite3
    with sqlite3.connect(store.root / "market/market.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM market_quotes").fetchone()[0] == 3
    second_path, second = market_p2a.append_p2a_market_lock(
        store, prediction_id, http_client=http,
        manual_directory=tmp_path / "no_manual")
    assert second_path != lock_path
    assert second["market"]["movement_status"] == "PREVIOUS_OBSERVED_TO_CURRENT"
    assert second["market"]["previous_market_snapshot_id"] == lock["market"]["snapshot_id"]
    assert second["market"]["movement"]["MAX_ABS_MOVE_PP"] == pytest.approx(0.0)
    assert http.calls == 2
    assert sha256(base_path.read_bytes()) == base_hash
    with sqlite3.connect(store.root / "market/market.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM market_movement").fetchone()[0] == 1


def test_p2a_no_key_still_appends_model_only_lock(tmp_path, monkeypatch) -> None:
    store, frozen, fixture, slate_match, payload, release = _synthetic_case(tmp_path)
    prediction_id = store.append_prediction(payload)
    store.lock_prediction(prediction_id)
    publish_locked(store, prediction_id, frozen, fixture, slate_match, release)
    monkeypatch.delenv("YYCORE_THE_ODDS_API_KEY", raising=False)
    monkeypatch.setattr(market_p2a, "verify_p2a_release", lambda: {
        "version": "R3_THE_ODDS_API_P2A_V1", "model_snapshot_id": frozen["snapshot_id"]})
    synthetic_release = tmp_path / "p2a_release.json"
    synthetic_release.write_text('{"version":"R3_THE_ODDS_API_P2A_V1"}',
                                 encoding="utf-8")
    monkeypatch.setattr(market_p2a, "RELEASE_PATH", synthetic_release)
    path, lock = market_p2a.append_p2a_market_lock(store, prediction_id,
        manual_directory=tmp_path / "no_manual")
    assert path.exists()
    assert lock["market"]["status"] == "UNAVAILABLE"
    assert lock["market"]["provider_prefetch_status"] == "NOT_CONFIGURED"
    assert lock["market"]["provider_status"]["THE_ODDS_API_V4"]["status"] == "NOT_CONFIGURED"
    assert lock["market"]["fusion"]["mode"] == "MODEL_ONLY"
    assert lock["final_fusion_probability"] == lock["model_probability"]
