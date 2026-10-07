"""One-pass user-authoritative blind-test inference; never fit or scrape."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from erguoyuan_football.blind_test_r3.capability import (
    CLUB,
    NATIONAL_TEAM,
    UNKNOWN,
    ModelCapabilityRouter,
    ModelRegistration,
)
from erguoyuan_football.blind_test_r3.derivation import standardize
from erguoyuan_football.blind_test_r3.market_intelligence import (
    fuse_model_market,
    load_r3_market_config,
    r5_market_only,
)
from erguoyuan_football.blind_test_r3.model_domain import classify_fixture
from erguoyuan_football.blind_test_r3.release_registry import load_verified_release
from erguoyuan_football.blind_test_r3.runner import _team_id
from erguoyuan_football.blind_test_r3.selection import ENGINE_VERSION, select
from erguoyuan_football.blind_test_r3.store import canonical_bytes, sha256, write_once
from erguoyuan_football.blind_test_r3.user_daily import (
    UserFixture,
    load_slate,
    screenshot_hash,
)
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.models.dixon_coles import CoreDixonColesModel
from erguoyuan_football.models.score_matrix import ScoreMatrix

ROOT = Path(__file__).resolve().parents[3]
RELEASE_PATH = ROOT / "cloud_release/release.json"
STATE_ROOT = ROOT / "cloud_state"
CONFIG_PATH = ROOT / "config/r3_market_intelligence_v1.yaml"
ROUTER_PATH = ROOT / "cloud_release/model_router.json"
RUN_VERSION = "BRAZIL_RELEASE_V1"


def init_state(root: Path = STATE_ROOT) -> None:
    for name in ("market", "predictions", "locks", "failure_audits", "results",
                 "evaluation", "summaries"):
        (root / name).mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(root / "market.sqlite")) as connection, connection:
        connection.execute("CREATE TABLE IF NOT EXISTS market_snapshots ("
            "snapshot_id TEXT PRIMARY KEY, fixture_id TEXT NOT NULL, "
            "captured_at TEXT NOT NULL, source TEXT NOT NULL, "
            "input_json_hash TEXT NOT NULL, screenshot_hash TEXT, "
            "payload_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL)")
        for name in ("blind_test_predictions", "results", "evaluations"):
            connection.execute(f"CREATE TABLE IF NOT EXISTS {name} ("
                "event_id TEXT PRIMARY KEY, fixture_id TEXT NOT NULL, "
                "created_at TEXT NOT NULL, payload_sha256 TEXT NOT NULL)")
        connection.execute("CREATE TRIGGER IF NOT EXISTS market_no_update "
            "BEFORE UPDATE ON market_snapshots BEGIN SELECT RAISE(ABORT, 'APPEND_ONLY'); END")
        connection.execute("CREATE TRIGGER IF NOT EXISTS market_no_delete "
            "BEFORE DELETE ON market_snapshots BEGIN SELECT RAISE(ABORT, 'APPEND_ONLY'); END")


def _verify_release(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify and load the entire immutable multi-model release before routing."""
    return load_verified_release(root)


def _market_payload(item: UserFixture, fixture_id: str, input_hash: str,
                    evidence_hash: str | None, now: datetime) -> dict[str, Any]:
    odds = {"SPF": item.spf.model_dump() if item.spf else None,
            "RQSPF": item.rqspf.model_dump()}
    observed_at = item.market_observed_at or now
    identity = {"fixture_id": fixture_id, "odds": odds, "input_json_hash": input_hash,
                "screenshot_hash": evidence_hash, "captured_at": observed_at.isoformat()}
    return {**identity, "market_snapshot_id": "JCU-" + sha256(canonical_bytes(identity))[:32],
            "jc_match_number": item.jc_match_number,
            "source": "USER_CONFIRMED_MARKET", "source_authority": "USER_AUTHORITATIVE",
            "market_type": "JC", "odds": odds,
            "no_vig": {"SPF": item.spf.no_vig() if item.spf else None,
                       "RQSPF": item.rqspf.no_vig()},
            "market_verified_external": False}


def _save_market(root: Path, payload: dict[str, Any]) -> Path:
    snapshot_id = payload["market_snapshot_id"]
    encoded = canonical_bytes(payload)
    path = root / "market" / f"{snapshot_id}.json"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("MARKET_SNAPSHOT_ID_CONTENT_CONFLICT")
    else:
        write_once(path, encoded)
    with closing(sqlite3.connect(root / "market.sqlite")) as connection, connection:
        existing = connection.execute(
            "SELECT payload_sha256 FROM market_snapshots WHERE snapshot_id=?",
            (snapshot_id,)).fetchone()
        if existing is None:
            connection.execute("INSERT INTO market_snapshots VALUES (?,?,?,?,?,?,?,?)",
                (snapshot_id, payload["fixture_id"], payload["captured_at"],
                 payload["source"], payload["input_json_hash"],
                 payload["screenshot_hash"], sha256(encoded), encoded.decode("utf-8")))
        elif existing[0] != sha256(encoded):
            raise ValueError("MARKET_SNAPSHOT_ID_DATABASE_CONFLICT")
    return path


def _model_output(model: CoreDixonColesModel, item: UserFixture,
                  fixture_id: str, now: datetime, policy: dict[str, Any],
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    if item.kickoff is None:
        raise ValueError("USER_KICKOFF_UNAVAILABLE")
    if now >= item.kickoff:
        raise ValueError("KICKOFF_NOT_FUTURE")
    if item.neutral_venue is None:
        raise ValueError("MODEL_NEUTRAL_VENUE_UNAVAILABLE")
    home_id, away_id = _team_id(item.home_team), _team_id(item.away_team)
    threshold = int(policy["minimum_team_training_matches"])
    counts = model.metadata.get("team_match_counts", {})
    if min(counts.get(home_id, 0), counts.get(away_id, 0)) < threshold:
        raise ValueError("MODEL_TEAM_SAMPLE_BELOW_FROZEN_THRESHOLD")
    fixture = Fixture(match_id=fixture_id, lottery_match_no=item.jc_match_number,
        competition_id="SENIOR_MENS_INTERNATIONAL", home_team_id=home_id,
        away_team_id=away_id, kickoff_time=item.kickoff,
        source="USER_AUTHORITATIVE", retrieved_at=now, as_of_time=now,
        data_version=hashlib.sha256(canonical_bytes(item.model_dump(mode="json"))).hexdigest(),
        neutral_venue=item.neutral_venue)
    snapshot = PredictionSnapshot(match_id=fixture_id, prediction_time=now,
                                  match_data_snapshot=fixture)
    raw = model.predict(fixture, snapshot).model_dump(mode="json")
    standard = standardize(raw, item.rqspf.handicap)
    return raw, standard


def _club_model_output(model: Any, item: UserFixture, fixture_id: str,
                       now: datetime, aliases: dict[str, str]) -> dict[str, Any]:
    """Predict with one frozen Brazil artifact after exact domain/team capability checks."""
    from erguoyuan_football.blind_test_r3.club_brazil import BRAZIL_SERIE_A

    if item.kickoff is None or now >= item.kickoff:
        raise ValueError("USER_KICKOFF_UNAVAILABLE_OR_NOT_FUTURE")
    if item.neutral_venue is None:
        raise ValueError("MODEL_NEUTRAL_VENUE_UNAVAILABLE")
    try:
        home_id, away_id = aliases[item.home_team], aliases[item.away_team]
    except KeyError as error:
        raise ValueError(f"BRAZIL_TEAM_ALIAS_UNAVAILABLE:{error.args[0]}") from error
    fixture = Fixture(match_id=fixture_id, lottery_match_no=item.jc_match_number,
        competition_id=BRAZIL_SERIE_A, home_team_id=home_id, away_team_id=away_id,
        kickoff_time=item.kickoff, source="USER_AUTHORITATIVE", retrieved_at=now,
        as_of_time=now,
        data_version=hashlib.sha256(canonical_bytes(item.model_dump(mode="json"))).hexdigest(),
        neutral_venue=item.neutral_venue)
    if not model.supports_fixture(fixture):
        raise ValueError("MODEL_SUPPORTS_FIXTURE_FALSE")
    snapshot = PredictionSnapshot(match_id=fixture_id, prediction_time=now,
                                  match_data_snapshot=fixture)
    result = model.predict(fixture, snapshot).model_dump(mode="json")
    if result.get("execution_status") != "SUCCESS":
        raise ValueError(f"FROZEN_MODEL_PREDICTION_{result.get('execution_status')}:{result.get('reason')}")
    return result


def _ensemble_score_output(model_outputs: dict[str, dict[str, Any]],
                           probabilities: dict[str, float], handicap: int
                           ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Average only real available score matrices and preserve ensemble H/D/A separately."""
    matrices: list[ScoreMatrix] = []
    sources: list[str] = []
    for name, raw in sorted(model_outputs.items()):
        values = raw.get("metadata", {}).get("score_matrix")
        if values is None:
            continue
        matrices.append(ScoreMatrix.model_validate(values))
        sources.append(name)
    if not matrices:
        unavailable = _hda_standard(probabilities, handicap,
                                    "NO_AVAILABLE_FROZEN_SCORE_MATRIX")
        unavailable["probability_sources"] = {"one_x_two": "AVAILABLE_MODEL_ENSEMBLE"}
        return {"execution_status": "SUCCESS", "model_id": "AVAILABLE_MODEL_ENSEMBLE",
                "model_version": RUN_VERSION, "p_home": probabilities["HOME"],
                "p_draw": probabilities["DRAW"], "p_away": probabilities["AWAY"]}, unavailable
    if len({matrix.max_goals for matrix in matrices}) != 1:
        unavailable = _hda_standard(probabilities, handicap,
                                    "FROZEN_SCORE_MATRIX_SUPPORT_MISMATCH")
        return {"execution_status": "SUCCESS", "model_id": "AVAILABLE_MODEL_ENSEMBLE",
                "model_version": RUN_VERSION, "p_home": probabilities["HOME"],
                "p_draw": probabilities["DRAW"], "p_away": probabilities["AWAY"]}, unavailable
    count = len(matrices)
    cells = [[sum(matrix.values[row][column] for matrix in matrices) / count
              for column in range(matrices[0].max_goals + 1)]
             for row in range(matrices[0].max_goals + 1)]
    retained = sum(matrix.retained_mass for matrix in matrices) / count
    matrix = ScoreMatrix(values=tuple(tuple(row) for row in cells),
                         max_goals=matrices[0].max_goals,
                         retained_mass=retained, tail_mass=1.0 - retained)
    matrix_output = {"execution_status": "SUCCESS", "model_id": "AVAILABLE_SCORE_MATRIX_ENSEMBLE",
        "model_version": "EQUAL_WEIGHT_AVAILABLE_SCORE_MATRICES_V1",
        "p_home": matrix.outcome().p_home, "p_draw": matrix.outcome().p_draw,
        "p_away": matrix.outcome().p_away,
        "metadata": {"score_matrix": matrix.model_dump(mode="json"),
                     "score_matrix_models": sources}}
    standardized = standardize(matrix_output, handicap)
    standardized["one_x_two"] = _hda_standard(probabilities, handicap, "UNUSED")["one_x_two"]
    standardized["probability_sources"] = {
        "one_x_two": "EQUAL_WEIGHT_AVAILABLE_MODELS_V1",
        "score_matrix_derivations": sources,
        "score_matrix": "EQUAL_WEIGHT_AVAILABLE_SCORE_MATRICES_V1"}
    return matrix_output, standardized


def _hda_standard(probabilities: dict[str, float], handicap: int,
                  reason: str) -> dict[str, Any]:
    """Represent genuine H/D/A output when no compatible score matrix exists."""
    top2 = [{"selection": key, "probability": value} for key, value in sorted(
        probabilities.items(), key=lambda pair: (-pair[1], pair[0]))[:2]]
    unavailable = {"status": "UNAVAILABLE", "reason": reason}
    return {"status": "AVAILABLE_PARTIAL",
            "one_x_two": {"probabilities": probabilities, "top2": top2},
            "handicap_one_x_two": {**unavailable, "official_handicap": handicap},
            "total_goals": unavailable,
            "half_full_time": {"status": "UNAVAILABLE",
                               "reason": "INDEPENDENT_HTFT_MODEL_NOT_FROZEN"},
            "exact_score": unavailable,
            "score_matrix": unavailable,
            "daily_combinations": unavailable,
            "reference_plans_400_100_20": unavailable,
            "market_verified": False}


def _save_failure_audit(root: Path, payload: dict[str, Any]) -> tuple[str, str]:
    """Append a model-failure audit and a non-prediction audit lock."""
    audit_id = "FAIL-" + str(uuid4())
    record = {"audit_id": audit_id, "record_type": "MODEL_EXECUTION_FAILURE_AUDIT",
              "result_status": "MODEL_EXECUTION_FAILED",
              "r3_official_blind_test": False, "oos_eligible": False,
              "model_probabilities": None, **payload}
    data = canonical_bytes(record)
    write_once(root / "failure_audits" / f"{audit_id}.json", data)
    lock_id = "AUDIT-LOCK-" + str(uuid4())
    lock = {"event": "MODEL_EXECUTION_FAILURE_AUDIT_LOCK", "lock_id": lock_id,
            "audit_id": audit_id, "audit_sha256": sha256(data),
            "locked_at": datetime.now(UTC).isoformat(),
            "r3_official_blind_test": False, "oos_eligible": False}
    write_once(root / "locks" / f"{lock_id}.json", canonical_bytes(lock))
    return audit_id, lock_id


def _router_for_fixture(model: CoreDixonColesModel, item: UserFixture,
                        fixture_id: str, now: datetime,
                        policy: dict[str, Any], release: dict[str, Any],
                        router_manifest: dict[str, Any], aliases: dict[str, str]
                        ) -> tuple[str, dict[str, Any]]:
    """Route only entries bound to artifacts loaded by the verified release registry."""
    domain = classify_fixture(item, _team_id)
    registration_rows = router_manifest.get("models", [])
    registrations: list[ModelRegistration] = []
    release_models = {row["model_id"]: row for row in release["models"]}
    loaded_models = release["loaded_models"]
    for row in registration_rows:
        name = f"{row['model_id']}@{row['model_version']}"
        bound = release_models.get(row["model_id"])
        if (bound is None or row["model_version"] != bound["model_version"]
                or row["artifact_id"] != bound["artifact_id"]
                or row["artifact_sha256"] != bound["artifact_sha256"]
                or row["domain"] != bound["domain"]):
            def invalid_release(row_name: str = name) -> dict[str, Any]:
                raise ValueError(f"MODEL_NOT_BOUND_TO_VERIFIED_RELEASE:{row_name}")
            registrations.append(ModelRegistration(name, row["domain"], invalid_release))
            continue
        model = loaded_models[row["model_id"]]
        if row["domain"] == CLUB:
            def run_club(model: Any = model) -> dict[str, Any]:
                return _club_model_output(model, item, fixture_id, now, aliases)
            runner = run_club
        else:
            def run_national(model: CoreDixonColesModel = model) -> dict[str, Any]:
                raw, _ = _model_output(model, item, fixture_id, now, policy)
                return raw
            runner = run_national
        registrations.append(ModelRegistration(name, row["domain"], runner))
    minimum = int(router_manifest.get("minimum_available_models", 1))
    route = ModelCapabilityRouter(registrations, minimum).execute(domain)
    if domain not in (NATIONAL_TEAM, CLUB, UNKNOWN):
        route["model_coverage"] = "UNAVAILABLE"
        route["reason"] = "UNKNOWN_MODEL_DOMAIN"
    return domain, route


def _save_prediction(root: Path, payload: dict[str, Any]) -> tuple[str, str]:
    prediction_id = str(uuid4())
    record = {"prediction_id": prediction_id, **payload}
    data = canonical_bytes(record)
    write_once(root / "predictions" / f"{prediction_id}.json", data)
    lock_id = "LOCK-" + str(uuid4())
    lock = {"event": "PREDICTION_LOCK", "lock_id": lock_id,
            "prediction_id": prediction_id, "prediction_sha256": sha256(data),
            "locked_at": datetime.now(UTC).isoformat()}
    write_once(root / "locks" / f"{lock_id}.json", canonical_bytes(lock))
    return prediction_id, lock_id


def run_daily(input_path: Path, *, root: Path = ROOT,
              state_root: Path | None = None) -> dict[str, Any]:
    """Route every authoritative fixture and separate prediction locks from failure audits."""
    state = state_root or root / "cloud_state"
    slate, input_hash = load_slate(input_path)
    release, policy = _verify_release(root)
    models = release["loaded_models"]
    national_model = models["DIXON_COLES_V1"]
    aliases = release["brazil_mapping"]["aliases"]
    release_models = {row["model_id"]: row for row in release["models"]}
    router_manifest = json.loads((root / "cloud_release/model_router.json").read_text(
        encoding="utf-8"))
    if (router_manifest.get("router_version") != RUN_VERSION or
            router_manifest.get("ensemble_method") != "EQUAL_WEIGHT_AVAILABLE_MODELS_V1"):
        raise ValueError("MODEL_ROUTER_MANIFEST_INVALID")
    rule = json.loads((root / "cloud_release/selection_rule.json").read_text(encoding="utf-8"))
    if (rule["engine_version"] != ENGINE_VERSION or rule["source_sha256"] !=
            sha256((root / "src/erguoyuan_football/blind_test_r3/selection.py").read_bytes())):
        raise ValueError("CLOUD_SELECTION_ENGINE_CHANGED")
    config = load_r3_market_config(root / "config/r3_market_intelligence_v1.yaml")
    init_state(state)
    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    selectable: list[dict[str, Any]] = []
    for item in slate.fixtures:
        fixture_id = item.fixture_id(slate.slate_date)
        evidence_hash = screenshot_hash(root, item.screenshot_path)
        market = _market_payload(item, fixture_id, input_hash, evidence_hash, now)
        market_path = _save_market(state, market)
        domain, route = _router_for_fixture(
            national_model, item, fixture_id, now, policy, release, router_manifest, aliases)
        available_count = int(route["available_model_count"])
        model_status = "AVAILABLE" if available_count >= int(
            router_manifest.get("minimum_available_models", 1)) else "MODEL_EXECUTION_FAILED"
        raw: dict[str, Any] | None = None
        standard: dict[str, Any] | None = None
        ensemble = route.get("available_model_ensemble")
        model_outputs = route.get("model_outputs", {})
        if model_status == "AVAILABLE" and ensemble is not None:
            if domain == CLUB:
                raw, standard = _ensemble_score_output(
                    dict(model_outputs), dict(ensemble), item.rqspf.handicap)
            elif available_count == 1:
                raw = dict(next(iter(model_outputs.values())))
                try:
                    if item.rqspf is None:
                        raise ValueError("USER_HANDICAP_UNAVAILABLE")
                    standard = standardize(raw, item.rqspf.handicap)
                except (ValueError, TypeError, KeyError, RuntimeError) as error:
                    standard = _hda_standard(
                        ensemble, item.rqspf.handicap,
                        f"SCORE_MATRIX_UNAVAILABLE:{type(error).__name__}:{error}")
            else:
                raw = {"execution_status": "SUCCESS",
                       "model_id": "AVAILABLE_MODEL_ENSEMBLE",
                       "model_version": RUN_VERSION,
                       "lottery_match_no": item.jc_match_number,
                       "model_names": route["available_models"],
                       "p_home": ensemble["HOME"], "p_draw": ensemble["DRAW"],
                       "p_away": ensemble["AWAY"]}
                standard = _hda_standard(
                    ensemble, item.rqspf.handicap,
                    "ENSEMBLE_HAS_NO_COMMON_FROZEN_SCORE_MATRIX")
        market_spf = market["no_vig"]["SPF"]
        market_rqspf = market["no_vig"]["RQSPF"]
        if standard is not None:
            probabilities = standard["one_x_two"]["probabilities"]
            fusion = fuse_model_market(probabilities, market_spf, config)
            if market_spf is None:
                edge = None
                divergence = "UNAVAILABLE_MARKET"
            else:
                edge = {key: (probabilities[key] - market_spf[key]) * 100
                        for key in probabilities}
                max_edge = max(abs(value) for value in edge.values())
                divergence = ("SEVERE" if max_edge >= config["divergence"]["severe_pp"]
                              else "WARNING" if max_edge >= config["divergence"]["warning_pp"]
                              else "NORMAL")
            handicap_head = standard.get("handicap_one_x_two", {})
            if handicap_head.get("status") == "UNAVAILABLE":
                handicap_fusion = {"status": "UNAVAILABLE", "reason": handicap_head.get("reason")}
                handicap_edge = None
                handicap_divergence = "UNAVAILABLE"
            else:
                handicap_probabilities = handicap_head["probabilities"]
                handicap_fusion = fuse_model_market(handicap_probabilities, market_rqspf, config)
                handicap_edge = {key: (handicap_probabilities[key] - market_rqspf[key]) * 100
                                 for key in handicap_probabilities}
                handicap_max_edge = max(abs(value) for value in handicap_edge.values())
                handicap_divergence = (
                    "SEVERE" if handicap_max_edge >= config["divergence"]["severe_pp"]
                    else "WARNING" if handicap_max_edge >= config["divergence"]["warning_pp"]
                    else "NORMAL")
        else:
            fusion = {"status": "UNAVAILABLE", "reason": "MODEL_UNAVAILABLE"}
            handicap_fusion = {"status": "UNAVAILABLE", "reason": "MODEL_UNAVAILABLE"}
            edge = None
            handicap_edge = None
            divergence = "UNAVAILABLE"
            handicap_divergence = "UNAVAILABLE"
        payload = {"mode": "BLIND_TEST_R3", "run_version": RUN_VERSION,
            "record_type": "R3_MODEL_PREDICTION" if model_status == "AVAILABLE" else
                          "MODEL_EXECUTION_FAILURE_AUDIT",
            "result_status": "OFFICIAL_BLIND_TEST" if model_status == "AVAILABLE" else
                             "MODEL_EXECUTION_FAILED",
            "r3_official_blind_test": model_status == "AVAILABLE",
            "oos_eligible": model_status == "AVAILABLE",
            "slate_date": slate.slate_date.isoformat(),
            "fixture_id": fixture_id, "jc_match_number": item.jc_match_number,
            "competition": item.competition, "home_team": item.home_team,
            "away_team": item.away_team,
            "kickoff_time": item.kickoff.isoformat() if item.kickoff else None,
            "run_time": now.isoformat(),
            "data_as_of": max((release_models[model_id]["data_as_of"]
                               for model_id in release_models if model_id in {
                                   row.split("@")[0] for row in route["available_models"]}),
                              default=release_models["DIXON_COLES_V1"]["data_as_of"]),
            "model_snapshot_id": release["release_id"],
            "model_snapshot_ids": {row["model_id"]: row["artifact_id"]
                                   for row in release["models"]},
            "model_name": "AVAILABLE_MODEL_ENSEMBLE" if domain == CLUB else release["models"][0]["model_id"],
            "model_version": RUN_VERSION if domain == CLUB else release["models"][0]["model_version"],
            "release_id": release["release_id"],
            "market_snapshot_id": market["market_snapshot_id"],
            "market_snapshot_sha256": sha256(market_path.read_bytes()),
            "market_source": "USER_CONFIRMED_MARKET", "input_json_hash": input_hash,
            "screenshot_hash": evidence_hash, "market_verified_external": False,
            "user_authoritative_input": True, "fixture_reverification": False,
            "market_reverification": False,
            "external_research_mode": "OPTIONAL_ENRICHMENT",
            "match_domain": domain, "available_models": route["available_models"],
            "unavailable_models": route["unavailable_models"],
            "model_errors": route["model_errors"],
            "available_model_count": available_count,
            "registered_model_count": route["registered_model_count"],
            "model_coverage": route["model_coverage"],
            "model_status": model_status,
            "individual_model_outputs": model_outputs,
            "individual_model_probabilities": route["model_distributions"],
            "available_model_ensemble": ensemble,
            "raw_model_output": raw, "standardized_output": standard,
            "jc_market": market, "r3": standard if standard is not None else
                  {"status": "UNAVAILABLE", "reason": model_status},
            "r5": {"SPF": r5_market_only(market_spf),
                   "RQSPF": r5_market_only(market["no_vig"]["RQSPF"])},
            "edge_pp": edge, "divergence": divergence,
            "handicap_edge_pp": handicap_edge,
            "handicap_divergence": handicap_divergence,
            "fusion": fusion, "handicap_fusion": handicap_fusion,
            "global_market": {"status": "UNAVAILABLE"},
            "external_research": {"status": "OPTIONAL_NOT_REQUESTED"},
            "external_research_warning": item.external_research_warnings or None,
            "r3_status": "AVAILABLE" if model_status == "AVAILABLE" else "UNAVAILABLE",
            "r5_status": "AVAILABLE_MARKET_ONLY"}
        if model_status == "AVAILABLE" and raw is not None and standard is not None:
            prediction_id, lock_id = _save_prediction(state, payload)
            audit_id = None
        else:
            audit_payload = {**payload, "failure_reason": route.get("reason") or
                             route.get("model_errors") or route["unavailable_models"]}
            audit_id, lock_id = _save_failure_audit(state, audit_payload)
            prediction_id = None
        row = {"fixture_id": fixture_id, "jc_match_number": item.jc_match_number,
               "market_snapshot_id": market["market_snapshot_id"],
               "prediction_id": prediction_id, "failure_audit_id": audit_id,
               "lock_id": lock_id,
               "match_domain": domain, "available_models": route["available_models"],
               "unavailable_models": route["unavailable_models"],
               "model_errors": route["model_errors"],
               "model_coverage": route["model_coverage"],
               "model_status": model_status, "r3": payload["r3"], "r5": payload["r5"],
               "edge_pp": edge, "divergence": divergence, "fusion": fusion,
               "handicap_edge_pp": handicap_edge,
               "handicap_divergence": handicap_divergence,
               "handicap_fusion": handicap_fusion}
        rows.append(row)
        if prediction_id is not None and raw is not None and standard is not None:
            selectable.append({"prediction_id": prediction_id, "fixture_id": fixture_id,
                "home_team": item.home_team, "away_team": item.away_team,
                "raw_model_output": raw, "standardized_output": standard})
    plans = select(selectable) if selectable else {"status": "UNAVAILABLE",
        "reason": "NO_MODEL_PROBABILITIES"}
    official_count = sum(row["model_status"] == "AVAILABLE" for row in rows)
    failure_count = sum(row["model_status"] == "MODEL_EXECUTION_FAILED" for row in rows)
    summary = {"source": "USER_AUTHORITATIVE", "run_version": RUN_VERSION,
               "user_authoritative_input": True,
               "fixture_reverification": False, "market_reverification": False,
               "public_schedule_conflict": "WARNING_ONLY",
               "total_user_fixtures": len(slate.fixtures),
               "total_processed": len(rows),
               "official_r3_predictions": official_count,
               "model_execution_failures": failure_count,
               "oos_eligible_count": official_count,
               "input_rejected": 0,
               "slate_date": slate.slate_date.isoformat(),
               "release_id": release["release_id"],
               "model_snapshot_ids": {row["model_id"]: row["artifact_id"]
                                      for row in release["models"]},
               "model_versions": {row["model_id"]: row["model_version"]
                                  for row in release["models"]},
               "fusion_version": config["fusion"]["version"],
               "market_source": "USER_CONFIRMED_MARKET", "results": rows,
               "selection_engine": plans,
               "production_ready": False,
               "blind_test_r3_ready": official_count > 0}
    summary_id = str(uuid4())
    summary_data = canonical_bytes(summary)
    write_once(state / "summaries" / f"{summary_id}.json", summary_data)
    write_once(state / "locks" / f"FINAL-{summary_id}.json", canonical_bytes({
        "event": "FINAL_OUTPUT_LOCK", "summary_id": summary_id,
        "summary_sha256": sha256(summary_data),
        "locked_at": datetime.now(UTC).isoformat()}))
    summary["summary_id"] = summary_id
    summary["final_output_lock_id"] = f"FINAL-{summary_id}"
    return summary
