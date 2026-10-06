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

from erguoyuan_football.blind_test_r3.derivation import standardize
from erguoyuan_football.blind_test_r3.market_intelligence import (
    fuse_model_market,
    load_r3_market_config,
    r5_market_only,
)
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

ROOT = Path(__file__).resolve().parents[3]
RELEASE_PATH = ROOT / "cloud_release/release.json"
STATE_ROOT = ROOT / "cloud_state"
CONFIG_PATH = ROOT / "config/r3_market_intelligence_v1.yaml"


def init_state(root: Path = STATE_ROOT) -> None:
    for name in ("market", "predictions", "locks", "results", "evaluation", "summaries"):
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


def _verify_release(root: Path) -> tuple[CoreDixonColesModel, dict[str, Any], dict[str, Any]]:
    release = json.loads((root / "cloud_release/release.json").read_text(encoding="utf-8"))
    paths = release["paths"]
    artifact = root / paths["model_directory"]
    policy_path = root / paths["policy"]
    expected = {"model_payload_sha256": artifact / "model.joblib",
                "model_manifest_sha256": artifact / "manifest.json",
                "model_policy_sha256": policy_path}
    if any(sha256(path.read_bytes()) != release[key] for key, path in expected.items()):
        raise ValueError("CLOUD_FROZEN_RELEASE_HASH_MISMATCH")
    code_paths = {"model_source": "src/erguoyuan_football/models/dixon_coles.py",
                  "adapter_source": "src/erguoyuan_football/models/penaltyblog_adapter.py",
                  "score_matrix_source": "src/erguoyuan_football/models/score_matrix.py",
                  "derivation_source": "src/erguoyuan_football/blind_test_r3/derivation.py",
                  "team_mapping": "src/erguoyuan_football/knowledge/entities/country_aliases.json"}
    for name, relative in code_paths.items():
        if sha256((root / relative).read_bytes()) != release["code_sha256"][name]:
            raise ValueError(f"CLOUD_FROZEN_CODE_CHANGED:{name}")
    model = CoreDixonColesModel.load(artifact)
    if (not model.fitted or model.model_id != release["model_id"]
            or model.model_version != release["model_version"]
            or model.trained_until is None
            or model.trained_until.isoformat() != release["data_as_of"]):
        raise ValueError("CLOUD_FROZEN_MODEL_LINEAGE_MISMATCH")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    return model, release, policy


def _market_payload(item: UserFixture, fixture_id: str, input_hash: str,
                    evidence_hash: str | None, now: datetime) -> dict[str, Any]:
    odds = {"SPF": item.spf.model_dump(), "RQSPF": item.rqspf.model_dump()}
    identity = {"fixture_id": fixture_id, "odds": odds, "input_json_hash": input_hash,
                "screenshot_hash": evidence_hash, "captured_at": now.isoformat()}
    return {**identity, "market_snapshot_id": "JCU-" + sha256(canonical_bytes(identity))[:32],
            "jc_match_number": item.jc_match_number,
            "source": "USER_CONFIRMED_MARKET", "source_authority": "USER_AUTHORITATIVE",
            "market_type": "JC", "odds": odds,
            "no_vig": {"SPF": item.spf.no_vig(), "RQSPF": item.rqspf.no_vig()},
            "market_verified_external": False}


def _save_market(root: Path, payload: dict[str, Any]) -> Path:
    snapshot_id = payload["market_snapshot_id"]
    encoded = canonical_bytes(payload)
    path = root / "market" / f"{snapshot_id}.json"
    write_once(path, encoded)
    with closing(sqlite3.connect(root / "market.sqlite")) as connection, connection:
        connection.execute("INSERT INTO market_snapshots VALUES (?,?,?,?,?,?,?,?)",
            (snapshot_id, payload["fixture_id"], payload["captured_at"],
             payload["source"], payload["input_json_hash"],
             payload["screenshot_hash"], sha256(encoded), encoded.decode("utf-8")))
    return path


def _model_output(model: CoreDixonColesModel, item: UserFixture,
                  fixture_id: str, now: datetime, policy: dict[str, Any],
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
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
    """Load frozen artifact and append one market/prediction/lock per valid pre-match fixture."""
    state = state_root or root / "cloud_state"
    slate, input_hash = load_slate(input_path)
    model, release, policy = _verify_release(root)
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
        if now >= item.kickoff:
            rows.append({"fixture_id": fixture_id, "status": "UNAVAILABLE",
                         "reason": "KICKOFF_NOT_FUTURE"})
            continue
        evidence_hash = screenshot_hash(root, item.screenshot_path)
        market = _market_payload(item, fixture_id, input_hash, evidence_hash, now)
        market_path = _save_market(state, market)
        model_status = "AVAILABLE"
        raw: dict[str, Any] | None = None
        standard: dict[str, Any] | None = None
        try:
            raw, standard = _model_output(model, item, fixture_id, now, policy)
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            model_status = f"UNAVAILABLE:{type(error).__name__}:{error}"
        market_spf = market["no_vig"]["SPF"]
        if standard is not None:
            probabilities = standard["one_x_two"]["probabilities"]
            fusion = fuse_model_market(probabilities, market_spf, config)
            edge = {key: (probabilities[key] - market_spf[key]) * 100
                    for key in probabilities}
            max_edge = max(abs(value) for value in edge.values())
            divergence = ("SEVERE" if max_edge >= config["divergence"]["severe_pp"]
                          else "WARNING" if max_edge >= config["divergence"]["warning_pp"]
                          else "NORMAL")
        else:
            fusion = {"status": "UNAVAILABLE", "reason": "MODEL_UNAVAILABLE"}
            edge = None
            divergence = "UNAVAILABLE"
        payload = {"mode": "FROZEN_BLIND_TEST", "slate_date": slate.slate_date.isoformat(),
            "fixture_id": fixture_id, "jc_match_number": item.jc_match_number,
            "competition": item.competition, "home_team": item.home_team,
            "away_team": item.away_team, "kickoff_time": item.kickoff.isoformat(),
            "run_time": now.isoformat(), "data_as_of": release["data_as_of"],
            "model_snapshot_id": release["source_snapshot_id"],
            "model_name": release["model_id"], "model_version": release["model_version"],
            "market_snapshot_id": market["market_snapshot_id"],
            "market_snapshot_sha256": sha256(market_path.read_bytes()),
            "market_source": "USER_CONFIRMED_MARKET", "input_json_hash": input_hash,
            "screenshot_hash": evidence_hash, "model_status": model_status,
            "raw_model_output": raw, "standardized_output": standard,
            "jc_market": market, "r3": standard if standard is not None else
                  {"status": "UNAVAILABLE", "reason": model_status},
            "r5": {"SPF": r5_market_only(market_spf),
                   "RQSPF": r5_market_only(market["no_vig"]["RQSPF"])},
            "edge_pp": edge, "divergence": divergence,
            "fusion": fusion, "global_market": {"status": "UNAVAILABLE"},
            "external_research": {"status": "OPTIONAL_NOT_REQUESTED"}}
        prediction_id, lock_id = _save_prediction(state, payload)
        row = {"fixture_id": fixture_id, "jc_match_number": item.jc_match_number,
               "market_snapshot_id": market["market_snapshot_id"],
               "prediction_id": prediction_id, "lock_id": lock_id,
               "model_status": model_status, "r3": payload["r3"], "r5": payload["r5"],
               "edge_pp": edge, "divergence": divergence, "fusion": fusion}
        rows.append(row)
        if raw is not None and standard is not None:
            selectable.append({"prediction_id": prediction_id, "fixture_id": fixture_id,
                "home_team": item.home_team, "away_team": item.away_team,
                "raw_model_output": raw, "standardized_output": standard})
    plans = select(selectable) if selectable else {"status": "UNAVAILABLE",
        "reason": "NO_MODEL_PROBABILITIES"}
    summary = {"source": "USER_AUTHORITATIVE", "slate_date": slate.slate_date.isoformat(),
               "model_snapshot_id": release["source_snapshot_id"],
               "model_version": release["model_version"],
               "fusion_version": config["fusion"]["version"],
               "market_source": "USER_CONFIRMED_MARKET", "results": rows,
               "selection_engine": plans,
               "production_ready": False}
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
