"""R3 live inference: load a frozen artifact, predict, persist, lock; never fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from erguoyuan_football.blind_test_r3.derivation import standardize
from erguoyuan_football.blind_test_r3.store import R3Store, canonical_bytes, sha256
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.knowledge.entities.country_alias_registry import (
    CountryAliasRegistry,
)
from erguoyuan_football.models.dixon_coles import CoreDixonColesModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
R3_ROOT = PROJECT_ROOT / "blind_test" / "r3"
INPUT_ALIASES = {"波黑": "BIH"}


def _team_id(name: str) -> str:
    registry = CountryAliasRegistry()
    country = registry.resolve(name)
    if country is None:
        iso = INPUT_ALIASES.get(name)
        country = next((item for item in registry.all() if item.iso3 == iso), None)
    if country is None:
        raise ValueError(f"R3_TEAM_UNRESOLVED:{name}")
    return f"NATIONAL_{country.iso3}_M_SENIOR"


def _verify_live_code(record: dict[str, Any]) -> None:
    """Prevent running changed imported code against frozen snapshot bytes."""
    paths = {
        "model_source": PROJECT_ROOT / "src/erguoyuan_football/models/dixon_coles.py",
        "adapter_source": PROJECT_ROOT / "src/erguoyuan_football/models/penaltyblog_adapter.py",
        "score_matrix_source": PROJECT_ROOT / "src/erguoyuan_football/models/score_matrix.py",
        "derivation_source": PROJECT_ROOT / "src/erguoyuan_football/blind_test_r3/derivation.py",
        "runner_source": Path(__file__),
    }
    for name, path in paths.items():
        if sha256(path.read_bytes()) != record["file_sha256"].get(name):
            raise ValueError(f"R3_IMPORTED_CODE_CHANGED:{name}")


def run_slate(store: R3Store, snapshot_id: str, slate_path: Path,
              fixtures_path: Path) -> list[dict[str, Any]]:
    """Run each fixture independently and lock every persisted successful output."""
    frozen = store.verify_snapshot(snapshot_id)
    _verify_live_code(frozen)
    spec = frozen["specification"]
    model = CoreDixonColesModel.load(
        Path(frozen["file_paths"]["model_artifact"]).parent)
    if (model.model_id != spec["model_id"] or model.model_version != spec["model_version"]
            or model.training_data_hash != spec["data_version"]):
        raise ValueError("R3_MODEL_SNAPSHOT_LINEAGE_MISMATCH")
    policy = json.loads(Path(frozen["file_paths"]["model_policy"]).read_text(encoding="utf-8"))
    slate = json.loads(slate_path.read_text(encoding="utf-8"))
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    if slate.get("mode") != "BLIND_TEST_R3" or slate.get("market_verified") is not False:
        raise ValueError("R3_SLATE_MODE_OR_MARKET_STATUS_INVALID")
    if fixtures.get("fixture_status") != "VERIFIED_PREMATCH":
        raise ValueError("R3_FIXTURE_EVIDENCE_INVALID")
    fixture_time = datetime.fromisoformat(fixtures["retrieved_at"])
    fixture_by_code = {row["jc_code"]: row for row in fixtures["fixtures"]}
    if len(fixture_by_code) != len(fixtures["fixtures"]):
        raise ValueError("R3_DUPLICATE_FIXTURE_CODE")
    results: list[dict[str, Any]] = []
    for match in slate["matches"]:
        code = match["jc_code"]
        evidence = fixture_by_code.get(code)
        fixture_id = evidence["fixture_id"] if evidence else code
        try:
            if evidence is None:
                raise ValueError("R3_FIXTURE_EVIDENCE_MISSING")
            home_id, away_id = _team_id(match["home"]), _team_id(match["away"])
            if (home_id != f"NATIONAL_{evidence['home_iso3']}_M_SENIOR"
                    or away_id != f"NATIONAL_{evidence['away_iso3']}_M_SENIOR"):
                raise ValueError("R3_FIXTURE_TEAM_CONFLICT")
            kickoff = datetime.fromisoformat(evidence["kickoff_utc"])
            now = datetime.now(UTC)
            if fixture_time > now or kickoff <= now:
                raise ValueError("R3_PIT_FIXTURE_NOT_PREMATCH")
            if evidence["neutral_venue"] not in (True, False):
                raise ValueError("R3_NEUTRAL_VENUE_UNKNOWN")
            if (datetime.fromisoformat(match["kickoff_display"].replace(" ", "T"))
                    != (kickoff.astimezone(ZoneInfo("Asia/Shanghai"))
                        .replace(tzinfo=None))):
                raise ValueError("R3_SCREENSHOT_KICKOFF_CONFLICT")
            counts = model.metadata.get("team_match_counts", {})
            threshold = int(policy["minimum_team_training_matches"])
            if min(counts.get(home_id, 0), counts.get(away_id, 0)) < threshold:
                raise ValueError("R3_TEAM_SAMPLE_BELOW_FROZEN_THRESHOLD")
            if model.trained_until is None or model.trained_until > now:
                raise ValueError("R3_MODEL_TRAINING_AFTER_RUN")
            version = hashlib.sha256(canonical_bytes(evidence)).hexdigest()
            fixture = Fixture(match_id=str(fixture_id), lottery_match_no=code,
                competition_id="SENIOR_MENS_INTERNATIONAL",
                home_team_id=home_id, away_team_id=away_id, kickoff_time=kickoff,
                source="UEFA_OFFICIAL_MATCH_PAGE", retrieved_at=fixture_time,
                as_of_time=fixture_time, data_version=version,
                season="2026-27", neutral_venue=evidence["neutral_venue"])
            snapshot = PredictionSnapshot(match_id=fixture.match_id,
                prediction_time=now, match_data_snapshot=fixture)
            prediction = model.predict(fixture, snapshot)
            raw = prediction.model_dump(mode="json")
            standardized = standardize(raw, match["handicap"])
            execution_id = "R3EXEC-" + sha256(canonical_bytes(raw))[:32]
            prediction_id = store.append_prediction({
                "prediction_date": slate["display_date"],
                "competition": match["competition"], "fixture_id": fixture_id,
                "home_team": match["home"], "away_team": match["away"],
                "kickoff_time": kickoff.isoformat(), "handicap": match["handicap"],
                "model_snapshot_id": snapshot_id,
                "data_as_of": spec["data_as_of"], "run_time": now.isoformat(),
                "raw_model_output": raw, "standardized_output": standardized,
                "model_execution_record_id": execution_id,
                "prediction_snapshot_id": snapshot.prediction_snapshot_id,
                "fixture_evidence_url": evidence["source_url"],
                "fixture_evidence_retrieved_at": fixture_time.isoformat(),
                "market_verified": False,
            })
            lock = store.lock_prediction(prediction_id)
            results.append({"jc_code": code, "status": "LOCKED",
                "prediction_id": prediction_id, "lock_id": prediction_id,
                "lock_path": str(lock), "standardized_output": standardized})
        except (OSError, ValueError, KeyError, TypeError) as error:
            reason = f"{type(error).__name__}:{error}"
            failure = store.append_failure({"fixture_id": fixture_id,
                "jc_code": code, "reason": reason,
                "model_snapshot_id": snapshot_id})
            results.append({"jc_code": code, "status": "UNAVAILABLE",
                            "reason": reason, "failure_path": str(failure)})
    return results


def main(argv: list[str] | None = None) -> int:
    """Execute one saved R3 slate with no fit or market dependence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--slate", type=Path,
        default=R3_ROOT / "manifest/2026-10-05_user_slate.json")
    parser.add_argument("--fixtures", type=Path,
        default=R3_ROOT / "manifest/2026-10-05_uefa_fixtures.json")
    args = parser.parse_args(argv)
    rows = run_slate(R3Store(R3_ROOT), args.snapshot_id, args.slate, args.fixtures)
    print(json.dumps({"mode": "BLIND_TEST_R3", "model_snapshot_id": args.snapshot_id,
                      "results": rows}, ensure_ascii=False, indent=2))
    return 0 if all(row["status"] == "LOCKED" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
