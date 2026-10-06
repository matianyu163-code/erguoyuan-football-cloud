"""Deterministic, probability-only R3 selection over immutable predictions."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from itertools import combinations, product
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.store import (
    R3Store,
    canonical_bytes,
    sha256,
    write_once,
)

ENGINE_VERSION = "R3_SELECTION_ENGINE_V1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
R3_ROOT = PROJECT_ROOT / "blind_test" / "r3"
RULE_PATH = R3_ROOT / "manifest" / f"{ENGINE_VERSION}.json"


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("R3_INVALID_PROBABILITY")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("R3_INVALID_PROBABILITY")
    return result


def _top2(module: Any) -> list[dict[str, Any]] | None:
    if not isinstance(module, dict) or not isinstance(module.get("top2"), list):
        return None
    rows = module["top2"]
    if len(rows) != 2 or any(not isinstance(row, dict) for row in rows):
        return None
    result = [{"selection": row["selection"],
               "probability": _probability(row["probability"])} for row in rows]
    if result[0]["selection"] == result[1]["selection"]:
        raise ValueError("R3_DUPLICATE_TOP2")
    return result


def _entry(record: dict[str, Any], module: str, coverage: bool) -> dict[str, Any] | None:
    top2 = _top2(record["standardized_output"].get(module))
    if top2 is None:
        return None
    probability = sum(item["probability"] for item in top2) if coverage else top2[0]["probability"]
    if probability > 1.0 + 1e-9:
        raise ValueError("R3_COVERAGE_OVER_ONE")
    return {"prediction_id": record["prediction_id"],
            "fixture_id": record["fixture_id"],
            "jc_code": record["raw_model_output"]["lottery_match_no"]
                if record["raw_model_output"].get("lottery_match_no") else
                record["fixture_id"],
            "home_team": record["home_team"], "away_team": record["away_team"],
            "selection": [top2[0], top2[1]] if coverage else top2[0],
            "probability": probability}


def _pair(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(entries) < 2:
        return {"status": "UNAVAILABLE", "reason": "FEWER_THAN_TWO_ELIGIBLE_MATCHES"}
    pairs = [(a, b) for a, b in combinations(entries, 2)
             if a["fixture_id"] != b["fixture_id"]]
    if not pairs:
        return {"status": "UNAVAILABLE", "reason": "FEWER_THAN_TWO_DISTINCT_MATCHES"}
    a, b = min(pairs, key=lambda pair: (-pair[0]["probability"] * pair[1]["probability"],
                                      tuple(sorted((pair[0]["fixture_id"], pair[1]["fixture_id"])))))
    matches = sorted((a, b), key=lambda entry: entry["fixture_id"])
    return {"status": "AVAILABLE", "matches": matches,
            "joint_probability": a["probability"] * b["probability"],
            "joint_method": "PRODUCT_DISTINCT_FIXTURES_INDEPENDENCE_ASSUMPTION"}


def _best_single(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {"status": "UNAVAILABLE", "reason": "NO_ELIGIBLE_MATCH"}
    return {"status": "AVAILABLE", "match": min(
        entries, key=lambda entry: (-entry["probability"], entry["fixture_id"]))}


def _score_pair(entries: list[dict[str, Any]]) -> dict[str, Any]:
    result = _pair(entries)
    if result["status"] != "AVAILABLE":
        return result
    a, b = result["matches"]
    combos = [{"score_1": x["selection"], "score_2": y["selection"],
               "joint_probability": x["probability"] * y["probability"]}
              for x, y in product(a["selection"], b["selection"])]
    result["four_combinations"] = combos
    return result


def _allocate_units(total_units: int, weights: list[float], keys: list[str]) -> dict[str, int]:
    """Largest-remainder allocation; each unit is one 2 yuan stake."""
    if len(weights) != len(keys) or not weights or sum(weights) <= 0:
        raise ValueError("R3_ALLOCATION_WEIGHTS_INVALID")
    raw = [total_units * weight / sum(weights) for weight in weights]
    units = [math.floor(value) for value in raw]
    extras = total_units - sum(units)
    order = sorted(range(len(keys)), key=lambda i: (-(raw[i] - units[i]), keys[i]))
    for i in order[:extras]:
        units[i] += 1
    return {key: 2 * unit for key, unit in zip(keys, units)}


def _standard_plan(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible = [(name, results[name]["joint_probability"] if name != "total_goals_duplex"
                 else results[name]["match"]["probability"])
                for name in ("one_x_two_pair", "handicap_pair", "total_goals_duplex", "score_pair_duplex")
                if results[name]["status"] == "AVAILABLE"]
    if not eligible:
        return {"status": "UNAVAILABLE", "reason": "NO_PROBABILITY_ONLY_COMPONENT"}
    allocations = _allocate_units(200, [probability for _, probability in eligible],
                                  [name for name, _ in eligible])
    tickets = []
    for name, stake in allocations.items():
        selection = results[name]
        if name in ("one_x_two_pair", "handicap_pair"):
            tickets.append({"component": name, "matches": selection["matches"],
                            "stake_yuan": stake})
        elif name == "total_goals_duplex":
            match = selection["match"]
            choices = match["selection"]
            split = _allocate_units(stake // 2,
                                    [choice["probability"] for choice in choices],
                                    [choice["selection"] for choice in choices])
            tickets.extend({"component": name, "match": match["fixture_id"],
                            "selection": choice["selection"],
                            "stake_yuan": split[choice["selection"]]} for choice in choices)
        else:
            combos = selection["four_combinations"]
            keys = [f"{row['score_1']}|{row['score_2']}" for row in combos]
            split = _allocate_units(stake // 2,
                                    [row["joint_probability"] for row in combos], keys)
            tickets.extend({"component": name, "matches": [
                row["fixture_id"] for row in selection["matches"]],
                "scores": [combo["score_1"], combo["score_2"]],
                "stake_yuan": split[key]} for combo, key in zip(combos, keys))
    return {"status": "AVAILABLE", "name": "400_STANDARD_PROBABILITY_PLAN",
            "total_yuan": 400, "allocations_yuan": allocations,
            "tickets": tickets,
            "component_success_probabilities": dict(eligible),
            "allocation_method": "PROPORTIONAL_SUCCESS_PROBABILITY_LARGEST_REMAINDER_2_YUAN",
            "unavailable_component_redistribution": "RENORMALIZE_ACROSS_AVAILABLE_COMPONENTS",
            "htft_component": "UNAVAILABLE", "market_ev_validated": False,
            "odds_used": False}


def _free_plan(ones: list[dict[str, Any]], handicaps: list[dict[str, Any]],
               totals: list[dict[str, Any]]) -> dict[str, Any]:
    if not ones or not handicaps or not totals:
        return {"status": "UNAVAILABLE", "reason": "REQUIRED_PROBABILITY_CATEGORY_MISSING"}
    candidates = [(a, b, c) for a, b, c in product(ones, handicaps, totals)
                  if len({a["fixture_id"], b["fixture_id"], c["fixture_id"]}) == 3]
    if not candidates:
        return {"status": "UNAVAILABLE", "reason": "FEWER_THAN_THREE_DISTINCT_MATCHES"}
    selected = min(candidates, key=lambda rows: (-sum(row["probability"] for row in rows),
                                                tuple(row["fixture_id"] for row in rows)))
    ranked = sorted(zip(("one_x_two", "handicap_one_x_two", "total_goals"), selected),
                    key=lambda item: (-item[1]["probability"], item[1]["fixture_id"], item[0]))
    allocations = {name: amount for (name, _), amount in zip(ranked, (50, 30, 20))}
    selections = []
    for name, entry in ranked:
        stake = allocations[name]
        if name == "total_goals":
            split = _allocate_units(stake // 2,
                                    [item["probability"] for item in entry["selection"]],
                                    [item["selection"] for item in entry["selection"]])
            ticket_stakes = [{"selection": item["selection"],
                              "stake_yuan": split[item["selection"]]}
                             for item in entry["selection"]]
        else:
            ticket_stakes = [{"selection": entry["selection"]["selection"],
                              "stake_yuan": stake}]
        selections.append({"module": name, "match": entry,
                           "stake_yuan": stake, "tickets": ticket_stakes})
    return {"status": "AVAILABLE", "name": "100_PROBABILITY_ONLY_PLAN",
            "total_yuan": 100, "selections": selections,
            "selection_method": "MAX_SUM_COVERAGE_ONE_PER_CATEGORY_DISTINCT_FIXTURES",
            "stake_method": "50_30_20_DESCENDING_COVERAGE",
            "probability_margin_status": "RECORDED_NOT_TEMPORAL_STABILITY",
            "temporal_stability_status": "UNAVAILABLE_NO_MULTI_RUN_EVIDENCE",
            "model_consistency_status": "UNAVAILABLE_SINGLE_FROZEN_MODEL",
            "risk_diversification": "THREE_DISTINCT_FIXTURES_AND_CATEGORIES",
            "ev_status": "UNAVAILABLE", "market_ev_validated": False,
            "odds_used": False}


def select(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply fixed mathematical rules; never change or create model probabilities."""
    if len({record["fixture_id"] for record in records}) != len(records):
        raise ValueError("R3_DUPLICATE_FIXTURE")
    ones = [item for record in records if (item := _entry(record, "one_x_two", False))]
    handicaps = [item for record in records if (item := _entry(record, "handicap_one_x_two", False))]
    totals = [item for record in records if (item := _entry(record, "total_goals", True))]
    scores = [item for record in records if (item := _entry(record, "exact_score", True))]
    results = {"one_x_two_pair": _pair(ones), "handicap_pair": _pair(handicaps),
               "total_goals_duplex": _best_single(totals),
               "score_pair_duplex": _score_pair(scores),
               "half_full_time_duplex": {"status": "UNAVAILABLE",
                   "reason": "INDEPENDENT_HTFT_MODEL_NOT_FROZEN"},
               "high_odds_20": {"status": "UNAVAILABLE",
                   "reason": "MARKET_VERIFIED_FALSE"}}
    results["standard_plan_400"] = _standard_plan(results)
    results["free_plan_100"] = _free_plan(ones, handicaps, totals)
    return results


def run_locked(store: R3Store, prediction_ids: list[str]) -> tuple[Path, dict[str, Any]]:
    """Read verified locks, append a content-addressed derived result once."""
    if len(set(prediction_ids)) != len(prediction_ids) or not prediction_ids:
        raise ValueError("R3_DUPLICATE_OR_EMPTY_PREDICTION_IDS")
    rules = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    if rules["engine_version"] != ENGINE_VERSION or rules["source_sha256"] != sha256(
            Path(__file__).read_bytes()):
        raise ValueError("R3_SELECTION_ENGINE_FROZEN_CODE_MISMATCH")
    records = []
    lock_hashes = {}
    for prediction_id in sorted(prediction_ids):
        if not store.verify_lock(prediction_id):
            raise ValueError(f"R3_LOCK_INVALID:{prediction_id}")
        path = store.root / "predictions" / f"{prediction_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["prediction_id"] != prediction_id or record["mode"] != "BLIND_TEST_R3":
            raise ValueError("R3_PREDICTION_ID_OR_MODE_INVALID")
        store.verify_snapshot(record["model_snapshot_id"])
        records.append(record)
        lock_hashes[prediction_id] = sha256((store.root / "locks" /
                                            f"{prediction_id}.json").read_bytes())
    if len({row["model_snapshot_id"] for row in records}) != 1 or len({
            row["prediction_date"] for row in records}) != 1:
        raise ValueError("R3_MIXED_SNAPSHOT_OR_DATE")
    if any(row.get("market_verified") is not False for row in records):
        raise ValueError("R3_MARKET_STATUS_CONFLICT")
    identity = {"engine_version": ENGINE_VERSION, "engine_sha256": rules["source_sha256"],
                "rules_sha256": sha256(RULE_PATH.read_bytes()),
                "prediction_ids": sorted(prediction_ids), "lock_sha256": lock_hashes}
    selection_id = "R3SEL-" + sha256(canonical_bytes(identity))[:32]
    path = store.root / "selections" / f"{selection_id}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["identity"] != identity or existing["selection_id"] != selection_id:
            raise ValueError("R3_SELECTION_EXISTING_RECORD_CONFLICT")
        if existing["results"] != select(records):
            raise ValueError("R3_SELECTION_RECOMPUTE_MISMATCH")
        return path, existing
    result = {"mode": "BLIND_TEST_R3", "event": "DERIVED_SELECTION",
              "selection_id": selection_id, "prediction_date": records[0]["prediction_date"],
              "model_snapshot_id": records[0]["model_snapshot_id"],
              "created_at": datetime.now(UTC).isoformat(), "identity": identity,
              "results": select(records)}
    write_once(path, canonical_bytes(result))
    return path, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_ids", nargs="+")
    parser.add_argument("--root", type=Path, default=R3_ROOT)
    args = parser.parse_args(argv)
    path, result = run_locked(R3Store(args.root), args.prediction_ids)
    print(json.dumps({"path": str(path), **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
