"""Deterministic selection tests over synthetic locked-style model output."""

from __future__ import annotations

from copy import deepcopy

from erguoyuan_football.blind_test_r3.selection import select


def _synthetic_record(number: int, probability: float) -> dict:
    return {"fixture_id": f"SYNTHETIC_TEST_{number}",
            "prediction_id": f"SYNTHETIC_PREDICTION_{number}",
            "home_team": f"Home {number}", "away_team": f"Away {number}",
            "raw_model_output": {},
            "standardized_output": {
                "one_x_two": {"top2": [
                    {"selection": "HOME", "probability": probability},
                    {"selection": "DRAW", "probability": 0.20}]},
                "handicap_one_x_two": {"top2": [
                    {"selection": "AWAY", "probability": probability - 0.05},
                    {"selection": "DRAW", "probability": 0.25}]},
                "total_goals": {"top2": [
                    {"selection": "2", "probability": probability / 2},
                    {"selection": "3", "probability": probability / 3}]},
                "exact_score": {"top2": [
                    {"selection": "1-0", "probability": probability / 4},
                    {"selection": "1-1", "probability": probability / 5}]},
            }}


def test_selection_is_deterministic_and_never_mutates_probabilities() -> None:
    records = [_synthetic_record(1, 0.70), _synthetic_record(2, 0.65),
               _synthetic_record(3, 0.55), _synthetic_record(4, 0.50)]
    original = deepcopy(records)
    result = select(records)
    assert records == original
    assert result == select(list(reversed(records)))
    assert result["one_x_two_pair"]["joint_probability"] == 0.70 * 0.65
    assert result["one_x_two_pair"]["status"] == "AVAILABLE"
    assert result["handicap_pair"]["status"] == "AVAILABLE"
    assert result["total_goals_duplex"]["status"] == "AVAILABLE"
    assert result["score_pair_duplex"]["status"] == "AVAILABLE"
    assert len(result["score_pair_duplex"]["four_combinations"]) == 4
    assert result["half_full_time_duplex"]["status"] == "UNAVAILABLE"
    assert sum(result["standard_plan_400"]["allocations_yuan"].values()) == 400
    assert sum(ticket["stake_yuan"] for ticket in
               result["standard_plan_400"]["tickets"]) == 400
    assert sum(row["stake_yuan"] for row in result["free_plan_100"]["selections"]) == 100
    assert result["high_odds_20"]["status"] == "UNAVAILABLE"


def test_missing_handicap_only_excludes_that_match() -> None:
    records = [_synthetic_record(1, 0.70), _synthetic_record(2, 0.65),
               _synthetic_record(3, 0.55), _synthetic_record(4, 0.50)]
    records[0]["standardized_output"]["handicap_one_x_two"] = {
        "status": "UNAVAILABLE"}
    result = select(records)
    assert result["handicap_pair"]["status"] == "AVAILABLE"
    assert all(item["fixture_id"] != "SYNTHETIC_TEST_1" for item in
               result["handicap_pair"]["matches"])
