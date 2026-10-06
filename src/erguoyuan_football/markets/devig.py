"""Fixed-policy bookmaker de-vig methods and chronological validation tools."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.optimize import brentq

from erguoyuan_football.contracts.common import utc
from erguoyuan_football.markets.schemas import (
    DeVigMethod,
    DeVigPolicy,
    DeVigResult,
    MarketType,
)


class DeVigError(ValueError):
    """Invalid or numerically undefined de-vig market."""


def calculate_devig(market_id: str, odds: dict[str, float], method: DeVigMethod) -> DeVigResult:
    """Remove margin from one bookmaker's mutually exclusive price set."""
    if len(odds) < 2 or len(set(odds)) != len(odds):
        raise DeVigError("COMPLETE_MARKET_REQUIRES_AT_LEAST_TWO_UNIQUE_SELECTIONS")
    prices = np.asarray(list(odds.values()), dtype=float)
    if not np.isfinite(prices).all() or np.any(prices <= 1):
        raise DeVigError("INVALID_DECIMAL_ODDS")
    raw = 1.0 / prices
    overround = float(raw.sum())
    if overround <= 0:
        raise DeVigError("INVALID_OVERROUND")
    parameters: dict[str, float] = {}
    if method == DeVigMethod.MULTIPLICATIVE:
        fair = raw / overround
    elif method == DeVigMethod.ADDITIVE:
        fair = raw - (overround - 1.0) / len(raw)
        if np.any(fair < 0):
            raise DeVigError("ADDITIVE_DEVIG_NEGATIVE_PROBABILITY")
        fair /= fair.sum()
    elif method == DeVigMethod.POWER:
        exponent = _root_for_sum(lambda power: float(np.sum(raw**power)) - 1.0, 0.0, 64.0)
        fair = raw**exponent
        parameters["exponent"] = exponent
    elif method == DeVigMethod.SHIN:
        if len(raw) < 3:
            raise DeVigError("SHIN_REQUIRES_THREE_OR_MORE_OUTCOMES")
        inverse_sum = float(raw.sum())
        fair = _shin_probabilities(raw, inverse_sum)
        parameters["insider_fraction"] = _shin_parameter(raw, inverse_sum)
    elif method == DeVigMethod.ODDS_RATIO:
        ratio = _root_for_sum(lambda scale: float(np.sum(raw / (raw + scale * (1.0 - raw)))) - 1.0,
                              1e-10, max(2.0, overround * 100.0))
        fair = raw / (raw + ratio * (1.0 - raw))
        parameters["odds_ratio_scale"] = ratio
    else:  # pragma: no cover - enum exhaustiveness guard
        raise DeVigError(f"UNSUPPORTED_DEVIG_METHOD:{method}")
    fair = np.asarray(fair, dtype=float)
    if not np.isfinite(fair).all() or np.any(fair < 0) or not math.isclose(
        float(fair.sum()), 1.0, abs_tol=1e-8, rel_tol=0
    ):
        raise DeVigError("DEVIG_NUMERICAL_FAILURE")
    mapping = {key: float(value) for key, value in zip(odds, fair, strict=True)}
    return DeVigResult(market_id=market_id, method=method,
                       raw_implied_probabilities={key: float(value) for key, value in zip(odds, raw, strict=True)},
                       devig_probabilities=mapping, overround=overround,
                       method_parameters=parameters, success=True)


def _root_for_sum(function, left: float, right: float) -> float:
    f_left, f_right = function(left), function(right)
    if f_left == 0:
        return left
    if f_left * f_right > 0:
        raise DeVigError("DEVIG_ROOT_NOT_BRACKETED")
    return float(brentq(function, left, right, xtol=1e-12, rtol=1e-12, maxiter=500))


def _shin_probs_at(raw: np.ndarray, total: float, insider_fraction: float) -> np.ndarray:
    z = insider_fraction
    if z >= 1:
        raise DeVigError("SHIN_PARAMETER_OUT_OF_RANGE")
    radicand = z * z + 4.0 * (1.0 - z) * raw * raw / total
    return (np.sqrt(radicand) - z) / (2.0 * (1.0 - z))


def _shin_parameter(raw: np.ndarray, total: float) -> float:
    return _root_for_sum(lambda z: float(_shin_probs_at(raw, total, z).sum()) - 1.0,
                         0.0, 1.0 - 1e-10)


def _shin_probabilities(raw: np.ndarray, total: float) -> np.ndarray:
    return _shin_probs_at(raw, total, _shin_parameter(raw, total))


def overround(odds: dict[str, float]) -> float:
    """Return the sum of raw implied probabilities minus one."""
    prices = np.asarray(list(odds.values()), dtype=float)
    if len(prices) < 2 or not np.isfinite(prices).all() or np.any(prices <= 1):
        raise DeVigError("INVALID_DECIMAL_ODDS")
    return float((1.0 / prices).sum() - 1.0)


@dataclass(frozen=True)
class DeVigValidationCase:
    """One bookmaker market observed before the match with a later known outcome."""

    market_id: str
    market_type: MarketType
    prediction_time: datetime
    kickoff_time: datetime
    odds: dict[str, float]
    outcome: str
    competition_id: str


def devig_validation_report(cases: tuple[DeVigValidationCase, ...], *,
                            validation_start: datetime, final_test_start: datetime) -> dict:
    """Score candidate policies using only chronological validation observations."""
    validation_at, test_at = utc(validation_start), utc(final_test_start)
    if not validation_at < test_at or not cases:
        raise ValueError("NONEMPTY_VALIDATION_BEFORE_FROZEN_TEST_REQUIRED")
    methods = tuple(DeVigMethod)
    rows = []
    for case in cases:
        prediction_at, kickoff = utc(case.prediction_time), utc(case.kickoff_time)
        if not validation_at <= prediction_at < kickoff < test_at:
            raise ValueError("DEVIG_VALIDATION_TIME_LEAKAGE")
        if case.outcome not in case.odds:
            raise ValueError("VALIDATION_OUTCOME_NOT_IN_MARKET")
        row: dict[str, dict] = {}
        for method in methods:
            try:
                result = calculate_devig(case.market_id, case.odds, method)
            except DeVigError as error:
                row[method.value] = {"available": False, "reason": str(error)}
                continue
            probabilities = result.devig_probabilities
            ordered = list(probabilities.values())
            label = list(probabilities).index(case.outcome)
            brier = math.fsum((value - (index == label)) ** 2 for index, value in enumerate(ordered))
            confidence = max(ordered)
            row[method.value] = {"available": True,
                                 "log_loss": -math.log(max(1e-15, probabilities[case.outcome])),
                                 "brier": brier, "confidence": confidence,
                                 "correct": int(max(probabilities, key=lambda key: probabilities[key]) == case.outcome)}
        rows.append((case, row))
    methods_report: dict[str, dict] = {}
    selected_by_market: dict[MarketType, DeVigMethod] = {}
    for market_type in sorted({case.market_type for case in cases}, key=lambda item: item.value):
        selected_cases = [(case, row) for case, row in rows if case.market_type == market_type]
        market_report: dict[str, dict] = {}
        for method in methods:
            samples = [row[method.value] for _, row in selected_cases if row[method.value]["available"]]
            market_report[method.value] = {
                "sample_count": len(samples),
                "excluded_count": len(selected_cases) - len(samples),
                "log_loss": float(np.mean([row["log_loss"] for row in samples])) if samples else None,
                "brier": float(np.mean([row["brier"] for row in samples])) if samples else None,
                "accuracy": float(np.mean([row["correct"] for row in samples])) if samples else None,
                "top_class_calibration_error": _top_class_ece(samples) if samples else None,
            }
        methods_report[market_type.value] = market_report
        comparable = [method for method in methods
                      if market_report[method.value]["sample_count"] == len(selected_cases)]
        if not comparable:
            raise ValueError(f"NO_COMPARABLE_DEVIG_METHODS:{market_type.value}")
        score_order = sorted(comparable, key=lambda method: (
            market_report[method.value]["log_loss"], market_report[method.value]["brier"], method.value
        ))
        selected_by_market[market_type] = score_order[0]
    payload = [{"market_id": case.market_id, "prediction_time": utc(case.prediction_time).isoformat(),
                "outcome": case.outcome, "methods": row} for case, row in rows]
    evidence_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    policy = DeVigPolicy(policy_version=f"DEVIG_VALIDATED_{evidence_hash[:12]}",
                         method_by_market=selected_by_market,
                         status="VALIDATED", validation_evidence_hash=evidence_hash)
    return {"validation_start": validation_at.isoformat(), "final_test_start": test_at.isoformat(),
            "sample_count": len(cases), "methods": methods_report,
            "selected_method_by_market": {key.value: value.value for key, value in policy.method_by_market.items()},
            "policy": policy.model_dump(mode="json"), "evidence_hash": evidence_hash,
            "final_test_used": False}


def _top_class_ece(rows: list[dict]) -> float:
    """Compute a compact 10-bin multiclass top-choice calibration diagnostic."""
    if not rows:
        raise ValueError("empty calibration rows")
    confidence = np.asarray([item["confidence"] for item in rows])
    accuracy = np.asarray([item["correct"] for item in rows])
    error = 0.0
    for lower in np.linspace(0, 0.9, 10):
        upper = lower + 0.1
        members = (confidence >= lower) & ((confidence < upper) | ((upper >= 1) & (confidence <= 1)))
        if members.any():
            error += float(members.mean()) * abs(float(confidence[members].mean() - accuracy[members].mean()))
    return error
