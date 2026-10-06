"""Independent, fail-closed formal Production promotion gate."""

from __future__ import annotations

from typing import Any

REQUIRED_EVIDENCE = (
    "approved_release_manifest", "golden_oos", "calibration",
    "long_term_blind_test", "model_stability", "data_quality",
    "risk_controls", "market_validation", "runtime_isolation",
    "result_evaluation_lifecycle", "owner_approval",
)


def evaluate_production_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    """Never infer approval from a blind-test publication or model registry."""
    missing = [name for name in REQUIRED_EVIDENCE if evidence.get(name) is not True]
    return {"production_eligible": not missing, "missing_evidence": missing,
            "r3_blind_test_is_production_approval": False}
