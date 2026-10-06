"""ENHANCED_ONLY network gate used by later production prediction pipelines."""

from __future__ import annotations

from typing import Literal

from erguoyuan_football.contracts.common import Contract
from erguoyuan_football.network.schemas import NetworkHealthReport


class ProductionNetworkGateDecision(Contract):
    """Production permission is distinct from an offline development result."""

    status: Literal["PASS", "PREDICTION_BLOCKED", "DEVELOPMENT_ONLY"]
    network_policy: Literal["ENHANCED_ONLY"] = "ENHANCED_ONLY"
    reason: str
    production_eligible: bool


class ProductionNetworkGate:
    """Block production continuation unless international required sources pass."""

    def evaluate(self, report: NetworkHealthReport, *, offline_test_mode: bool = False) -> ProductionNetworkGateDecision:
        if report.status == "PASS":
            return ProductionNetworkGateDecision(status="PASS", reason="REQUIRED_EXTERNAL_SOURCES_HEALTHY",
                                                 production_eligible=True)
        if offline_test_mode:
            return ProductionNetworkGateDecision(status="DEVELOPMENT_ONLY", reason="OFFLINE_TEST_MODE_EXPLICIT",
                                                 production_eligible=False)
        return ProductionNetworkGateDecision(status="PREDICTION_BLOCKED",
            reason=report.reason or "NETWORK_PREFLIGHT_NOT_PASS", production_eligible=False)
