"""Fail-closed promotion review; development training never grants live status."""

from __future__ import annotations

from typing import Any

from erguoyuan_football.meta.contracts import MetaPromotionDecision


class MetaPromotionGate:
    """Evaluate documented evidence without running or tuning a model."""

    def evaluate(self, report: dict[str, Any], *, artifact_verified: bool) -> MetaPromotionDecision:
        """Keep Phase 9 unpromoted while final holdout and live/market evidence are absent."""
        return MetaPromotionDecision(
            engineering_ready=report.get("engineering_status") == "PASS",
            artifact_ready=artifact_verified,
            development_validation_ready=report.get("calibration_status") == "DEVELOPMENT_ONLY",
            final_holdout_ready=False,
            superiority_evidence="NOT_ESTABLISHED" if
                report.get("calibrated_minus_dixon_coles_log_loss", 0) > 0 else "INSUFFICIENT_EVIDENCE",
            live_temporal_ready=False,
            market_ready=False,
            production_promoted=False,
            reason="FINAL_HOLDOUT_UNAVAILABLE; LIVE_TEMPORAL_AND_MARKET_VALIDATION_PENDING",
        )
