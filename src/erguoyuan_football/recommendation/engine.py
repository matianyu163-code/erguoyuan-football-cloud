"""Transparent fair-price and EV policy with fail-closed market evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from erguoyuan_football.recommendation.schemas import BetAdvice, MarketPriceEvidence
from erguoyuan_football.selection.schemas import RankedCandidate


@dataclass(frozen=True)
class RecommendationPolicy:
    required_ev: float
    max_quote_age_minutes: int = 120
    status: str = "DEVELOPMENT_POLICY"

    def __post_init__(self) -> None:
        if (not 0 <= self.required_ev < 1 or self.max_quote_age_minutes <= 0 or
                self.status != "DEVELOPMENT_POLICY"):
            raise ValueError("INVALID_DEVELOPMENT_EV_POLICY")


class RecommendationEngine:
    """Compute advice from an existing candidate; never erase or reselect it."""

    def __init__(self, policy: RecommendationPolicy, *, allow_synthetic_test: bool = False) -> None:
        self.policy = policy
        self.allow_synthetic_test = allow_synthetic_test

    def advise(self, candidate: RankedCandidate, price: MarketPriceEvidence | None,
               *, reference_allocation: float = 0) -> BetAdvice:
        common = {"candidate_id": candidate.candidate_id,
            "reference_allocation": reference_allocation,
            "validation_status": candidate.validation_status,
            "production_status": candidate.production_status}
        if price is None:
            return BetAdvice.model_validate({**common, "advice_status": "UNAVAILABLE", "recommendation": None,
                "recommended_stake": 0, "reason_codes": ("PRICE_UNAVAILABLE",),
                "market_status": "UNAVAILABLE", "minimum_acceptable_price": None,
                "expected_value": None, "risk_level": "UNAVAILABLE"})
        if (price.candidate_id != candidate.candidate_id or
            price.market_type != candidate.play_type or
            len(price.kickoff_times) != len(candidate.match_ids) or
            price.prediction_time - price.as_of_time > timedelta(minutes=self.policy.max_quote_age_minutes) or
            (price.synthetic_test_only and not self.allow_synthetic_test)):
            return BetAdvice.model_validate({**common, "advice_status": "BLOCKED", "recommendation": None,
                "recommended_stake": 0, "reason_codes": ("MARKET_EVIDENCE_MISMATCH_OR_TEST_ONLY",),
                "market_status": "BLOCKED", "minimum_acceptable_price": None,
                "expected_value": None, "risk_level": "BLOCKED"})
        probability = candidate.joint_probability
        if probability <= 0:
            return BetAdvice.model_validate({**common, "advice_status": "UNAVAILABLE", "recommendation": None,
                "recommended_stake": 0, "reason_codes": ("ZERO_MODEL_PROBABILITY",),
                "market_status": "AVAILABLE", "minimum_acceptable_price": None,
                "expected_value": None, "risk_level": "UNAVAILABLE"})
        minimum_price = (1 + self.policy.required_ev) / probability
        ev = probability * price.decimal_odds - 1
        recommendation = "BET" if ev >= self.policy.required_ev else "NO_BET"
        return BetAdvice.model_validate({**common, "advice_status": "AVAILABLE", "recommendation": recommendation,
            "recommended_stake": reference_allocation if recommendation == "BET" else 0,
            "reason_codes": ("DEVELOPMENT_EV_POLICY",) if recommendation == "BET" else ("EV_BELOW_POLICY",),
            "market_status": "AVAILABLE", "minimum_acceptable_price": minimum_price,
            "expected_value": ev, "risk_level": "EXPERIMENTAL_NOT_PROMOTED"})
