"""Evidence and advice contracts; missing market is never NO_BET."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import Contract, UTCTime


class MarketPriceEvidence(Contract):
    candidate_id: str
    decimal_odds: float = Field(gt=1, allow_inf_nan=False)
    market_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    quote_ids: tuple[str, ...]
    source: Literal["SPORTTERY_OFFICIAL_PUBLIC", "AUTHORIZED_PROVIDER", "SYNTHETIC_TEST"]
    provider_id: str
    bookmaker_id: str
    market_type: str
    as_of_time: UTCTime
    retrieved_at: UTCTime
    prediction_time: UTCTime
    kickoff_times: tuple[UTCTime, ...]
    quality_status: Literal["GOOD", "ACCEPTABLE"]
    devig_method: Literal["MULTIPLICATIVE", "ADDITIVE", "POWER", "SHIN", "ODDS_RATIO"]
    devig_evidence_id: str
    purchasable: bool
    synthetic_test_only: bool = False

    @model_validator(mode="after")
    def validate_price(self) -> MarketPriceEvidence:
        if (not self.quote_ids or len(self.quote_ids) != len(self.kickoff_times) or
                not self.devig_evidence_id or not self.provider_id or
                not self.bookmaker_id or
                not self.as_of_time <= self.retrieved_at <= self.prediction_time or
                any(not self.prediction_time < kickoff for kickoff in self.kickoff_times) or
                not self.purchasable or self.quality_status not in {"GOOD", "ACCEPTABLE"}):
            raise ValueError("MARKET_PRICE_PIT_OR_QUALITY_INVALID")
        if (self.source == "SYNTHETIC_TEST") != self.synthetic_test_only:
            raise ValueError("SYNTHETIC_MARKET_FLAG_MISMATCH")
        return self


class BetAdvice(Contract):
    candidate_id: str
    advice_status: Literal["AVAILABLE", "UNAVAILABLE", "BLOCKED", "FAILED"]
    recommendation: Literal["STRONG_BET", "BET", "SMALL_BET", "WATCH", "NO_BET"] | None
    recommended_stake: float = Field(ge=0, allow_inf_nan=False)
    reference_allocation: float = Field(ge=0, allow_inf_nan=False)
    reason_codes: tuple[str, ...]
    market_status: str
    minimum_acceptable_price: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    expected_value: float | None = Field(default=None, allow_inf_nan=False)
    risk_level: str
    validation_status: str
    production_status: str

    @model_validator(mode="after")
    def validate_advice(self) -> BetAdvice:
        if (self.advice_status == "AVAILABLE") != (self.recommendation is not None):
            raise ValueError("ADVICE_STATUS_RECOMMENDATION_MISMATCH")
        if self.recommendation == "NO_BET" and self.recommended_stake != 0:
            raise ValueError("NO_BET_STAKE_MUST_BE_ZERO")
        if self.recommended_stake > 0 and (self.advice_status != "AVAILABLE" or
                self.recommendation not in {"STRONG_BET", "BET", "SMALL_BET"}):
            raise ValueError("STAKE_WITHOUT_BUY_ADVICE")
        return self


class LongshotValueScore(Contract):
    status: Literal["NOT_TRAINED_NO_HISTORICAL_MARKET"] = "NOT_TRAINED_NO_HISTORICAL_MARKET"
    value: None = None
