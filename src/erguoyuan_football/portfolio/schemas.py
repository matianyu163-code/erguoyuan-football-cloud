"""Account, ledger and settlement contracts."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import Contract
from erguoyuan_football.recommendation.schemas import BetAdvice
from erguoyuan_football.selection.schemas import RankedCandidate


class AccountEntry(Contract):
    slot: str
    candidate: RankedCandidate | None
    advice: BetAdvice | None
    reference_allocation: float = Field(ge=0, allow_inf_nan=False)
    recommended_stake: float = Field(ge=0, allow_inf_nan=False)
    status: str
    reason: str | None = None

    @model_validator(mode="after")
    def consistent(self) -> AccountEntry:
        if self.candidate is None and (self.advice is not None or self.recommended_stake > 0):
            raise ValueError("PORTFOLIO_CANNOT_FUND_MISSING_CANDIDATE")
        if self.advice is not None and (self.candidate is None or
                self.advice.candidate_id != self.candidate.candidate_id or
                self.recommended_stake != self.advice.recommended_stake):
            raise ValueError("PORTFOLIO_CANNOT_CHANGE_SELECTION_OR_ADVICE")
        return self


class ExperimentalAccount(Contract):
    account_id: Literal["HIGH_HIT_400", "VALUE_100", "LONGSHOT_20"]
    budget_cap: float
    status: str
    entries: tuple[AccountEntry, ...]
    reference_total: float
    recommended_total: float

    @model_validator(mode="after")
    def capped(self) -> ExperimentalAccount:
        caps = {"HIGH_HIT_400": 400, "VALUE_100": 100, "LONGSHOT_20": 20}
        if (self.budget_cap != caps[self.account_id] or
            self.reference_total > self.budget_cap + 1e-9 or
            self.recommended_total > self.budget_cap + 1e-9 or
            abs(self.reference_total - sum(e.reference_allocation for e in self.entries)) > 1e-8 or
            abs(self.recommended_total - sum(e.recommended_stake for e in self.entries)) > 1e-8):
            raise ValueError("PORTFOLIO_BUDGET_CAP_OR_TOTAL_INVALID")
        return self


class CandidateLedger(Contract):
    candidates: tuple[RankedCandidate, ...]


class BetAdviceLedger(Contract):
    advices: tuple[BetAdvice, ...]


class PortfolioLedger(Contract):
    accounts: tuple[ExperimentalAccount, ...]


class SettlementEntry(Contract):
    candidate_id: str
    settled_at: date
    actual_quote_id: str | None
    actual_stake: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    return_amount: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    outcome: Literal["WIN", "LOSS", "VOID", "UNAVAILABLE"]
    counterfactual: bool = False


class SettlementLedger(Contract):
    entries: tuple[SettlementEntry, ...] = ()
