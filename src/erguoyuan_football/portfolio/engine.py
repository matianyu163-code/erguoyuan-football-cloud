"""Fixed-slot, capped allocation without changing candidates."""

from __future__ import annotations

from typing import Literal

from erguoyuan_football.portfolio.schemas import AccountEntry, ExperimentalAccount
from erguoyuan_football.recommendation.engine import RecommendationEngine
from erguoyuan_football.recommendation.schemas import MarketPriceEvidence
from erguoyuan_football.selection.engine import ACCOUNT_400_ORDER
from erguoyuan_football.selection.schemas import CandidateSlot


class PortfolioEngine:
    """Allocate a fixed reference cap; actual stakes require valid advice."""

    def __init__(self, recommendation: RecommendationEngine) -> None:
        self.recommendation = recommendation

    def high_hit_400(self, slots: tuple[CandidateSlot, ...],
                     prices: dict[str, MarketPriceEvidence] | None = None) -> ExperimentalAccount:
        by_type = {slot.play_type: slot for slot in slots}
        available = [by_type[key] for key in ACCOUNT_400_ORDER if by_type[key].candidate is not None]
        # Transparent fixed reference ceiling per available slot; no claim of optimal allocation.
        allocation = 400 / len(available) if available else 0
        entries = []
        for key in ACCOUNT_400_ORDER:
            slot = by_type[key]
            candidate = slot.candidate
            advice = (self.recommendation.advise(candidate, (prices or {}).get(candidate.candidate_id),
                reference_allocation=allocation) if candidate is not None else None)
            entries.append(AccountEntry(slot=key, candidate=candidate, advice=advice,
                reference_allocation=allocation if candidate else 0,
                recommended_stake=advice.recommended_stake if advice else 0,
                status=slot.status, reason=slot.reason))
        return ExperimentalAccount(account_id="HIGH_HIT_400", budget_cap=400,
            status="DEVELOPMENT_ONLY", entries=tuple(entries),
            reference_total=sum(e.reference_allocation for e in entries),
            recommended_total=sum(e.recommended_stake for e in entries))

    def market_blocked(self, account_id: Literal["VALUE_100", "LONGSHOT_20"]) -> ExperimentalAccount:
        cap = {"VALUE_100": 100, "LONGSHOT_20": 20}[account_id]
        return ExperimentalAccount(account_id=account_id, budget_cap=cap,
            status="UNAVAILABLE_MARKET_DATA", entries=(), reference_total=0,
            recommended_total=0)

    def priced_account(self, account_id: Literal["VALUE_100", "LONGSHOT_20"],
                       candidates, prices) -> ExperimentalAccount:
        """Fund at most two pre-ranked priced candidates without changing selection."""
        cap = {"VALUE_100": 100, "LONGSHOT_20": 20}[account_id]
        chosen = tuple(candidates[:2])
        allocation = cap / len(chosen) if chosen else 0
        entries = []
        for candidate in chosen:
            advice = self.recommendation.advise(candidate, prices.get(candidate.candidate_id),
                                                reference_allocation=allocation)
            entries.append(AccountEntry(slot=f"RANK_{candidate.rank}", candidate=candidate,
                advice=advice, reference_allocation=allocation,
                recommended_stake=advice.recommended_stake, status=advice.advice_status))
        return ExperimentalAccount(account_id=account_id, budget_cap=cap,
            status="DEVELOPMENT_ONLY" if entries else "NO_QUALIFIED_VALUE_CANDIDATES",
            entries=tuple(entries), reference_total=sum(e.reference_allocation for e in entries),
            recommended_total=sum(e.recommended_stake for e in entries))
