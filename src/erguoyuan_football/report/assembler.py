"""Pure Phase 11 report assembly, separate from databases and source adapters."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from erguoyuan_football.portfolio.engine import PortfolioEngine
from erguoyuan_football.portfolio.schemas import (
    BetAdviceLedger,
    CandidateLedger,
    PortfolioLedger,
    SettlementLedger,
)
from erguoyuan_football.portfolio.weekly import weekly_metrics
from erguoyuan_football.recommendation.engine import (
    RecommendationEngine,
    RecommendationPolicy,
)
from erguoyuan_football.recommendation.schemas import MarketPriceEvidence
from erguoyuan_football.report.schemas import CoreReportV2Schema
from erguoyuan_football.selection.engine import SelectionEngine
from erguoyuan_football.selection.schemas import MatchHeads, RankedCandidate


class CoreReportV2Assembler:
    """Join already selected options, advice and capped accounts without source I/O."""

    def __init__(self, *, required_ev: float = 0.03,
                 allow_synthetic_test: bool = False) -> None:
        self.recommendation = RecommendationEngine(RecommendationPolicy(required_ev),
                                                   allow_synthetic_test=allow_synthetic_test)

    def build(self, *, heads: tuple[MatchHeads, ...], status: dict[str, object],
              highest_hit_prices: dict[str, MarketPriceEvidence] | None = None,
              value_candidates: tuple[RankedCandidate, ...] = (),
              longshot_candidates: tuple[RankedCandidate, ...] = (),
              value_prices: dict[str, MarketPriceEvidence] | None = None,
              ) -> tuple[CoreReportV2Schema, dict[str, float]]:
        if not heads:
            raise ValueError("ALL_MATCH_ROWS_REQUIRED")
        prices = {**(highest_hit_prices or {}), **(value_prices or {})}
        if prices and any(head.temporal_mode != "EXACT_UTC" for head in heads):
            raise ValueError("DATE_SAFE_BATCH_MARKET_PRICE_BLOCKED")
        timings: dict[str, float] = {}
        start = time.perf_counter()
        slots = SelectionEngine().highest_hit(heads)
        timings["selection_ms"] = round((time.perf_counter() - start) * 1000, 3)
        start = time.perf_counter()
        portfolio = PortfolioEngine(self.recommendation)
        account400 = portfolio.high_hit_400(slots, prices)
        account100 = (portfolio.priced_account("VALUE_100", value_candidates, prices)
                      if value_candidates else portfolio.market_blocked("VALUE_100"))
        account20 = (portfolio.priced_account("LONGSHOT_20", longshot_candidates, prices)
                     if longshot_candidates else portfolio.market_blocked("LONGSHOT_20"))
        timings["portfolio_ms"] = round((time.perf_counter() - start) * 1000, 3)
        start = time.perf_counter()
        top_candidates = tuple(slot.candidate for slot in slots if slot.candidate is not None)
        all_candidates = (*top_candidates, *value_candidates, *longshot_candidates)
        account_advices = tuple(entry.advice for account in (account400, account100, account20)
                                for entry in account.entries if entry.advice is not None)
        advice_by_id = {item.candidate_id: item for item in account_advices}
        remaining_advices = tuple(self.recommendation.advise(candidate, prices.get(candidate.candidate_id))
            for candidate in top_candidates if candidate.candidate_id not in advice_by_id)
        advices = (*account_advices, *remaining_advices)
        timings["recommendation_ms"] = round((time.perf_counter() - start) * 1000, 3)
        candidates = CandidateLedger(candidates=all_candidates)
        advice_ledger = BetAdviceLedger(advices=advices)
        portfolios = PortfolioLedger(accounts=(account400, account100, account20))
        settlements = SettlementLedger()
        report = CoreReportV2Schema(created_at=datetime.now(UTC), all_matches=heads,
            highest_hit=slots, high_hit_400=account400, value_100=account100,
            longshot_20=account20, model_data_status=status,
            candidate_ledger=candidates, bet_advice_ledger=advice_ledger,
            portfolio_ledger=portfolios, settlement_ledger=settlements,
            weekly_metrics=weekly_metrics(candidates, advice_ledger, settlements))
        return report, timings
