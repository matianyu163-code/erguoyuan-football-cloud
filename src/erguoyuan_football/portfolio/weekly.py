"""Weekly experimental metrics require actual priced, settled bets."""

from __future__ import annotations

from erguoyuan_football.portfolio.schemas import (
    BetAdviceLedger,
    CandidateLedger,
    SettlementLedger,
)


def weekly_metrics(candidates: CandidateLedger, advices: BetAdviceLedger,
                   settlements: SettlementLedger) -> dict[str, object]:
    """Return null economic statistics until real quotes and settlements exist."""
    priced = [item for item in settlements.entries if item.actual_quote_id is not None and
              item.actual_stake is not None and item.return_amount is not None and not item.counterfactual]
    base: dict[str, object] = {
        "candidate_count": len(candidates.candidates),
        "bet_count": sum(item.recommended_stake > 0 for item in advices.advices),
        "no_bet_count": sum(item.recommendation == "NO_BET" for item in advices.advices),
        "unavailable_advice_count": sum(item.advice_status == "UNAVAILABLE" for item in advices.advices),
        "settlement_count": len(priced),
        "counterfactual_count": sum(item.counterfactual for item in settlements.entries),
    }
    if not priced:
        return {**base, "status": "UNAVAILABLE_NO_REAL_PRICED_SETTLEMENT",
            "actual_stake": None, "return": None, "net_result": None,
            "roi": None, "hit_rate": None, "average_model_probability": None,
            "average_market_probability": None, "average_ev": None,
            "closing_line": None, "max_drawdown": None}
    stake = sum(item.actual_stake or 0 for item in priced)
    returns = sum(item.return_amount or 0 for item in priced)
    by_id = {item.candidate_id: item for item in candidates.candidates}
    joined = [by_id[item.candidate_id] for item in priced if item.candidate_id in by_id]
    def mean_available(values: list[float | None]) -> float | None:
        available = [value for value in values if value is not None]
        return sum(available) / len(available) if len(available) == len(values) and values else None
    balance = peak = drawdown = 0.0
    for item in sorted(priced, key=lambda row: (row.settled_at, row.candidate_id)):
        balance += (item.return_amount or 0) - (item.actual_stake or 0)
        peak = max(peak, balance)
        drawdown = max(drawdown, peak - balance)
    return {**base, "status": "AVAILABLE", "actual_stake": stake,
        "return": returns, "net_result": returns - stake,
        "roi": (returns - stake) / stake if stake else None,
        "hit_rate": sum(item.outcome == "WIN" for item in priced) / len(priced),
        "average_model_probability": mean_available([item.joint_probability for item in joined])
            if len(joined) == len(priced) else None,
        "average_market_probability": mean_available([item.market_probability for item in joined])
            if len(joined) == len(priced) else None,
        "average_ev": mean_available([item.expected_value for item in joined])
            if len(joined) == len(priced) else None,
        "closing_line": None, "max_drawdown": drawdown}
