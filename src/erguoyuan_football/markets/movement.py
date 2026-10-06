"""Point-in-time opening-to-current price, probability and line movement."""

from __future__ import annotations

import math
from datetime import datetime

from erguoyuan_football.contracts.common import utc
from erguoyuan_football.markets.devig import calculate_devig
from erguoyuan_football.markets.schemas import (
    DeVigPolicy,
    MarketMovement,
    OddsQuote,
    QuoteQuality,
)
from erguoyuan_football.markets.snapshot import current_quotes_at


class MarketMovementEngine:
    """Compute movement from confirmed opening quotes to a frozen current snapshot."""

    def build(self, opening_quotes: tuple[OddsQuote, ...], all_quotes: tuple[OddsQuote, ...], *,
              match_id: str, prediction_time: datetime, policy: DeVigPolicy) -> tuple[MarketMovement, ...]:
        at = utc(prediction_time)
        current = current_quotes_at(all_quotes, match_id=match_id, prediction_time=at)
        opens = tuple(item for item in opening_quotes if item.match_id == match_id
                      and item.is_opening_confirmed and not item.is_live and not item.is_suspended
                      and item.quality_status == QuoteQuality.GOOD and item.as_of_time is not None
                      and utc(item.as_of_time) <= at and utc(item.retrieved_at) <= at)
        opening_streams: dict[tuple, list[OddsQuote]] = {}
        current_streams: dict[tuple, list[OddsQuote]] = {}
        for quote in opens:
            key = (quote.provider_id, quote.bookmaker_id, quote.market_type)
            opening_streams.setdefault(key, []).append(quote)
        for quote in current:
            if quote.quality_status == QuoteQuality.GOOD and not quote.is_suspended:
                key = (quote.provider_id, quote.bookmaker_id, quote.market_type)
                current_streams.setdefault(key, []).append(quote)
        movements: list[MarketMovement] = []
        for key in sorted(set(opening_streams) & set(current_streams), key=str):
            old_quotes, new_quotes = opening_streams[key], current_streams[key]
            market_type = key[2]
            devig_method = policy.method_by_market.get(market_type)
            if devig_method is None:
                continue
            for current_quote in new_quotes:
                prior_candidates = [item for item in old_quotes if item.selection == current_quote.selection]
                if not prior_candidates:
                    continue
                opening = max(prior_candidates, key=lambda item: (item.as_of_time or at, item.retrieved_at))
                current_time, opening_time = current_quote.as_of_time, opening.as_of_time
                current_price, opening_price = current_quote.odds_decimal, opening.odds_decimal
                if current_time is None or opening_time is None or current_price is None or opening_price is None:
                    continue
                old_probs = _fair_prices(old_quotes, opening.line_quarters, devig_method)
                new_probs = _fair_prices(new_quotes, current_quote.line_quarters, devig_method)
                old_probability = old_probs.get(opening.selection.value)
                new_probability = new_probs.get(current_quote.selection.value)
                delta = new_probability - old_probability if old_probability is not None and new_probability is not None else None
                logit_delta = _logit(new_probability) - _logit(old_probability) if (
                    old_probability is not None and new_probability is not None
                ) else None
                elapsed = max(0, int((utc(current_time) - utc(opening_time)).total_seconds()))
                old_line, new_line = opening.line_quarters, current_quote.line_quarters
                movements.append(MarketMovement(
                    match_id=match_id, market_type=market_type, selection=current_quote.selection,
                    line_quarters=new_line, opening_quote_id=opening.quote_id,
                    current_quote_id=current_quote.quote_id, opening_probability=old_probability,
                    current_probability=new_probability, delta_probability=delta,
                    delta_logit_probability=logit_delta, opening_line_quarters=old_line,
                    current_line_quarters=new_line,
                    line_delta_quarters=(new_line - old_line if new_line is not None and old_line is not None else None),
                    opening_odds_decimal=opening_price, current_odds_decimal=current_price,
                    delta_odds=current_price - opening_price,
                    elapsed_seconds=elapsed, as_of_time=utc(current_time),
                ))
        return tuple(movements)


def _fair_prices(quotes: list[OddsQuote], line: int | None, method) -> dict[str, float]:
    selected = [quote for quote in quotes if quote.line_quarters == line and quote.odds_decimal is not None]
    required = {item.selection.value for item in selected}
    if len(selected) < 2 or len(required) != len(selected):
        return {}
    try:
        return calculate_devig("MOVEMENT", {item.selection.value: float(item.odds_decimal)
            for item in selected if item.odds_decimal is not None}, method).devig_probabilities
    except ValueError:
        return {}


def _logit(probability: float) -> float:
    clipped = min(1 - 1e-12, max(1e-12, probability))
    return math.log(clipped / (1 - clipped))
