"""Convert complete source-backed Phase 6 1X2 quotes to priced evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, cast

from erguoyuan_football.markets.devig import calculate_devig
from erguoyuan_football.markets.schemas import (
    DeVigMethod,
    MarketType,
    OddsQuote,
    QuoteQuality,
    Selection,
    SourceType,
    TimestampQuality,
)
from erguoyuan_football.recommendation.schemas import MarketPriceEvidence
from erguoyuan_football.selection.schemas import RankedCandidate


class MarketValueEngine:
    """Require complete, same-bookmaker, timestamped purchasable 1X2 prices."""

    def evidence(self, candidate: RankedCandidate, quotes: tuple[OddsQuote, ...], *,
                 prediction_time: datetime, kickoff_times: tuple[datetime, ...],
                 devig_method: DeVigMethod = DeVigMethod.MULTIPLICATIVE,
                 synthetic_test_only: bool = False) -> MarketPriceEvidence:
        if candidate.play_type not in {"MATCH_1X2", "MATCH_1X2_PAIR", "OFFICIAL_HANDICAP_1X2",
                                       "OFFICIAL_HANDICAP_PAIR"}:
            raise ValueError("COMPOSITE_MARKET_PAYOFF_NOT_DEFINED")
        if len(candidate.match_ids) != len(kickoff_times):
            raise ValueError("MARKET_KICKOFF_COUNT_MISMATCH")
        selected: list[OddsQuote] = []
        market_probs: list[float] = []
        for index, (match_id, direction, kickoff) in enumerate(zip(candidate.match_ids,
                candidate.selections, kickoff_times, strict=True)):
            if direction not in {"HOME", "DRAW", "AWAY"}:
                raise ValueError("MARKET_SELECTION_UNSUPPORTED")
            market_type = (MarketType.SPORTTERY_HANDICAP_1X2 if "HANDICAP" in candidate.play_type
                           else MarketType.SPORTTERY_1X2)
            relevant = [q for q in quotes if q.match_id == match_id and q.market_type == market_type]
            if len(relevant) != 3 or {q.selection for q in relevant} != {
                    Selection.HOME, Selection.DRAW, Selection.AWAY}:
                raise ValueError("COMPLETE_MARKET_NOT_AVAILABLE")
            first = relevant[0]
            if (any(q.bookmaker_id != first.bookmaker_id or q.provider_id != first.provider_id or
                    q.line_quarters != first.line_quarters or q.as_of_time != first.as_of_time or
                    q.source_type != first.source_type for q in relevant) or
                first.source_type not in {SourceType.SPORTTERY_OFFICIAL_PUBLIC, SourceType.AUTHORIZED_PROVIDER,
                                          SourceType.SYNTHETIC_TEST} or
                (first.source_type == SourceType.SYNTHETIC_TEST) != synthetic_test_only or
                any(q.quality_status != QuoteQuality.GOOD or q.is_live or q.is_suspended or
                    q.timestamp_quality not in {TimestampQuality.SOURCE_NATIVE,
                                                TimestampQuality.PROVIDER_NATIVE} or
                    q.as_of_time is None or q.retrieved_at > prediction_time or
                    q.as_of_time > prediction_time or prediction_time >= kickoff
                    for q in relevant)):
                raise ValueError("MARKET_QUOTE_PIT_OR_SOURCE_INVALID")
            if market_type == MarketType.SPORTTERY_HANDICAP_1X2 and first.line_quarters is None:
                raise ValueError("OFFICIAL_HANDICAP_LINE_MISSING")
            if market_type == MarketType.SPORTTERY_HANDICAP_1X2 and (
                    first.line_quarters != candidate.official_handicap_lines[index] * 4):
                raise ValueError("OFFICIAL_HANDICAP_LINE_MISMATCH")
            odds = {q.selection.value: float(q.odds_decimal) for q in relevant if q.odds_decimal is not None}
            devig = calculate_devig(f"{match_id}:{market_type.value}:{first.bookmaker_id}", odds, devig_method)
            chosen = next(q for q in relevant if q.selection.value == direction)
            assert chosen.odds_decimal is not None and chosen.as_of_time is not None
            selected.append(chosen)
            market_probs.append(devig.devig_probabilities[direction])
        if not selected:
            raise ValueError("NO_MARKET_LEGS")
        if (len({q.provider_id for q in selected}) != 1 or
                len({q.bookmaker_id for q in selected}) != 1):
            raise ValueError("PARLAY_NOT_PURCHASABLE_ACROSS_BOOKMAKERS")
        combined_odds = 1.0
        combined_probability = 1.0
        for quote, probability in zip(selected, market_probs, strict=True):
            assert quote.odds_decimal is not None
            combined_odds *= float(quote.odds_decimal)
            combined_probability *= probability
        as_of_times = [q.as_of_time for q in selected if q.as_of_time is not None]
        if len(as_of_times) != len(selected):
            raise ValueError("MARKET_SOURCE_TIME_MISSING")
        return MarketPriceEvidence(candidate_id=candidate.candidate_id,
            decimal_odds=combined_odds, market_probability=combined_probability,
            quote_ids=tuple(q.quote_id for q in selected),
            source=cast(Literal["SPORTTERY_OFFICIAL_PUBLIC", "AUTHORIZED_PROVIDER", "SYNTHETIC_TEST"],
                        selected[0].source_type.value),
            provider_id=selected[0].provider_id, bookmaker_id=selected[0].bookmaker_id,
            market_type=candidate.play_type, as_of_time=min(as_of_times),
            retrieved_at=max(q.retrieved_at for q in selected), prediction_time=prediction_time,
            kickoff_times=kickoff_times, quality_status="GOOD", devig_method=devig_method.value,
            devig_evidence_id="PHASE6_DEVIG:" + ":".join(q.quote_id for q in selected),
            purchasable=True, synthetic_test_only=synthetic_test_only)
