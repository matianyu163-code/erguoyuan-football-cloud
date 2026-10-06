"""Per-bookmaker de-vig, quality filtering and market consensus aggregation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import timedelta

import numpy as np

from erguoyuan_football.contracts.common import utc
from erguoyuan_football.markets.config import MarketConfig
from erguoyuan_football.markets.devig import calculate_devig
from erguoyuan_football.markets.dispersion import market_dispersion
from erguoyuan_football.markets.schemas import (
    ConsensusMethod,
    DeVigPolicy,
    MarketConsensus,
    MarketSnapshot,
    MarketType,
    OddsQuote,
    QuoteQuality,
    Selection,
    TimestampQuality,
)


class MarketConsensusEngine:
    """Aggregate only after one coherent bookmaker market is normalized and de-vigged."""

    def __init__(self, config: MarketConfig, policy: DeVigPolicy, *,
                 bookmaker_weights: dict[str, float] | None = None,
                 weight_evidence_hash: str | None = None) -> None:
        self.config = config
        self.policy = policy
        self.bookmaker_weights = bookmaker_weights or {}
        self.weight_evidence_hash = weight_evidence_hash
        if config.consensus_method == ConsensusMethod.WEIGHTED_MEAN and (
            not self.bookmaker_weights or not weight_evidence_hash
        ):
            raise ValueError("WEIGHTED_CONSENSUS_REQUIRES_VALIDATED_OOS_WEIGHTS")
        if any(not math.isfinite(value) or value <= 0 for value in self.bookmaker_weights.values()):
            raise ValueError("bookmaker weights must be finite and positive")

    def build(self, snapshot: MarketSnapshot) -> tuple[MarketConsensus, ...]:
        """Create aligned consensus rows; no complete source/bookmaker market means no consensus."""
        grouped: dict[tuple[MarketType, int | None], list[OddsQuote]] = defaultdict(list)
        for quote in snapshot.quotes:
            grouped[(quote.market_type, quote.line_quarters)].append(quote)
        result: list[MarketConsensus] = []
        for (market_type, line), quotes in sorted(grouped.items(), key=lambda item: (
            item[0][0].value, item[0][1] if item[0][1] is not None else -10_000
        )):
            method = self.policy.method_by_market.get(market_type)
            if method is None:
                continue
            by_bookmaker_provider: dict[tuple[str, str], list[OddsQuote]] = defaultdict(list)
            expected = _selections(market_type)
            horizon_age = self.config.freshness_seconds_by_horizon.get(
                snapshot.prediction_horizon.value, self.config.maximum_market_freshness_seconds
            )
            for quote in quotes:
                if quote.quality_status != QuoteQuality.GOOD or quote.is_live or quote.is_suspended:
                    continue
                if quote.as_of_time is None or quote.odds_decimal is None:
                    continue
                if utc(snapshot.prediction_time) - utc(quote.as_of_time) > timedelta(seconds=horizon_age):
                    continue
                if quote.timestamp_quality == TimestampQuality.UNKNOWN:
                    continue
                by_bookmaker_provider[(quote.bookmaker_id, quote.provider_id)].append(quote)
            fair_by_bookmaker: dict[str, dict[str, float]] = {}
            selected_quotes: dict[str, tuple[OddsQuote, ...]] = {}
            overrounds: dict[str, float] = {}
            for (bookmaker_id, provider_id), market_quotes in sorted(by_bookmaker_provider.items()):
                selection_quote = {quote.selection: quote for quote in market_quotes}
                if set(selection_quote) != expected:
                    continue
                ordered_quotes = tuple(selection_quote[key] for key in sorted(expected, key=lambda item: item.value))
                odds = {quote.selection.value: float(quote.odds_decimal) for quote in ordered_quotes
                        if quote.odds_decimal is not None}
                if len(odds) != len(ordered_quotes):
                    continue
                market_id = _market_id(snapshot.match_id, market_type, line, bookmaker_id, provider_id)
                try:
                    devig = calculate_devig(market_id, odds, method)
                except ValueError:
                    continue
                ceiling = self.config.maximum_overround.get(market_type)
                if ceiling is not None and devig.overround - 1.0 > ceiling:
                    continue
                existing = selected_quotes.get(bookmaker_id)
                if existing is not None:
                    old_time = max(utc(item.as_of_time) for item in existing if item.as_of_time is not None)
                    new_time = max(utc(item.as_of_time) for item in ordered_quotes if item.as_of_time is not None)
                    if (new_time, provider_id) <= (old_time, min(item.provider_id for item in existing)):
                        continue
                fair_by_bookmaker[bookmaker_id] = devig.devig_probabilities
                selected_quotes[bookmaker_id] = ordered_quotes
                overrounds[bookmaker_id] = devig.overround
            if len(fair_by_bookmaker) < self.config.minimum_bookmakers:
                continue
            probabilities, ids = self._aggregate(fair_by_bookmaker, selected_quotes)
            dispersion = market_dispersion(tuple(fair_by_bookmaker.values()))
            participating = tuple(quote for bookmaker_quotes in selected_quotes.values() for quote in bookmaker_quotes)
            freshness = max(0, int(max((utc(snapshot.prediction_time) - utc(item.as_of_time)).total_seconds()
                                       for item in participating if item.as_of_time is not None)))
            quality_status = snapshot.quality_status
            result.append(MarketConsensus(
                match_id=snapshot.match_id, market_snapshot_id=snapshot.market_snapshot_id,
                market_type=market_type, line_quarters=line, prediction_time=snapshot.prediction_time,
                probabilities=probabilities, bookmaker_count=len(fair_by_bookmaker),
                provider_count=len({item.provider_id for item in participating}),
                overround_mean=float(np.mean(tuple(overrounds.values()))), dispersion={
                    f"{selection}.{metric}": float(value)
                    for selection, metrics in dispersion.items() for metric, value in metrics.items()
                }, freshness_seconds=freshness, consensus_method=self.config.consensus_method,
                devig_method=method, devig_policy_version=self.policy.policy_version,
                quality_status=quality_status, quote_ids=ids,
            ))
        return tuple(result)

    def _aggregate(self, markets: dict[str, dict[str, float]],
                   selected_quotes: dict[str, tuple[OddsQuote, ...]]) -> tuple[dict[str, float], tuple[str, ...]]:
        bookmaker_ids = sorted(markets)
        selections = sorted(next(iter(markets.values())))
        matrix = np.asarray([[markets[bookmaker][selection] for selection in selections]
                             for bookmaker in bookmaker_ids], dtype=float)
        if self.config.consensus_method == ConsensusMethod.MEDIAN:
            aggregate = np.median(matrix, axis=0)
        elif self.config.consensus_method == ConsensusMethod.TRIMMED_MEAN:
            trim_count = math.floor(len(matrix) * self.config.trim_fraction)
            sorted_matrix = np.sort(matrix, axis=0)
            aggregate = np.mean(sorted_matrix[trim_count:len(matrix) - trim_count or None], axis=0)
        else:
            weights = np.asarray([self.bookmaker_weights.get(bookmaker, 0.0) for bookmaker in bookmaker_ids])
            if np.any(weights <= 0):
                raise ValueError("validated weights missing for a participating bookmaker")
            aggregate = np.average(matrix, axis=0, weights=weights)
        total = float(aggregate.sum())
        if total <= 0 or not math.isfinite(total):
            raise ValueError("INVALID_CONSENSUS_PROBABILITY")
        probabilities = {selection: float(value / total) for selection, value in zip(selections, aggregate, strict=True)}
        quote_ids = tuple(sorted(quote.quote_id for values in selected_quotes.values() for quote in values))
        return probabilities, quote_ids


def _selections(market_type: MarketType) -> set[Selection]:
    if market_type in {MarketType.MATCH_1X2, MarketType.SPORTTERY_1X2}:
        return {Selection.HOME, Selection.DRAW, Selection.AWAY}
    if market_type == MarketType.TOTALS:
        return {Selection.OVER, Selection.UNDER}
    if market_type == MarketType.ASIAN_HANDICAP:
        return {Selection.HOME, Selection.AWAY}
    return {Selection.HOME, Selection.DRAW, Selection.AWAY}


def _market_id(match_id: str, market: MarketType, line: int | None,
               bookmaker: str, provider: str) -> str:
    payload = json.dumps([match_id, market.value, line, bookmaker, provider], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]
