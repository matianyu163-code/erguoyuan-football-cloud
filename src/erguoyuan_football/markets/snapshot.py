"""Point-in-time market selection, horizon labeling and closing-price leakage guards."""

from __future__ import annotations

from datetime import datetime

from erguoyuan_football.contracts.common import Availability, utc
from erguoyuan_football.markets.normalizer import deduplicate_quotes
from erguoyuan_football.markets.schemas import (
    MarketQualityStatus,
    MarketSnapshot,
    OddsQuote,
    PredictionHorizon,
    QuoteQuality,
    QuoteSelectionResult,
)


class FutureClosingOddsLeakageError(ValueError):
    """A post-prediction closing price was found in a forecast feature set."""


def current_quotes_at(quotes: tuple[OddsQuote, ...], *, match_id: str,
                      prediction_time: datetime) -> tuple[OddsQuote, ...]:
    """Select latest source-time quote per provider/bookmaker/market/selection/line."""
    at = utc(prediction_time)
    streams: dict[tuple, OddsQuote] = {}
    for quote in quotes:
        if quote.match_id != match_id or quote.as_of_time is None:
            continue
        if utc(quote.as_of_time) > at or utc(quote.retrieved_at) > at:
            continue
        key = (quote.provider_id, quote.bookmaker_id, quote.market_type, quote.line_quarters, quote.selection)
        previous = streams.get(key)
        if previous is None or (quote.as_of_time, quote.retrieved_at, quote.quote_id) > (
            previous.as_of_time, previous.retrieved_at, previous.quote_id
        ):
            streams[key] = quote
    return deduplicate_quotes(tuple(streams.values()))


def opening_quote(quotes: tuple[OddsQuote, ...], *, match_id: str,
                  bookmaker_id: str, market_type, selection, prediction_time: datetime,
                  line_quarters: int | None = None) -> QuoteSelectionResult:
    """Return only source-confirmed openings; first observed quote is not assumed open."""
    candidates = [quote for quote in quotes if quote.match_id == match_id
                  and quote.bookmaker_id == bookmaker_id and quote.market_type == market_type
                  and quote.selection == selection and quote.line_quarters == line_quarters
                  and quote.is_opening_confirmed and quote.as_of_time is not None
                  and utc(quote.as_of_time) <= utc(prediction_time)
                  and utc(quote.retrieved_at) <= utc(prediction_time)
                  and not quote.is_live and not quote.is_suspended
                  and quote.quality_status == QuoteQuality.GOOD]
    if not candidates:
        return QuoteSelectionResult(availability=Availability.UNAVAILABLE, reason="OPENING_NOT_SOURCE_CONFIRMED")
    quote = min(candidates, key=lambda item: (item.as_of_time, item.retrieved_at, item.quote_id))
    return QuoteSelectionResult(availability=Availability.AVAILABLE, quote=quote, reason="SOURCE_CONFIRMED_OPENING")


def closing_quote(quotes: tuple[OddsQuote, ...], *, match_id: str, kickoff_time: datetime,
                  bookmaker_id: str, market_type, selection, line_quarters: int | None = None) -> QuoteSelectionResult:
    """Select the last valid pre-kickoff quote strictly for post-hoc benchmarking."""
    kickoff = utc(kickoff_time)
    candidates = [quote for quote in quotes if quote.match_id == match_id
                  and quote.bookmaker_id == bookmaker_id and quote.market_type == market_type
                  and quote.selection == selection and quote.line_quarters == line_quarters
                  and quote.as_of_time is not None and utc(quote.as_of_time) < kickoff
                  and not quote.is_live and not quote.is_suspended
                  and quote.quality_status == QuoteQuality.GOOD]
    if not candidates:
        return QuoteSelectionResult(availability=Availability.UNAVAILABLE, reason="NO_ARCHIVED_PRE_KICKOFF_CLOSE")
    quote = max(candidates, key=lambda item: (item.as_of_time, item.retrieved_at, item.quote_id))
    return QuoteSelectionResult(availability=Availability.AVAILABLE, quote=quote,
                                reason="POST_HOC_LAST_VALID_PRE_KICKOFF_QUOTE")


def assert_no_future_closing_quotes(quote_ids: tuple[str, ...], quotes: tuple[OddsQuote, ...], *,
                                    prediction_time: datetime) -> None:
    """Reject a close or any quote not actually available at the historical cutoff."""
    by_id = {quote.quote_id: quote for quote in quotes}
    at = utc(prediction_time)
    for quote_id in quote_ids:
        quote = by_id.get(quote_id)
        if quote is None:
            raise ValueError("FEATURE_QUOTE_LINEAGE_MISSING")
        if quote.as_of_time is None or utc(quote.as_of_time) > at or utc(quote.retrieved_at) > at:
            raise FutureClosingOddsLeakageError("DATA_LEAKAGE_BLOCKED:future closing/current quote in features")


def build_market_snapshot(quotes: tuple[OddsQuote, ...], *, match_id: str,
                          prediction_time: datetime, kickoff_time: datetime,
                          max_age_seconds: int, quality_status: MarketQualityStatus,
                          horizon_tolerance_seconds: dict[str, int] | None = None) -> MarketSnapshot:
    """Freeze all current source-timestamp-safe quotes available at one prediction time."""
    at, kickoff = utc(prediction_time), utc(kickoff_time)
    if at >= kickoff:
        raise ValueError("MARKET_PREDICTION_AFTER_KICKOFF")
    selected = tuple(quote for quote in current_quotes_at(quotes, match_id=match_id, prediction_time=at)
                     if not quote.is_live and quote.quality_status not in {
                         QuoteQuality.INVALID_QUOTE, QuoteQuality.TIMESTAMP_UNAVAILABLE,
                         QuoteQuality.LIVE_EXCLUDED,
                     })
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must be nonnegative")
    fresh = [quote for quote in selected if quote.as_of_time is not None
             and (at - utc(quote.as_of_time)).total_seconds() <= max_age_seconds]
    oldest = min((utc(quote.as_of_time) for quote in fresh if quote.as_of_time is not None), default=None)
    freshness = int((at - oldest).total_seconds()) if oldest else None
    horizon_seconds = int((kickoff - at).total_seconds())
    horizon = resolve_horizon(horizon_seconds, horizon_tolerance_seconds)
    return MarketSnapshot(
        match_id=match_id, prediction_time=at, kickoff_time=kickoff,
        prediction_horizon=horizon, horizon_seconds=horizon_seconds,
        source_count=len({quote.provider_id for quote in selected}),
        bookmaker_count=len({quote.bookmaker_id for quote in selected}), quote_count=len(selected),
        markets_available=tuple(sorted({quote.market_type for quote in fresh}, key=lambda item: item.value)),
        freshness_seconds=freshness, quality_status=quality_status,
        included_quote_ids=tuple(quote.quote_id for quote in selected), quotes=selected,
    )


def resolve_horizon(seconds_to_kickoff: int,
                    tolerances: dict[str, int] | None = None) -> PredictionHorizon:
    """Label the nearest configured reference horizon only inside explicit configured windows."""
    candidates = {
        PredictionHorizon.T_MINUS_24H: 24 * 3600,
        PredictionHorizon.T_MINUS_6H: 6 * 3600,
        PredictionHorizon.T_MINUS_3H: 3 * 3600,
        PredictionHorizon.T_MINUS_60M: 3600,
        PredictionHorizon.T_MINUS_15M: 900,
    }
    eligible = [horizon for horizon, target in candidates.items()
                if abs(target - seconds_to_kickoff) <= (tolerances or {}).get(horizon.value, 0)]
    return min(eligible, key=lambda item: (abs(candidates[item] - seconds_to_kickoff), item.value)) \
        if eligible else PredictionHorizon.CUSTOM


class FutureClosingOddsLeakageGuard:
    """Named guard used by model features and regression tests."""

    def validate(self, feature_quote_ids: tuple[str, ...], quotes: tuple[OddsQuote, ...], *,
                 prediction_time: datetime) -> None:
        assert_no_future_closing_quotes(feature_quote_ids, quotes, prediction_time=prediction_time)
