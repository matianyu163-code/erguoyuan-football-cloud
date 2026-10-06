"""Provider/bookmaker/selection normalization before any market calculation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from erguoyuan_football.contracts.common import utc
from erguoyuan_football.markets.schemas import (
    Bookmaker,
    MarketType,
    OddsQuote,
    QuoteQuality,
    Selection,
    SourceType,
    TimestampQuality,
)


class BookmakerRegistry:
    """Resolve provider-specific bookmaker labels to configured canonical IDs."""

    def __init__(self, bookmakers: tuple[Bookmaker, ...] = ()) -> None:
        self._by_id = {item.bookmaker_id: item for item in bookmakers}
        if len(self._by_id) != len(bookmakers):
            raise ValueError("DUPLICATE_BOOKMAKER_ID")

    def resolve(self, provider_id: str, label: str) -> Bookmaker:
        """Require an explicit canonical name, alias, or provider mapping."""
        normalized = _normalize_name(label)
        found = []
        for bookmaker in self._by_id.values():
            provider_value = bookmaker.provider_mapping.get(provider_id)
            names = {bookmaker.canonical_name, *bookmaker.aliases}
            if bookmaker.active and ((provider_value is not None and _normalize_name(provider_value) == normalized)
                                     or any(_normalize_name(name) == normalized for name in names)):
                found.append(bookmaker)
        unique = {item.bookmaker_id: item for item in found}
        if len(unique) != 1:
            reason = "UNKNOWN_BOOKMAKER" if not unique else "AMBIGUOUS_BOOKMAKER"
            raise ValueError(reason)
        return next(iter(unique.values()))


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


def canonical_market_type(value: str) -> MarketType:
    """Translate documented provider market names; do not infer unknown types."""
    aliases = {
        "1X2": MarketType.MATCH_1X2, "MATCH_ODDS": MarketType.MATCH_1X2,
        "MATCH_1X2": MarketType.MATCH_1X2, "ASIAN_HANDICAP": MarketType.ASIAN_HANDICAP,
        "AH": MarketType.ASIAN_HANDICAP, "TOTALS": MarketType.TOTALS,
        "OVER_UNDER": MarketType.TOTALS, "SPORTTERY_1X2": MarketType.SPORTTERY_1X2,
        "SPORTS_LOTTERY": MarketType.SPORTTERY_1X2,
        "SPORTTERY_HANDICAP_1X2": MarketType.SPORTTERY_HANDICAP_1X2,
    }
    try:
        return aliases[value.strip().upper()]
    except KeyError as error:
        raise ValueError(f"UNSUPPORTED_MARKET_TYPE:{value}") from error


def canonical_selection(market_type: MarketType, value: str) -> Selection:
    """Map common literal codes into the single internal selection vocabulary."""
    text = value.strip().upper()
    one_x_two = {"HOME": Selection.HOME, "H": Selection.HOME, "1": Selection.HOME,
                 "DRAW": Selection.DRAW, "D": Selection.DRAW, "X": Selection.DRAW,
                 "AWAY": Selection.AWAY, "A": Selection.AWAY, "2": Selection.AWAY}
    totals = {"OVER": Selection.OVER, "O": Selection.OVER, "UNDER": Selection.UNDER, "U": Selection.UNDER}
    choices = totals if market_type == MarketType.TOTALS else one_x_two if market_type in {
        MarketType.MATCH_1X2, MarketType.SPORTTERY_1X2, MarketType.SPORTTERY_HANDICAP_1X2
    } else {"HOME": Selection.HOME, "H": Selection.HOME, "1": Selection.HOME,
            "AWAY": Selection.AWAY, "A": Selection.AWAY, "2": Selection.AWAY}
    try:
        return choices[text]
    except KeyError as error:
        raise ValueError(f"UNSUPPORTED_SELECTION:{value}") from error


def decimal_odds(value: Any, odds_format: str) -> Decimal:
    """Convert explicitly declared common price formats to decimal odds."""
    kind = odds_format.strip().upper()
    if kind == "DECIMAL":
        result = Decimal(str(value))
    elif kind == "AMERICAN":
        american = Decimal(str(value))
        if american == 0:
            raise ValueError("INVALID_AMERICAN_ODDS")
        result = Decimal(1) + (american / 100 if american > 0 else Decimal(100) / abs(american))
    elif kind == "FRACTIONAL":
        text = str(value).strip()
        try:
            numerator, denominator = text.split("/", maxsplit=1)
            result = Decimal(1) + Decimal(numerator) / Decimal(denominator)
        except (ValueError, InvalidOperation, ZeroDivisionError) as error:
            raise ValueError("INVALID_FRACTIONAL_ODDS") from error
    elif kind == "HONG_KONG":
        result = Decimal(1) + Decimal(str(value))
    elif kind == "MALAY":
        price = Decimal(str(value))
        if 0 < price <= 1:
            result = Decimal(1) + price
        elif -1 <= price < 0:
            result = Decimal(1) + Decimal(1) / abs(price)
        else:
            raise ValueError("INVALID_MALAY_ODDS")
    elif kind == "INDONESIAN":
        price = Decimal(str(value))
        if price >= 1:
            result = Decimal(1) + price
        elif price <= -1:
            result = Decimal(1) + Decimal(1) / abs(price)
        else:
            raise ValueError("INVALID_INDONESIAN_ODDS")
    else:
        raise ValueError(f"UNKNOWN_ODDS_FORMAT:{odds_format}")
    if not result.is_finite() or result <= 1:
        raise ValueError("INVALID_DECIMAL_ODDS")
    return result


def quarter_units(value: Decimal | str | float) -> int:
    """Convert a quarter-compatible line to an exact integer identity."""
    line = Decimal(str(value))
    quarters = line * Decimal(4)
    if not quarters.is_finite() or quarters != quarters.to_integral_value():
        raise ValueError("LINE_NOT_QUARTER_COMPATIBLE")
    return int(quarters)


class QuoteNormalizer:
    """Normalize a single provider record while retaining invalid-quote provenance."""

    def __init__(self, provider_id: str, source_type: SourceType,
                 bookmakers: BookmakerRegistry, *, schema_version: str) -> None:
        self.provider_id = provider_id
        self.source_type = source_type
        self.bookmakers = bookmakers
        self.schema_version = schema_version

    def normalize(self, payload: dict[str, Any], *, retrieved_at: datetime) -> OddsQuote:
        """Canonicalize team viewpoint, line, price, time and business identities."""
        market = canonical_market_type(str(payload["market_type"]))
        selection = canonical_selection(market, str(payload["selection"]))
        bookmaker = self.bookmakers.resolve(self.provider_id, str(payload["bookmaker"]))
        line = None
        if market in {MarketType.ASIAN_HANDICAP, MarketType.TOTALS, MarketType.SPORTTERY_HANDICAP_1X2}:
            line = quarter_units(payload["line"])
            if market in {MarketType.ASIAN_HANDICAP, MarketType.SPORTTERY_HANDICAP_1X2} and str(
                payload.get("line_perspective", "HOME")
            ).upper() == "AWAY":
                line = -line
            elif str(payload.get("line_perspective", "HOME")).upper() != "HOME":
                raise ValueError("UNKNOWN_LINE_PERSPECTIVE")
        timestamp_quality = TimestampQuality.UNKNOWN
        source_time = _parse_time(payload.get("source_time"))
        as_of_time = None
        if source_time is not None:
            declared = str(payload.get("timestamp_quality", "SOURCE_NATIVE")).upper()
            timestamp_quality = TimestampQuality(declared)
            as_of_time = source_time
        elif payload.get("timestamp_quality") == TimestampQuality.RETRIEVAL_ONLY.value:
            timestamp_quality = TimestampQuality.RETRIEVAL_ONLY
        status = QuoteQuality.GOOD
        reason = None
        price: Decimal | None
        try:
            price = decimal_odds(payload["odds"], str(payload.get("odds_format", "DECIMAL")))
        except (ValueError, InvalidOperation, KeyError) as error:
            price = None
            status = QuoteQuality.INVALID_QUOTE
            reason = str(error)
        is_live = bool(payload.get("is_live", False))
        is_suspended = bool(payload.get("is_suspended", False))
        if is_live and status != QuoteQuality.INVALID_QUOTE:
            status, reason = QuoteQuality.LIVE_EXCLUDED, "IN_PLAY_PRICE"
        elif is_suspended and status != QuoteQuality.INVALID_QUOTE:
            status, reason = QuoteQuality.SUSPENDED, "BOOKMAKER_MARKET_SUSPENDED"
        elif as_of_time is None and status != QuoteQuality.INVALID_QUOTE:
            status, reason = QuoteQuality.TIMESTAMP_UNAVAILABLE, "SOURCE_QUOTE_TIME_UNKNOWN"
        canonical: dict[str, object] = {key: str(value) for key, value in payload.items()}
        canonical.update({"provider_id": self.provider_id, "bookmaker_id": bookmaker.bookmaker_id,
                          "market_type": market.value, "selection": selection.value,
                          "line_quarters": line, "odds_decimal": str(price) if price is not None else None})
        content_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        source_type = SourceType(str(payload.get("source_type", self.source_type.value)))
        return OddsQuote(
            quote_id=hashlib.sha256(f"{self.provider_id}:{content_hash}".encode()).hexdigest()[:32],
            match_id=str(payload["match_id"]), provider_id=self.provider_id,
            bookmaker_id=bookmaker.bookmaker_id, market_type=market, selection=selection,
            line_quarters=line, odds_decimal=price, original_odds=str(payload.get("odds")),
            original_format=str(payload.get("odds_format", "DECIMAL")).upper(), source_type=source_type,
            source_event_id=payload.get("source_event_id"), source_quote_id=payload.get("source_quote_id"),
            source_time=source_time, retrieved_at=utc(retrieved_at), as_of_time=as_of_time,
            timestamp_quality=timestamp_quality, is_live=is_live, is_suspended=is_suspended,
            is_opening_confirmed=bool(payload.get("is_opening_confirmed", False)),
            schema_version=self.schema_version, content_hash=content_hash, quality_status=status,
            quality_reason=reason, screenshot_image_hash=payload.get("screenshot_image_hash"),
            ocr_confidence=payload.get("ocr_confidence"),
        )


def deduplicate_quotes(quotes: tuple[OddsQuote, ...]) -> tuple[OddsQuote, ...]:
    """Keep one canonical exact duplicate and preserve deterministic provider priority."""
    selected: dict[tuple[Any, ...], OddsQuote] = {}
    for quote in sorted(quotes, key=lambda item: (item.provider_id, item.quote_id)):
        key = (quote.match_id, quote.bookmaker_id, quote.market_type, quote.selection,
               quote.line_quarters, quote.as_of_time, quote.odds_decimal, quote.is_live, quote.is_suspended)
        selected.setdefault(key, quote)
    return tuple(sorted(selected.values(), key=lambda item: (
        item.as_of_time or item.retrieved_at, item.bookmaker_id, item.market_type.value,
        item.line_quarters if item.line_quarters is not None else -10_000, item.selection.value,
    )))


def swap_home_away_quote(quote: OddsQuote, *, new_quote_id: str | None = None) -> OddsQuote:
    """Apply a fixture team swap, reversing both AH line and selection direction."""
    if quote.market_type == MarketType.ASIAN_HANDICAP:
        selection = Selection.AWAY if quote.selection == Selection.HOME else Selection.HOME
        line = -quote.line_quarters if quote.line_quarters is not None else None
    elif quote.market_type in {MarketType.MATCH_1X2, MarketType.SPORTTERY_1X2}:
        selection = {Selection.HOME: Selection.AWAY, Selection.AWAY: Selection.HOME,
                     Selection.DRAW: Selection.DRAW}[quote.selection]
        line = quote.line_quarters
    elif quote.market_type == MarketType.SPORTTERY_HANDICAP_1X2:
        selection = {Selection.HOME: Selection.AWAY, Selection.AWAY: Selection.HOME,
                     Selection.DRAW: Selection.DRAW}[quote.selection]
        line = -quote.line_quarters if quote.line_quarters is not None else None
    else:
        selection, line = quote.selection, quote.line_quarters
    return quote.model_copy(update={"selection": selection, "line_quarters": line,
                                    "quote_id": new_quote_id or f"swap:{quote.quote_id}"})


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return utc(value)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError("INVALID_SOURCE_TIMESTAMP") from error
    return utc(parsed)
