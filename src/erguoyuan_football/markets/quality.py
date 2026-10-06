"""Market data quality diagnostics and ENHANCED_ONLY source eligibility gate."""

from __future__ import annotations

from typing import Literal

from erguoyuan_football.contracts.common import Contract
from erguoyuan_football.markets.config import MarketConfig
from erguoyuan_football.markets.dispersion import mean_dispersion
from erguoyuan_football.markets.identity import EventBinding
from erguoyuan_football.markets.schemas import (
    MarketConsensus,
    MarketQualityReport,
    MarketQualityStatus,
    MarketSnapshot,
    MarketType,
    OddsQuote,
    QuoteQuality,
    TimestampQuality,
)
from erguoyuan_football.network.schemas import NetworkHealthReport, SourceHealthStatus


def build_quality_report(snapshot: MarketSnapshot, consensuses: tuple[MarketConsensus, ...], *,
                         raw_quotes: tuple[OddsQuote, ...], config: MarketConfig,
                         devig_policy_validated: bool, fit_error: float | None = None) -> MarketQualityReport:
    """Report observable evidence; do not synthesize an unvalidated quality score."""
    relevant = [quote for quote in raw_quotes if quote.match_id == snapshot.match_id]
    included = list(snapshot.quotes)
    quote_by_id = {quote.quote_id: quote for quote in included}
    used_quote_ids = {quote_id for item in consensuses for quote_id in item.quote_ids}
    used_quotes = [quote_by_id[item] for item in used_quote_ids if item in quote_by_id]
    sources = {quote.provider_id for quote in used_quotes}
    bookmakers = {quote.bookmaker_id for quote in used_quotes}
    freshness = snapshot.freshness_seconds
    overround_values = [value.overround_mean for value in consensuses]
    dispersions = [mean_dispersion({selection.removesuffix(".mad"): {"mad": value}
                                    for selection, value in item.dispersion.items()
                                    if selection.endswith(".mad")}) for item in consensuses]
    dispersion = float(sum(value for value in dispersions if value is not None) /
                       len([value for value in dispersions if value is not None])) if any(
                           value is not None for value in dispersions) else None
    time_qualities = {quote.timestamp_quality for quote in used_quotes}
    if not included:
        time_quality = TimestampQuality.UNKNOWN
    elif time_qualities == {TimestampQuality.SOURCE_NATIVE}:
        time_quality = TimestampQuality.SOURCE_NATIVE
    elif time_qualities <= {TimestampQuality.SOURCE_NATIVE, TimestampQuality.PROVIDER_NATIVE}:
        time_quality = TimestampQuality.PROVIDER_NATIVE
    elif TimestampQuality.RETRIEVAL_ONLY in time_qualities:
        time_quality = TimestampQuality.RETRIEVAL_ONLY
    else:
        time_quality = TimestampQuality.UNKNOWN
    markets = {item.market_type for item in consensuses}
    reasons: list[str] = []
    if not consensuses:
        status = MarketQualityStatus.UNAVAILABLE
        reasons.append("NO_COMPLETE_MARKET_CONSENSUS")
    elif not devig_policy_validated:
        status = MarketQualityStatus.POOR
        reasons.append("DEVIG_POLICY_NOT_VALIDATED_ON_OOS_DATA")
    elif freshness is None or freshness > config.maximum_market_freshness_seconds:
        status = MarketQualityStatus.POOR
        reasons.append("MARKET_STALE")
    elif len(bookmakers) < config.minimum_bookmakers:
        status = MarketQualityStatus.POOR
        reasons.append("INSUFFICIENT_BOOKMAKERS")
    elif time_quality in {TimestampQuality.UNKNOWN, TimestampQuality.RETRIEVAL_ONLY}:
        status = MarketQualityStatus.POOR
        reasons.append("SOURCE_TIME_NOT_HISTORICALLY_VERIFIABLE")
    elif MarketType.MATCH_1X2 not in markets:
        status = MarketQualityStatus.ACCEPTABLE
        reasons.append("1X2_CONSENSUS_UNAVAILABLE")
    else:
        status = MarketQualityStatus.GOOD
    return MarketQualityReport(
        match_id=snapshot.match_id, prediction_time=snapshot.prediction_time,
        source_count=len(sources), bookmaker_count=len(bookmakers), freshness_seconds=freshness,
        timestamp_quality=time_quality, overround=(sum(overround_values) / len(overround_values)
                                                   if overround_values else None),
        dispersion={"mean_mad": dispersion} if dispersion is not None else {},
        suspended_quote_count=sum(item.is_suspended for item in relevant),
        invalid_quote_count=sum(item.quality_status == QuoteQuality.INVALID_QUOTE for item in relevant),
        fit_error=fit_error, markets_available=tuple(sorted(markets, key=lambda item: item.value)),
        quality_score=None, status=status, reason_codes=tuple(reasons),
    )


class MarketSourceGateDecision(Contract):
    status: Literal["PASS", "PREDICTION_BLOCKED"]
    production_eligible: bool
    reason_codes: tuple[str, ...]
    required_source_ids: tuple[str, ...]
    required_markets: tuple[MarketType, ...]


class MarketSourceGate:
    """Require a healthy authorized market source and required fresh consensus markets."""

    def __init__(self, config: MarketConfig) -> None:
        self.config = config

    def evaluate(self, network_report: NetworkHealthReport, snapshot: MarketSnapshot,
                 consensuses: tuple[MarketConsensus, ...], quality: MarketQualityReport, *,
                 required_source_ids: tuple[str, ...], event_binding: EventBinding | None = None,
                 required_markets: tuple[MarketType, ...] = (
                     MarketType.MATCH_1X2,
                 )) -> MarketSourceGateDecision:
        reasons: list[str] = []
        if event_binding is None:
            reasons.append("MARKET_EVENT_BINDING_MISSING")
        elif (event_binding.match_id != snapshot.match_id or
              event_binding.provider_id not in required_source_ids or
              event_binding.event_identity.kickoff_time != snapshot.kickoff_time or
              event_binding.checked_at > snapshot.prediction_time or
              any(quote.provider_id != event_binding.provider_id or
                  quote.source_event_id != event_binding.source_event_id or
                  quote.quote_id not in event_binding.quote_ids for quote in snapshot.quotes)):
            reasons.append("MARKET_EVENT_BINDING_SNAPSHOT_MISMATCH")
        health = {item.source_id: item for item in network_report.sources}
        if (quality.match_id, quality.prediction_time) != (snapshot.match_id, snapshot.prediction_time):
            reasons.append("QUALITY_REPORT_SNAPSHOT_IDENTITY_MISMATCH")
        if any((item.match_id, item.market_snapshot_id, item.prediction_time) != (
                snapshot.match_id, snapshot.market_snapshot_id, snapshot.prediction_time)
               for item in consensuses):
            reasons.append("CONSENSUS_SNAPSHOT_IDENTITY_MISMATCH")
        if network_report.status != "PASS":
            reasons.append("NETWORK_PREFLIGHT_NOT_PASS")
        if not required_source_ids:
            reasons.append("NO_REQUIRED_INTERNATIONAL_MARKET_SOURCE_CONFIGURED")
        if any(source_id not in health or health[source_id].status != SourceHealthStatus.HEALTHY
               for source_id in required_source_ids):
            reasons.append("REQUIRED_MARKET_SOURCE_UNHEALTHY")
        source_ids = {quote.provider_id for quote in snapshot.quotes if not quote.is_live
                      and not quote.is_suspended and quote.quality_status == QuoteQuality.GOOD}
        if not source_ids.intersection(required_source_ids):
            reasons.append("NO_AUTHORIZED_MARKET_QUOTES")
        available = {item.market_type for item in consensuses
                     if item.quality_status in self.config.allowed_quality_statuses}
        missing = set(required_markets) - available
        if missing:
            reasons.append("REQUIRED_MARKETS_UNAVAILABLE:" + ",".join(sorted(item.value for item in missing)))
        if quality.status not in {MarketQualityStatus(item) for item in self.config.allowed_quality_statuses}:
            reasons.append("MARKET_QUALITY_NOT_ALLOWED")
        if snapshot.quality_status not in {MarketQualityStatus(item) for item in self.config.allowed_quality_statuses}:
            reasons.append("SNAPSHOT_QUALITY_NOT_ALLOWED")
        return MarketSourceGateDecision(status="PREDICTION_BLOCKED" if reasons else "PASS",
            production_eligible=not reasons, reason_codes=tuple(reasons),
            required_source_ids=required_source_ids, required_markets=required_markets)
