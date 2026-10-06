"""Market pipeline composition: freeze -> per-bookmaker de-vig -> quality -> goals."""

from __future__ import annotations

from datetime import datetime

from erguoyuan_football.contracts.common import utc
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.markets.config import MarketConfig
from erguoyuan_football.markets.consensus import MarketConsensusEngine
from erguoyuan_football.markets.devig import DeVigPolicy
from erguoyuan_football.markets.identity import EventBinding
from erguoyuan_football.markets.implied_goals import MarketImpliedGoalEngine
from erguoyuan_football.markets.movement import MarketMovementEngine
from erguoyuan_football.markets.quality import MarketSourceGate, build_quality_report
from erguoyuan_football.markets.schemas import (
    MarketEngineResult,
    MarketQualityStatus,
    MarketType,
    OddsQuote,
    QuoteQuality,
)
from erguoyuan_football.markets.snapshot import (
    build_market_snapshot,
    current_quotes_at,
    resolve_horizon,
)
from erguoyuan_football.network.schemas import NetworkHealthReport


class MarketEngine:
    """Pure market transformation; it never fetches data or creates placeholder odds."""

    def __init__(self, config: MarketConfig, devig_policy: DeVigPolicy) -> None:
        self.config = config
        self.devig_policy = devig_policy
        self.consensus_engine = MarketConsensusEngine(config, devig_policy,
            bookmaker_weights=config.bookmaker_weights,
            weight_evidence_hash=config.weight_evidence_hash)
        self.goal_engine = MarketImpliedGoalEngine(max_goals=config.implied_goals_max_score,
                                                   maximum_fit_error=config.maximum_goal_fit_error)
        self.movement_engine = MarketMovementEngine()
        self.source_gate = MarketSourceGate(config)

    def run(self, fixture: Fixture, quotes: tuple[OddsQuote, ...], *, prediction_time: datetime,
            opening_quotes: tuple[OddsQuote, ...] = (), production: bool = False,
            network_report: NetworkHealthReport | None = None,
            event_binding: EventBinding | None = None,
            required_source_ids: tuple[str, ...] = (),
            required_markets: tuple[MarketType, ...] = ()) -> MarketEngineResult:
        """Build one deterministic market snapshot, with production eligibility separate."""
        at = utc(prediction_time)
        if max(utc(fixture.as_of_time), utc(fixture.retrieved_at)) > at:
            raise ValueError("MARKET_FIXTURE_POINT_IN_TIME_VIOLATION")
        horizon_seconds = int((utc(fixture.kickoff_time) - at).total_seconds())
        if horizon_seconds <= 0:
            raise ValueError("MARKET_PREDICTION_AFTER_KICKOFF")
        current = current_quotes_at(quotes, match_id=fixture.match_id, prediction_time=at)
        usable_current = any(item.quality_status == QuoteQuality.GOOD and not item.is_live
                             and not item.is_suspended for item in current)
        preliminary_quality = MarketQualityStatus.ACCEPTABLE if usable_current else MarketQualityStatus.UNAVAILABLE
        horizon = resolve_horizon(horizon_seconds, self.config.horizon_tolerance_seconds)
        max_age = self.config.freshness_seconds_by_horizon.get(
            horizon.value, self.config.maximum_market_freshness_seconds)
        snapshot = build_market_snapshot(quotes, match_id=fixture.match_id,
            prediction_time=at, kickoff_time=fixture.kickoff_time,
            max_age_seconds=max_age, quality_status=preliminary_quality,
            horizon_tolerance_seconds=self.config.horizon_tolerance_seconds)
        consensuses = self.consensus_engine.build(snapshot)
        goal_features = self.goal_engine.fit(snapshot, consensuses)
        movements = self.movement_engine.build(opening_quotes, quotes, match_id=fixture.match_id,
            prediction_time=at, policy=self.devig_policy)
        quality = build_quality_report(snapshot, consensuses, raw_quotes=quotes,
            config=self.config, devig_policy_validated=self.devig_policy.status == "VALIDATED",
            fit_error=goal_features.fit_error)
        gate_status: str
        reason: str | None
        if not production:
            gate_status, reason = "DEVELOPMENT_ONLY", "OFFLINE_MARKET_ENGINE_DOES_NOT_GRANT_PRODUCTION_ELIGIBILITY"
        elif network_report is None:
            gate_status, reason = "PREDICTION_BLOCKED", "NETWORK_PREFLIGHT_NOT_PROVIDED"
        else:
            decision = self.source_gate.evaluate(network_report, snapshot, consensuses, quality,
                required_source_ids=required_source_ids,
                event_binding=event_binding,
                required_markets=required_markets or (MarketType.MATCH_1X2,))
            gate_status = decision.status or "PREDICTION_BLOCKED"
            reason = ";".join(decision.reason_codes) or None
        return MarketEngineResult(market_snapshot=snapshot, consensuses=consensuses,
            quality_report=quality, market_goal_features=goal_features, movements=movements,
            production_gate_status=gate_status, reason=reason)
