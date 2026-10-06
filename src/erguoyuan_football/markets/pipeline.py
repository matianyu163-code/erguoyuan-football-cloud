"""Network-gated live market acquisition and offline point-in-time composition."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from erguoyuan_football.contracts.common import Contract, now, utc
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.store import Store
from erguoyuan_football.external.market import MarketOddsProvider, MarketProviderResult
from erguoyuan_football.gates.production_network import (
    ProductionNetworkGate,
    ProductionNetworkGateDecision,
)
from erguoyuan_football.markets.engine import MarketEngine
from erguoyuan_football.markets.identity import EventBinding, MarketEventBinder
from erguoyuan_football.markets.schemas import MarketEngineResult, MarketType, OddsQuote
from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.health import run_network_preflight
from erguoyuan_football.network.schemas import NetworkHealthReport


class MarketPipelineResult(Contract):
    """Auditable source and transformation outcome for one requested match."""

    request_time: datetime
    prediction_time: datetime
    network_report: NetworkHealthReport
    network_gate: ProductionNetworkGateDecision
    provider_result: MarketProviderResult | None = None
    event_binding: EventBinding | None = None
    engine_result: MarketEngineResult


class MarketDataPipeline:
    """Run preflight, fetch through CORE networking, then freeze immutable PIT market data."""

    def __init__(self, engine: MarketEngine, network_client: CoreNetworkClient,
                 provider: MarketOddsProvider, *,
                 schema_validators: dict[str, Callable[[Any], bool]],
                 store: Store | None = None, clock: Callable[[], datetime] = now) -> None:
        self.engine = engine
        self.network_client = network_client
        self.provider = provider
        self.schema_validators = schema_validators
        self.store = store
        self.clock = clock

    def execute_live(self, fixture: Fixture, source_event_id: str, *,
                     opening_quotes: tuple[OddsQuote, ...] = (),
                     required_markets: tuple[MarketType, ...] = (MarketType.MATCH_1X2,)) -> MarketPipelineResult:
        """Preflight before fetching; fail closed when no critical authorized source is configured."""
        request_time = utc(self.clock())
        network_report = run_network_preflight(self.network_client.registry, self.network_client,
                                               schema_validators=self.schema_validators)
        network_gate = ProductionNetworkGate().evaluate(network_report)
        required_sources = tuple(item.source_id for item in self.network_client.registry.required())
        provider_result: MarketProviderResult | None = None
        event_binding: EventBinding | None = None
        quotes: tuple[OddsQuote, ...] = ()
        if network_gate.production_eligible and self.provider.source_id in required_sources:
            if self.store is None:
                provider_result = MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider.source_id,
                                                       reason="MATCH_RESOLVER_STORE_UNAVAILABLE")
            else:
                provider_result = self.provider.fetch_quotes(fixture.match_id, source_event_id, request_time)
            if provider_result.status == "AVAILABLE":
                if any(quote.match_id != fixture.match_id or quote.provider_id != provider_result.source_id
                       or quote.source_event_id != source_event_id for quote in provider_result.quotes) or (
                           provider_result.event_identity is None or self.store is None
                       ):
                    provider_result = provider_result.model_copy(update={
                        "status": "DATA_UNAVAILABLE", "quotes": (),
                        "reason": "MARKET_PROVIDER_RESULT_IDENTITY_MISMATCH",
                    })
                else:
                    try:
                        binding_time = max([utc(self.clock()),
                                            *(utc(item.retrieved_at) for item in provider_result.quotes)])
                        if binding_time >= utc(fixture.kickoff_time):
                            raise ValueError("MARKET_EVENT_BINDING_AFTER_KICKOFF")
                        event_binding, quotes = MarketEventBinder(self.store).bind(
                            fixture, provider_result.event_identity, provider_result.quotes,
                            prediction_time=binding_time)
                    except ValueError as error:
                        provider_result = provider_result.model_copy(update={
                            "status": "DATA_UNAVAILABLE", "quotes": (), "reason": str(error),
                        })
        freeze_time = max([utc(self.clock()), *(utc(item.retrieved_at) for item in quotes)])
        if freeze_time >= utc(fixture.kickoff_time):
            raise ValueError("MARKET_FREEZE_NOT_BEFORE_KICKOFF")
        if opening_quotes and quotes:
            self._validate_live_openings(fixture, source_event_id, opening_quotes,
                                         fetched_quotes=quotes, freeze_time=freeze_time)
        engine_result = self.engine.run(fixture, quotes, prediction_time=freeze_time,
            opening_quotes=opening_quotes if quotes else (), production=True, network_report=network_report,
            event_binding=event_binding, required_source_ids=required_sources,
            required_markets=required_markets)
        result = MarketPipelineResult(request_time=request_time, prediction_time=freeze_time,
            network_report=network_report, network_gate=network_gate,
            provider_result=provider_result, event_binding=event_binding, engine_result=engine_result)
        if self.store is not None:
            self._persist(result, quotes)
        return result

    def execute_offline(self, fixture: Fixture, quotes: tuple[OddsQuote, ...], *,
                        prediction_time: datetime,
                        opening_quotes: tuple[OddsQuote, ...] = ()) -> MarketEngineResult:
        """Build development/backtest features from archived quotes without external access."""
        return self.engine.run(fixture, quotes, prediction_time=prediction_time,
                               opening_quotes=opening_quotes, production=False)

    def _validate_live_openings(self, fixture: Fixture, source_event_id: str,
                                opening_quotes: tuple[OddsQuote, ...], *,
                                fetched_quotes: tuple[OddsQuote, ...], freeze_time: datetime) -> None:
        """Accept only already-bound source openings or openings in this verified fetch."""
        if self.store is None:
            raise ValueError("UNVERIFIED_OPENING_QUOTES:NO_STORE")
        fetched = {quote.quote_id: quote for quote in fetched_quotes}
        persisted = {quote.quote_id: quote for quote in self.store.odds_quotes_at(
            fixture.match_id, freeze_time)}
        for opening in opening_quotes:
            if (opening.match_id != fixture.match_id or
                    opening.provider_id != self.provider.source_id or
                    opening.source_event_id != source_event_id or
                    not opening.is_opening_confirmed or opening.as_of_time is None or
                    max(utc(opening.as_of_time), utc(opening.retrieved_at)) > freeze_time):
                raise ValueError("UNVERIFIED_OPENING_QUOTES:IDENTITY_OR_TIME")
            if fetched.get(opening.quote_id) == opening:
                continue
            if persisted.get(opening.quote_id) != opening:
                raise ValueError("UNVERIFIED_OPENING_QUOTES:NOT_PERSISTED")
            bound = self.store.connection.execute(
                "SELECT 1 FROM market_event_bindings WHERE match_id=? AND provider_id=? "
                "AND source_event_id=? LIMIT 1",
                [fixture.match_id, opening.provider_id, source_event_id],
            ).fetchone()
            if bound is None:
                raise ValueError("UNVERIFIED_OPENING_QUOTES:EVENT_NOT_BOUND")

    def _persist(self, result: MarketPipelineResult, fetched_quotes: tuple[OddsQuote, ...]) -> None:
        """Persist each run atomically; exact repeats are idempotent and conflicts fail closed."""
        if self.store is None:
            return
        connection = self.store.connection
        connection.execute("BEGIN TRANSACTION")
        try:
            if result.event_binding is not None:
                self.store.save_market_event_binding(result.event_binding)
            for quote in fetched_quotes:
                existing = connection.execute("SELECT payload FROM odds_quotes WHERE quote_id=?",
                                              [quote.quote_id]).fetchone()
                if existing is None:
                    self.store.add_odds_quote(quote)
                elif existing[0] != quote.model_dump_json():
                    raise ValueError("ODDS_QUOTE_ID_CONTENT_CONFLICT")
            self.store.save_market_snapshot(result.engine_result.market_snapshot)
            for consensus in result.engine_result.consensuses:
                self.store.save_market_consensus(consensus)
            for movement in result.engine_result.movements:
                self.store.save_market_movement(movement)
            self.store.save_market_quality(result.engine_result.quality_report)
            self.store.save_market_implied_goals(result.engine_result.market_goal_features)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
