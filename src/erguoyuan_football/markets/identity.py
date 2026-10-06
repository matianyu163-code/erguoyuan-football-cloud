"""Bind a provider event to one catalog fixture before market prices are stored."""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum

from pydantic import model_validator

from erguoyuan_football.contracts.common import Contract, Identifier, UTCTime, utc
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.store import Store
from erguoyuan_football.input.match_resolver import MatchResolver
from erguoyuan_football.input.schemas import InputType, MatchRequest, ResolutionStatus
from erguoyuan_football.markets.normalizer import swap_home_away_quote
from erguoyuan_football.markets.schemas import OddsQuote


class EventOrientation(StrEnum):
    NORMAL = "NORMAL"
    SWAPPED = "SWAPPED"


class ProviderEventIdentity(Contract):
    """Names and schedule asserted by the provider, independent of our match_id."""

    source_event_id: Identifier
    competition_name: Identifier
    home_team_name: Identifier
    away_team_name: Identifier
    kickoff_time: UTCTime

    @model_validator(mode="after")
    def distinct_teams(self) -> ProviderEventIdentity:
        if self.home_team_name.casefold() == self.away_team_name.casefold():
            raise ValueError("PROVIDER_EVENT_SAME_TEAM")
        return self


class EventBinding(Contract):
    binding_id: Identifier
    match_id: Identifier
    provider_id: Identifier
    source_event_id: Identifier
    orientation: EventOrientation
    event_identity: ProviderEventIdentity
    checked_at: UTCTime
    quote_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def consistent_identity(self) -> EventBinding:
        if self.event_identity.source_event_id != self.source_event_id:
            raise ValueError("MARKET_BINDING_EVENT_ID_MISMATCH")
        if not self.quote_ids or len(set(self.quote_ids)) != len(self.quote_ids):
            raise ValueError("MARKET_BINDING_QUOTE_IDS_INVALID")
        return self


class MarketEventBinder:
    """Use catalog aliases and a unique kickoff/competition fixture, never fuzzy names."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.resolver = MatchResolver(store)

    def bind(self, fixture: Fixture, event: ProviderEventIdentity,
             quotes: tuple[OddsQuote, ...], *, prediction_time: datetime) -> tuple[EventBinding, tuple[OddsQuote, ...]]:
        try:
            catalog_fixture = self.store.fixture_at(fixture.match_id, prediction_time)
        except (KeyError, ValueError) as error:
            raise ValueError("MARKET_FIXTURE_NOT_FOUND_AT_PREDICTION_TIME") from error
        if catalog_fixture != fixture:
            raise ValueError("MARKET_FIXTURE_NOT_CURRENT_CATALOG_VERSION")
        if not quotes or any(quote.match_id != fixture.match_id or
                             quote.source_event_id != event.source_event_id or
                             quote.provider_id != quotes[0].provider_id for quote in quotes):
            raise ValueError("MARKET_EVENT_QUOTE_IDENTITY_MISMATCH")
        if event.kickoff_time != fixture.kickoff_time:
            raise ValueError("MARKET_EVENT_KICKOFF_MISMATCH")
        direct = self._resolve(event, prediction_time)
        reversed_event = event.model_copy(update={
            "home_team_name": event.away_team_name, "away_team_name": event.home_team_name,
        })
        reverse = self._resolve(reversed_event, prediction_time)
        direct_match = direct.resolution_status == ResolutionStatus.RESOLVED and direct.match_id == fixture.match_id
        reverse_match = reverse.resolution_status == ResolutionStatus.RESOLVED and reverse.match_id == fixture.match_id
        if (direct.resolution_status == ResolutionStatus.AMBIGUOUS or
                reverse.resolution_status == ResolutionStatus.AMBIGUOUS):
            raise ValueError("MARKET_EVENT_AMBIGUOUS_OR_UNRESOLVED")
        if direct.resolution_status == ResolutionStatus.RESOLVED and reverse.resolution_status == ResolutionStatus.RESOLVED:
            raise ValueError("MARKET_EVENT_AMBIGUOUS_OR_UNRESOLVED")
        if direct_match == reverse_match:
            raise ValueError("MARKET_EVENT_AMBIGUOUS_OR_UNRESOLVED")
        if direct_match:
            orientation = EventOrientation.NORMAL
            bound_quotes = quotes
        else:
            orientation = EventOrientation.SWAPPED
            bound_quotes = tuple(swap_home_away_quote(quote) for quote in quotes)
        quote_ids = tuple(quote.quote_id for quote in bound_quotes)
        provider_id = quotes[0].provider_id
        checked_at = utc(prediction_time)
        fingerprint = "|".join((provider_id, event.source_event_id, fixture.match_id,
                                orientation.value, checked_at.isoformat(), *quote_ids))
        binding = EventBinding(binding_id=hashlib.sha256(fingerprint.encode()).hexdigest(),
            match_id=fixture.match_id, provider_id=provider_id,
            source_event_id=event.source_event_id, orientation=orientation,
            event_identity=event, checked_at=checked_at, quote_ids=quote_ids)
        return binding, bound_quotes

    def _resolve(self, event: ProviderEventIdentity, prediction_time: datetime) -> MatchRequest:
        request = MatchRequest(input_type=InputType.MANUAL, source="MARKET_PROVIDER_EVENT",
            competition_name=event.competition_name, home_team_name=event.home_team_name,
            away_team_name=event.away_team_name, kickoff_time=event.kickoff_time)
        return self.resolver.resolve(request, prediction_time=prediction_time)
