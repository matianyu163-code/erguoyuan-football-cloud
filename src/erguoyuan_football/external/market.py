"""Authorized market data adapter; every remote response comes through CoreNetworkClient."""

from __future__ import annotations

from typing import Literal, Protocol

from erguoyuan_football.contracts.common import Contract, UTCTime
from erguoyuan_football.markets.identity import ProviderEventIdentity
from erguoyuan_football.markets.normalizer import BookmakerRegistry, QuoteNormalizer
from erguoyuan_football.markets.schemas import OddsQuote, SourceType
from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.errors import (
    NetworkDependencyUnavailable,
    RetryExhausted,
)


class MarketProviderResult(Contract):
    status: Literal["AVAILABLE", "NETWORK_UNAVAILABLE", "AUTH_FAILED", "RATE_LIMITED", "DATA_UNAVAILABLE"]
    source_id: str
    retrieved_at: UTCTime | None = None
    quotes: tuple[OddsQuote, ...] = ()
    event_identity: ProviderEventIdentity | None = None
    reason: str | None = None


class MarketOddsProvider(Protocol):
    """Provider contract returning canonical rows only after match identity is resolved."""

    source_id: str

    def fetch_quotes(self, match_id: str, source_event_id: str, prediction_time) -> MarketProviderResult: ...


class CoreNetworkMarketProvider:
    """Small schema adapter for one explicitly registered source/endpoint."""

    def __init__(self, client: CoreNetworkClient, *, source_id: str, endpoint_id: str,
                 source_type: SourceType, bookmaker_registry: BookmakerRegistry,
                 schema_version: str) -> None:
        self.client = client
        self.source_id = source_id
        self.endpoint_id = endpoint_id
        self.normalizer = QuoteNormalizer(source_id, source_type, bookmaker_registry,
                                         schema_version=schema_version)

    def fetch_quotes(self, match_id: str, source_event_id: str, prediction_time) -> MarketProviderResult:
        """Fetch one resolved event; malformed or mismatched data remains unavailable."""
        try:
            response = self.client.fetch_json(self.source_id, self.endpoint_id,
                params={"event_id": source_event_id}, prediction_time=prediction_time)
        except KeyError as error:
            return MarketProviderResult(status="NETWORK_UNAVAILABLE", source_id=self.source_id,
                                        reason=str(error))
        except ValueError as error:
            message = str(error)
            auth_status: Literal["AUTH_FAILED", "DATA_UNAVAILABLE"] = (
                "AUTH_FAILED" if message.startswith("AUTH_FAILED") else "DATA_UNAVAILABLE"
            )
            return MarketProviderResult(status=auth_status, source_id=self.source_id, reason=message)
        except RetryExhausted as error:
            retry_status: Literal["RATE_LIMITED", "NETWORK_UNAVAILABLE"] = (
                "RATE_LIMITED" if error.code == "RATE_LIMITED" or str(error).startswith("RATE_LIMITED")
                else "NETWORK_UNAVAILABLE"
            )
            return MarketProviderResult(status=retry_status, source_id=self.source_id, reason=str(error))
        except (NetworkDependencyUnavailable, TimeoutError, ConnectionError, OSError, RuntimeError) as error:
            return MarketProviderResult(status="NETWORK_UNAVAILABLE", source_id=self.source_id,
                                        reason=f"{type(error).__name__}:{error}")
        payload = response.body
        if not isinstance(payload, dict) or not isinstance(payload.get("quotes"), list):
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.source_id,
                                        retrieved_at=response.retrieved_at, reason="INVALID_MARKET_RESPONSE_SCHEMA")
        if not payload["quotes"]:
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.source_id,
                                        retrieved_at=response.retrieved_at, reason="EMPTY_MARKET_QUOTE_SET")
        try:
            event_payload = payload["event"]
            if not isinstance(event_payload, dict):
                raise TypeError("PROVIDER_EVENT_IDENTITY_MISSING")
            event_identity = ProviderEventIdentity.model_validate(event_payload)
            if event_identity.source_event_id != source_event_id:
                raise ValueError("PROVIDER_EVENT_ID_MISMATCH")
        except (KeyError, TypeError, ValueError) as error:
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.source_id,
                                        retrieved_at=response.retrieved_at,
                                        reason=f"PROVIDER_EVENT_IDENTITY_INVALID:{error}")
        normalized = []
        try:
            for row in payload["quotes"]:
                if not isinstance(row, dict) or (row.get("match_id") is not None
                                                and str(row["match_id"]) != match_id):
                    raise ValueError("PROVIDER_MATCH_ID_MISMATCH")
                if str(row.get("source_event_id")) != source_event_id:
                    raise ValueError("PROVIDER_EVENT_ID_MISMATCH")
                # External feeds need not know our catalog ID; the event binder checks
                # provider teams, competition and kickoff before this quote is persisted.
                quote = self.normalizer.normalize({**row, "match_id": match_id},
                                                  retrieved_at=response.retrieved_at)
                if quote.as_of_time is None or quote.as_of_time > response.retrieved_at:
                    raise ValueError("PROVIDER_QUOTE_TIME_UNAVAILABLE_OR_FUTURE")
                if response.as_of_time is not None and quote.as_of_time > response.as_of_time:
                    # An enclosing response timestamp must not be older than its contained quotes.
                    raise ValueError("QUOTE_NEWER_THAN_RESPONSE_TIMESTAMP")
                if quote.match_id != match_id:
                    raise ValueError("PROVIDER_MATCH_ID_MISMATCH")
                normalized.append(quote)
        except (KeyError, TypeError, ValueError) as error:
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.source_id,
                                        retrieved_at=response.retrieved_at, reason=str(error))
        return MarketProviderResult(status="AVAILABLE", source_id=self.source_id,
                                    retrieved_at=response.retrieved_at, quotes=tuple(normalized),
                                    event_identity=event_identity)
