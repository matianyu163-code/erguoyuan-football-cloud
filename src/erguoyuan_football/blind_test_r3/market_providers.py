"""R3 adapters over the existing canonical market provider and quote interfaces."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Protocol

from erguoyuan_football.external.market import MarketProviderResult
from erguoyuan_football.markets.identity import ProviderEventIdentity
from erguoyuan_football.markets.normalizer import BookmakerRegistry, QuoteNormalizer
from erguoyuan_football.markets.schemas import (
    Bookmaker,
    OddsQuote,
    SourceType,
)


class R3MarketProvider(Protocol):
    provider_name: str
    status: str

    def fetch_1x2(self, fixture_id: str, home_team: str, away_team: str,
                  kickoff_utc: datetime, at: datetime) -> MarketProviderResult: ...


class MarketProviderRegistry:
    """Deterministic registry; one failed provider cannot hide another's quotes."""

    def __init__(self) -> None:
        self._providers: dict[str, R3MarketProvider] = {}

    def register(self, provider: R3MarketProvider) -> None:
        if provider.provider_name in self._providers:
            raise ValueError("R3_DUPLICATE_MARKET_PROVIDER")
        self._providers[provider.provider_name] = provider

    def all(self) -> tuple[R3MarketProvider, ...]:
        return tuple(self._providers[name] for name in sorted(self._providers))

    def get(self, name: str) -> R3MarketProvider:
        return self._providers[name]


class NotConfiguredMarketProvider:
    """Explicit JC/global placeholder until a legal endpoint is configured."""

    status = "NOT_CONFIGURED"

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    def fetch_1x2(self, fixture_id: str, home_team: str, away_team: str,
                  kickoff_utc: datetime, at: datetime) -> MarketProviderResult:
        return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider_name,
                                    reason="NOT_CONFIGURED")


class ManualMarketProvider:
    """Read explicit user-confirmed files; screenshots alone never qualify."""

    provider_name = "MANUAL_MARKET_PROVIDER"
    status = "AVAILABLE_IF_USER_CONFIRMED_INPUT"

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.last_evidence: dict[str, str] | None = None

    def fetch_1x2(self, fixture_id: str, home_team: str, away_team: str,
                  kickoff_utc: datetime, at: datetime) -> MarketProviderResult:
        self.last_evidence = None
        if not self.directory.is_dir():
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider_name,
                                        reason="NO_USER_CONFIRMED_MARKET_FILE")
        eligible: list[tuple[datetime, Path, dict]] = []
        for path in sorted(self.directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fixture_id") != fixture_id:
                continue
            confirmation = payload.get("confirmation", {})
            if confirmation.get("status") != "USER_CONFIRMED" or (
                payload.get("home_team") != home_team or
                payload.get("away_team") != away_team or
                datetime.fromisoformat(payload["kickoff_utc"]) != kickoff_utc
            ):
                raise ValueError("USER_CONFIRMED_MARKET_FIXTURE_IDENTITY_INVALID")
            confirmed_at = datetime.fromisoformat(confirmation["confirmed_at"])
            if confirmed_at.tzinfo is None or confirmed_at > at or at >= kickoff_utc:
                raise ValueError("USER_CONFIRMED_MARKET_POINT_IN_TIME_INVALID")
            evidence_hash = str(confirmation.get("source_artifact_sha256", ""))
            if len(evidence_hash) != 64 or any(ch not in "0123456789abcdef" for ch in evidence_hash.lower()):
                raise ValueError("USER_CONFIRMED_MARKET_EVIDENCE_HASH_REQUIRED")
            eligible.append((confirmed_at, path, payload))
        if not eligible:
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider_name,
                                        reason="NO_USER_CONFIRMED_MARKET_FOR_FIXTURE")
        confirmed_at, path, payload = max(eligible, key=lambda row: (row[0], str(row[1])))
        confirmation = payload["confirmation"]
        names = sorted({str(row["bookmaker"]) for row in payload["quotes"]})
        books = tuple(Bookmaker(bookmaker_id="USER_" + hashlib.sha256(name.encode()).hexdigest()[:12],
                                canonical_name=name) for name in names)
        normalizer = QuoteNormalizer(self.provider_name, SourceType.USER_CONFIRMED_MARKET,
                                     BookmakerRegistry(books), schema_version="R3_MANUAL_MARKET_V1")
        quotes: list[OddsQuote] = []
        for index, row in enumerate(payload["quotes"]):
            fetched_at = datetime.fromisoformat(row["fetched_at"])
            if fetched_at.tzinfo is None or fetched_at > confirmed_at or fetched_at >= kickoff_utc:
                raise ValueError("USER_CONFIRMED_QUOTE_TIME_INVALID")
            for selection, field in (("HOME", "home_odds"), ("DRAW", "draw_odds"),
                                     ("AWAY", "away_odds")):
                quote = normalizer.normalize({"match_id": fixture_id,
                    "source_event_id": str(confirmation["confirmation_id"]),
                    "source_quote_id": f"{confirmation['confirmation_id']}:{index}:{selection}",
                    "bookmaker": row["bookmaker"], "market_type": "MATCH_1X2",
                    "selection": selection, "odds": row[field],
                    "source_time": fetched_at.isoformat(),
                    "timestamp_quality": "RETRIEVAL_ONLY",
                    "source_artifact_sha256": confirmation["source_artifact_sha256"],
                    "is_opening_confirmed": bool(row.get("is_opening_confirmed", False))},
                    retrieved_at=confirmed_at)
                if quote.odds_decimal is None or quote.as_of_time is None:
                    raise ValueError("USER_CONFIRMED_ODDS_INVALID")
                quotes.append(quote)
        identity = ProviderEventIdentity(source_event_id=str(confirmation["confirmation_id"]),
            competition_name=str(payload["competition"]), home_team_name=home_team,
            away_team_name=away_team, kickoff_time=kickoff_utc)
        self.last_evidence = {"path": str(path.resolve()),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_artifact_sha256": confirmation["source_artifact_sha256"],
            "confirmation_id": str(confirmation["confirmation_id"]),
            "confirmed_at": confirmed_at.isoformat(), "source_quality": "USER_CONFIRMED"}
        return MarketProviderResult(status="AVAILABLE", source_id=self.provider_name,
            retrieved_at=confirmed_at, quotes=tuple(quotes), event_identity=identity)


def default_registry(manual_directory: Path) -> MarketProviderRegistry:
    registry = MarketProviderRegistry()
    registry.register(NotConfiguredMarketProvider("JC_OFFICIAL_PROVIDER"))
    # A key alone is not an endpoint or authorization to guess an Odds API URL.
    _ = os.environ.get("YYCORE_ODDS_API_KEY")
    registry.register(NotConfiguredMarketProvider("GLOBAL_ODDS_PROVIDER"))
    registry.register(ManualMarketProvider(manual_directory))
    return registry
