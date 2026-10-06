"""Conservative event matching and 1X2 adapter for The Odds API V4."""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import UTC, datetime
from typing import Any

from erguoyuan_football.blind_test_r3.the_odds_api_v4 import TheOddsApiV4Client
from erguoyuan_football.external.market import MarketProviderResult
from erguoyuan_football.knowledge.entities.universal_team_resolver import (
    UniversalTeamResolver,
)
from erguoyuan_football.markets.identity import ProviderEventIdentity
from erguoyuan_football.markets.normalizer import BookmakerRegistry, QuoteNormalizer
from erguoyuan_football.markets.schemas import Bookmaker, OddsQuote, SourceType


def _name(value: str) -> str:
    """Conservative fallback: punctuation/spacing only, identity suffixes preserved."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise TypeError("ODDS_API_TIMESTAMP_MISSING")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("ODDS_API_TIMESTAMP_TIMEZONE_MISSING")
    return parsed.astimezone(UTC)


def resolve_sport_key(competition: str, sport_map: dict[str, Any],
                      catalog: list[dict[str, Any]]) -> str | None:
    """Use a verified mapping or confirm an exact candidate in the live catalog."""
    mappings = sport_map.get("mappings", {})
    for name, mapping in mappings.items():
        if _name(name) == _name(competition) and mapping.get("verified") is True and (
            mapping.get("source") == "THE_ODDS_API_SPORTS_ENDPOINT"
        ):
            return str(mapping["sport_key"])
    by_key = {str(row.get("key")): row for row in catalog if isinstance(row, dict)}
    for candidate in sport_map.get("candidates", []):
        names = [candidate["competition"], *candidate.get("aliases", [])]
        if _name(competition) not in {_name(str(name)) for name in names}:
            continue
        sport_key = str(candidate["sport_key"])
        row = by_key.get(sport_key)
        if row and row.get("active") is True and (
            str(row.get("title")) == str(candidate["required_catalog_title"])
        ):
            return sport_key
    return None


def match_fixture(events: list[dict[str, Any]], *, home_team: str,
                  away_team: str, kickoff_utc: datetime,
                  tolerance_minutes: int, normalized_tolerance_minutes: int,
                  resolver: UniversalTeamResolver | None = None) -> tuple[dict[str, Any], str] | None:
    """Require exact orientation, both identities, and one unique kickoff match."""
    if kickoff_utc.tzinfo is None or tolerance_minutes < 0:
        raise ValueError("ODDS_API_FIXTURE_TARGET_INVALID")
    identity = resolver or UniversalTeamResolver()
    target_home = identity.resolve(home_team)
    target_away = identity.resolve(away_team)
    candidates: list[tuple[dict[str, Any], str]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            event_home = str(event["home_team"])
            event_away = str(event["away_team"])
            kickoff = _time(event["commence_time"])
        except (KeyError, TypeError, ValueError):
            continue
        remote_home = identity.resolve(event_home)
        remote_away = identity.resolve(event_away)
        if (target_home.identity is not None and target_away.identity is not None and
            remote_home.identity is not None and remote_away.identity is not None):
            matches = (target_home.identity.team_id == remote_home.identity.team_id and
                       target_away.identity.team_id == remote_away.identity.team_id and
                       target_home.parsed.squad_level_hint == remote_home.parsed.squad_level_hint and
                       target_away.parsed.squad_level_hint == remote_away.parsed.squad_level_hint)
            method = "CANONICAL_IDS"
            tolerance = tolerance_minutes
        else:
            matches = (_name(home_team) == _name(event_home) and
                       _name(away_team) == _name(event_away))
            method = "NORMALIZED_EXACT"
            tolerance = normalized_tolerance_minutes
        if matches and abs((kickoff - kickoff_utc).total_seconds()) <= tolerance * 60:
            candidates.append((event, method))
    if len(candidates) > 1:
        raise ValueError("AMBIGUOUS_FIXTURE_MATCH")
    return candidates[0] if candidates else None


def parse_complete_h2h(event: dict[str, Any], *, fixture_id: str,
                       fetched_at: datetime, raw_payload_sha256: str,
                       maximum_age_seconds: int, maximum_overround: float,
                       minimum_decimal_odds: float) -> tuple[tuple[OddsQuote, ...], dict[str, Any]]:
    """Normalize each complete bookmaker independently; reject stale/bad sets."""
    if fetched_at.tzinfo is None:
        raise ValueError("ODDS_API_FETCH_TIME_INVALID")
    raw_books = event.get("bookmakers")
    books = ([row for row in raw_books if isinstance(row, dict)
              and isinstance(row.get("key"), str) and isinstance(row.get("title"), str)]
             if isinstance(raw_books, list) else [])
    registry = BookmakerRegistry(tuple(Bookmaker(bookmaker_id=str(row["key"]),
        canonical_name=str(row["title"]), provider_mapping={"THE_ODDS_API_V4": str(row["key"])})
        for row in {row["key"]: row for row in books}.values()))
    normalizer = QuoteNormalizer("THE_ODDS_API_V4", SourceType.AUTHORIZED_PROVIDER,
        registry, schema_version="THE_ODDS_API_V4_H2H_V1")
    quotes: list[OddsQuote] = []
    accepted: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1
    for book in books:
        markets = book.get("markets")
        if not isinstance(markets, list):
            reject("NO_COMPLETE_1X2")
            continue
        for market in markets:
            if not isinstance(market, dict) or market.get("key") != "h2h":
                continue
            try:
                updated = _time(market.get("last_update") or book.get("last_update"))
            except (TypeError, ValueError):
                reject("TIMESTAMP_UNAVAILABLE")
                continue
            age = (fetched_at.astimezone(UTC) - updated).total_seconds()
            if age < 0:
                reject("FUTURE_BOOKMAKER_UPDATE")
                continue
            if age > maximum_age_seconds:
                reject("STALE_MARKET")
                continue
            prices: dict[str, float] = {}
            outcomes = market.get("outcomes")
            if not isinstance(outcomes, list):
                reject("NO_COMPLETE_1X2")
                continue
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue
                name = str(outcome.get("name", ""))
                side = ("HOME" if name == event["home_team"] else
                        "AWAY" if name == event["away_team"] else
                        "DRAW" if _name(name) == "draw" else None)
                if side is None or side in prices:
                    continue
                try:
                    price = float(outcome["price"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(price) and price >= minimum_decimal_odds:
                    prices[side] = price
            if set(prices) != {"HOME", "DRAW", "AWAY"}:
                reject("NO_COMPLETE_1X2")
                continue
            overround = sum(1 / value for value in prices.values()) - 1
            if not 0 <= overround <= maximum_overround:
                reject("DATA_QUALITY_WARNING")
                continue
            for side in ("HOME", "DRAW", "AWAY"):
                quote = normalizer.normalize({"match_id": fixture_id,
                    "source_event_id": str(event["id"]),
                    "source_quote_id": f"{event['id']}:{book['key']}:{updated.isoformat()}:{side}",
                    "bookmaker": book["key"], "market_type": "MATCH_1X2",
                    "selection": side, "odds": prices[side],
                    "source_time": updated.isoformat(),
                    "timestamp_quality": "PROVIDER_NATIVE",
                    "source_artifact_sha256": raw_payload_sha256},
                    retrieved_at=fetched_at)
                quotes.append(quote)
            accepted.append({"bookmaker_key": book["key"],
                             "bookmaker": book["title"],
                             "market_last_update": updated.isoformat(),
                             "market_age_seconds": age,
                             "overround": overround,
                             "raw_1x2": prices})
    return tuple(quotes), {"bookmakers": accepted, "rejected": rejected}


class TheOddsApiV4Provider:
    """Return canonical quotes from the cached, precisely matched API event."""

    provider_name = "THE_ODDS_API_V4"
    status = "AVAILABLE_IF_CONFIGURED"

    def __init__(self, client: TheOddsApiV4Client, *, competition: str) -> None:
        self.client = client
        self.competition = competition
        self.last_evidence: dict[str, Any] | None = None
        self.prefetch_status: str | None = None

    def prefetch(self) -> str:
        """Fetch before choosing prediction_time, allowing a true PIT snapshot."""
        if not self.client.configured:
            self.prefetch_status = "NOT_CONFIGURED"
            return self.prefetch_status
        try:
            catalog = self.client.refresh_sport_catalog()
            sport_key = resolve_sport_key(self.competition, self.client.sport_map,
                                          catalog["payload"])
            if sport_key is None:
                self.prefetch_status = "SPORT_KEY_UNRESOLVED"
                return self.prefetch_status
            self.client.get_h2h(sport_key)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
            self.prefetch_status = f"ERROR:{type(error).__name__}:{error}"
            return self.prefetch_status
        self.prefetch_status = "FETCHED"
        return self.prefetch_status

    def fetch_1x2(self, fixture_id: str, home_team: str, away_team: str,
                  kickoff_utc: datetime, at: datetime) -> MarketProviderResult:
        self.last_evidence = None
        if not self.client.configured:
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider_name,
                                        reason="NOT_CONFIGURED")
        if self.prefetch_status not in (None, "FETCHED"):
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider_name,
                                        reason=self.prefetch_status)
        catalog = self.client.cache.fresh(endpoint="sports", sport_key=None,
                                          regions=None, at=at)
        if catalog is None or not isinstance(catalog["payload"], list):
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider_name,
                                        reason="SPORT_CATALOG_UNAVAILABLE")
        sport_key = resolve_sport_key(self.competition, self.client.sport_map,
                                      catalog["payload"])
        if sport_key is None:
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider_name,
                                        reason="SPORT_KEY_UNRESOLVED")
        regions = ",".join(self.client.config["regions"])
        raw = self.client.cache.fresh(endpoint=f"odds_{sport_key}", sport_key=sport_key,
                                      regions=regions, at=at)
        if raw is None or not isinstance(raw["payload"], list):
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider_name,
                                        reason="CURRENT_ODDS_UNAVAILABLE")
        try:
            matched = match_fixture(raw["payload"], home_team=home_team,
                away_team=away_team, kickoff_utc=kickoff_utc,
                tolerance_minutes=int(self.client.config["fixture_match"]["kickoff_tolerance_minutes"]),
                normalized_tolerance_minutes=int(self.client.config["fixture_match"][
                    "normalized_exact_tolerance_minutes"]))
        except ValueError as error:
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider_name,
                                        reason=str(error))
        if matched is None:
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider_name,
                                        reason="FIXTURE_NOT_FOUND")
        event, method = matched
        if _time(event["commence_time"]) <= raw["fetched_at"] or (
            raw["fetched_at"] > at or at >= kickoff_utc
        ):
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider_name,
                                        reason="ODDS_API_NOT_PREMATCH")
        quotes, detail = parse_complete_h2h(event, fixture_id=fixture_id,
            fetched_at=raw["fetched_at"], raw_payload_sha256=raw["payload_sha256"],
            maximum_age_seconds=int(self.client.config["maximum_market_age_seconds"]),
            maximum_overround=float(self.client.config["maximum_overround"]),
            minimum_decimal_odds=float(self.client.config["minimum_decimal_odds"]))
        if not quotes:
            reason = "STALE_ONLY" if detail["rejected"].get("STALE_MARKET") else "NO_COMPLETE_1X2"
            return MarketProviderResult(status="DATA_UNAVAILABLE", source_id=self.provider_name,
                                        reason=reason)
        quota_remaining = raw["quota_remaining"]
        self.last_evidence = {"provider": self.provider_name,
            "provider_category": "GLOBAL_ODDS_PROVIDER", "source_quality": "TRUSTED_API",
            "provider_event_id": str(event["id"]), "sport_key": sport_key,
            "regions": regions, "fixture_match_method": method,
            "bookmakers": detail["bookmakers"], "rejected": detail["rejected"],
            "market_fetched_at": raw["fetched_at"].isoformat(),
            "quota_remaining": quota_remaining, "quota_used": raw["quota_used"],
            "quota_last_cost": raw["quota_last_cost"],
            "quota_warning": quota_remaining is not None and quota_remaining <= int(
                self.client.config["quota_warning_threshold"]),
            "raw_payload_sha256": raw["payload_sha256"],
            "raw_response_id": raw["response_id"]}
        identity = ProviderEventIdentity(source_event_id=str(event["id"]),
            competition_name=self.competition, home_team_name=home_team,
            away_team_name=away_team, kickoff_time=kickoff_utc)
        return MarketProviderResult(status="AVAILABLE", source_id=self.provider_name,
            retrieved_at=raw["fetched_at"], quotes=quotes, event_identity=identity)
