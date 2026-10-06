"""Provider-bound, bounded historical result acquisition contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from erguoyuan_football.research.historical_provider_router import (
    HistoricalCapability,
    HistoricalProviderOption,
    HistoricalProviderRouter,
)
from erguoyuan_football.research.historical_result_cache import HistoricalResultCache
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBProvider,
    SeasonBatch,
)


class HistoricalProviderUnavailable(RuntimeError):
    """No provider returned cutoff-safe historical rows for the requested season."""


@dataclass(frozen=True)
class HistoricalAcquisition:
    """Acquisition counts retained for the pipeline audit report."""

    batch: SeasonBatch
    pages_fetched: int
    records_received: int
    pagination_supported: bool
    cache_status: str
    batches: tuple[SeasonBatch, ...] = ()
    batch_provider_ids: tuple[str, ...] = ()
    fallback_attempts: tuple[tuple[str, str], ...] = ()


class HistoricalDataAcquirer:
    """Acquire through an already registered provider; never invent endpoints."""

    def __init__(self, cache: HistoricalResultCache | None = None, *,
                 fallback_providers: tuple[HistoricalProviderOption, ...] = ()) -> None:
        self.cache = cache or HistoricalResultCache(
            Path(__file__).resolve().parents[3] / "data/historical_result_cache.sqlite")
        self._owns_cache = cache is None
        self.fallback_providers = fallback_providers
        self.router = HistoricalProviderRouter()

    def close(self) -> None:
        """Close the default-owned local cache."""
        if self._owns_cache:
            self.cache.close()

    def acquire(self, provider: OpenLigaDBProvider, *, cutoff: datetime
                ) -> HistoricalAcquisition:
        """Fetch the active season and PIT-safe configured history seasons."""
        current = provider.fetch_season(bypass_cache=True)
        batches = [current]
        batch_provider_ids = [provider.config.provider_id]
        cache_states: list[str] = []
        fallback_attempts: list[tuple[str, str]] = []
        for season in provider.config.history_seasons:
            cached = self.cache.get(provider.config.provider_id,
                                    provider.config.competition_id, season, cutoff=cutoff)
            if cached is None:
                try:
                    historical = provider.fetch_season(season, bypass_cache=True)
                    if historical.retrieved_at > cutoff:
                        raise ValueError("PRIMARY_EVIDENCE_AFTER_CUTOFF")
                    selected_provider_id = provider.config.provider_id
                    self.cache.put(selected_provider_id,
                                   provider.config.competition_id, historical)
                    cache_states.append("LIVE")
                except (OSError, TimeoutError, ValueError, RuntimeError) as primary_error:
                    fallback_rows = self._fetch_fallback(
                        provider, season, cutoff, fallback_attempts,
                        primary_error)
                    cache_states.append("FALLBACK")
                    for historical, selected_provider_id in fallback_rows:
                        if historical.retrieved_at <= cutoff:
                            batches.append(historical)
                            batch_provider_ids.append(selected_provider_id)
                        else:
                            fallback_attempts.append((selected_provider_id,
                                                      "EVIDENCE_AFTER_CUTOFF"))
                else:
                    if historical.retrieved_at <= cutoff:
                        batches.append(historical)
                        batch_provider_ids.append(selected_provider_id)
                    else:
                        fallback_attempts.append((selected_provider_id,
                                                  "EVIDENCE_AFTER_CUTOFF"))
            else:
                batches.append(cached)
                batch_provider_ids.append(provider.config.provider_id)
                cache_states.append("CACHE_FRESH")
        status = "NOT_CONFIGURED" if not provider.config.history_seasons else (
            "MIXED" if "LIVE" in cache_states and "CACHE_FRESH" in cache_states else
            "CACHE_FRESH" if cache_states and all(x == "CACHE_FRESH" for x in cache_states)
            else "FALLBACK" if "FALLBACK" in cache_states else "LIVE")
        return HistoricalAcquisition(current, len(batches),
            sum(len(batch.matches) for batch in batches), False, status, tuple(batches),
            tuple(batch_provider_ids), tuple(fallback_attempts))

    def _fetch_fallback(self, primary: OpenLigaDBProvider, season: int,
                        cutoff: datetime,
                        attempts: list[tuple[str, str]],
                        primary_error: BaseException) -> tuple[tuple[SeasonBatch, str], ...]:
        """Fetch every exact-scope source so later sample merging retains lineage."""
        options = self.router.resolve(
            competition=primary.config.competition_id,
            country=primary.config.team_country, season=season,
            capability=HistoricalCapability.RESULTS,
            providers=self.fallback_providers)
        results: list[tuple[SeasonBatch, str]] = []
        for option in options:
            if (not option.canonical_team_map_complete or
                    option.competition_id != primary.config.competition_id):
                attempts.append((option.provider_id, "CANONICAL_SCOPE_MISMATCH"))
                continue
            try:
                cached = self.cache.get(option.provider_id, option.competition_id,
                                        season, cutoff=cutoff)
                batch = cached if cached is not None else option.fetch(season)
                if batch.season != season:
                    raise ValueError("FALLBACK_SEASON_MISMATCH")
                if batch.retrieved_at > cutoff:
                    raise ValueError("FALLBACK_EVIDENCE_AFTER_CUTOFF")
                if cached is None:
                    self.cache.put(option.provider_id, option.competition_id, batch)
                attempts.append((option.provider_id, "AVAILABLE"))
                results.append((batch, option.provider_id))
            except (OSError, TimeoutError, ValueError, RuntimeError) as error:
                attempts.append((option.provider_id,
                                 f"{type(error).__name__}:{error}"))
        if results:
            return tuple(results)
        detail = ";".join(f"{provider}:{state}" for provider, state in attempts)
        raise HistoricalProviderUnavailable(
            f"HISTORY_PROVIDER_UNAVAILABLE:{primary.config.competition_id}:"
            f"{season}:primary={type(primary_error).__name__}:{detail}") from primary_error
