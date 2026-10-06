"""Allow-listed provider contracts; transport adapters must use CoreNetworkClient."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import IntEnum
from typing import Protocol

from erguoyuan_football.jc_verification.jc_schema import JCMatch


class ProviderTier(IntEnum):
    """Provenance rank for provider selection."""

    OFFICIAL = 1
    LICENSED_STABLE = 2
    MANUAL_IMPORT = 3


class JCProvider(Protocol):
    """Query one provider's complete JC slate for a UTC calendar date."""

    provider_id: str
    tier: ProviderTier

    def get_daily_matches(self, match_date: date) -> list[JCMatch]:
        """Return sourced fixtures or raise a typed/provider exception."""


@dataclass(frozen=True)
class ProviderRegistration:
    """Explicit registration metadata; no URL is accepted by this boundary."""

    provider: JCProvider
    complete_daily_coverage: bool = False

    def __post_init__(self) -> None:
        if not self.provider.provider_id:
            raise ValueError("JC_PROVIDER_ID_REQUIRED")


class JCProviderRegistry:
    """Stable, explicit provider allow-list ordered by trust tier."""

    def __init__(self, providers: Iterable[ProviderRegistration] = ()) -> None:
        registrations = tuple(providers)
        ids = [item.provider.provider_id for item in registrations]
        if len(set(ids)) != len(ids):
            raise ValueError("DUPLICATE_JC_PROVIDER_ID")
        self._providers = tuple(sorted(registrations, key=lambda item: (
            item.provider.tier, item.provider.provider_id)))

    def list(self) -> tuple[ProviderRegistration, ...]:
        """Return registered providers in deterministic priority order."""
        return self._providers
