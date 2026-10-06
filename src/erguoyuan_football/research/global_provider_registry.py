"""Finite registry for configured providers and their allowed network hosts."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from erguoyuan_football.research.provider_coverage_profile import (
    ProviderCoverageProfile,
)


@dataclass(frozen=True)
class RegisteredProvider:
    """Provider profile plus explicitly allowlisted domains and runtime object."""

    profile: ProviderCoverageProfile
    allowed_domains: frozenset[str]
    configured: bool
    health: str = "UNKNOWN"
    provider: object | None = None

    def __post_init__(self) -> None:
        if not self.allowed_domains:
            raise ValueError("PROVIDER_ALLOWED_DOMAIN_REQUIRED")
        if any(not domain or "/" in domain or ":" in domain for domain in self.allowed_domains):
            raise ValueError("INVALID_ALLOWED_DOMAIN")


class GlobalProviderRegistry:
    """Reject duplicate providers and expose stable configured listings."""

    def __init__(self, providers: tuple[RegisteredProvider, ...] = ()) -> None:
        self._providers = {row.profile.provider_id: row for row in providers}
        if len(self._providers) != len(providers):
            raise ValueError("DUPLICATE_PROVIDER_ID")

    def get(self, provider_id: str) -> RegisteredProvider:
        """Return one explicitly configured source."""
        try:
            return self._providers[provider_id]
        except KeyError as error:
            raise KeyError(f"PROVIDER_NOT_REGISTERED:{provider_id}") from error

    def list(self) -> tuple[RegisteredProvider, ...]:
        """List profiles in deterministic identifier order."""
        return tuple(self._providers[key] for key in sorted(self._providers))

    @classmethod
    def from_source_definitions(cls, rows: tuple[tuple[ProviderCoverageProfile, str, bool, object | None], ...]
                                ) -> GlobalProviderRegistry:
        """Create entries from reviewed HTTPS base URLs and optional provider objects."""
        providers = []
        for profile, base_url, configured, provider in rows:
            parsed = urlparse(base_url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("PROVIDER_BASE_URL_MUST_BE_HTTPS")
            providers.append(RegisteredProvider(profile, frozenset({parsed.hostname}),
                                                configured, provider=provider))
        return cls(tuple(providers))
