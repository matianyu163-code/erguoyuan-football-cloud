"""Auditable provider capability and scope profiles for global routing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class CoverageStatus(StrEnum):
    """Evidence level for one provider capability in a declared scope."""

    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    UNVERIFIED = "UNVERIFIED"
    UNSUPPORTED = "UNSUPPORTED"
    DEGRADED = "DEGRADED"


@dataclass(frozen=True)
class ProviderCoverageProfile:
    """Declared coverage; declaration alone never constitutes verification."""

    provider_id: str
    source_tier: int
    regions: frozenset[str]
    countries: frozenset[str]
    federations: frozenset[str]
    genders: frozenset[str]
    age_groups: frozenset[str]
    entity_types: frozenset[str]
    squad_levels: frozenset[str]
    competition_types: frozenset[str]
    capabilities: frozenset[str]
    historical_depth_years: int | None
    live_data: bool
    requires_api_key: bool
    rate_limit_known: bool
    license_note: str | None
    last_verified_at: datetime | None = None
    capability_status: tuple[tuple[str, CoverageStatus], ...] = ()

    def __post_init__(self) -> None:
        if not self.provider_id or not 1 <= self.source_tier <= 5:
            raise ValueError("INVALID_PROVIDER_PROFILE")
        if self.historical_depth_years is not None and self.historical_depth_years < 0:
            raise ValueError("INVALID_HISTORICAL_DEPTH")
        if self.last_verified_at is not None and self.last_verified_at.tzinfo is None:
            raise ValueError("UTC_VERIFICATION_TIME_REQUIRED")
        if len(dict(self.capability_status)) != len(self.capability_status):
            raise ValueError("DUPLICATE_CAPABILITY_STATUS")
        declared_status = dict(self.capability_status)
        if not set(declared_status) <= self.capabilities:
            raise ValueError("STATUS_FOR_UNDECLARED_CAPABILITY")

    def status_for(self, capability: str) -> CoverageStatus:
        """Return UNVERIFIED for a merely declared capability."""
        return dict(self.capability_status).get(
            capability, CoverageStatus.UNVERIFIED
            if capability in self.capabilities else CoverageStatus.UNSUPPORTED)

