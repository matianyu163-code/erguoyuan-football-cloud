"""Build provider routing registry from explicitly configured source profiles."""

from __future__ import annotations

from pathlib import Path

from erguoyuan_football.research.global_provider_registry import GlobalProviderRegistry
from erguoyuan_football.research.live_data.football_data_org_directory import (
    FootballDataOrgDirectoryProvider,
)
from erguoyuan_football.research.live_data.openligadb_directory import (
    OpenLigaDBDirectoryProvider,
)
from erguoyuan_football.research.provider_coverage_profile import (
    ProviderCoverageProfile,
)


def build_global_provider_registry(
        openligadb_config: Path | str) -> GlobalProviderRegistry:
    """Register public directory endpoints with UNVERIFIED declarations.

    Live verification is a separate operation; constructing a registry never
    claims that every configured capability was successfully fetched.
    """
    directory = OpenLigaDBDirectoryProvider(openligadb_config)
    football_data = FootballDataOrgDirectoryProvider()
    try:
        profiles: list[ProviderCoverageProfile] = [
            directory.profile_declaration(scope)
            for scope in directory.scopes.values()
        ]
        openligadb = _merge_profiles(profiles)
        return GlobalProviderRegistry.from_source_definitions((
            (openligadb, directory.base_url, True, directory),
            (FootballDataOrgDirectoryProvider.profile_declaration(),
             football_data.base_url, True, football_data),
        ))
    except Exception:
        directory.close()
        football_data.close()
        raise


def _merge_profiles(profiles: list[ProviderCoverageProfile]
                    ) -> ProviderCoverageProfile:
    """Combine configured scopes without promoting unverified capability."""
    if not profiles:
        raise ValueError("OPENLIGADB_SCOPE_REQUIRED")
    first = profiles[0]
    return ProviderCoverageProfile(
        first.provider_id, first.source_tier,
        frozenset().union(*(row.regions for row in profiles)),
        frozenset().union(*(row.countries for row in profiles)),
        frozenset().union(*(row.federations for row in profiles)),
        frozenset().union(*(row.genders for row in profiles)),
        frozenset().union(*(row.age_groups for row in profiles)),
        frozenset().union(*(row.entity_types for row in profiles)),
        frozenset().union(*(row.squad_levels for row in profiles)),
        frozenset().union(*(row.competition_types for row in profiles)),
        frozenset().union(*(row.capabilities for row in profiles)),
        max((row.historical_depth_years or 0 for row in profiles), default=0),
        any(row.live_data for row in profiles),
        any(row.requires_api_key for row in profiles),
        all(row.rate_limit_known for row in profiles),
        "; ".join(sorted({row.license_note or "" for row in profiles})),
    )

