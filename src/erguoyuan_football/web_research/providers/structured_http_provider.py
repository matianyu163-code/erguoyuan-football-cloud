"""Structured-data JSON adapter; no implied odds or xG feed."""

from __future__ import annotations

from erguoyuan_football.web_research.providers.capabilities import ProviderCapability
from erguoyuan_football.web_research.providers.registered_json_provider import (
    RegisteredJsonProvider,
)


class StructuredHttpProvider(RegisteredJsonProvider):
    """Configured fixture, team-statistics or odds endpoint adapter."""

    allowed_capabilities = frozenset({ProviderCapability.FIXTURE,
                                      ProviderCapability.TEAM_STATS,
                                      ProviderCapability.ODDS})
