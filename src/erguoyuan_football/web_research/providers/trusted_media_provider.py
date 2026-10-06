"""Trusted-media JSON adapter; reports never become official facts."""

from __future__ import annotations

from erguoyuan_football.web_research.providers.capabilities import ProviderCapability
from erguoyuan_football.web_research.providers.registered_json_provider import (
    RegisteredJsonProvider,
)


class TrustedMediaProvider(RegisteredJsonProvider):
    """Configured news or injury reports; never odds or official lineups."""

    allowed_capabilities = frozenset({ProviderCapability.NEWS,
                                      ProviderCapability.INJURY})
