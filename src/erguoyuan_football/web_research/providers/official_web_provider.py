"""Official-source JSON adapter; capabilities require explicit endpoints."""

from __future__ import annotations

from erguoyuan_football.web_research.providers.capabilities import ProviderCapability
from erguoyuan_football.web_research.providers.registered_json_provider import (
    RegisteredJsonProvider,
)


class OfficialWebProvider(RegisteredJsonProvider):
    """Official fixtures, lineups, injuries or news when separately configured."""

    allowed_capabilities = frozenset({ProviderCapability.FIXTURE,
                                      ProviderCapability.LINEUP,
                                      ProviderCapability.INJURY,
                                      ProviderCapability.NEWS})
