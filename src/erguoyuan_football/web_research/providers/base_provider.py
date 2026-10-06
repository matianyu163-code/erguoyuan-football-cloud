"""Read-only external research provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from erguoyuan_football.web_research.providers.capabilities import ProviderCapability
from erguoyuan_football.web_research.providers.provider_result import ProviderResult
from erguoyuan_football.web_research.search.search_task import SearchTask


class BaseProvider(ABC):
    """Fetch evidence only; future HTTP adapters must use CoreNetworkClient."""

    provider_name: str
    provider_id: str = ""
    source_tier: int = 0
    capabilities: frozenset[ProviderCapability] = frozenset()

    def supports(self, capability: ProviderCapability) -> bool:
        """Require an explicit capability declaration for router selection."""
        return capability in self.capabilities

    @abstractmethod
    def fetch(self, task: SearchTask) -> ProviderResult:
        """Fetch a task without changing models, predictions, or output contracts."""
