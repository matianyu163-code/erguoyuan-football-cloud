"""Capability and registry-tier aware provider routing."""

from __future__ import annotations

from collections.abc import Iterable

from erguoyuan_football.web_research.providers.base_provider import BaseProvider
from erguoyuan_football.web_research.providers.capabilities import ProviderCapability
from erguoyuan_football.web_research.search.search_task import SearchTask
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry


class ProviderRouter:
    """Route only to configured, enabled, allowed and capable adapters."""

    def __init__(self, sources: SourceRegistry, providers: Iterable[BaseProvider]) -> None:
        self.sources = sources
        self.providers = tuple(providers)

    def providers_for(self, task: SearchTask) -> tuple[BaseProvider, ...]:
        """Return supported providers in descending registry tier order."""
        capability = ProviderCapability(task.task_type)
        eligible: list[tuple[int, BaseProvider]] = []
        for provider in self.providers:
            if not provider.supports(capability) or not getattr(provider, "configured", False):
                continue
            try:
                source = self.sources.get(provider.provider_id)
            except (KeyError, ValueError):
                continue
            if (provider.source_tier != source.tier or source.source_type not in task.required_sources):
                continue
            eligible.append((source.tier, provider))
        return tuple(provider for _, provider in sorted(
            eligible, key=lambda pair: (-pair[0], pair[1].provider_id)
        ))
