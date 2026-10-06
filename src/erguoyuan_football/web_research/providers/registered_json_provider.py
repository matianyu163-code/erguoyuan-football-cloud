"""Registered JSON adapter through the existing audited CoreNetworkClient."""

from __future__ import annotations

from collections.abc import Mapping

from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.web_research.providers.base_provider import BaseProvider
from erguoyuan_football.web_research.providers.capabilities import ProviderCapability
from erguoyuan_football.web_research.providers.provider_result import ProviderResult
from erguoyuan_football.web_research.search.search_task import SearchTask
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.time_utils import parse_utc, utc_iso


class RegisteredJsonProvider(BaseProvider):
    """Capability-scoped JSON endpoint adapter; no direct HTTPX calls."""

    allowed_capabilities: frozenset[ProviderCapability] = frozenset()

    def __init__(
        self,
        provider_id: str,
        sources: SourceRegistry,
        client: CoreNetworkClient,
        endpoint_ids: Mapping[ProviderCapability, str],
    ) -> None:
        if any(capability not in self.allowed_capabilities for capability in endpoint_ids):
            raise ValueError("PROVIDER_CAPABILITY_NOT_IMPLEMENTED")
        source = sources.get(provider_id)
        if any(endpoint_id not in sources.network_registry.get(provider_id).endpoints
               for endpoint_id in endpoint_ids.values()):
            raise ValueError("ENDPOINT_NOT_REGISTERED")
        self.provider_id = provider_id
        self.provider_name = source.name
        self.source_tier = source.tier
        self.capabilities = frozenset(endpoint_ids)
        self.endpoint_ids = dict(endpoint_ids)
        self.sources = sources
        self.client = client
        self.configured = bool(endpoint_ids)

    def fetch(self, task: SearchTask) -> ProviderResult:
        """Fetch a declared task with CoreNetworkClient and source timestamps."""
        capability = ProviderCapability(task.task_type)
        if not self.supports(capability):
            return ProviderResult(False, error_code="PROVIDER_CAPABILITY_UNAVAILABLE",
                                  provider_id=self.provider_id, task_type=task.task_type)
        endpoint_id = self.endpoint_ids[capability]
        response = self.client.fetch_json(
            self.provider_id, endpoint_id,
            params={"query": task.query, "task_id": task.task_id},
            prediction_time=None, bypass_cache=True,
        )
        if not isinstance(response.body, dict):
            raise TypeError("INVALID_PROVIDER_RESPONSE:JSON_OBJECT_REQUIRED")
        observed = response.body.get("observed_at")
        published = response.body.get("published_at")
        if not isinstance(observed, str):
            raise TypeError("INVALID_PROVIDER_RESPONSE:OBSERVED_AT_REQUIRED")
        observed_at = parse_utc(observed)
        if observed_at > response.retrieved_at:
            raise ValueError("INVALID_PROVIDER_RESPONSE:FUTURE_OBSERVATION")
        if published is not None and not isinstance(published, str):
            raise ValueError("INVALID_PROVIDER_RESPONSE:PUBLISHED_AT_INVALID")
        source_url = response.final_url or self.sources.network_registry.get(
            self.provider_id).endpoints[endpoint_id]
        self.sources.validate_result_url(self.provider_id, source_url)
        return ProviderResult(
            True, response.body, self.provider_name, source_url,
            utc_iso(response.retrieved_at), "LOW", utc_iso(observed_at),
            provider_id=self.provider_id, source_tier=self.source_tier,
            task_type=task.task_type, published_time=published,
            observed_time=utc_iso(observed_at), cache_status="LIVE",
            http_status=response.status_code, retry_count=response.retry_count,
            endpoint_id=endpoint_id,
        )
