"""Explicit environment-backed provider registration; no guessed endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.config import NetworkConfig, RetryPolicy, TimeoutPolicy
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from erguoyuan_football.web_research.providers.base_provider import BaseProvider
from erguoyuan_football.web_research.providers.capabilities import ProviderCapability
from erguoyuan_football.web_research.providers.official_web_provider import (
    OfficialWebProvider,
)
from erguoyuan_football.web_research.providers.registered_json_provider import (
    RegisteredJsonProvider,
)
from erguoyuan_football.web_research.providers.structured_http_provider import (
    StructuredHttpProvider,
)
from erguoyuan_football.web_research.providers.trusted_media_provider import (
    TrustedMediaProvider,
)
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.sources.source_schema import SourceRecord

_TIERS = (
    ("OFFICIAL", 3, "OFFICIAL", OfficialWebProvider),
    ("STRUCTURED", 2, "STRUCTURED", StructuredHttpProvider),
    ("MEDIA", 1, "MEDIA", TrustedMediaProvider),
)


@dataclass
class LiveProviderEnvironment:
    """Registries, one HTTP transport and explicitly configured adapters."""

    sources: SourceRegistry
    client: CoreNetworkClient
    providers: tuple[BaseProvider, ...]

    def close(self) -> None:
        """Close the shared HTTP pool."""
        self.client.close()


def _network_config() -> NetworkConfig:
    timeout = float(os.environ.get("CORE_HTTP_TIMEOUT", "30"))
    retries = int(os.environ.get("CORE_NETWORK_RETRIES", "2"))
    if not 1 <= timeout <= 180 or not 0 <= retries <= 7:
        raise ValueError("INVALID_NETWORK_ENV_CONFIG")
    return NetworkConfig(
        timeout=TimeoutPolicy(connect_timeout=min(10.0, timeout),
                              read_timeout=min(20.0, timeout),
                              total_timeout=timeout),
        retry=RetryPolicy(max_attempts=retries + 1),
    )


def build_live_provider_environment() -> LiveProviderEnvironment:
    """Register only HTTPS URLs and capabilities expressly present in environment."""
    definitions: list[SourceDefinition] = []
    records: list[SourceRecord] = []
    plans: list[tuple[str, type[RegisteredJsonProvider], dict[ProviderCapability, str]]] = []
    network_config = _network_config()
    for label, tier, source_type, adapter_type in _TIERS:
        prefix = f"CORE_{label}_PROVIDER"
        endpoint_url = os.environ.get(f"{prefix}_URL", "").strip()
        if not endpoint_url:
            continue
        parsed = urlparse(endpoint_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError(f"INVALID_PROVIDER_URL:{label}")
        requested = os.environ.get(f"{prefix}_CAPABILITIES", "")
        capabilities = tuple(ProviderCapability(part.strip().upper())
                             for part in requested.split(",") if part.strip())
        if any(capability not in adapter_type.allowed_capabilities
               for capability in capabilities):
            raise ValueError(f"PROVIDER_CAPABILITY_NOT_IMPLEMENTED:{label}")
        endpoint_ids = {capability: capability.value.lower() for capability in capabilities}
        source_id = f"CORE_{label}"
        base_url = f"https://{parsed.netloc}"
        key_name = f"CORE_{label}_API_KEY"
        auth_required = (os.environ.get(f"{prefix}_REQUIRES_AUTH", "").lower()
                         in {"1", "true", "yes"} or bool(os.environ.get(key_name)))
        definitions.append(SourceDefinition(
            source_id=source_id, display_name=f"Configured {label} Source",
            category="WEB_RESEARCH", base_url=base_url,
            endpoints={endpoint_id: endpoint_url for endpoint_id in endpoint_ids.values()},
            requires_auth=auth_required,
            auth_env_var=key_name if auth_required else None,
            source_timestamp_field="observed_at", schema_version="PHASE13_4_V1",
            retry_policy=network_config.retry,
            timeout_policy=network_config.timeout,
        ))
        records.append(SourceRecord(source_id, f"Configured {label} Source",
                                    source_type, tier, base_url, True,
                                    allowed_domains=(parsed.hostname,)))
        plans.append((source_id, adapter_type, endpoint_ids))
    network_registry = ExternalSourceRegistry(tuple(definitions))
    sources = SourceRegistry(network_registry)
    for record in records:
        sources.add(record)
    client = CoreNetworkClient(network_registry, network_config)
    providers = tuple(adapter_type(source_id, sources, client, endpoint_ids)
                      for source_id, adapter_type, endpoint_ids in plans)
    return LiveProviderEnvironment(sources, client, providers)
