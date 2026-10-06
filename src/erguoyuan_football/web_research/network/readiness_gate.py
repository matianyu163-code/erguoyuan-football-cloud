"""Fail-closed readiness of configured provider domains, without arbitrary probes."""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from urllib.parse import urlparse

from erguoyuan_football.web_research.providers.base_provider import BaseProvider
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry


def _dns_probe(host: str) -> bool:
    """Resolve only a registered provider hostname, not a model service."""
    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class NetworkReadinessReport:
    """Configuration and DNS gate; the actual HTTP request remains authoritative."""

    internet_available: bool
    dns_ok: bool
    proxy_detected: bool
    official_provider_ready: bool
    structured_provider_ready: bool
    media_provider_ready: bool
    overall_ready: bool
    failures: tuple[str, ...]
    ready_provider_ids: tuple[str, ...] = ()


class NetworkReadinessGate:
    """Check only explicitly configured sources; no unauthorised probe URL."""

    def __init__(
        self,
        sources: SourceRegistry,
        providers: Iterable[BaseProvider],
        *,
        dns_probe: Callable[[str], bool] = _dns_probe,
    ) -> None:
        self.sources = sources
        self.providers = tuple(providers)
        self.dns_probe = dns_probe

    def check(self) -> NetworkReadinessReport:
        """Return task-independent readiness without touching prediction services."""
        ready = {3: False, 2: False, 1: False}
        failures: list[str] = []
        configured = False
        ready_providers: list[str] = []
        proxy_detected = bool(os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY"))
        resolved_host = False
        for provider in self.providers:
            if not getattr(provider, "configured", False):
                continue
            configured = True
            try:
                source = self.sources.get(provider.provider_id)
            except (KeyError, ValueError):
                failures.append("SOURCE_NOT_REGISTERED")
                continue
            host = urlparse(source.url).hostname
            dns_result = host is not None and self.dns_probe(host)
            resolved_host = resolved_host or dns_result
            if not dns_result and not proxy_detected:
                failures.append(f"DNS_UNAVAILABLE:{provider.provider_id}")
                continue
            ready[source.tier] = True
            ready_providers.append(provider.provider_id)
        if not configured:
            failures.append("NO_PRODUCTION_ENDPOINT_CONFIGURED")
        dns_ok = resolved_host
        return NetworkReadinessReport(
            internet_available=dns_ok, dns_ok=dns_ok,
            proxy_detected=proxy_detected,
            official_provider_ready=ready[3], structured_provider_ready=ready[2],
            media_provider_ready=ready[1], overall_ready=bool(ready_providers),
            failures=tuple(failures),
            ready_provider_ids=tuple(ready_providers),
        )
