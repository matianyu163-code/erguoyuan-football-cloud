"""Research classification over the existing network allowlist."""

from __future__ import annotations

from urllib.parse import urlparse

from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from erguoyuan_football.web_research.sources.source_schema import SourceRecord


class SourceRegistry:
    """Register only sources that also exist in the transport allowlist."""

    def __init__(self, network_registry: ExternalSourceRegistry) -> None:
        self.network_registry = network_registry
        self._sources: dict[str, SourceRecord] = {}

    def add(self, source: SourceRecord) -> None:
        """Reject undocumented URLs and conflicting source identities."""
        definition = self.network_registry.get(source.source_id)
        if definition.base_url is None or definition.base_url.rstrip("/") != source.url.rstrip("/"):
            raise ValueError("SOURCE_URL_NOT_ALLOWLISTED")
        if source.source_id in self._sources and self._sources[source.source_id] != source:
            raise ValueError("SOURCE_ID_CONFLICT")
        self._sources[source.source_id] = source

    def get(self, source_id: str) -> SourceRecord:
        """Return an enabled source or fail closed."""
        try:
            source = self._sources[source_id]
        except KeyError as error:
            raise KeyError(f"SOURCE_NOT_REGISTERED:{source_id}") from error
        if not source.enabled:
            raise ValueError("SOURCE_DISABLED")
        return source

    def list(self) -> tuple[SourceRecord, ...]:
        """List registered metadata, including disabled sources."""
        return tuple(self._sources[key] for key in sorted(self._sources))

    def validate_result_url(self, source_id: str, url: str) -> SourceRecord:
        """Accept only the source's registered base or explicit endpoint URL."""
        source = self.get(source_id)
        definition = self.network_registry.get(source_id)
        allowed = {source.url, *definition.endpoints.values()}
        parsed = urlparse(url)
        domain_allowed = (parsed.scheme == "https" and parsed.hostname is not None
                          and parsed.hostname.casefold() in {
                              domain.casefold() for domain in source.allowed_domains
                          } and not parsed.username and not parsed.password
                          and not parsed.query and not parsed.fragment)
        if url not in allowed and not domain_allowed:
            raise ValueError("RESULT_URL_NOT_ALLOWLISTED")
        return source
