"""Explicit external source allowlist and provider registration."""

from __future__ import annotations

from erguoyuan_football.network.schemas import RequiredLevel, SourceDefinition


class ExternalSourceRegistry:
    """Lookup source definitions without inventing undocumented endpoints."""

    def __init__(self, sources: tuple[SourceDefinition, ...] = ()) -> None:
        self._sources = {source.source_id: source for source in sources}
        if len(self._sources) != len(sources):
            raise ValueError("duplicate source_id")

    def get(self, source_id: str) -> SourceDefinition:
        """Resolve a registered source or fail before making a request."""
        try:
            return self._sources[source_id]
        except KeyError as error:
            raise KeyError(f"SOURCE_NOT_CONFIGURED:{source_id}") from error

    def list(self) -> tuple[SourceDefinition, ...]:
        """Return sources in stable order."""
        return tuple(self._sources[key] for key in sorted(self._sources))

    def required(self) -> tuple[SourceDefinition, ...]:
        """Return CRITICAL and REQUIRED sources."""
        return tuple(source for source in self.list() if source.required_level != RequiredLevel.OPTIONAL)

