"""Actual network smoke, excluded from default regression by an explicit marker gate."""

from __future__ import annotations

import os

import pytest

from erguoyuan_football.web_research.providers.provider_config import (
    build_live_provider_environment,
)
from erguoyuan_football.web_research.search.search_task import SearchTask
from erguoyuan_football.web_research.time_utils import parse_utc


@pytest.mark.live_network
def test_live_registered_provider(request: pytest.FixtureRequest) -> None:
    """Make a real HTTP request only with a configured endpoint and explicit marker."""
    if "live_network" not in request.config.option.markexpr:
        pytest.skip("explicit -m live_network required; default tests do not access network")
    if not any(os.environ.get(f"CORE_{tier}_PROVIDER_URL")
               for tier in ("OFFICIAL", "STRUCTURED", "MEDIA")):
        pytest.skip("production endpoint not configured")
    environment = build_live_provider_environment()
    try:
        providers = [provider for provider in environment.providers if provider.capabilities]
        assert providers, "configured endpoint has no declared supported capability"
        provider = providers[0]
        capability = min(provider.capabilities, key=lambda value: value.value)
        task = SearchTask("LIVE_SMOKE", capability.value,
                          os.environ.get("CORE_LIVE_SMOKE_QUERY", "football fixture"),
                          (environment.sources.get(provider.provider_id).source_type,))
        result = provider.fetch(task)
        assert result.success
        assert result.source_url.startswith("https://")
        assert result.fetched_at is not None
        assert parse_utc(result.as_of_time or "") <= result.fetched_at
        assert environment.client.audit.records()
    finally:
        environment.close()
