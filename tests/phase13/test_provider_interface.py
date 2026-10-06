"""SYNTHETIC_TEST providers only; these tests make no network requests."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from erguoyuan_football.web_research.cache import ResearchCache
from erguoyuan_football.web_research.providers.base_provider import BaseProvider
from erguoyuan_football.web_research.providers.provider_result import ProviderResult
from erguoyuan_football.web_research.research_service import ResearchService
from erguoyuan_football.web_research.search.query_builder import QueryBuilder
from erguoyuan_football.web_research.search.search_task import SearchTask
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.sources.source_schema import SourceRecord

_BASE = "https://official.example"
_ENDPOINT = "https://official.example/fixture"
_NOW = datetime(2026, 10, 1, 12, tzinfo=UTC)


def _registry() -> SourceRegistry:
    network = ExternalSourceRegistry((SourceDefinition(
        source_id="TEST_OFFICIAL", display_name="Test Official", category="fixture",
        base_url=_BASE, endpoints={"fixture": _ENDPOINT}, schema_version="SYNTHETIC_TEST_V1"
    ),))
    sources = SourceRegistry(network)
    sources.add(SourceRecord("TEST_OFFICIAL", "Test Official", "OFFICIAL", 3,
                             _BASE, True))
    return sources


class _FakeProvider(BaseProvider):
    """SYNTHETIC_TEST: deterministic response; no HTTP client or model output."""

    provider_name = "TEST_OFFICIAL"

    def __init__(self, fetched_time: str = "2026-10-01T11:59:00Z") -> None:
        self.calls = 0
        self.fetched_time = fetched_time

    def fetch(self, task: SearchTask) -> ProviderResult:
        self.calls += 1
        return ProviderResult(True, {"task": task.task_type}, "Test Official",
                              _ENDPOINT, self.fetched_time, "HIGH",
                              "2026-10-01T11:58:00Z")


def test_provider_interface_is_abstract() -> None:
    """Only an implementing adapter can fetch; no real provider is installed."""
    with pytest.raises(TypeError):
        BaseProvider()  # type: ignore[abstract]
    result = _FakeProvider().fetch(QueryBuilder.build("Arsenal", "Liverpool")[0])
    assert result.success and result.confidence == "HIGH"


def test_same_task_uses_persistent_cache_and_audit(tmp_path: Path) -> None:
    """The second request, including after reopening the cache, never fetches."""
    task = QueryBuilder.build("Arsenal", "Liverpool")[0]
    path = tmp_path / "research.sqlite"
    first_cache = ResearchCache(path)
    provider = _FakeProvider()
    service = ResearchService(_registry(), first_cache, clock=lambda: _NOW)
    first = service.execute(task, "TEST_OFFICIAL", provider)
    assert first.success and provider.calls == 1
    first_cache.close()

    reopened = ResearchCache(path)
    second = ResearchService(_registry(), reopened, clock=lambda: _NOW + timedelta(minutes=1))
    assert second.execute(task, "TEST_OFFICIAL", provider) == first
    assert provider.calls == 1
    assert [item.outcome for item in reopened.audit_records()] == [
        "FETCH_SUCCESS", "CACHE_HIT"
    ]
    assert reopened.get("TEST_OFFICIAL", task, _NOW - timedelta(minutes=1)) is None
    reopened.close()


def test_future_or_unregistered_provider_response_rejected(tmp_path: Path) -> None:
    """A future retrieval is not cached or treated as historical evidence."""
    task = QueryBuilder.build("Arsenal", "Liverpool")[0]
    cache = ResearchCache(tmp_path / "research.sqlite")
    provider = _FakeProvider("2026-10-01T12:01:00Z")
    service = ResearchService(_registry(), cache, clock=lambda: _NOW)
    with pytest.raises(ValueError, match="FUTURE_RETRIEVAL_FORBIDDEN"):
        service.execute(task, "TEST_OFFICIAL", provider)
    assert cache.get("TEST_OFFICIAL", task, _NOW + timedelta(minutes=2)) is None
    assert cache.audit_records()[0].outcome == "FETCH_FAILED"
    cache.close()


def test_fetch_completion_time_allows_actual_retrieval(tmp_path: Path) -> None:
    """SYNTHETIC_TEST: retrieval during a request is not mistaken for future data."""
    task = QueryBuilder.build("Arsenal", "Liverpool")[0]
    moments = iter((_NOW, _NOW + timedelta(seconds=2)))
    cache = ResearchCache(tmp_path / "research.sqlite")
    provider = _FakeProvider("2026-10-01T12:00:01Z")
    result = ResearchService(_registry(), cache, clock=lambda: next(moments)).execute(
        task, "TEST_OFFICIAL", provider
    )
    assert result.success and provider.calls == 1
    assert cache.get("TEST_OFFICIAL", task, _NOW + timedelta(seconds=2)) == result
    cache.close()


def test_provider_result_requires_temporal_provenance() -> None:
    """SYNTHETIC_TEST: source timestamps cannot be omitted or reversed."""
    with pytest.raises(ValueError, match="PROVIDER_AS_OF_TIME_REQUIRED"):
        ProviderResult(True, {"x": 1}, "Test Official", _ENDPOINT,
                       "2026-10-01T12:00:00Z", "HIGH")
    with pytest.raises(ValueError, match="SOURCE_TIME_AFTER_RETRIEVAL"):
        ProviderResult(True, {"x": 1}, "Test Official", _ENDPOINT,
                       "2026-10-01T12:00:00Z", "HIGH", "2026-10-01T13:00:00Z")
