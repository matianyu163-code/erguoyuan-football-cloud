"""Synthetic transport tests for bounded live JSON requests and safe redirects."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.config import RetryPolicy
from erguoyuan_football.network.errors import RetryExhausted
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry


class _Response:
    def __init__(self, code: int, body: dict[str, Any] | None = None,
                 headers: dict[str, str] | None = None) -> None:
        self.status_code = code
        self.body = body or {}
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        """Return the synthetic body."""
        return self.body


class _Transport:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, *, params: dict[str, Any],
            headers: dict[str, str]) -> _Response:
        """Record the addressed URL without making a network request."""
        self.calls.append(url)
        return self.responses.pop(0)


def _client(responses: list[_Response]) -> tuple[CoreNetworkClient, _Transport]:
    source = SourceDefinition(
        source_id="SYNTHETIC", display_name="Synthetic", category="WEB_RESEARCH",
        base_url="https://source.example.test",
        endpoints={"fixture": "https://source.example.test/fixture"},
        source_timestamp_field="observed_at", schema_version="SYNTHETIC_V1",
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0,
                                 max_backoff_seconds=0),
    )
    transport = _Transport(responses)
    return CoreNetworkClient(ExternalSourceRegistry((source,)),
                             http_client=transport), transport


def _body() -> dict[str, str]:
    return {"observed_at": datetime.now(UTC).isoformat(), "fixture": "synthetic"}


def test_503_retries_are_bounded_and_audited() -> None:
    client, transport = _client([_Response(503), _Response(200, _body())])
    result = client.fetch_json("SYNTHETIC", "fixture", params={}, bypass_cache=True)
    assert result.status_code == 200
    assert result.retry_count == 1
    assert len(transport.calls) == 2
    assert client.audit.records()[-1].outcome == "SUCCESS"


def test_429_reports_exhausted_attempts() -> None:
    client, transport = _client([_Response(429), _Response(429), _Response(429)])
    with pytest.raises(RetryExhausted, match="RATE_LIMITED") as error:
        client.fetch_json("SYNTHETIC", "fixture", params={}, bypass_cache=True)
    assert error.value.attempts == 3
    assert len(transport.calls) == 3
    assert client.audit.records()[-1].outcome == "RATE_LIMITED"


def test_same_host_redirect_and_cross_host_rejection() -> None:
    client, transport = _client([
        _Response(302, headers={"Location": "/new-fixture"}),
        _Response(200, _body()),
    ])
    result = client.fetch_json("SYNTHETIC", "fixture", params={}, bypass_cache=True)
    assert result.final_url == "https://source.example.test/new-fixture"
    assert len(transport.calls) == 2
    client, _ = _client([_Response(302, headers={"Location": "https://evil.example/"})])
    with pytest.raises(ValueError, match="SOURCE_NOT_ALLOWED"):
        client.fetch_json("SYNTHETIC", "fixture", params={}, bypass_cache=True)
