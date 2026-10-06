"""Network tests use injected transports; default test runs never access the Internet."""

import sys
import types
from datetime import timedelta

import pytest

from erguoyuan_football.contracts.common import now
from erguoyuan_football.gates.production_network import ProductionNetworkGate
from erguoyuan_football.network.cache import NetworkCache, canonical_hash
from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.config import (
    NetworkConfig,
    NetworkMode,
    RateLimitPolicy,
    RetryPolicy,
)
from erguoyuan_football.network.health import (
    check_source_health,
    refresh_network_status,
    run_network_preflight,
)
from erguoyuan_football.network.schemas import (
    CachedPayload,
    CacheStatus,
    RequiredLevel,
    SourceDefinition,
    SourceHealthStatus,
)
from erguoyuan_football.network.source_registry import ExternalSourceRegistry

pytestmark = pytest.mark.fast


class FakeResponse:
    def __init__(
        self, status_code: int, body: dict | None = None, headers: dict | None = None
    ) -> None:
        self.status_code = status_code
        self._body = body or {}
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeHTTPClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def get(self, url, *, params, headers):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.responses.pop(0)


def source(
    *, required=RequiredLevel.REQUIRED, auth=False, retry=None
) -> SourceDefinition:
    return SourceDefinition(
        source_id="fixture-source",
        display_name="Fixture source",
        category="XG",
        base_url="https://data.example.test",
        endpoints={
            "health": "https://data.example.test/v1/health",
            "matches": "https://data.example.test/v1/matches",
        },
        required_level=required,
        requires_auth=auth,
        auth_env_var="FIXTURE_API_TOKEN" if auth else None,
        source_timestamp_field="data_timestamp",
        schema_version="fixture-v1",
        retry_policy=retry
        or RetryPolicy(
            max_attempts=2, initial_backoff_seconds=0, max_backoff_seconds=0
        ),
        rate_limit_policy=RateLimitPolicy(
            requests_per_second=1000, requests_per_minute=60000, burst=100
        ),
    )


def body(**values) -> dict:
    return {"data_timestamp": now().isoformat(), "items": [1], **values}


def test_network_config_modes_and_tls_verification() -> None:
    assert NetworkConfig().mode == NetworkMode.AUTO
    assert NetworkConfig(mode="DIRECT").mode == NetworkMode.DIRECT
    assert NetworkConfig(
        mode="EXPLICIT_PROXY", explicit_proxy_url="http://127.0.0.1:8080"
    ).explicit_proxy_url
    with pytest.raises(ValueError, match="cannot be disabled"):
        NetworkConfig(verify_tls=False)
    with pytest.raises(ValueError, match="PROXY_CONFIGURATION_UNSUPPORTED"):
        NetworkConfig(
            mode="EXPLICIT_PROXY", explicit_proxy_url="socks5://127.0.0.1:1080"
        )
    with pytest.raises(ValueError, match="only 429"):
        RetryPolicy(retry_status_codes=(401,))


def test_env_proxy_and_explicit_proxy_modes(monkeypatch) -> None:
    calls = []

    class HTTPXClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def close(self):
            pass

    module = types.SimpleNamespace(
        Client=HTTPXClient,
        Timeout=lambda **kwargs: kwargs,
        TransportError=type("TransportError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "httpx", module)
    registry = ExternalSourceRegistry((source(),))
    auto = CoreNetworkClient(registry, NetworkConfig(mode="AUTO"))
    auto.close()
    explicit = CoreNetworkClient(
        registry,
        NetworkConfig(
            mode="EXPLICIT_PROXY", explicit_proxy_url="http://127.0.0.1:8080"
        ),
    )
    explicit.close()
    assert calls[0]["trust_env"] is True
    assert calls[0]["verify"] is True
    assert calls[1]["trust_env"] is False
    assert calls[1]["proxy"] == "http://127.0.0.1:8080"


def test_network_timeout_policy_is_bounded() -> None:
    config = NetworkConfig()
    assert config.timeout.connect_timeout > 0
    assert config.timeout.read_timeout > 0
    assert config.timeout.total_timeout > 0


def test_official_html_fetch_is_endpoint_allowlisted_and_size_bounded() -> None:
    class HTMLResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers = {"Content-Type": "text/html; charset=utf-8"}
            self.url = "https://data.example.test/fixtures"
            self.encoding = "utf-8"
            self.content = b"<html>official</html>"

    class HTMLClient:
        def get(self, url, *, headers):
            assert url == "https://data.example.test/fixtures"
            return HTMLResponse()

    definition = SourceDefinition(
        source_id="official-test",
        display_name="Official test",
        category="OFFICIAL",
        base_url="https://data.example.test",
        endpoints={"fixtures": "https://data.example.test/fixtures"},
        schema_version="official-html-v1",
    )
    client = CoreNetworkClient(
        ExternalSourceRegistry((definition,)), http_client=HTMLClient()
    )
    result = client.fetch_text("official-test", "fixtures", max_bytes=100)
    assert result.body == "<html>official</html>"
    assert result.final_url == "https://data.example.test/fixtures"
    with pytest.raises(ValueError, match="ENDPOINT_NOT_REGISTERED"):
        client.fetch_text("official-test", "arbitrary")
    with pytest.raises(ValueError, match="CONTENT_LIMIT_EXCEEDED"):
        client.fetch_text("official-test", "fixtures", max_bytes=2)


def test_retry_503_then_success() -> None:
    responses = [FakeResponse(503), FakeResponse(200, body())]
    fake = FakeHTTPClient(responses)
    client = CoreNetworkClient(ExternalSourceRegistry((source(),)), http_client=fake)
    result = client.fetch_json(
        "fixture-source", "matches", params={}, prediction_time=now()
    )
    assert result.status_code == 200
    assert len(fake.calls) == 2


def test_no_retry_401(monkeypatch) -> None:
    monkeypatch.setenv("FIXTURE_API_TOKEN", "test-secret")
    fake = FakeHTTPClient([FakeResponse(401), FakeResponse(200, body())])
    client = CoreNetworkClient(ExternalSourceRegistry((source(),)), http_client=fake)
    with pytest.raises(ValueError, match="AUTH_FAILED"):
        client.fetch_json("fixture-source", "matches", params={}, prediction_time=now())
    assert len(fake.calls) == 1


def test_rate_limit_policy_and_source_pacing() -> None:
    policy = RateLimitPolicy(requests_per_second=20, requests_per_minute=100, burst=2)
    assert policy.burst == 2
    client = CoreNetworkClient(
        ExternalSourceRegistry((source(),)),
        http_client=FakeHTTPClient(
            [FakeResponse(200, body()), FakeResponse(200, body())]
        ),
    )
    client.fetch_json(
        "fixture-source", "matches", params={"a": 1}, prediction_time=now()
    )
    client.fetch_json(
        "fixture-source", "matches", params={"a": 2}, prediction_time=now()
    )


def test_network_cache_fresh_stale_and_key_versioning() -> None:
    cache = NetworkCache()
    at = now()
    params_hash = canonical_hash({"team": "t"})
    key = cache.key("s", "e", params_hash, "schema-v1")
    payload = CachedPayload(
        cache_key=key,
        source_id="s",
        endpoint_id="e",
        canonical_params_hash=params_hash,
        schema_version="schema-v1",
        created_at=at,
        expires_at=at + timedelta(seconds=60),
        source_timestamp=at,
        content_hash="hash",
        payload={"x": 1},
    )
    cache.put(payload)
    assert cache.get(key, at + timedelta(seconds=1))[1] == CacheStatus.FRESH
    assert cache.get(key, at + timedelta(seconds=61))[1] == CacheStatus.EXPIRED
    assert cache.key("s", "e", params_hash, "schema-v2") != key


def test_network_cache_hit_is_audited() -> None:
    fake = FakeHTTPClient([FakeResponse(200, body())])
    client = CoreNetworkClient(ExternalSourceRegistry((source(),)), http_client=fake)
    at = now()
    first = client.fetch_json(
        "fixture-source", "matches", params={"a": 1}, prediction_time=at
    )
    second = client.fetch_json(
        "fixture-source",
        "matches",
        params={"a": 1},
        prediction_time=at + timedelta(seconds=1),
    )
    assert first.body == second.body
    assert len(fake.calls) == 1
    assert any(record.cache_hit for record in client.audit.records())


def test_future_cache_source_timestamp_rejected() -> None:
    fake = FakeHTTPClient(
        [
            FakeResponse(
                200,
                {
                    "data_timestamp": (now() + timedelta(days=1)).isoformat(),
                    "items": [1],
                },
            )
        ]
    )
    client = CoreNetworkClient(ExternalSourceRegistry((source(),)), http_client=fake)
    with pytest.raises(ValueError, match="POINT_IN_TIME_GUARD_V1"):
        client.fetch_json("fixture-source", "matches", params={}, prediction_time=now())


def test_health_valid_and_preflight_pass() -> None:
    definition = source(required=RequiredLevel.CRITICAL)
    fake = FakeHTTPClient([FakeResponse(200, body())])
    registry = ExternalSourceRegistry((definition,))
    client = CoreNetworkClient(registry, http_client=fake)
    result = check_source_health(
        client,
        definition.source_id,
        schema_validator=lambda value: bool(value.get("items")),
    )
    assert result.status == SourceHealthStatus.HEALTHY
    fake2 = FakeHTTPClient([FakeResponse(200, body())])
    client2 = CoreNetworkClient(registry, http_client=fake2)
    report = run_network_preflight(
        registry,
        client2,
        schema_validators={
            definition.source_id: lambda value: bool(value.get("items"))
        },
    )
    assert report.status == "PASS"


def test_health_auth_failed_and_invalid_schema(monkeypatch) -> None:
    monkeypatch.setenv("FIXTURE_API_TOKEN", "test-secret")
    definition = source(auth=True)
    registry = ExternalSourceRegistry((definition,))
    auth_client = CoreNetworkClient(
        registry, http_client=FakeHTTPClient([FakeResponse(401)])
    )
    auth_result = check_source_health(
        auth_client, definition.source_id, schema_validator=lambda _: True
    )
    assert auth_result.status == SourceHealthStatus.AUTH_FAILED
    invalid_client = CoreNetworkClient(
        registry, http_client=FakeHTTPClient([FakeResponse(200, body())])
    )
    invalid_result = check_source_health(
        invalid_client, definition.source_id, schema_validator=lambda _: False
    )
    assert invalid_result.status == SourceHealthStatus.INVALID_RESPONSE


def test_preflight_fails_closed_without_required_source() -> None:
    registry = ExternalSourceRegistry((source(required=RequiredLevel.OPTIONAL),))
    client = CoreNetworkClient(registry, http_client=FakeHTTPClient([]))
    report = run_network_preflight(registry, client, schema_validators={})
    assert report.status == "FAILED"
    assert report.reason == "NO_CRITICAL_OR_REQUIRED_SOURCES_CONFIGURED"
    assert ProductionNetworkGate().evaluate(report).status == "PREDICTION_BLOCKED"
    dev = ProductionNetworkGate().evaluate(report, offline_test_mode=True)
    assert dev.status == "DEVELOPMENT_ONLY" and not dev.production_eligible


def test_network_audit_and_secret_not_logged(monkeypatch) -> None:
    token = "do-not-log-this-token"
    monkeypatch.setenv("FIXTURE_API_TOKEN", token)
    definition = source(auth=True)
    fake = FakeHTTPClient([FakeResponse(200, body())])
    client = CoreNetworkClient(ExternalSourceRegistry((definition,)), http_client=fake)
    client.fetch_json(definition.source_id, "matches", params={}, prediction_time=now())
    audit_text = " ".join(record.model_dump_json() for record in client.audit.records())
    assert fake.calls[0]["headers"]["Authorization"] == f"Bearer {token}"
    assert token not in audit_text


def test_source_url_allowlist() -> None:
    with pytest.raises(ValueError, match="registered HTTPS host"):
        SourceDefinition(
            source_id="bad",
            display_name="bad",
            category="test",
            base_url="https://good.example",
            endpoints={"read": "https://evil.example/data"},
            schema_version="v1",
        )
    client = CoreNetworkClient(
        ExternalSourceRegistry((source(),)), http_client=FakeHTTPClient([])
    )
    with pytest.raises(ValueError, match="ENDPOINT_NOT_REGISTERED"):
        client.fetch_json(
            "fixture-source", "user-controlled", params={}, prediction_time=now()
        )
    with pytest.raises(ValueError, match="URL_SECRET_FORBIDDEN"):
        client.fetch_json(
            "fixture-source",
            "matches",
            params={"api_key": "hidden"},
            prediction_time=now(),
        )


def test_refresh_network_status_rechecks_without_vpn_control() -> None:
    definition = source(required=RequiredLevel.REQUIRED)
    registry = ExternalSourceRegistry((definition,))
    client = CoreNetworkClient(
        registry, http_client=FakeHTTPClient([FakeResponse(200, body())])
    )
    report = refresh_network_status(
        registry,
        client,
        schema_validators={definition.source_id: lambda value: "items" in value},
    )
    assert report.status == "PASS"


@pytest.mark.network_live
def test_network_live_requires_explicit_source_configuration() -> None:
    pytest.skip(
        "No legal production endpoint is configured; no real source URL is guessed"
    )
