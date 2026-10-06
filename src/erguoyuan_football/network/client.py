"""Shared synchronous HTTP transport with bounded retries, audit and PIT cache."""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import time
from datetime import timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from erguoyuan_football.contracts.common import now, utc
from erguoyuan_football.network.audit import NetworkAudit
from erguoyuan_football.network.cache import NetworkCache, canonical_hash
from erguoyuan_football.network.config import NetworkConfig, NetworkMode
from erguoyuan_football.network.errors import (
    NetworkDependencyUnavailable,
    RetryExhausted,
)
from erguoyuan_football.network.rate_limit import RateLimiter
from erguoyuan_football.network.schemas import (
    CachedPayload,
    NetworkAuditRecord,
    NetworkResponse,
)
from erguoyuan_football.network.source_registry import ExternalSourceRegistry


def _quota_header(headers: Any, name: str) -> int | None:
    """Parse safe numeric quota metadata without retaining response headers."""
    if headers is None:
        return None
    try:
        value = int(headers.get(name, ""))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


class CoreNetworkClient:
    """All external HTTP requests pass through the explicit source registry."""

    def __init__(
        self,
        registry: ExternalSourceRegistry,
        config: NetworkConfig | None = None,
        *,
        http_client: Any | None = None,
        cache: NetworkCache | None = None,
        audit: NetworkAudit | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or NetworkConfig()
        self.cache = cache or NetworkCache()
        self.audit = audit or NetworkAudit()
        self.rate_limiter = RateLimiter()
        self._owns_client = http_client is None
        self._client = http_client or self._make_http_client()

    def _make_http_client(self) -> Any:
        try:
            import httpx
        except ImportError as error:
            raise NetworkDependencyUnavailable(
                "Install the declared httpx dependency.", cause=error
            ) from error
        timeout = httpx.Timeout(
            connect=self.config.timeout.connect_timeout,
            read=min(
                self.config.timeout.read_timeout, self.config.timeout.total_timeout
            ),
            write=min(
                self.config.timeout.read_timeout, self.config.timeout.total_timeout
            ),
            pool=self.config.timeout.connect_timeout,
        )
        options: dict[str, Any] = {
            "timeout": timeout,
            "verify": True,
            "trust_env": self.config.mode == NetworkMode.AUTO,
        }
        if self.config.mode == NetworkMode.EXPLICIT_PROXY:
            options["proxy"] = self.config.explicit_proxy_url
            options["trust_env"] = False
        return httpx.Client(**options)

    def close(self) -> None:
        """Close the shared pooled client when owned by this wrapper."""
        if self._owns_client:
            self._client.close()

    @staticmethod
    def _retry_after(response: Any, fallback: float) -> float:
        value = response.headers.get("Retry-After")
        if not value:
            return fallback
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(value) - now()).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return fallback

    def fetch_json(
        self,
        source_id: str,
        endpoint_id: str,
        *,
        params: dict[str, Any],
        prediction_time=None,
        as_of_time=None,
        allow_stale: bool = False,
        bypass_cache: bool = False,
    ) -> NetworkResponse:
        """Fetch a registered JSON endpoint, respecting source PIT and retry policy."""
        source = self.registry.get(source_id)
        if endpoint_id not in source.endpoints:
            raise ValueError(f"ENDPOINT_NOT_REGISTERED:{source_id}:{endpoint_id}")
        sensitive_keys = {
            "api_key",
            "apikey",
            "token",
            "password",
            "authorization",
            "secret",
        }
        if any(key.lower() in sensitive_keys for key in params):
            raise ValueError(
                "URL_SECRET_FORBIDDEN:credentials must come from declared environment variables"
            )
        endpoint = source.endpoints[endpoint_id]
        at = utc(prediction_time) if prediction_time is not None else now()
        params_hash = canonical_hash(params)
        key = self.cache.key(source_id, endpoint_id, params_hash, source.schema_version)
        cached = None if bypass_cache else self.cache.get(key, at)
        if cached and cached[1].value == "FRESH":
            item, status = cached
            cached_at = now()
            self._record(
                source_id,
                endpoint_id,
                cached_at,
                cached_at,
                200,
                0,
                item.content_hash,
                source.schema_version,
                "CACHE_HIT",
                item.source_timestamp,
                cache_hit=True,
            )
            return NetworkResponse(
                source_id=source_id,
                endpoint_id=endpoint_id,
                status_code=200,
                body=item.payload,
                retrieved_at=item.created_at,
                as_of_time=item.source_timestamp,
                content_hash=item.content_hash,
                schema_version=source.schema_version,
                cache_status=status,
                latency_ms=0,
                final_url=endpoint,
                retry_count=0,
            )
        auth_value = (
            os.environ.get(source.auth_env_var) if source.auth_env_var else None
        )
        if source.requires_auth and not auth_value:
            self._audit_failure(
                source_id, endpoint_id, source.schema_version, "AUTH_FAILED"
            )
            raise ValueError(f"AUTH_FAILED:{source.auth_env_var}")
        headers = {
            "Accept": "application/json",
            "User-Agent": "CORE-Football-Research/13.4",
        }
        if auth_value and source.auth_query_name is None:
            if source.auth_header_name == "X-Auth-Token":
                headers["X-Auth-Token"] = auth_value
            else:
                headers["Authorization"] = f"Bearer {auth_value}"
        self.rate_limiter.acquire(source_id, source.rate_limit_policy)
        started = time.monotonic()
        requested_at = now()
        response = None
        request_params = {**params, source.auth_query_name: auth_value} if (
            auth_value and source.auth_query_name is not None) else params
        final_url = endpoint
        attempt = 0
        last_error: Exception | None = None
        while attempt < source.retry_policy.max_attempts:
            attempt += 1
            try:
                request_url = endpoint
                for redirect_count in range(4):
                    current_response = self._client.get(
                        request_url,
                        params=request_params if redirect_count == 0 else {},
                        headers=headers,
                    )
                    response = current_response
                    if current_response.status_code not in {301, 302, 303, 307, 308}:
                        break
                    location = current_response.headers.get("Location")
                    if not location or redirect_count == 3:
                        raise ValueError(
                            "INVALID_RESPONSE:redirect limit or Location missing"
                        )
                    next_url = urljoin(request_url, location)
                    parsed_next = urlparse(next_url)
                    parsed_base = urlparse(source.base_url or "")
                    if (
                        parsed_next.scheme != "https"
                        or parsed_next.hostname != parsed_base.hostname
                        or parsed_next.username
                        or parsed_next.password
                        or (source.auth_query_name is not None and parsed_next.query)
                    ):
                        self._audit_failure(
                            source_id,
                            endpoint_id,
                            source.schema_version,
                            "SOURCE_NOT_ALLOWED",
                        )
                        raise ValueError("SOURCE_NOT_ALLOWED:redirect domain")
                    request_url = next_url
                final_url = request_url
                assert response is not None
                if response.status_code not in source.retry_policy.retry_status_codes:
                    break
                if attempt >= source.retry_policy.max_attempts:
                    break
                fallback = min(
                    source.retry_policy.max_backoff_seconds,
                    source.retry_policy.initial_backoff_seconds * (2 ** (attempt - 1)),
                )
                if response.status_code == 429:
                    wait_seconds = self._retry_after(response, fallback)
                    if wait_seconds > source.retry_policy.max_backoff_seconds:
                        break
                else:
                    wait_seconds = fallback
                time.sleep(wait_seconds)
            except Exception as error:
                retryable = isinstance(
                    error, (ConnectionError, TimeoutError, socket.timeout, OSError)
                )
                try:
                    import httpx

                    retryable = retryable or isinstance(error, httpx.TransportError)
                except ImportError:
                    pass
                if not retryable:
                    if source.auth_query_name is not None:
                        raise ValueError(f"AUTH_QUERY_REQUEST_FAILED:{type(error).__name__}") from None
                    raise
                last_error = error
                if attempt >= source.retry_policy.max_attempts:
                    response_at = now()
                    elapsed_ms = (time.monotonic() - started) * 1000
                    self._record(
                        source_id,
                        endpoint_id,
                        requested_at,
                        response_at,
                        None,
                        elapsed_ms,
                        None,
                        source.schema_version,
                        "TRANSPORT_RETRY_EXHAUSTED",
                    )
                    if source.auth_query_name is not None:
                        raise RetryExhausted(
                            f"{type(error).__name__} after {attempt} attempts",
                            attempts=attempt,
                        ) from None
                    raise RetryExhausted(
                        f"{type(error).__name__} after {attempt} attempts",
                        cause=error,
                        attempts=attempt,
                    ) from error
                delay = min(
                    source.retry_policy.max_backoff_seconds,
                    source.retry_policy.initial_backoff_seconds * (2 ** (attempt - 1)),
                )
                time.sleep(
                    delay + random.Random(attempt).uniform(0, min(delay * 0.1, 0.05))
                )
        if response is None:
            raise RetryExhausted(
                "request returned no response", cause=last_error, attempts=attempt
            )
        assert response is not None
        received_at = now()
        elapsed_ms = (time.monotonic() - started) * 1000
        if elapsed_ms > self.config.timeout.total_timeout * 1000:
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                response.status_code,
                elapsed_ms,
                None,
                source.schema_version,
                "TOTAL_TIMEOUT",
            )
            raise TimeoutError("NETWORK_TOTAL_TIMEOUT")
        if response.status_code in {401, 403}:
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                response.status_code,
                elapsed_ms,
                None,
                source.schema_version,
                "AUTH_FAILED",
            )
            raise ValueError(f"AUTH_FAILED:{response.status_code}")
        if response.status_code == 429:
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                429,
                elapsed_ms,
                None,
                source.schema_version,
                "RATE_LIMITED",
            )
            raise RetryExhausted(
                "RATE_LIMITED: retry budget exhausted", attempts=attempt
            )
        if response.status_code in source.retry_policy.retry_status_codes:
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                response.status_code,
                elapsed_ms,
                None,
                source.schema_version,
                "RETRY_EXHAUSTED",
            )
            raise RetryExhausted(
                f"RETRY_EXHAUSTED: HTTP {response.status_code}", attempts=attempt
            )
        if response.status_code >= 400:
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                response.status_code,
                elapsed_ms,
                None,
                source.schema_version,
                "HTTP_ERROR",
            )
            raise ValueError(f"HTTP_ERROR:{response.status_code}")
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError) as error:
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                response.status_code,
                elapsed_ms,
                None,
                source.schema_version,
                "INVALID_RESPONSE",
            )
            raise ValueError("INVALID_RESPONSE:expected JSON") from error
        if body is None or (not source.allow_empty_json and body in ({}, [])):
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                response.status_code,
                elapsed_ms,
                None,
                source.schema_version,
                "INVALID_RESPONSE",
            )
            raise ValueError("INVALID_RESPONSE:empty body")
        content_hash = hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode()
        ).hexdigest()
        supplied_source_time = as_of_time
        if supplied_source_time is None and source.source_timestamp_field:
            raw_timestamp = (
                body.get(source.source_timestamp_field)
                if isinstance(body, dict)
                else None
            )
            if not isinstance(raw_timestamp, str):
                self._record(
                    source_id,
                    endpoint_id,
                    requested_at,
                    received_at,
                    response.status_code,
                    elapsed_ms,
                    content_hash,
                    source.schema_version,
                    "INVALID_RESPONSE",
                )
                raise ValueError("INVALID_RESPONSE:declared source timestamp missing")
            from datetime import datetime

            try:
                supplied_source_time = datetime.fromisoformat(raw_timestamp)
            except ValueError as error:
                self._record(
                    source_id,
                    endpoint_id,
                    requested_at,
                    received_at,
                    response.status_code,
                    elapsed_ms,
                    content_hash,
                    source.schema_version,
                    "INVALID_RESPONSE",
                )
                raise ValueError(
                    "INVALID_RESPONSE:declared source timestamp malformed"
                ) from error
        source_time = (
            utc(supplied_source_time)
            if supplied_source_time is not None
            else received_at
        )
        cutoff = received_at if prediction_time is None else at
        if source_time > cutoff:
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                response.status_code,
                elapsed_ms,
                content_hash,
                source.schema_version,
                "FUTURE_DATA_REJECTED",
                source_time,
            )
            raise ValueError(
                "POINT_IN_TIME_GUARD_V1:source timestamp after prediction time"
            )
        self._record(
            source_id,
            endpoint_id,
            requested_at,
            received_at,
            response.status_code,
            elapsed_ms,
            content_hash,
            source.schema_version,
            "SUCCESS",
            source_time,
        )
        ttl = int(source.cache_policy.get("ttl_seconds", self.config.cache_ttl_seconds))
        if ttl > 0:
            self.cache.put(
                CachedPayload(
                    cache_key=key,
                    source_id=source_id,
                    endpoint_id=endpoint_id,
                    canonical_params_hash=params_hash,
                    schema_version=source.schema_version,
                    created_at=received_at,
                    expires_at=received_at + timedelta(seconds=ttl),
                    source_timestamp=source_time,
                    content_hash=content_hash,
                    payload=body,
                )
            )
        cache_status = None
        if cached and allow_stale:
            cache_status = cached[1]
        return NetworkResponse(
            source_id=source_id,
            endpoint_id=endpoint_id,
            status_code=response.status_code,
            body=body,
            retrieved_at=received_at,
            as_of_time=source_time,
            content_hash=content_hash,
            schema_version=source.schema_version,
            cache_status=cache_status,
            latency_ms=elapsed_ms,
            final_url=final_url,
            retry_count=max(0, attempt - 1),
            quota_remaining=_quota_header(response.headers, "x-requests-remaining"),
            quota_used=_quota_header(response.headers, "x-requests-used"),
            quota_last_cost=_quota_header(response.headers, "x-requests-last"),
        )

    def fetch_text(
        self,
        source_id: str,
        endpoint_id: str,
        *,
        prediction_time=None,
        max_bytes: int = 2_000_000,
    ) -> NetworkResponse:
        """Fetch bounded official HTML/text from an explicitly registered endpoint.

        This deliberately has no arbitrary-URL argument: callers can only request
        endpoint IDs already present in the source registry. Redirects remain on
        the registered host, TLS verification stays enabled, and the response is
        audited without interpreting page content as a model feature.
        """
        source = self.registry.get(source_id)
        if endpoint_id not in source.endpoints:
            raise ValueError(f"ENDPOINT_NOT_REGISTERED:{source_id}:{endpoint_id}")
        endpoint = source.endpoints[endpoint_id]
        if prediction_time is not None:
            utc(prediction_time)
        requested_at = now()
        started = time.monotonic()
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
            "User-Agent": "CORE-Football-Official-Research/1.0",
        }
        request_url = endpoint
        response = None
        base = urlparse(source.base_url or "")
        for redirect_count in range(4):
            response = self._client.get(request_url, headers=headers)
            if response.status_code not in {301, 302, 303, 307, 308}:
                break
            location = response.headers.get("Location")
            if not location or redirect_count == 3:
                raise ValueError("INVALID_RESPONSE:redirect limit or Location missing")
            next_url = urljoin(request_url, location)
            parsed_next = urlparse(next_url)
            if (
                parsed_next.scheme != "https"
                or parsed_next.hostname != base.hostname
                or parsed_next.username
                or parsed_next.password
            ):
                raise ValueError("SOURCE_NOT_ALLOWED:redirect domain")
            request_url = next_url
        assert response is not None
        received_at = now()
        elapsed_ms = (time.monotonic() - started) * 1000
        final = urlparse(str(response.url))
        if (
            final.scheme != "https"
            or final.hostname != base.hostname
            or final.username
            or final.password
        ):
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                response.status_code,
                elapsed_ms,
                None,
                source.schema_version,
                "SOURCE_NOT_ALLOWED",
            )
            raise ValueError("SOURCE_NOT_ALLOWED:redirect domain")
        if elapsed_ms > self.config.timeout.total_timeout * 1000:
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                response.status_code,
                elapsed_ms,
                None,
                source.schema_version,
                "TOTAL_TIMEOUT",
            )
            raise TimeoutError("NETWORK_TOTAL_TIMEOUT")
        if response.status_code >= 400:
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                response.status_code,
                elapsed_ms,
                None,
                source.schema_version,
                "HTTP_ERROR",
            )
            raise ValueError(f"HTTP_ERROR:{response.status_code}")
        raw = bytes(response.content)
        if len(raw) > max_bytes:
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                response.status_code,
                elapsed_ms,
                None,
                source.schema_version,
                "CONTENT_LIMIT_EXCEEDED",
            )
            raise ValueError("CONTENT_LIMIT_EXCEEDED")
        content_type = str(response.headers.get("Content-Type", "")).casefold()
        if not any(
            kind in content_type
            for kind in (
                "text/html",
                "application/xhtml+xml",
                "application/xml",
                "text/xml",
            )
        ):
            self._record(
                source_id,
                endpoint_id,
                requested_at,
                received_at,
                response.status_code,
                elapsed_ms,
                None,
                source.schema_version,
                "INVALID_RESPONSE",
            )
            raise ValueError("INVALID_RESPONSE:expected official HTML/XML")
        body = raw.decode(response.encoding or "utf-8", errors="replace")
        content_hash = hashlib.sha256(raw).hexdigest()
        self._record(
            source_id,
            endpoint_id,
            requested_at,
            received_at,
            response.status_code,
            elapsed_ms,
            content_hash,
            source.schema_version,
            "SUCCESS",
            received_at,
        )
        return NetworkResponse(
            source_id=source_id,
            endpoint_id=endpoint_id,
            status_code=response.status_code,
            body=body,
            retrieved_at=received_at,
            as_of_time=received_at,
            content_hash=content_hash,
            schema_version=source.schema_version,
            latency_ms=elapsed_ms,
            final_url=str(response.url),
            retry_count=0,
        )

    def _audit_failure(
        self, source_id: str, endpoint_id: str, schema_version: str, outcome: str
    ) -> None:
        at = now()
        self._record(
            source_id, endpoint_id, at, at, None, 0, None, schema_version, outcome
        )

    def _record(
        self,
        source_id: str,
        endpoint_id: str,
        requested_at,
        response_at,
        status_code,
        latency_ms,
        content_hash,
        schema_version,
        outcome,
        as_of_time=None,
        cache_hit: bool = False,
    ) -> None:
        self.audit.record(
            NetworkAuditRecord(
                source_id=source_id,
                endpoint_id=endpoint_id,
                requested_at=requested_at,
                response_at=response_at,
                as_of_time=as_of_time,
                status_code=status_code,
                cache_hit=cache_hit,
                latency_ms=max(0, latency_ms),
                content_hash=content_hash,
                schema_version=schema_version,
                outcome=outcome,
            )
        )
