"""Source-aware preflight; it probes CORE registered sources, never unrelated websites."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from erguoyuan_football.contracts.common import now
from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.errors import RetryExhausted
from erguoyuan_football.network.schemas import (
    NetworkHealthReport,
    SourceHealth,
    SourceHealthStatus,
)
from erguoyuan_football.network.source_registry import ExternalSourceRegistry


def check_source_health(client: CoreNetworkClient, source_id: str, *,
                        schema_validator: Callable[[Any], bool] | None,
                        max_age_seconds: int = 3600) -> SourceHealth:
    """Check actual endpoint, auth, JSON shape, nonempty body and freshness."""
    source = client.registry.get(source_id)
    checked_at = now()
    if "health" not in source.endpoints or schema_validator is None or not source.source_timestamp_field:
        return SourceHealth(source_id=source_id, status=SourceHealthStatus.UNAVAILABLE,
            checked_at=checked_at, authentication_status="NOT_CHECKED", schema_valid=None,
            response_non_empty=None, reason="HEALTH_ENDPOINT_SCHEMA_OR_TIMESTAMP_NOT_CONFIGURED")
    started = time.monotonic()
    try:
        response = client.fetch_json(source_id, "health", params={}, prediction_time=checked_at,
                                     bypass_cache=True)
    except ValueError as error:
        message = str(error)
        status = SourceHealthStatus.AUTH_FAILED if message.startswith("AUTH_FAILED") else (
            SourceHealthStatus.INVALID_RESPONSE if message.startswith("INVALID_RESPONSE") else SourceHealthStatus.UNAVAILABLE)
        return SourceHealth(source_id=source_id, status=status, checked_at=checked_at,
            latency_ms=(time.monotonic() - started) * 1000, authentication_status="FAILED" if status == SourceHealthStatus.AUTH_FAILED else "UNKNOWN",
            schema_valid=False if status == SourceHealthStatus.INVALID_RESPONSE else None,
            response_non_empty=False if status == SourceHealthStatus.INVALID_RESPONSE else None, reason=message)
    except RetryExhausted as error:
        status = SourceHealthStatus.RATE_LIMITED if "RATE_LIMITED" in str(error) else SourceHealthStatus.UNAVAILABLE
        return SourceHealth(source_id=source_id, status=status, checked_at=checked_at,
            latency_ms=(time.monotonic() - started) * 1000, authentication_status="UNKNOWN", reason=f"{error.code}:{error}")
    except (TimeoutError, ConnectionError, OSError, RuntimeError) as error:
        return SourceHealth(source_id=source_id, status=SourceHealthStatus.UNAVAILABLE,
            checked_at=checked_at, latency_ms=(time.monotonic() - started) * 1000,
            authentication_status="UNKNOWN", reason=f"{type(error).__name__}:{error}")
    valid = bool(schema_validator(response.body))
    if not valid:
        return SourceHealth(source_id=source_id, status=SourceHealthStatus.INVALID_RESPONSE,
            checked_at=checked_at, latency_ms=response.latency_ms, http_status=response.status_code,
            authentication_status="PASSED", schema_valid=False, response_non_empty=True,
            data_timestamp=response.as_of_time, reason="RESPONSE_SCHEMA_INVALID")
    stale = response.as_of_time is None or checked_at - response.as_of_time > timedelta(seconds=max_age_seconds)
    return SourceHealth(source_id=source_id,
        status=SourceHealthStatus.STALE if stale else SourceHealthStatus.HEALTHY,
        checked_at=checked_at, latency_ms=response.latency_ms, http_status=response.status_code,
        authentication_status="PASSED", schema_valid=True, response_non_empty=True,
        data_timestamp=response.as_of_time, last_success_at=response.retrieved_at,
        reason="SOURCE_DATA_STALE" if stale else None)


def run_network_preflight(registry: ExternalSourceRegistry, client: CoreNetworkClient, *,
                          schema_validators: dict[str, Callable[[Any], bool]],
                          max_age_seconds: int = 3600) -> NetworkHealthReport:
    """Run checks for configured critical/required sources and fail closed if none exist."""
    required = registry.required()
    checked_at = now()
    if not required:
        return NetworkHealthReport(status="FAILED", checked_at=checked_at, sources=(),
                                   reason="NO_CRITICAL_OR_REQUIRED_SOURCES_CONFIGURED")
    results = tuple(check_source_health(client, source.source_id,
        schema_validator=schema_validators.get(source.source_id), max_age_seconds=max_age_seconds)
        for source in required)
    required_failed = [item for item, source in zip(results, required, strict=True)
                       if item.status != SourceHealthStatus.HEALTHY]
    if required_failed:
        return NetworkHealthReport(status="FAILED", checked_at=checked_at, sources=results,
                                   reason="REQUIRED_SOURCE_PREFLIGHT_FAILED")
    return NetworkHealthReport(status="PASS", checked_at=checked_at, sources=results)


def refresh_network_status(registry: ExternalSourceRegistry, client: CoreNetworkClient, *,
                           schema_validators: dict[str, Callable[[Any], bool]],
                           max_age_seconds: int = 3600) -> NetworkHealthReport:
    """Repeat live source checks without restarting CORE or modifying networking."""
    return run_network_preflight(registry, client, schema_validators=schema_validators,
                                 max_age_seconds=max_age_seconds)
