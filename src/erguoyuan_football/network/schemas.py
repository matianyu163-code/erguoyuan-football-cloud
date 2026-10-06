"""Audit-safe source, response, health and cache records."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import Contract, UTCTime
from erguoyuan_football.network.config import (
    RateLimitPolicy,
    RetryPolicy,
    TimeoutPolicy,
)


class RequiredLevel(StrEnum):
    CRITICAL = "CRITICAL"
    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"


class SourceHealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    AUTH_FAILED = "AUTH_FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    STALE = "STALE"


class CacheStatus(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    EXPIRED = "EXPIRED"


class SourceDefinition(Contract):
    """Allowlisted source metadata. Unknown endpoints remain absent, never guessed."""

    source_id: str
    display_name: str
    category: str
    base_url: str | None = None
    endpoints: dict[str, str] = Field(default_factory=dict)
    required_level: RequiredLevel = RequiredLevel.OPTIONAL
    requires_auth: bool = False
    supports_history: bool = False
    supports_live: bool = False
    timeout_policy: TimeoutPolicy = Field(default_factory=TimeoutPolicy)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    cache_policy: dict[str, int] = Field(default_factory=lambda: {"ttl_seconds": 900})
    rate_limit_policy: RateLimitPolicy = Field(default_factory=RateLimitPolicy)
    schema_version: str
    auth_env_var: str | None = None
    auth_header_name: Literal["Authorization", "X-Auth-Token"] = "Authorization"
    auth_query_name: Literal["apiKey"] | None = None
    source_timestamp_field: str | None = None
    allow_empty_json: bool = False

    @model_validator(mode="after")
    def safe_allowlist(self) -> SourceDefinition:
        if self.base_url is None:
            if self.endpoints:
                raise ValueError("endpoints require an explicitly configured base_url")
            return self
        parsed_base = urlparse(self.base_url)
        if (parsed_base.scheme != "https" or not parsed_base.hostname or parsed_base.username
                or parsed_base.password or parsed_base.query or parsed_base.fragment):
            raise ValueError("source base_url must be an explicit HTTPS URL")
        for endpoint in self.endpoints.values():
            parsed = urlparse(endpoint)
            if (parsed.scheme != "https" or parsed.netloc != parsed_base.netloc or parsed.username
                    or parsed.password or parsed.fragment):
                raise ValueError("endpoint must use the registered HTTPS host")
        if self.requires_auth and not self.auth_env_var:
            raise ValueError("authenticated source requires an environment variable name")
        return self


class NetworkAuditRecord(Contract):
    """One request event; credentials are deliberately not represented."""

    source_id: str
    endpoint_id: str
    requested_at: UTCTime
    response_at: UTCTime
    as_of_time: UTCTime | None = None
    status_code: int | None = None
    cache_hit: bool = False
    latency_ms: float = Field(ge=0)
    content_hash: str | None = None
    schema_version: str
    outcome: str


class CachedPayload(Contract):
    """Content-addressed response with explicit temporal bounds."""

    cache_key: str
    source_id: str
    endpoint_id: str
    canonical_params_hash: str
    schema_version: str
    created_at: UTCTime
    expires_at: UTCTime
    source_timestamp: UTCTime | None
    content_hash: str
    payload: Any


class SourceHealth(Contract):
    """Validated health result, including response and freshness evidence."""

    source_id: str
    status: SourceHealthStatus
    checked_at: UTCTime
    latency_ms: float | None = Field(default=None, ge=0)
    http_status: int | None = None
    authentication_status: str
    schema_valid: bool | None = None
    response_non_empty: bool | None = None
    data_timestamp: UTCTime | None = None
    last_success_at: UTCTime | None = None
    reason: str | None = None


class NetworkHealthReport(Contract):
    """Aggregate network status. Empty configured-source sets fail closed."""

    status: Literal["PASS", "DEGRADED", "FAILED"]
    checked_at: UTCTime
    sources: tuple[SourceHealth, ...]
    reason: str | None = None


class NetworkResponse(Contract):
    """Transport result with body, source timestamp and cache freshness separated."""

    source_id: str
    endpoint_id: str
    status_code: int
    body: Any
    retrieved_at: UTCTime
    as_of_time: UTCTime | None = None
    content_hash: str
    schema_version: str
    cache_status: CacheStatus | None = None
    latency_ms: float = Field(ge=0)
    final_url: str | None = None
    retry_count: int = 0
    quota_remaining: int | None = None
    quota_used: int | None = None
    quota_last_cost: int | None = None
