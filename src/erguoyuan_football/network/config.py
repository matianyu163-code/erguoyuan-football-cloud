"""Network configuration keeps transport policy separate from model hashes."""

from __future__ import annotations

from enum import StrEnum
from urllib.parse import urlparse

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import Contract


class NetworkMode(StrEnum):
    """Supported Windows transport choices."""

    AUTO = "AUTO"
    DIRECT = "DIRECT"
    EXPLICIT_PROXY = "EXPLICIT_PROXY"


class RetryPolicy(Contract):
    """Finite retry policy for transient network failures only."""

    max_attempts: int = Field(default=3, ge=1, le=8)
    initial_backoff_seconds: float = Field(default=0.2, ge=0, le=10)
    max_backoff_seconds: float = Field(default=2.0, ge=0, le=30)
    retry_status_codes: tuple[int, ...] = (429, 502, 503, 504)

    @model_validator(mode="after")
    def transient_only(self) -> RetryPolicy:
        if any(code not in {429, 502, 503, 504} for code in self.retry_status_codes):
            raise ValueError("only 429, 502, 503 and 504 may be retried")
        return self


class TimeoutPolicy(Contract):
    """Bound connect, read and total request duration."""

    connect_timeout: float = Field(default=5.0, gt=0, le=60)
    read_timeout: float = Field(default=15.0, gt=0, le=120)
    total_timeout: float = Field(default=25.0, gt=0, le=180)


class RateLimitPolicy(Contract):
    """Per-source request pacing."""

    requests_per_second: float = Field(default=2.0, gt=0, le=1000)
    requests_per_minute: int = Field(default=60, ge=1, le=60000)
    burst: int = Field(default=2, ge=1, le=1000)


class NetworkConfig(Contract):
    """Network-only settings; AUTO respects existing OS routes and HTTPX env proxies."""

    mode: NetworkMode = NetworkMode.AUTO
    explicit_proxy_url: str | None = None
    verify_tls: bool = True
    timeout: TimeoutPolicy = Field(default_factory=TimeoutPolicy)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    default_rate_limit: RateLimitPolicy = Field(default_factory=RateLimitPolicy)
    cache_ttl_seconds: int = Field(default=900, ge=0, le=86400)

    @model_validator(mode="after")
    def proxy_policy(self) -> NetworkConfig:
        """Reject ambiguous, malformed, and unsupported proxy configuration."""
        if not self.verify_tls:
            raise ValueError("TLS verification cannot be disabled")
        if self.mode == NetworkMode.EXPLICIT_PROXY and not self.explicit_proxy_url:
            raise ValueError("EXPLICIT_PROXY requires explicit_proxy_url")
        if self.explicit_proxy_url:
            parsed = urlparse(self.explicit_proxy_url)
            if parsed.scheme not in {"http", "https", "socks5", "socks5h"} or not parsed.hostname:
                raise ValueError("invalid explicit proxy URL")
            if parsed.scheme.startswith("socks"):
                raise ValueError("PROXY_CONFIGURATION_UNSUPPORTED: install and verify HTTPX SOCKS support first")
        return self
