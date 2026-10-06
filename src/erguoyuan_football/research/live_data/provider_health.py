"""Classify actual provider transport and schema outcomes without another HTTP stack."""

from __future__ import annotations

from dataclasses import dataclass

from erguoyuan_football.research.live_data.openligadb import SeasonBatch


@dataclass(frozen=True)
class ProviderHealthReport:
    """One observed live response, or an explicit unavailable/failure state."""

    provider_id: str
    configured: bool
    reachable: bool
    authenticated: bool | None
    schema_valid: bool
    latency_ms: int | None
    status: str
    failures: tuple[str, ...]
    http_status: int | None = None


class ProviderHealthChecker:
    """Use a fetched batch as evidence of health; never infer reachability from DNS."""

    @staticmethod
    def check(provider_id: str, configured: bool, batch: SeasonBatch | None,
              error_code: str | None = None) -> ProviderHealthReport:
        """Return READY only after a successful typed real response."""
        if not configured:
            return ProviderHealthReport(provider_id, False, False, None, False,
                                        None, "UNCONFIGURED", ("ENDPOINT_NOT_CONFIGURED",))
        if batch is None:
            code = error_code or "PROVIDER_UNREACHABLE"
            status = ("AUTH_FAILED" if "AUTH" in code else
                      "INVALID_SCHEMA" if "SCHEMA" in code or "INVALID" in code
                      else "UNREACHABLE")
            return ProviderHealthReport(provider_id, True, False, None, False,
                                        None, status, (code,))
        if batch.http_status != 200 or not batch.matches:
            return ProviderHealthReport(provider_id, True, True, None, False,
                                        round(batch.latency_ms), "INVALID_SCHEMA",
                                        ("EMPTY_OR_BAD_HTTP_RESPONSE",), batch.http_status)
        return ProviderHealthReport(provider_id, True, True, None, True,
                                    round(batch.latency_ms), "READY", (), batch.http_status)
