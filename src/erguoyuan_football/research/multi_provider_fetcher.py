"""Bounded provider fallback with validation and per-attempt lineage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from erguoyuan_football.network.errors import NetworkError
from erguoyuan_football.research.global_provider_registry import RegisteredProvider


@dataclass(frozen=True)
class ProviderAttempt:
    """Outcome of one explicit provider attempt."""

    provider_id: str
    status: str
    evidence_ids: tuple[str, ...]
    error_code: str | None = None


@dataclass(frozen=True)
class MultiProviderResult:
    """First validated result and all preceding audit attempts."""

    status: str
    provider_id: str | None
    data: Any | None
    evidence_ids: tuple[str, ...]
    attempts: tuple[ProviderAttempt, ...]


class MultiProviderFetcher:
    """Call only registered providers, in resolver order, with a hard attempt limit."""

    def __init__(self, max_providers_per_task: int = 3) -> None:
        if max_providers_per_task < 1:
            raise ValueError("MAX_PROVIDERS_MUST_BE_POSITIVE")
        self.max_providers_per_task = max_providers_per_task

    def fetch(self, providers: tuple[RegisteredProvider, ...],
              fetchers: dict[str, Callable[[], Any]],
              validators: dict[str, Callable[[Any], bool]]) -> MultiProviderResult:
        """Fallback on classified provider errors or invalid payloads; no silent success."""
        attempts: list[ProviderAttempt] = []
        tried = providers[:self.max_providers_per_task]
        for registered in tried:
            provider_id = registered.profile.provider_id
            fetch = fetchers.get(provider_id)
            validate = validators.get(provider_id)
            if fetch is None or validate is None:
                attempts.append(ProviderAttempt(provider_id, "NOT_CONFIGURED", (),
                                                "FETCH_OR_VALIDATOR_MISSING"))
                continue
            try:
                data = fetch()
                valid = validate(data)
            except (NetworkError, OSError, TimeoutError, ValueError, TypeError,
                    RuntimeError) as error:
                code = type(error).__name__
                attempts.append(ProviderAttempt(provider_id, "FAILED", (), code))
                continue
            if not valid:
                attempts.append(ProviderAttempt(provider_id, "INVALID", (), "PAYLOAD_VALIDATION_FAILED"))
                continue
            evidence = tuple(getattr(data, "evidence_ids", ()))
            attempts.append(ProviderAttempt(provider_id, "SUCCESS", evidence))
            return MultiProviderResult("AVAILABLE", provider_id, data, evidence, tuple(attempts))
        return MultiProviderResult("UNAVAILABLE", None, None, (), tuple(attempts))
