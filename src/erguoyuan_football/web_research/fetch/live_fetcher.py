"""Cache-first, fail-closed live acquisition into the research-only evidence store."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from erguoyuan_football.network.errors import RetryExhausted
from erguoyuan_football.web_research.cache import (
    ResearchAuditRecord,
    ResearchCache,
)
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.fetch.fixture_identity import (
    FixtureExpectation,
    verify_fixture,
)
from erguoyuan_football.web_research.fetch.schema_validation import (
    validate_provider_payload,
)
from erguoyuan_football.web_research.network.readiness_gate import NetworkReadinessGate
from erguoyuan_football.web_research.policies.conflict_policy import (
    ConflictRecord,
    SourcedValue,
    resolve_conflicts,
)
from erguoyuan_football.web_research.policies.freshness_policy import FreshnessPolicy
from erguoyuan_football.web_research.providers.base_provider import BaseProvider
from erguoyuan_football.web_research.providers.capabilities import ProviderCapability
from erguoyuan_football.web_research.providers.provider_result import ProviderResult
from erguoyuan_football.web_research.providers.provider_router import ProviderRouter
from erguoyuan_football.web_research.search.search_task import SearchTask
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.time_utils import parse_utc, utc_iso


@dataclass(frozen=True)
class LiveTaskResult:
    """Research-only task outcome with explicit cache and fixture state."""

    status: str
    evidence: tuple[EvidenceRecord, ...] = ()
    conflicts: tuple[ConflictRecord, ...] = ()
    cache_status: str = "MISS"
    error_code: str | None = None
    verified_match_id: str | None = None
    fixture_candidates: tuple[dict[str, Any], ...] = ()
    selected_value: SourcedValue | None = None


class LiveResearchFetcher:
    """Fetch only registered task capabilities; never calls any predictor."""

    def __init__(
        self,
        router: ProviderRouter,
        gate: NetworkReadinessGate,
        sources: SourceRegistry,
        cache: ResearchCache,
        evidence_store: EvidenceStore,
        *,
        freshness: FreshnessPolicy | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.router = router
        self.gate = gate
        self.sources = sources
        self.cache = cache
        self.evidence_store = evidence_store
        self.freshness = freshness or FreshnessPolicy()
        self.clock = clock

    @staticmethod
    def _evidence_id(provider_id: str, task: SearchTask, result: ProviderResult) -> str:
        content = json.dumps(result.data, sort_keys=True, ensure_ascii=False,
                             allow_nan=False).encode()
        digest = hashlib.sha256(provider_id.encode() + task.task_id.encode()
                                + result.fetched_time.encode() + content).hexdigest()
        return f"WEB_{digest}"

    def _evidence(self, provider: BaseProvider, task: SearchTask,
                  result: ProviderResult, match_key: str) -> EvidenceRecord:
        claim_type = result.data.get("claim_type") if task.task_type in {"NEWS", "INJURY"} else None
        return EvidenceRecord(
            self._evidence_id(provider.provider_id, task, result),
            task.task_type, result.data, provider.provider_id,
            result.published_time, result.fetched_time, result.confidence,
            result.source_url, result.as_of_time or "",
            provider_id=provider.provider_id, source_tier=provider.source_tier,
            observed_time=result.observed_time, match_key=match_key,
            claim_type=claim_type,
        )

    def _audit(self, provider: BaseProvider, task: SearchTask, started: datetime,
               outcome: str, *, result: ProviderResult | None = None,
               error_code: str | None = None, cache_status: str = "MISS",
               retry_count: int = 0) -> None:
        endpoint_ids = getattr(provider, "endpoint_ids", {})
        endpoint_id = (result.endpoint_id if result is not None else
                       endpoint_ids.get(ProviderCapability(task.task_type)))
        query_digest = hashlib.sha256(task.query.encode()).hexdigest()
        self.cache.record_audit(ResearchAuditRecord(
            task.task_id, provider.provider_id, f"sha256:{query_digest}",
            utc_iso(started), outcome,
            cache_status == "CACHE_FRESH", result.fetched_time if result else None,
            error_code, str(uuid4()), provider.provider_id,
            endpoint_id, utc_iso(self.clock()),
            result.http_status if result else None,
            result.retry_count if result else retry_count, cache_status,
        ))

    @staticmethod
    def _failure_code(error: Exception) -> str:
        if isinstance(error, RetryExhausted) and "RATE_LIMITED" in str(error):
            return "PROVIDER_RATE_LIMIT"
        if isinstance(error, TimeoutError) or (isinstance(error, RetryExhausted)
                                               and error.__cause__ is not None
                                               and "Timeout" in type(error.__cause__).__name__):
            return "PROVIDER_TIMEOUT"
        if isinstance(error, ValueError):
            message = str(error)
            if message.startswith("AUTH_FAILED"):
                return "PROVIDER_AUTHENTICATION_FAILED"
            if message.startswith("SOURCE_NOT_ALLOWED"):
                return "SOURCE_NOT_ALLOWED"
            if message.startswith("FIXTURE_AMBIGUOUS"):
                return "FIXTURE_AMBIGUOUS"
            if message.startswith("FIXTURE_NOT_FOUND"):
                return "FIXTURE_NOT_FOUND"
            if message.startswith("REVERSE_FIXTURE"):
                return "REVERSE_FIXTURE"
            if message.startswith("OFFICIAL_LINEUP_UNAVAILABLE"):
                return "LINEUP_UNAVAILABLE"
            return "INVALID_PROVIDER_RESPONSE"
        if isinstance(error, (ConnectionError, OSError, RetryExhausted)):
            return "NETWORK_REQUEST_FAILED"
        return "PROVIDER_FAILED"

    @staticmethod
    def _comparable_value(task: SearchTask, evidence: EvidenceRecord) -> Any:
        data = evidence.value
        if task.task_type == "FIXTURE":
            return data.get("kickoff_time")
        if task.task_type == "ODDS":
            return (data.get("market_type"), data.get("selection"), data.get("bookmaker"),
                    data.get("odds"))
        if task.task_type == "TEAM_STATS":
            return (data.get("team_id"), data.get("metric"), data.get("value"))
        return data.get("text") if isinstance(data, dict) else data

    def fetch_task(
        self,
        task: SearchTask,
        *,
        match_key: str,
        fixture_expectation: FixtureExpectation | None = None,
    ) -> LiveTaskResult:
        """Use fresh cache before DNS, then bounded registered providers only."""
        if not match_key or not match_key.startswith(("RESEARCH_SESSION:", "MATCH:")):
            raise ValueError("EXPLICIT_MATCH_OR_RESEARCH_SESSION_KEY_REQUIRED")
        providers = self.router.providers_for(task)
        if not providers:
            code = ("STRUCTURED_PROVIDER_UNAVAILABLE" if task.task_type == "ODDS"
                    else "PROVIDER_UNAVAILABLE")
            return LiveTaskResult("UNAVAILABLE", error_code=code)
        now_at = self.clock()
        records: list[EvidenceRecord] = []
        stale_seen = False
        errors: list[str] = []
        cache_used = False
        verified_id: str | None = None
        fixture_candidates: tuple[dict[str, Any], ...] = ()
        gate_report = None
        for provider in providers:
            started = self.clock()
            cached = self.cache.latest(provider.provider_id, task, now_at)
            if cached is not None:
                old, cache_state, age = cached
                item = self.evidence_store.get(self._evidence_id(provider.provider_id, task, old))
                if item is not None and item.match_key != match_key:
                    item = None
                if (cache_state == "CACHE_FRESH" and old.provider_id == provider.provider_id
                        and old.observed_time is not None
                        and old.as_of_time is not None and item is not None
                        and self.freshness.is_fresh(task.task_type,
                                                    parse_utc(old.observed_time), now_at)):
                    if task.task_type == "FIXTURE" and fixture_expectation is not None:
                        checked_fixture = verify_fixture(old.data, fixture_expectation)
                        if checked_fixture.status != "VERIFIED":
                            item = None
                        else:
                            if provider.source_tier == 3:
                                fixture_candidates = checked_fixture.candidates
                                verified_id = checked_fixture.match_id
                            elif not fixture_candidates:
                                fixture_candidates = checked_fixture.candidates
                    if item is not None:
                        records.append(item)
                        cache_used = True
                        self._audit(provider, task, started, "CACHE_FRESH",
                                    result=replace(old, is_cache_hit=True,
                                                   cache_age_seconds=age,
                                                   cache_status="CACHE_FRESH"),
                                    cache_status="CACHE_FRESH")
                        continue
                stale_seen = stale_seen or item is not None
            if gate_report is None:
                gate_report = self.gate.check()
            if provider.provider_id not in gate_report.ready_provider_ids:
                errors.append("NETWORK_UNAVAILABLE")
                self._audit(provider, task, started, "NETWORK_UNAVAILABLE",
                            cache_status="CACHE_STALE" if stale_seen else "MISS",
                            error_code="NETWORK_UNAVAILABLE")
                continue
            result: ProviderResult | None = None
            try:
                result = provider.fetch(task)
                if not result.success:
                    errors.append(result.error_code or "PROVIDER_UNAVAILABLE")
                    self._audit(provider, task, started, "PROVIDER_UNAVAILABLE",
                                result=result, error_code=result.error_code)
                    continue
                source = self.sources.validate_result_url(provider.provider_id,
                                                          result.source_url)
                if (result.provider_id != provider.provider_id
                        or result.source_tier != source.tier):
                    raise ValueError("PROVIDER_SOURCE_ID_OR_TIER_MISMATCH")
                finished = self.clock()
                if (result.fetched_at is None or result.fetched_at > finished
                        or result.observed_time is None or not self.freshness.is_fresh(
                            task.task_type, parse_utc(result.observed_time), finished)):
                    raise ValueError("STALE_OR_FUTURE_SOURCE_DATA")
                verification = validate_provider_payload(
                    task, result, fixture_expectation=fixture_expectation
                )
                if verification is not None:
                    if source.tier == 3:
                        fixture_candidates = verification.candidates
                        verified_id = verification.match_id
                    elif not fixture_candidates:
                        fixture_candidates = verification.candidates
                    result = replace(result, data=verification.candidates[0])
                item = self._evidence(provider, task, result, match_key)
                self.evidence_store.save(item)
                self.evidence_store.link_to_task(item.evidence_id, task.task_id)
                self.cache.put(provider.provider_id, task, result, finished,
                               self.freshness.ttl(task.task_type))
                records.append(item)
                self._audit(provider, task, started, "LIVE_SUCCESS", result=result,
                            cache_status="LIVE")
            except Exception as error:  # noqa: BLE001 - isolate and audit each provider failure
                code = self._failure_code(error)
                if (code in {"FIXTURE_AMBIGUOUS", "REVERSE_FIXTURE", "FIXTURE_NOT_FOUND"}
                        and task.task_type == "FIXTURE" and fixture_expectation is not None
                        and result is not None and result.success):
                    fixture_candidates = verify_fixture(
                        result.data, fixture_expectation).candidates
                errors.append(code)
                self._audit(provider, task, started, "FETCH_FAILED", error_code=code,
                            retry_count=max(0, (error.attempts or 1) - 1)
                            if isinstance(error, RetryExhausted) else 0)
        if records:
            decision = resolve_conflicts(tuple(SourcedValue(
                self._comparable_value(task, item), item.source_id, item.source_tier or 0
            ) for item in records))
            if decision.status == "CONFLICT_UNRESOLVED":
                status = "CONFLICT_UNRESOLVED"
            else:
                status = "CACHE_FRESH" if cache_used and len(records) == 1 else "LIVE"
            return LiveTaskResult(status, tuple(records), decision.conflicts,
                                  "CACHE_FRESH" if cache_used and len(records) == 1 else "LIVE",
                                  verified_match_id=verified_id,
                                  fixture_candidates=fixture_candidates,
                                  selected_value=decision.selected)
        if stale_seen:
            return LiveTaskResult("CACHE_STALE", cache_status="CACHE_STALE",
                                  error_code=errors[0] if errors else "STALE_CACHE",
                                  fixture_candidates=fixture_candidates)
        return LiveTaskResult("UNAVAILABLE", error_code=errors[0] if errors else "NO_DATA",
                              fixture_candidates=fixture_candidates)
