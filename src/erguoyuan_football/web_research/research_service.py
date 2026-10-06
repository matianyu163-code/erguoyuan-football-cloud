"""Provider-independent source gate, persistent cache and research audit."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from erguoyuan_football.web_research.cache import ResearchAuditRecord, ResearchCache
from erguoyuan_football.web_research.providers.base_provider import BaseProvider
from erguoyuan_football.web_research.providers.provider_result import ProviderResult
from erguoyuan_football.web_research.search.search_task import SearchTask
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.time_utils import parse_utc, utc_iso


class ResearchService:
    """Call only explicitly supplied providers; no live provider ships in this phase."""

    def __init__(
        self,
        sources: SourceRegistry,
        cache: ResearchCache,
        *,
        ttl: timedelta = timedelta(minutes=15),
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if ttl.total_seconds() <= 0:
            raise ValueError("POSITIVE_CACHE_TTL_REQUIRED")
        self.sources = sources
        self.cache = cache
        self.ttl = ttl
        self.clock = clock

    def execute(self, task: SearchTask, source_id: str,
                provider: BaseProvider) -> ProviderResult:
        """Fetch or reuse evidence with source, tier, time and audit checks."""
        source = self.sources.get(source_id)
        if source.source_type not in task.required_sources:
            raise ValueError("SOURCE_TYPE_NOT_ALLOWED_FOR_TASK")
        if provider.provider_name != source_id:
            raise ValueError("PROVIDER_SOURCE_MISMATCH")
        requested_at = self.clock()
        utc_iso(requested_at)
        cached = self.cache.get(source_id, task, requested_at)
        if cached is not None:
            self.cache.record_audit(ResearchAuditRecord(
                task.task_id, source_id, task.query, utc_iso(requested_at), "CACHE_HIT", True,
                cached.fetched_time
            ))
            return cached
        try:
            result = provider.fetch(task)
            completed_at = self.clock()
            if completed_at < requested_at:
                raise ValueError("RESEARCH_CLOCK_MOVED_BACKWARDS")
            if result.success:
                self.sources.validate_result_url(source_id, result.source_url)
                if result.source_name != source.name:
                    raise ValueError("PROVIDER_SOURCE_NAME_MISMATCH")
                if parse_utc(result.fetched_time) > completed_at:
                    raise ValueError("FUTURE_RETRIEVAL_FORBIDDEN")
                self.cache.put(source_id, task, result, completed_at, self.ttl)
                outcome = "FETCH_SUCCESS"
            else:
                outcome = "FETCH_UNAVAILABLE"
            self.cache.record_audit(ResearchAuditRecord(
                task.task_id, source_id, task.query, utc_iso(requested_at), outcome, False,
                result.fetched_time or None, result.error_code
            ))
            return result
        except Exception as error:
            self.cache.record_audit(ResearchAuditRecord(
                task.task_id, source_id, task.query, utc_iso(requested_at), "FETCH_FAILED", False,
                error_code=type(error).__name__
            ))
            raise
