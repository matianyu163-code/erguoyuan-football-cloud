"""Offline pre-match identity, task and evidence composition."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from erguoyuan_football.app.input.match_input import MatchRequest
from erguoyuan_football.knowledge.match_identity import GlobalKnowledgeResolver
from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from erguoyuan_football.research.data_quality import DataQualityEvaluator
from erguoyuan_football.research.match_package import MatchResearchPackage
from erguoyuan_football.research.research_status import ResearchStatus
from erguoyuan_football.research.validators import (
    validate_identity,
    validate_package,
    validate_sources,
)
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.fetch.fixture_identity import FixtureExpectation
from erguoyuan_football.web_research.fetch.live_fetcher import LiveResearchFetcher
from erguoyuan_football.web_research.policies.conflict_policy import ConflictRecord
from erguoyuan_football.web_research.policies.freshness_policy import (
    is_valid_for_cutoff,
)
from erguoyuan_football.web_research.search.query_builder import QueryBuilder
from erguoyuan_football.web_research.search.search_task import SearchTask
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.time_utils import utc_iso

_TASK_DATA_KEY = {
    "FIXTURE": "fixture",
    "INJURY": "injury",
    "LINEUP": "lineup",
    "ODDS": "odds",
    "TEAM_STATS": "xg",
    "NEWS": "news",
}


class MatchResearchOrchestrator:
    """Compose a research package, with explicit opt-in research fetching only."""

    def __init__(
        self,
        *,
        resolver: GlobalKnowledgeResolver | None = None,
        sources: SourceRegistry | None = None,
        evidence_store: EvidenceStore | None = None,
        quality: DataQualityEvaluator | None = None,
        live_fetcher: LiveResearchFetcher | None = None,
    ) -> None:
        if evidence_store is not None and sources is None and live_fetcher is None:
            raise ValueError("SOURCE_REGISTRY_REQUIRED_WITH_EVIDENCE_STORE")
        self.resolver = resolver or GlobalKnowledgeResolver()
        self.sources = sources or (live_fetcher.sources if live_fetcher is not None
                                   else SourceRegistry(ExternalSourceRegistry()))
        self.evidence_store = evidence_store or (
            live_fetcher.evidence_store if live_fetcher is not None
            else EvidenceStore(":memory:", self.sources))
        if live_fetcher is not None and (live_fetcher.sources is not self.sources
                                         or live_fetcher.evidence_store is not self.evidence_store):
            raise ValueError("LIVE_FETCHER_STORE_OR_SOURCE_MISMATCH")
        self.quality = quality or DataQualityEvaluator()
        self.live_fetcher = live_fetcher

    @staticmethod
    def _failed(as_of: str, code: str) -> MatchResearchPackage:
        package = MatchResearchPackage(None, None, None, None,
                                       status=ResearchStatus.FAILED,
                                       as_of_time=as_of, error_code=code)
        validate_package(package)
        return package

    def _evidence_for_task(
        self,
        task: SearchTask,
        as_of: datetime,
    ) -> list[EvidenceRecord]:
        expected = {task.task_type}
        if task.task_type == "TEAM_STATS":
            expected.add("XG")
        evidence = list(self.evidence_store.for_task(task.task_id, as_of))
        if any(item.data_type.upper() not in expected for item in evidence):
            raise ValueError("EVIDENCE_TASK_TYPE_MISMATCH")
        if any(self.sources.get(item.source_id).source_type not in task.required_sources
               for item in evidence):
            raise ValueError("EVIDENCE_SOURCE_TIER_NOT_ALLOWED_FOR_TASK")
        return evidence

    def build(
        self,
        request: MatchRequest,
        *,
        as_of_time: datetime | None = None,
        fetch_live: bool = False,
    ) -> MatchResearchPackage:
        """Build one PIT research view; live fetch is opt-in and never predicts."""
        at = as_of_time or datetime.now(UTC)
        as_of = utc_iso(at)
        identity = validate_identity(request, self.resolver)
        if identity.status not in {"IDENTITIES_RESOLVED_NO_FIXTURE"}:
            return self._failed(as_of, identity.status)
        home, away = identity.home_team, identity.away_team
        assert home is not None and away is not None
        verified_match_id = (request.match_id.strip()
                             if request.validation_status == "RESOLVED"
                             and request.match_id and request.match_id.strip() else None)
        session_id = str(uuid4()) if fetch_live and verified_match_id is None else None
        fixture_key = verified_match_id or session_id
        tasks = list(QueryBuilder.build(home.official_name, away.official_name,
                                        identity.competition.name if identity.competition else None,
                                        fixture_key=fixture_key))
        available: dict[str, list[EvidenceRecord]] = {}
        conflicts: list[ConflictRecord] = []
        live_results: list[str] = []
        cache_states: list[str] = []
        if fetch_live:
            if self.live_fetcher is None:
                live_results.append("LIVE_FETCHER_NOT_CONFIGURED")
            else:
                match_key = (f"MATCH:{verified_match_id}" if verified_match_id else
                             f"RESEARCH_SESSION:{session_id}")
                expectation = FixtureExpectation(
                    home.team_id, away.team_id,
                    identity.competition.competition_id if identity.competition else None,
                    request.date,
                )
                for task in tasks:
                    if task.task_type != "FIXTURE" and verified_match_id is None:
                        continue
                    outcome = self.live_fetcher.fetch_task(
                        task, match_key=match_key,
                        fixture_expectation=expectation if task.task_type == "FIXTURE" else None,
                    )
                    live_results.append(outcome.status)
                    cache_states.append(outcome.cache_status)
                    conflicts.extend(outcome.conflicts)
                    if (task.task_type == "FIXTURE" and outcome.verified_match_id
                            and outcome.status != "CONFLICT_UNRESOLVED"
                            and any(is_valid_for_cutoff(item, at)
                                    for item in outcome.evidence if item.source_tier == 3)):
                        verified_match_id = outcome.verified_match_id
                at = datetime.now(UTC) if as_of_time is None else at
                as_of = utc_iso(at)
        try:
            if verified_match_id is not None:
                for task in tasks:
                    evidence = self._evidence_for_task(task, at)
                    if evidence:
                        available[_TASK_DATA_KEY[task.task_type]] = evidence
            source_records = validate_sources(
                (item for records in available.values() for item in records),
                self.sources,
            )
        except (KeyError, ValueError):
            return self._failed(as_of, "EVIDENCE_VALIDATION_FAILED")
        missing = [key for key in _TASK_DATA_KEY.values() if key not in available]
        if identity.competition is None:
            missing.insert(0, "competition")
        if not missing:
            status = ResearchStatus.FOUND
        elif available:
            status = ResearchStatus.PARTIAL
        else:
            status = ResearchStatus.MISSING
        package = MatchResearchPackage(
            match_id=verified_match_id,
            home_team=home,
            away_team=away,
            competition=identity.competition,
            research_tasks=tasks,
            available_data=available,
            missing_data=missing,
            sources=source_records,
            status=status,
            as_of_time=as_of,
            research_session_id=session_id,
            network_status=("NOT_REQUESTED" if not fetch_live else
                            "AVAILABLE" if any(state in {"LIVE", "CACHE_FRESH"}
                                               for state in live_results) else "UNAVAILABLE"),
            live_fetch_attempted=fetch_live,
            live_fetch_completed=bool(self.live_fetcher is not None and live_results),
            cache_status=("CACHE_STALE" if "CACHE_STALE" in cache_states else
                          "CACHE_FRESH" if cache_states and all(
                              state == "CACHE_FRESH" for state in cache_states) else
                          "MIXED" if cache_states else "NOT_CHECKED"),
            conflicts=conflicts,
        )
        package.quality_score = self.quality.evaluate(package)
        validate_package(package)
        return package
