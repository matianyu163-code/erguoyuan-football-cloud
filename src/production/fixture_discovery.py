"""Provider-routed fixture discovery before competition is required."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse

from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.research.global_provider_registry import (
    GlobalProviderRegistry,
    RegisteredProvider,
)
from erguoyuan_football.web_research.time_utils import parse_utc


@dataclass(frozen=True)
class FixtureDiscoveryRequest:
    """Exact team pair and optional search hints, with an explicit PIT cutoff."""

    home_team: TeamIdentity
    away_team: TeamIdentity
    prediction_time: datetime
    date_hint: date | None = None
    competition_hint: str | None = None

    def __post_init__(self) -> None:
        if (
            self.prediction_time.tzinfo is None
            or self.prediction_time.utcoffset() is None
        ):
            raise ValueError("FIXTURE_DISCOVERY_TIMEZONE_REQUIRED")


@dataclass(frozen=True)
class FixtureCandidate:
    """A provider-evidenced fixture identity; this is not a prediction snapshot."""

    fixture_id: str
    home_team_id: str
    away_team_id: str
    competition_id: str
    competition_name: str
    kickoff_time: datetime
    source: str
    evidence_id: str
    venue: str | None = None

    def __post_init__(self) -> None:
        if not all(
            (
                self.fixture_id,
                self.home_team_id,
                self.away_team_id,
                self.competition_id,
                self.competition_name,
                self.source,
                self.evidence_id,
            )
        ):
            raise ValueError("FIXTURE_CANDIDATE_PROVENANCE_REQUIRED")
        if self.kickoff_time.tzinfo is None or self.kickoff_time.utcoffset() is None:
            raise ValueError("FIXTURE_KICKOFF_TIMEZONE_REQUIRED")


class FixtureSearchProvider(Protocol):
    """Optional, explicit provider capability; generic endpoints are not guessed."""

    def find_fixtures(
        self, request: FixtureDiscoveryRequest
    ) -> tuple[FixtureCandidate, ...]:
        """Search an allowlisted provider for this exact pair and optional hints."""


class FixtureResearchResultLike(Protocol):
    """Structural contract for the provider-independent evidence research layer."""

    @property
    def status(self) -> str: ...

    @property
    def candidates(self) -> tuple[FixtureCandidate, ...]: ...

    @property
    def source_attempts(self) -> tuple[str, ...]: ...

    @property
    def queries(self) -> tuple[str, ...]: ...

    @property
    def evidence_ids(self) -> tuple[str, ...]: ...

    @property
    def pit_classifications(self) -> tuple[str, ...]: ...

    @property
    def reason(self) -> str: ...


class FixtureResearchLike(Protocol):
    """Research fallback usable without structured provider coverage."""

    def discover(
        self, request: FixtureDiscoveryRequest, *, structured_result: str
    ) -> FixtureResearchResultLike:
        """Find and validate fixture evidence from registered public sources."""


@dataclass(frozen=True)
class FixtureDiscoveryResult:
    """Deterministic search outcome and its provider/cache trace."""

    status: str
    candidates: tuple[FixtureCandidate, ...]
    providers_registered: tuple[str, ...]
    providers_queried: tuple[str, ...]
    cache_candidates: int
    reason: str
    research_called: bool = False
    research_status: str = "NOT_CALLED"
    structured_status: str = "UNKNOWN"
    research_sources_attempted: tuple[str, ...] = ()
    research_queries: tuple[str, ...] = ()
    research_evidence_ids: tuple[str, ...] = ()
    pit_classifications: tuple[str, ...] = ()


class FixtureDiscoveryRouter:
    """Route exact-pair searches only to eligible registered fixture providers."""

    def __init__(
        self,
        registry: GlobalProviderRegistry,
        evidence_cache: Path | None = None,
        global_research: FixtureResearchLike | None = None,
    ) -> None:
        self.registry = registry
        self.evidence_cache = evidence_cache
        self.global_research = global_research

    def discover(self, request: FixtureDiscoveryRequest) -> FixtureDiscoveryResult:
        """Search verified provider capabilities and point-in-time fixture evidence."""
        rows = self.registry.list()
        registered = tuple(row.profile.provider_id for row in rows)
        eligible = tuple(
            row
            for row in rows
            if _provider_covers_pair(row, request)
            and callable(getattr(row.provider, "find_fixtures", None))
        )
        queried: list[str] = []
        candidates: list[FixtureCandidate] = []
        failures: list[str] = []
        for row in eligible:
            finder = cast(FixtureSearchProvider, row.provider)
            queried.append(row.profile.provider_id)
            try:
                candidates.extend(finder.find_fixtures(request))
            except (OSError, ValueError, TypeError, RuntimeError) as error:
                failures.append(f"{row.profile.provider_id}:{type(error).__name__}")
        cached = self._cached_candidates(request, rows)
        candidates.extend(cached)
        unique: dict[str, FixtureCandidate] = {
            candidate.fixture_id: candidate
            for candidate in candidates
            if _matches_request(candidate, request)
        }
        ordered = tuple(
            sorted(
                unique.values(),
                key=lambda item: (
                    item.kickoff_time,
                    item.competition_id,
                    item.fixture_id,
                ),
            )
        )
        if len(ordered) == 1:
            return FixtureDiscoveryResult(
                "FOUND",
                ordered,
                registered,
                tuple(queried),
                len(cached),
                "VERIFIED_FIXTURE_FOUND",
                structured_status="FOUND",
            )
        if len(ordered) > 1:
            return FixtureDiscoveryResult(
                "AMBIGUOUS_FIXTURE",
                ordered,
                registered,
                tuple(queried),
                len(cached),
                "MULTIPLE_VERIFIED_FIXTURES",
                structured_status="AMBIGUOUS_FIXTURE",
            )
        reason = (
            "PROVIDER_SEARCH_FAILED:" + ",".join(failures)
            if failures
            else "STRUCTURED_PROVIDER_COVERAGE_MISSING"
            if not eligible
            else "STRUCTURED_PROVIDER_NO_VERIFIED_FIXTURE"
        )
        if self.global_research is None:
            return FixtureDiscoveryResult(
                "NO_VERIFIED_FIXTURE_FOUND",
                (),
                registered,
                tuple(queried),
                len(cached),
                reason,
                structured_status="NO_COVERAGE"
                if not eligible
                else "NO_VERIFIED_FIXTURE",
            )
        research = self.global_research.discover(
            request,
            structured_result=(
                "NO_COVERAGE" if not eligible else "NO_VERIFIED_FIXTURE"
            ),
        )
        status = str(research.status)
        research_reason = str(research.reason)
        if status == "RESEARCH_FOUND":
            final_status = "RESEARCH_FOUND"
        elif status in {"AMBIGUOUS_FIXTURE", "OFFICIAL_FIXTURE_AMBIGUOUS"}:
            final_status = "OFFICIAL_FIXTURE_AMBIGUOUS"
        elif status in {
            "FIXTURE_EVIDENCE_CONFLICT",
            "KICKOFF_TIME_UNVERIFIED",
            "OFFICIAL_FIXTURE_NOT_FOUND",
            "OFFICIAL_SOURCE_UNREACHABLE",
            "OFFICIAL_SOURCE_COVERAGE_MISSING",
            "OFFICIAL_NETWORK_BLOCKED",
            "OFFICIAL_FIXTURE_ENTRYPOINT_NOT_CONFIGURED",
            "OFFICIAL_ADAPTER_UNSUPPORTED_PAGE",
        }:
            final_status = status
        elif status == "UNAVAILABLE":
            final_status = "NO_VERIFIED_FIXTURE_FOUND"
            research_reason = f"GLOBAL_FIXTURE_RESEARCH_UNAVAILABLE:{research_reason}"
        else:
            final_status = "NO_VERIFIED_FIXTURE_FOUND"
        return FixtureDiscoveryResult(
            final_status,
            tuple(research.candidates),
            registered,
            tuple(queried),
            len(cached),
            research_reason,
            True,
            status,
            "NO_COVERAGE" if not eligible else "NO_VERIFIED_FIXTURE",
            tuple(research.source_attempts),
            tuple(research.queries),
            tuple(research.evidence_ids),
            tuple(research.pit_classifications),
        )

    def _cached_candidates(
        self,
        request: FixtureDiscoveryRequest,
        providers: tuple[RegisteredProvider, ...],
    ) -> tuple[FixtureCandidate, ...]:
        """Read only PIT-safe fixture evidence from the local audited cache."""
        path = self.evidence_cache
        if path is None or not path.is_file():
            return ()
        domains = {domain for row in providers for domain in row.allowed_domains}
        provider_ids = {row.profile.provider_id for row in providers}
        # The live production runner uses this named OpenLigaDB adapter for verified
        # German fixtures; accept its persisted evidence only on the registered host.
        provider_ids.add("OPENLIGADB_PHASE16_REAL_ACCEPTANCE")
        try:
            uri = f"file:{path.resolve().as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                rows = connection.execute("""
                    SELECT evidence_id, value_json, source_id, source_url,
                           published_time, fetched_time, as_of_time,
                           observed_time, source_tier
                    FROM research_evidence WHERE data_type = 'FIXTURE'
                """).fetchall()
        except sqlite3.Error:
            return ()
        result: dict[str, FixtureCandidate] = {}
        for (
            evidence_id,
            value_json,
            source_id,
            source_url,
            published_time,
            fetched_time,
            as_of_time,
            observed_time,
            source_tier,
        ) in rows:
            host = urlparse(str(source_url)).hostname
            if source_id not in provider_ids or host not in domains or source_tier != 2:
                continue
            try:
                fetched = parse_utc(str(fetched_time))
                times = [fetched, parse_utc(str(as_of_time))]
                for stamp in (published_time, observed_time):
                    if stamp:
                        times.append(parse_utc(str(stamp)))
                if max(times) > request.prediction_time:
                    continue
                value = json.loads(value_json)
                if not isinstance(value, dict):
                    continue
                candidate = FixtureCandidate(
                    fixture_id=str(value["provider_match_id"]),
                    home_team_id=str(value["home_team_id"]),
                    away_team_id=str(value["away_team_id"]),
                    competition_id=str(value["competition_id"]),
                    competition_name=str(
                        value.get("competition_name") or value["competition_id"]
                    ),
                    kickoff_time=parse_utc(str(value["kickoff_at"])),
                    source=str(source_id),
                    evidence_id=str(evidence_id),
                    venue=(str(value["venue"]) if value.get("venue") else None),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not _matches_request(candidate, request):
                continue
            result[candidate.fixture_id] = candidate
        return tuple(result.values())


def _provider_covers_pair(
    row: RegisteredProvider, request: FixtureDiscoveryRequest
) -> bool:
    """Require verified FIXTURE scope and matching identity dimensions."""
    profile = row.profile
    if (
        not row.configured
        or not profile.live_data
        or profile.status_for("FIXTURE").value != "VERIFIED"
    ):
        return False
    for team in (request.home_team, request.away_team):
        if not _contains(profile.entity_types, team.entity_type):
            return False
        if not _contains(profile.genders, team.gender):
            return False
        if not _contains(profile.age_groups, team.age_group):
            return False
        if not _contains(profile.squad_levels, team.squad_level):
            return False
        if team.entity_type == "CLUB":
            if not _contains(profile.countries, team.country):
                return False
            if not _contains(profile.federations, team.federation):
                return False
    return True


def _contains(scope: frozenset[str], value: str) -> bool:
    return value == "UNKNOWN" or "*" in scope or value in scope


def _matches_request(
    candidate: FixtureCandidate, request: FixtureDiscoveryRequest
) -> bool:
    """Require exact orientation, PIT availability and an optional date hint."""
    kickoff = candidate.kickoff_time
    if kickoff.tzinfo is None or kickoff.utcoffset() is None:
        return False
    if not (request.prediction_time < kickoff):
        return False
    if (
        candidate.home_team_id != request.home_team.team_id
        or candidate.away_team_id != request.away_team.team_id
    ):
        return False
    if request.date_hint is not None and kickoff.date() != request.date_hint:
        return False
    if request.competition_hint:
        hint = request.competition_hint.casefold().strip()
        return hint in {
            candidate.competition_id.casefold(),
            candidate.competition_name.casefold(),
        }
    return True
