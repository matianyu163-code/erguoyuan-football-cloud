"""Provider-independent, source-audited global fixture research fallback."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Protocol
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from erguoyuan_football.knowledge.entities.country_alias_registry import (
    CountryAliasRegistry,
)
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.sources.source_schema import SourceRecord
from erguoyuan_football.web_research.time_utils import utc_iso
from production.association_sources import (
    AssociationSource,
    AssociationSourceRegistry,
)
from production.fixture_discovery import (
    FixtureCandidate,
    FixtureDiscoveryRequest,
)
from production.official_source_registry import OfficialFootballSourceRegistry

_AGE_LANGUAGE = {
    "nl": "O",
    "en": "U",
    "ga": "U",
    "de": "U",
    "fr": "U",
    "es": "U",
    "ru": "U",
    "kk": "U",
    "ro": "U",
    "cs": "U",
    "ja": "U",
    "ko": "U",
    "ar": "U",
}


@dataclass(frozen=True)
class FixtureResearchEvidence:
    """One source claim, kept separate from model-eligible pre-match features."""

    source_id: str
    source_url: str
    source_name: str
    source_tier: int
    retrieved_at: datetime
    home_entity_id: str
    away_entity_id: str
    competition_name: str
    confidence: str
    kickoff_utc: datetime | None = None
    kickoff_local: str | None = None
    kickoff_timezone: str | None = None
    venue: str | None = None
    published_at: datetime | None = None
    fixture_id: str | None = None
    source_content_hash: str | None = None


class GlobalFixtureResearchProvider(Protocol):
    """Search one declared source without relying on structured team IDs."""

    provider_id: str
    source_tier: int
    allowed_domains: frozenset[str]

    def search(
        self, query: str, *, source: AssociationSource | None, as_of_time: datetime
    ) -> tuple[FixtureResearchEvidence, ...]:
        """Return source-cited fixture claims; empty means no verified evidence."""


@dataclass(frozen=True)
class FixtureResearchResult:
    """Auditable global research outcome consumed by the fixture router."""

    status: str
    candidates: tuple[FixtureCandidate, ...]
    research_called: bool
    structured_result: str
    source_attempts: tuple[str, ...]
    queries: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    pit_classifications: tuple[str, ...]
    reason: str


def build_research_source_registry(
    associations: AssociationSourceRegistry,
    providers: tuple[GlobalFixtureResearchProvider, ...] = (),
    official_sources: OfficialFootballSourceRegistry | None = None,
) -> SourceRegistry:
    """Build evidence allowlists exclusively from official or structured sources."""
    definitions: list[SourceDefinition] = []
    records: list[SourceRecord] = []
    sources: dict[str, tuple[int, str, tuple[str, ...]]] = {}
    for source in associations.all():
        domains = tuple(
            sorted(
                {
                    domain
                    for base in source.official_domains
                    for domain in (base, f"www.{base}")
                }
            )
        )
        sources[source.association_id] = (3, source.official_domains[0], domains)
    for official_source in (official_sources or OfficialFootballSourceRegistry()).all():
        domains = tuple(
            sorted(
                {
                    official_source.official_domain,
                    f"www.{official_source.official_domain}",
                }
            )
        )
        sources[official_source.source_id] = (
            3,
            official_source.official_domain,
            domains,
        )
    for provider in providers:
        domains = tuple(sorted(provider.allowed_domains))
        if not domains:
            raise ValueError("GLOBAL_RESEARCH_PROVIDER_DOMAIN_REQUIRED")
        official_domains = {
            domain.casefold()
            for item in (official_sources or OfficialFootballSourceRegistry()).all()
            for domain in (item.official_domain, f"www.{item.official_domain}")
        }
        official_domains.update(
            domain.casefold()
            for item in associations.all()
            for base in item.official_domains
            for domain in (base, f"www.{base}")
        )
        if provider.source_tier != 3 or any(
            domain.casefold() not in official_domains for domain in domains
        ):
            raise ValueError("OFFICIAL_SOURCE_ONLY_PROVIDER_REQUIRED")
        sources[provider.provider_id] = (provider.source_tier, domains[0], domains)
    for source_id, (tier, primary_domain, domains) in sorted(sources.items()):
        source_type = (
            "OFFICIAL" if tier == 3 else "STRUCTURED" if tier == 2 else "MEDIA"
        )
        base_url = f"https://{primary_domain}"
        definitions.append(
            SourceDefinition(
                source_id=source_id,
                display_name=source_id,
                category="GLOBAL_FIXTURE_RESEARCH",
                base_url=base_url,
                schema_version="GLOBAL_FIXTURE_RESEARCH_V1",
            )
        )
        records.append(
            SourceRecord(
                source_id, source_id, source_type, tier, base_url, True, domains
            )
        )
    return (
        SourceRegistry(ExternalSourceRegistry(tuple(definitions)))
        if not records
        else (_source_registry(tuple(definitions), tuple(records)))
    )


def _source_registry(
    definitions: tuple[SourceDefinition, ...], records: tuple[SourceRecord, ...]
) -> SourceRegistry:
    """Construct validated source metadata without enabling any network endpoint."""
    external = ExternalSourceRegistry(definitions)
    registry = SourceRegistry(external)
    for record in records:
        registry.add(record)
    return registry


def localized_fixture_queries(
    home: TeamIdentity,
    away: TeamIdentity,
    sources: tuple[AssociationSource, ...],
    *,
    competition_hint: str | None = None,
    date_hint: date | None = None,
) -> tuple[str, ...]:
    """Create bounded deterministic multilingual queries from canonical identities."""
    countries = CountryAliasRegistry()
    source_by_country = {item.country_id: item for item in sources}
    home_country = countries.resolve(home.country)
    away_country = countries.resolve(away.country)
    language_set = {"en"}
    language_set.update(
        language for source in sources for language, _ in source.country_aliases
    )
    for country in (home_country, away_country):
        if country is not None and country.iso3 in source_by_country:
            language_set.update(source_by_country[country.iso3].supported_languages)
    languages = tuple(sorted(language_set))
    gender_term = (
        "men's international"
        if home.gender == away.gender == "MEN"
        else "women's international"
        if home.gender == away.gender == "WOMEN"
        else "international"
    )
    context_terms = " ".join(
        value
        for value in (
            competition_hint.strip() if competition_hint else None,
            date_hint.isoformat() if date_hint else None,
        )
        if value
    )

    def name_options(
        team: TeamIdentity, country_id: str | None, language: str
    ) -> tuple[str, ...]:
        source = source_by_country.get(country_id or "")
        localized = (
            dict(source.country_aliases).get(language) if source is not None else None
        )
        aliases = [team.official_name]
        if source is not None:
            aliases.extend(
                value for lang, value in source.country_aliases if lang == language
            )
        canonical = countries.resolve(team.country)
        if canonical is not None:
            aliases.append(canonical.name)
        aliases.extend(team.aliases)
        age_prefix = _AGE_LANGUAGE.get(language, "U")
        age = "" if team.age_group == "SENIOR" else f"{age_prefix}{team.age_group[1:]}"
        values: list[str] = []
        for alias in ([localized] if localized else []) + aliases:
            if not alias:
                continue
            rendered = f"{alias} {age}".strip()
            if rendered.casefold() not in {value.casefold() for value in values}:
                values.append(rendered)
        return tuple(values[:3])

    home_id = home_country.iso3 if home_country else None
    away_id = away_country.iso3 if away_country else None
    queries: list[str] = []
    for language in languages:
        home_names = name_options(home, home_id, language)
        away_names = name_options(away, away_id, language)
        if home_names and away_names:
            queries.append(f"{home_names[0]} {away_names[0]}")
            if len(queries) < 8:
                suffix = f"{gender_term} fixture {context_terms}".strip()
                queries.append(f"{home_names[-1]} vs {away_names[-1]} {suffix}")
    return tuple(dict.fromkeys(queries))[:8]


class GlobalFixtureResearch:
    """Country-source-driven research fallback; competition is only a search hint."""

    def __init__(
        self,
        *,
        associations: AssociationSourceRegistry | None = None,
        providers: tuple[GlobalFixtureResearchProvider, ...] = (),
        evidence_store: EvidenceStore | None = None,
        fixture_sources_enabled: bool = False,
    ) -> None:
        self.associations = associations or AssociationSourceRegistry()
        official_domains = {
            domain.casefold()
            for item in OfficialFootballSourceRegistry().all()
            for domain in (item.official_domain, f"www.{item.official_domain}")
        }
        official_domains.update(
            domain.casefold()
            for item in self.associations.all()
            for base in item.official_domains
            for domain in (base, f"www.{base}")
        )
        self.providers = tuple(
            provider
            for provider in providers
            if provider.source_tier == 3
            and provider.allowed_domains
            and all(
                domain.casefold() in official_domains
                for domain in provider.allowed_domains
            )
        )
        self.evidence_store = evidence_store
        self.fixture_sources_enabled = fixture_sources_enabled
        self.countries = CountryAliasRegistry()

    def discover(
        self, request: FixtureDiscoveryRequest, *, structured_result: str
    ) -> FixtureResearchResult:
        """Try source evidence even when no structured provider covers both teams."""
        home_country = self.countries.resolve(request.home_team.country)
        away_country = self.countries.resolve(request.away_team.country)
        source_rows = self.associations.for_countries(
            home_country.iso3 if home_country else "",
            away_country.iso3 if away_country else "",
        )
        queries = localized_fixture_queries(
            request.home_team,
            request.away_team,
            source_rows,
            competition_hint=request.competition_hint,
            date_hint=request.date_hint,
        )
        configured_ids = {provider.provider_id for provider in self.providers}
        attempts: list[str] = [
            f"{source.association_id}:NO_PROVIDER_CONFIGURED" for source in source_rows
            if source.association_id not in configured_ids
        ]
        evidence_ids: list[str] = []
        classifications: list[str] = []
        candidates: dict[tuple[str, str, str, datetime], FixtureCandidate] = {}
        saved_ids: set[str] = set()
        kickoff_unverified = False
        cache_records = self._cached(request)
        for record in cache_records:
            converted = self._candidate(record)
            if converted is not None:
                evidence_ids.append(record.evidence_id)
                classifications.append(
                    _pit_classification(converted.kickoff_time, request.prediction_time)
                )
                candidates[_candidate_identity(converted)] = converted
                attempts.append(f"{record.source_id}:CACHE_HIT")
            elif isinstance(record.value, dict) and not record.value.get("kickoff_utc"):
                evidence_ids.append(record.evidence_id)
                classifications.append("KICKOFF_TIME_UNVERIFIED")
                kickoff_unverified = True
                attempts.append(f"{record.source_id}:CACHE_HIT")

        official_sources = {item.association_id: item for item in source_rows}
        for provider in sorted(
            self.providers, key=lambda item: (-item.source_tier, item.provider_id)
        ):
            scopes: tuple[AssociationSource | None, ...] = (
                (official_sources[provider.provider_id],)
                if provider.provider_id in official_sources
                else (None,)
                if OfficialFootballSourceRegistry().get(provider.provider_id) is not None
                else tuple(official_sources.values())
            )
            for scope in scopes:
                relevant_queries = queries or (
                    f"{request.home_team.official_name} {request.away_team.official_name} fixture",
                )
                for query in relevant_queries:
                    scope_label = (
                        scope.association_id if scope else provider.provider_id
                    )
                    try:
                        results = provider.search(
                            query, source=scope, as_of_time=request.prediction_time
                        )
                    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                        failure = (
                            "NETWORK_POLICY_BLOCKED" if "NETWORK_POLICY_BLOCKED" in str(error)
                            else "OFFICIAL_PAGE_CLIENT_RENDERED_NO_FIXTURE_ROWS"
                            if "OFFICIAL_PAGE_CLIENT_RENDERED_NO_FIXTURE_ROWS" in str(error)
                            else type(error).__name__
                        )
                        attempts.append(f"{scope_label}:FAILED:{failure}")
                        continue
                    attempts.append(f"{scope_label}:QUERIED:{_query_hash(query)}")
                    for item in results:
                        if not self._valid_evidence(item, provider, scope, request):
                            continue
                        record, pit = self._to_record(item, request)
                        if (
                            self.evidence_store is not None
                            and record.evidence_id not in saved_ids
                        ):
                            try:
                                if self.evidence_store.get(record.evidence_id) is None:
                                    self.evidence_store.save(record)
                                saved_ids.add(record.evidence_id)
                            except (ValueError, RuntimeError):
                                attempts.append(f"{scope_label}:EVIDENCE_REJECTED")
                                continue
                        evidence_ids.append(record.evidence_id)
                        classifications.append(pit)
                        converted = self._candidate(record)
                        if converted is not None:
                            candidates[_candidate_identity(converted)] = converted
                        elif pit == "KICKOFF_TIME_UNVERIFIED":
                            kickoff_unverified = True

        ordered = tuple(
            sorted(
                candidates.values(),
                key=lambda value: (
                    value.kickoff_time,
                    value.competition_id,
                    value.fixture_id,
                ),
            )
        )
        if len(ordered) > 1:
            same_identity = {
                (item.home_team_id, item.away_team_id, item.competition_id)
                for item in ordered
            }
            kickoff_values = {item.kickoff_time for item in ordered}
            status = (
                "FIXTURE_EVIDENCE_CONFLICT"
                if len(same_identity) == 1 and len(kickoff_values) > 1
                else "OFFICIAL_FIXTURE_AMBIGUOUS"
            )
            reason = status
        elif kickoff_unverified:
            status, reason = (
                "KICKOFF_TIME_UNVERIFIED",
                "NO_RELIABLE_TIMEZONE_OR_KICKOFF",
            )
        elif len(ordered) == 1:
            status, reason = "RESEARCH_FOUND", "OFFICIAL_OR_REGISTERED_EVIDENCE_FOUND"
        elif not self.providers:
            status, reason = (
                (
                    "OFFICIAL_FIXTURE_ENTRYPOINT_NOT_CONFIGURED",
                    "SOURCE_HAS_NO_FIXTURE_ENTRYPOINT",
                )
                if self.fixture_sources_enabled and source_rows
                else (
                    "OFFICIAL_SOURCE_COVERAGE_MISSING",
                    "NO_OFFICIAL_SITE_ADAPTER_CONFIGURED",
                )
            )
        else:
            blocked = any(":FAILED:NETWORK_POLICY_BLOCKED" in attempt for attempt in attempts)
            unsupported = any(":FAILED:OFFICIAL_PAGE_CLIENT_RENDERED_NO_FIXTURE_ROWS" in attempt for attempt in attempts)
            failed = any(":FAILED:" in attempt for attempt in attempts)
            status, reason = (
                ("OFFICIAL_NETWORK_BLOCKED", "NETWORK_POLICY_BLOCKED") if blocked
                else ("OFFICIAL_FIXTURE_ENTRYPOINT_NOT_CONFIGURED", "SOURCE_HAS_NO_FIXTURE_ENTRYPOINT")
                if not any(":QUERIED:" in attempt for attempt in attempts)
                and not failed
                and any(":NO_PROVIDER_CONFIGURED" in attempt for attempt in attempts)
                else ("OFFICIAL_ADAPTER_UNSUPPORTED_PAGE", "OFFICIAL_PAGE_CLIENT_RENDERED_NO_FIXTURE_ROWS")
                if unsupported and not ordered
                else ("OFFICIAL_SOURCE_UNREACHABLE", "ALL_OFFICIAL_SOURCES_FAILED")
                if failed and not any(":QUERIED:" in attempt for attempt in attempts)
                else (
                    "OFFICIAL_FIXTURE_NOT_FOUND",
                    "OFFICIAL_PAGES_HAD_NO_EXACT_FIXTURE",
                )
            )
        if not source_rows and not self.providers:
            status = "OFFICIAL_SOURCE_COVERAGE_MISSING"
            reason = "NO_OFFICIAL_SOURCE_MAPPING_FOR_ENTITIES"
        return FixtureResearchResult(
            status,
            ordered,
            True,
            structured_result,
            tuple(attempts),
            queries,
            tuple(dict.fromkeys(evidence_ids)),
            tuple(dict.fromkeys(classifications)),
            reason,
        )

    def _cached(self, request: FixtureDiscoveryRequest) -> tuple[EvidenceRecord, ...]:
        if self.evidence_store is None:
            return ()
        home_id, away_id = request.home_team.team_id, request.away_team.team_id
        return tuple(
            item
            for item in self.evidence_store.available_at(request.prediction_time)
            if item.data_type == "FIXTURE"
            and isinstance(item.value, dict)
            and item.value.get("home_entity_id") == home_id
            and item.value.get("away_entity_id") == away_id
        )

    @staticmethod
    def _valid_evidence(
        item: FixtureResearchEvidence,
        provider: GlobalFixtureResearchProvider,
        source: AssociationSource | None,
        request: FixtureDiscoveryRequest,
    ) -> bool:
        parsed = urlparse(item.source_url)
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or not parsed.hostname
            or parsed.query
            or parsed.fragment
        ):
            return False
        allowed = (
            {domain for base in source.official_domains for domain in (base, f"www.{base}")}
            if source is not None
            else set(provider.allowed_domains)
        )
        if parsed.hostname.casefold() not in {domain.casefold() for domain in allowed}:
            return False
        if source is not None and (
            item.source_id != source.association_id or item.source_tier != 3
        ):
            return False
        if source is None and (
            item.source_id != provider.provider_id
            or item.source_tier != provider.source_tier
        ):
            return False
        if (
            item.home_entity_id != request.home_team.team_id
            or item.away_entity_id != request.away_team.team_id
            or item.confidence not in {"HIGH", "MEDIUM"}
        ):
            return False
        if (
            item.retrieved_at.tzinfo is None
            or item.retrieved_at.utcoffset() is None
            or (
                item.kickoff_utc is not None
                and (
                    item.kickoff_utc.tzinfo is None
                    or item.kickoff_utc.utcoffset() is None
                )
            )
        ):
            return False
        return not (item.published_at and item.published_at > item.retrieved_at)

    @staticmethod
    def _to_record(
        item: FixtureResearchEvidence, request: FixtureDiscoveryRequest
    ) -> tuple[EvidenceRecord, str]:
        kickoff = _kickoff_utc(item)
        pit = (
            _pit_classification(kickoff, request.prediction_time)
            if kickoff is not None
            else "KICKOFF_TIME_UNVERIFIED"
        )
        data = {
            "source_id": item.source_id,
            "source_url": item.source_url,
            "source_content_hash": item.source_content_hash,
            "source_tier_label": "A",
            "fetched_at": utc_iso(item.retrieved_at),
            "home_entity_id": item.home_entity_id,
            "away_entity_id": item.away_entity_id,
            "competition_name": item.competition_name,
            "competition_id": _competition_id(item.competition_name),
            "kickoff_utc": utc_iso(kickoff) if kickoff else None,
            "kickoff_local": item.kickoff_local,
            "kickoff_timezone": item.kickoff_timezone,
            "timezone_evidence": (
                "EXPLICIT_OFFSET_OR_ZONE" if kickoff is not None
                else "UNVERIFIED_LOCAL_TIME"
            ),
            "venue": item.venue,
            "fixture_id": item.fixture_id,
            "pit_classification": pit,
        }
        fetched = utc_iso(item.retrieved_at)
        digest = hashlib.sha256(
            json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        data["evidence_hash"] = digest
        record = EvidenceRecord(
            evidence_id=f"GLOBAL_FIXTURE_{digest}",
            data_type="FIXTURE",
            value=data,
            source_id=item.source_id,
            published_time=utc_iso(item.published_at) if item.published_at else None,
            fetched_time=fetched,
            confidence=item.confidence,
            source_url=item.source_url,
            as_of_time=fetched,
            provider_id=item.source_id,
            source_tier=item.source_tier,
            observed_time=fetched,
            match_key=f"{item.home_entity_id}:{item.away_entity_id}",
            claim_type="FACT",
        )
        return record, pit

    @staticmethod
    def _candidate(record: EvidenceRecord) -> FixtureCandidate | None:
        data = record.value
        if not isinstance(data, dict) or not data.get("kickoff_utc"):
            return None
        return FixtureCandidate(
            fixture_id=str(data.get("fixture_id") or record.evidence_id),
            home_team_id=str(data["home_entity_id"]),
            away_team_id=str(data["away_entity_id"]),
            competition_id=str(data["competition_id"]),
            competition_name=str(data["competition_name"]),
            kickoff_time=datetime.fromisoformat(str(data["kickoff_utc"])),
            source=record.source_id,
            evidence_id=record.evidence_id,
            venue=str(data["venue"]) if data.get("venue") else None,
        )


def _kickoff_utc(item: FixtureResearchEvidence) -> datetime | None:
    if item.kickoff_utc is not None:
        if item.kickoff_utc.tzinfo is None or item.kickoff_utc.utcoffset() is None:
            return None
        return item.kickoff_utc.astimezone(UTC)
    if not item.kickoff_local or not item.kickoff_timezone:
        return None
    try:
        local = datetime.fromisoformat(item.kickoff_local)
        zone = ZoneInfo(item.kickoff_timezone)
    except (ValueError, ZoneInfoNotFoundError):
        return None
    if local.tzinfo is None:
        first = local.replace(tzinfo=zone, fold=0)
        second = local.replace(tzinfo=zone, fold=1)
        if first.utcoffset() != second.utcoffset():
            return None
        local = first
        if local.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != local.replace(
            tzinfo=None
        ):
            return None
    return local.astimezone(UTC)


def _pit_classification(kickoff: datetime, as_of: datetime) -> str:
    return (
        "POST_MATCH_EVIDENCE"
        if kickoff.astimezone(UTC) <= as_of.astimezone(UTC)
        else "PRE_MATCH_EVIDENCE"
    )


def _competition_id(name: str) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
    return f"RESEARCH_COMPETITION_{slug or 'UNSPECIFIED'}"


def _candidate_identity(candidate: FixtureCandidate) -> tuple[str, str, str, datetime]:
    """Deduplicate corroborating sources without hiding time or competition conflicts."""
    return (
        candidate.home_team_id,
        candidate.away_team_id,
        candidate.competition_id,
        candidate.kickoff_time.astimezone(UTC),
    )


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
