"""Static, persisted and opt-in live identity resolution without guessing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

from erguoyuan_football.knowledge.entities.candidate_scoring import score_candidate
from erguoyuan_football.knowledge.entities.competition_hint_parser import (
    ParsedCompetitionHint,
)
from erguoyuan_football.knowledge.entities.discovery import (
    TeamDiscoveryCandidate,
    TeamDiscoveryProvider,
)
from erguoyuan_football.knowledge.entities.entity_constraints import hard_conflicts
from erguoyuan_football.knowledge.entities.entity_fingerprint import (
    canonical_entity_id,
    entity_fingerprint,
)
from erguoyuan_football.knowledge.entities.team_entity_parser import (
    ParsedTeamEntityHint,
    UniversalTeamNameParser,
)
from erguoyuan_football.knowledge.entities.verified_entity_store import (
    VerifiedEntityStore,
)
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.network.errors import NetworkError


@dataclass(frozen=True)
class EntityResolution:
    """Separate team identity confirmation from any fixture verification."""

    status: str
    identity: TeamIdentity | None
    parsed: ParsedTeamEntityHint
    source: str | None
    candidates: tuple[TeamDiscoveryCandidate, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def resolution_status(self) -> str:
        """Expose source-specific outcome without changing legacy status consumers."""
        if self.identity is not None and self.status == "VERIFIED":
            return {
                "STATIC_LOCAL": "RESOLVED_LOCAL",
                "COUNTRY_REGISTRY": "RESOLVED_STRUCTURED",
                "DYNAMIC_STORE": "RESOLVED_DYNAMIC",
                "LIVE_DISCOVERY": "RESOLVED_PROVIDER",
                "RESEARCH_DISCOVERY": "RESOLVED_RESEARCH",
            }.get(self.source or "", "RESOLVED_LOCAL")
        if "AMBIGUOUS" in self.status or "CONFLICT" in self.status:
            return "AMBIGUOUS"
        if self.status in {"TEAM_DISCOVERY_UNAVAILABLE", "PROVIDER_COVERAGE_MISSING",
                           "STALE_IDENTITY", "TEAM_NOT_FOUND"}:
            return "DISCOVERY_REQUIRED"
        return "FAILED"


def _apply_competition(hint: ParsedTeamEntityHint,
                       competition: ParsedCompetitionHint | None) -> ParsedTeamEntityHint:
    if competition is None:
        return hint
    changes: dict[str, str] = {}
    for field, target in (("gender", "gender_hint"), ("age_group", "age_group_hint"),
                          ("entity_type", "entity_type_hint")):
        value = getattr(competition, field)
        if value is None:
            continue
        current = getattr(hint, target)
        if current not in {"UNKNOWN", value} and not (
            target == "age_group_hint" and current == "SENIOR"):
            raise ValueError(f"TEAM_IDENTITY_CONFLICT:{field.upper()}")
        changes[target] = value
    return replace(hint, **changes)


def _complete(hint: ParsedTeamEntityHint, identity: TeamIdentity) -> bool:
    """A known user/competition dimension needs known matching identity evidence."""
    for expected, actual in ((hint.gender_hint, identity.gender),
                             (hint.age_group_hint, identity.age_group),
                             (hint.squad_level_hint, identity.squad_level)):
        if expected not in {"UNKNOWN", "SENIOR", "FIRST_TEAM"} and actual == "UNKNOWN":
            return False
    return True


class UniversalTeamResolver:
    """Resolve local/store identities, known national teams, then opt-in discovery."""

    version = "UNIVERSAL_ENTITY_RESOLVER_V1"

    def __init__(self, *, static: TeamResolver | None = None,
                 store: VerifiedEntityStore | None = None,
                 discovery: tuple[TeamDiscoveryProvider, ...] = ()) -> None:
        self.static = static or TeamResolver()
        self.store = store
        self.discovery = discovery
        self.parser = UniversalTeamNameParser()

    def _country_team(self, raw_name: str,
                      hint: ParsedTeamEntityHint) -> TeamIdentity | None:
        """Construct a stable national-team identity from a recognized country."""
        if hint.entity_type_hint != "NATIONAL":
            return None
        country = self.parser.countries.resolve(hint.base_name)
        if country is None:
            return None
        gender = hint.gender_hint if hint.gender_hint != "UNKNOWN" else "MEN"
        age_group = hint.age_group_hint if hint.age_group_hint != "UNKNOWN" else "SENIOR"
        gender_code = "W" if gender == "WOMEN" else "M"
        display_age = "Senior" if age_group == "SENIOR" else age_group
        gender_label = "Women's" if gender == "WOMEN" else "Men's"
        aliases = tuple(dict.fromkeys((raw_name, country.name, *country.aliases)))
        return TeamIdentity(
            team_id=f"NATIONAL_{country.iso3}_{gender_code}_{age_group}",
            official_name=f"{country.name} {display_age} {gender_label} National Team",
            country=country.name,
            federation=country.federation,
            aliases=list(aliases),
            entity_type="NATIONAL",
            gender=gender,
            age_group=age_group,
            squad_level=hint.squad_level_hint,
            identity_status="COUNTRY_REGISTRY_DERIVED",
        )

    def resolve(self, raw_name: str, *, competition: ParsedCompetitionHint | None = None,
                allow_discovery: bool = False,
                as_of: datetime | None = None) -> EntityResolution:
        """Return one source-confirmed entity or explicit ambiguity/unavailability."""
        at = as_of or datetime.now(UTC)
        try:
            hint = _apply_competition(self.parser.parse(raw_name), competition)
        except ValueError as error:
            fallback = self.parser.parse(raw_name) if raw_name.strip() else None
            if fallback is None:
                raise ValueError("TEAM_PARSE_FAILED") from error
            return EntityResolution(str(error), None, fallback, None)
        local = self.static.resolve(raw_name)
        if local is not None and not hard_conflicts(
            hint, country=None, gender=local.gender, age_group=local.age_group,
            entity_type=local.entity_type, squad_level=local.squad_level,
        ) and _complete(hint, local):
            return EntityResolution("VERIFIED", local, hint, "STATIC_LOCAL")
        if self.store is not None:
            stored = [entry for entry in self.store.find(raw_name, as_of=at)
                      if not hard_conflicts(hint, country=None,
                          gender=entry.identity.gender, age_group=entry.identity.age_group,
                          entity_type=entry.identity.entity_type,
                          squad_level=entry.identity.squad_level)
                      and _complete(hint, entry.identity)]
            if len(stored) > 1:
                return EntityResolution("TEAM_DISCOVERY_AMBIGUOUS", None, hint,
                                        "DYNAMIC_STORE")
            if len(stored) == 1 and stored[0].status != "STALE_IDENTITY":
                return EntityResolution("VERIFIED", stored[0].identity, hint,
                                        "DYNAMIC_STORE")
            if len(stored) == 1 and not allow_discovery:
                return EntityResolution("STALE_IDENTITY", stored[0].identity, hint,
                                        "DYNAMIC_STORE")
        if not allow_discovery:
            country_team = self._country_team(raw_name, hint)
            if country_team is not None:
                return EntityResolution("VERIFIED", country_team, hint, "COUNTRY_REGISTRY")
            return EntityResolution("TEAM_NOT_FOUND", None, hint, None)
        country_team = self._country_team(raw_name, hint)
        discovery_hint = (replace(hint, gender_hint="MEN")
                          if country_team is not None and hint.gender_hint == "UNKNOWN"
                          else hint)
        providers = [provider for provider in self.discovery
                     if provider.supports_team_discovery]
        if not providers:
            if country_team is not None:
                return EntityResolution("VERIFIED", country_team, hint,
                    "COUNTRY_REGISTRY", reasons=("PROVIDER_DISCOVERY_NOT_CONFIGURED",))
            return EntityResolution("TEAM_DISCOVERY_UNAVAILABLE", None, hint, None)
        covered = [provider for provider in providers
                   if provider.can_cover(discovery_hint, competition)]
        if not covered:
            if country_team is not None:
                return EntityResolution("VERIFIED", country_team, hint,
                    "COUNTRY_REGISTRY", reasons=("PROVIDER_COVERAGE_MISSING",))
            return EntityResolution("PROVIDER_COVERAGE_MISSING", None, hint, None)
        discovered: list[TeamDiscoveryCandidate] = []
        failures: list[str] = []
        for provider in covered:
            try:
                discovered.extend(provider.discover(discovery_hint, competition))
            except (NetworkError, OSError, ValueError, TypeError) as error:
                failures.append(getattr(error, "code", type(error).__name__))
        candidates = tuple(candidate for candidate in discovered
                           if candidate.fetched_at <= (
                               at if as_of is not None else datetime.now(UTC)))
        if not candidates and failures:
            if country_team is not None:
                return EntityResolution("VERIFIED", country_team, hint,
                                        "COUNTRY_REGISTRY", reasons=tuple(failures))
            return EntityResolution("TEAM_DISCOVERY_UNAVAILABLE", None, hint,
                                    "LIVE_DISCOVERY", reasons=tuple(failures))
        compatible = tuple(sorted((candidate for candidate in candidates
                            if score_candidate(discovery_hint, candidate, competition) is not None),
                            key=lambda candidate: score_candidate(discovery_hint, candidate,
                                                                    competition) or 0,
                            reverse=True))
        if not compatible:
            if country_team is not None:
                return EntityResolution("VERIFIED", country_team, hint,
                    "COUNTRY_REGISTRY", candidates=candidates,
                    reasons=("PROVIDER_IDENTITY_REJECTED",))
            return EntityResolution("TEAM_DISCOVERY_REJECTED" if candidates else
                                    "TEAM_NOT_FOUND", None, hint, "LIVE_DISCOVERY",
                                    candidates)
        distinct = {(item.provider_id, item.provider_team_id) for item in compatible}
        if len(distinct) != 1:
            if country_team is not None:
                return EntityResolution("VERIFIED", country_team, hint,
                    "COUNTRY_REGISTRY", compatible, ("PROVIDER_IDENTITY_AMBIGUOUS",))
            return EntityResolution("TEAM_DISCOVERY_AMBIGUOUS", None, hint,
                                    "LIVE_DISCOVERY", compatible)
        if len({(item.official_name, item.gender, item.age_group,
                 item.squad_level, item.entity_type) for item in compatible}) != 1:
            if country_team is not None:
                return EntityResolution("VERIFIED", country_team, hint,
                    "COUNTRY_REGISTRY", compatible, ("PROVIDER_IDENTITY_CONFLICT",))
            return EntityResolution("TEAM_PROVIDER_ID_CONFLICT", None, hint,
                                    "LIVE_DISCOVERY", compatible)
        candidate = compatible[0]
        if candidate.source_tier < 2 or not candidate.evidence_id or (
            discovery_hint.gender_hint != "UNKNOWN" and candidate.gender == "UNKNOWN"):
            if country_team is not None:
                return EntityResolution("VERIFIED", country_team, hint,
                    "COUNTRY_REGISTRY", compatible, ("INSUFFICIENT_IDENTITY_EVIDENCE",))
            return EntityResolution("TEAM_DISCOVERY_REJECTED", None, hint,
                                    "LIVE_DISCOVERY", compatible,
                                    ("INSUFFICIENT_IDENTITY_EVIDENCE",))
        fingerprint = entity_fingerprint(
            country=candidate.country or "UNKNOWN", entity_type=candidate.entity_type,
            gender=candidate.gender, age_group=candidate.age_group,
            squad_level=candidate.squad_level, name=candidate.official_name,
            provider_id=candidate.provider_id,
            provider_team_id=candidate.provider_team_id,
        )
        identity = TeamIdentity(
            canonical_entity_id(country=candidate.country or "UNKNOWN",
                                federation=candidate.federation or "UNKNOWN",
                                entity_type=candidate.entity_type,
                                gender=candidate.gender, age_group=candidate.age_group,
                                fingerprint=fingerprint,
                                provider_id=candidate.provider_id,
                                provider_team_id=candidate.provider_team_id),
            candidate.official_name, candidate.country or "UNKNOWN",
            candidate.federation or "UNKNOWN",
            list(dict.fromkeys((raw_name, *candidate.aliases))),
            candidate.entity_type, candidate.gender, candidate.age_group,
            candidate.squad_level, provider_ids={candidate.provider_id:
                                                 candidate.provider_team_id},
            identity_status="PROVIDER_VERIFIED",
            verification_evidence_ids=[candidate.evidence_id],
        )
        if self.store is not None:
            previous = self.store.by_provider(candidate.provider_id,
                                              candidate.provider_team_id, as_of=at)
            if previous is not None:
                old = previous.identity
                if (old.gender != identity.gender or old.age_group != identity.age_group
                        or old.squad_level != identity.squad_level or
                        old.entity_type != identity.entity_type):
                    return EntityResolution("TEAM_PROVIDER_ID_CONFLICT", None, hint,
                                            "LIVE_DISCOVERY", compatible)
                former = list(dict.fromkeys((*(old.former_names or []), old.official_name)))
                identity = replace(
                    identity, team_id=old.team_id,
                    aliases=list(dict.fromkeys((*old.aliases, *identity.aliases))),
                    former_names=former,
                    verification_evidence_ids=list(dict.fromkeys(
                        (*(old.verification_evidence_ids or []), candidate.evidence_id))),
                )
            self.store.save(identity, provider_id=candidate.provider_id,
                            provider_team_id=candidate.provider_team_id,
                            verified_at=candidate.fetched_at)
        return EntityResolution("VERIFIED", identity, hint, "LIVE_DISCOVERY", compatible)
