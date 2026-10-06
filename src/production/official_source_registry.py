"""Allowlisted official football sources and capability-based source routing."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from erguoyuan_football.knowledge.entities.country_alias_registry import (
    CountryAliasRegistry,
)
from production.association_sources import AssociationSourceRegistry

CAPABILITIES = frozenset(
    {"FIXTURES", "RESULTS", "TEAMS", "COMPETITIONS", "KICKOFF", "NEWS", "LINEUPS"}
)


@dataclass(frozen=True)
class OfficialFootballSource:
    """A declared official organization/domain scope; no endpoints are guessed."""

    source_id: str
    official_name: str
    official_domain: str
    source_type: str
    priority: int
    supported_languages: tuple[str, ...]
    capabilities: frozenset[str]
    country: str | None = None
    association: str | None = None
    competition: str | None = None
    entity_type: str | None = None
    base_url: str | None = None
    fixture_entrypoint: str | None = None
    schedule_entrypoint: str | None = None
    adapter: str | None = None
    timezone_policy: str = "SOURCE_OFFSET_REQUIRED"

    def __post_init__(self) -> None:
        if (
            not self.source_id
            or not self.official_name
            or not self.official_domain
            or self.official_domain.startswith(".")
            or "/" in self.official_domain
            or ":" in self.official_domain
        ):
            raise ValueError("OFFICIAL_SOURCE_IDENTITY_INVALID")
        if not self.capabilities or not self.capabilities <= CAPABILITIES:
            raise ValueError("OFFICIAL_SOURCE_CAPABILITY_INVALID")
        if self.source_type not in {
            "ASSOCIATION",
            "CONTINENTAL",
            "FIFA",
            "LEAGUE",
            "CLUB",
        }:
            raise ValueError("OFFICIAL_SOURCE_TYPE_INVALID")
        if "FIXTURES" in self.capabilities:
            entrypoint = self.fixture_entrypoint or self.schedule_entrypoint
            if not self.base_url or not entrypoint or not self.adapter:
                raise ValueError("OFFICIAL_FIXTURE_ENTRYPOINT_NOT_CONFIGURED")
            for url in (self.base_url, entrypoint):
                parsed = urlparse(url)
                if parsed.scheme != "https" or parsed.hostname not in {
                    self.official_domain, f"www.{self.official_domain}"
                } or parsed.username or parsed.password or parsed.query or parsed.fragment:
                    raise ValueError("OFFICIAL_FIXTURE_ENTRYPOINT_INVALID")
            if not self.timezone_policy:
                raise ValueError("OFFICIAL_TIMEZONE_POLICY_REQUIRED")


class OfficialFootballSourceRegistry:
    """Registry for trusted domains, routed by country/competition/entity scope."""

    version = "OFFICIAL_FOOTBALL_SOURCE_REGISTRY_V1"

    def __init__(
        self, sources: tuple[OfficialFootballSource, ...] | None = None
    ) -> None:
        self._sources = sources if sources is not None else _initial_sources()
        if len({source.source_id for source in self._sources}) != len(self._sources):
            raise ValueError("DUPLICATE_OFFICIAL_SOURCE_ID")
        self._domains = {source.official_domain.casefold() for source in self._sources}

    def all(self) -> tuple[OfficialFootballSource, ...]:
        """Return all registered sources in stable priority order."""
        return tuple(
            sorted(self._sources, key=lambda item: (item.priority, item.source_id))
        )

    def get(self, source_id: str) -> OfficialFootballSource | None:
        """Resolve an exact source identifier."""
        return next(
            (item for item in self._sources if item.source_id == source_id), None
        )

    def validate_domain(self, domain: str) -> bool:
        """Check an exact allowlisted hostname; subdomains are not implicit."""
        return domain.casefold() in self._domains

    def for_national_teams(
        self, *country_names: str
    ) -> tuple[OfficialFootballSource, ...]:
        """Resolve national association sources from canonical country identity."""
        countries = CountryAliasRegistry()
        ids = {
            entry.iso3
            for name in country_names
            if (entry := countries.resolve(name)) is not None
        }
        associations = AssociationSourceRegistry()
        source_ids = {
            row.association_id
            for country_id in ids
            if (row := associations.get(country_id)) is not None
        }
        return tuple(item for item in self.all() if item.association in source_ids)

    def for_competition(
        self, competition: str | None
    ) -> tuple[OfficialFootballSource, ...]:
        """Prioritize an official competition owner, if explicitly registered."""
        if not competition:
            return ()
        key = " ".join(competition.casefold().split())
        exact = tuple(
            item
            for item in self.all()
            if item.competition and " ".join(item.competition.casefold().split()) == key
        )
        return exact

    def fixture_sources(self) -> tuple[OfficialFootballSource, ...]:
        """Only explicitly configured official fixture entrypoints may be queried."""
        return tuple(item for item in self.all() if "FIXTURES" in item.capabilities)


def _initial_sources() -> tuple[OfficialFootballSource, ...]:
    sources: list[OfficialFootballSource] = []
    for row in AssociationSourceRegistry().all():
        configured = {
            "KNVB": ("https://www.onsoranje.nl", "https://www.onsoranje.nl/teams", "KNVB_FIXTURES_V1", "Europe/Amsterdam"),
            "DFB": ("https://datencenter.dfb.de", "https://datencenter.dfb.de/datencenter/naechste-spiele", "DFB_NATIONAL_SCHEDULE_V1", "Europe/Berlin"),
        }.get(row.association_id)
        sources.append(
            OfficialFootballSource(
                source_id=row.association_id,
                official_name=row.association_id,
                official_domain=("datencenter.dfb.de" if row.association_id == "DFB" else row.official_domains[0]),
                source_type="ASSOCIATION",
                priority=20,
                supported_languages=row.supported_languages,
                capabilities=frozenset(
                    {"FIXTURES", "RESULTS", "TEAMS", "COMPETITIONS", "KICKOFF"}
                    if configured else {"TEAMS", "COMPETITIONS", "NEWS"}
                ),
                country=row.country_id,
                association=row.association_id,
                entity_type="NATIONAL_TEAM",
                base_url=configured[0] if configured else None,
                schedule_entrypoint=configured[1] if configured else None,
                adapter=configured[2] if configured else None,
                timezone_policy=configured[3] if configured else "SOURCE_OFFSET_REQUIRED",
            )
        )
    for source_id, name, domain, kind, priority in (
        ("FIFA", "FIFA", "fifa.com", "FIFA", 10),
        ("UEFA", "UEFA", "uefa.com", "CONTINENTAL", 11),
        ("AFC", "Asian Football Confederation", "the-afc.com", "CONTINENTAL", 12),
        (
            "CAF",
            "Confederation of African Football",
            "cafonline.com",
            "CONTINENTAL",
            12,
        ),
        ("CONMEBOL", "CONMEBOL", "conmebol.com", "CONTINENTAL", 12),
        ("CONCACAF", "Concacaf", "concacaf.com", "CONTINENTAL", 12),
        (
            "OFC",
            "Oceania Football Confederation",
            "oceaniafootball.com",
            "CONTINENTAL",
            12,
        ),
    ):
        sources.append(
            OfficialFootballSource(
                source_id=source_id,
                official_name=name,
                official_domain=domain,
                source_type=kind,
                priority=priority,
                supported_languages=("en",),
                competition="UEFA Nations League" if source_id == "UEFA" else None,
                capabilities=frozenset(
                    {"FIXTURES", "RESULTS", "TEAMS", "COMPETITIONS", "KICKOFF"}
                    if source_id == "UEFA" else {"TEAMS", "COMPETITIONS"}
                ),
                base_url="https://www.uefa.com" if source_id == "UEFA" else None,
                fixture_entrypoint=(
                    "https://www.uefa.com/uefanationsleague/news/"
                    "02a2-1fea7079900a-801b2c5f6c9e-1000--"
                    "2026-27-uefa-nations-league-league-phase-fixtures-and-resul/"
                    if source_id == "UEFA" else None
                ),
                adapter="UEFA_FIXTURES_ARTICLE_V1" if source_id == "UEFA" else None,
                timezone_policy="Europe/Zurich" if source_id == "UEFA" else "SOURCE_OFFSET_REQUIRED",
            )
        )
    for source_id, name, domain, competition in (
        ("PREMIER_LEAGUE", "Premier League", "premierleague.com", "Premier League"),
        ("BUNDESLIGA", "Bundesliga", "bundesliga.com", "Bundesliga"),
        ("2_BUNDESLIGA", "2. Bundesliga", "bundesliga.com", "2. Bundesliga"),
        ("LALIGA", "LALIGA", "laliga.com", "LaLiga"),
        ("SERIE_A", "Lega Serie A", "legaseriea.it", "Serie A"),
        ("LIGUE_1", "Ligue 1", "ligue1.com", "Ligue 1"),
        ("EREDIVISIE", "Eredivisie", "eredivisie.nl", "Eredivisie"),
        (
            "UEFA_CHAMPIONS_LEAGUE",
            "UEFA Champions League",
            "uefa.com",
            "UEFA Champions League",
        ),
    ):
        sources.append(
            OfficialFootballSource(
                source_id=source_id,
                official_name=name,
                official_domain=domain,
                source_type="LEAGUE",
                priority=15,
                supported_languages=("en",),
                capabilities=frozenset({"TEAMS", "COMPETITIONS"}),
                competition=competition,
            )
        )
    return tuple(sources)
