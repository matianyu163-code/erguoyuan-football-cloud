"""Bounded site-level readers for explicitly registered public fixture pages."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import UTC, datetime
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from bs4 import BeautifulSoup

from erguoyuan_football.knowledge.entities.country_alias_registry import (
    CountryAliasRegistry,
)
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from production.association_sources import AssociationSource, AssociationSourceRegistry
from production.global_fixture_research import FixtureResearchEvidence
from production.network_diagnostics import network_policy_blocked
from production.official_adapters import _events, _extract_event, _JsonLdScripts
from production.official_source_registry import OfficialFootballSource


class OfficialLiveFixtureAdapter:
    """Read only declared entrypoints and same-host schedule links observed there."""

    source_tier = 3

    def __init__(self, source: OfficialFootballSource, home: TeamIdentity, away: TeamIdentity) -> None:
        if "FIXTURES" not in source.capabilities:
            raise ValueError("OFFICIAL_FIXTURE_ENTRYPOINT_NOT_CONFIGURED")
        self.source = source
        self.home = home
        self.away = away
        self.provider_id = source.source_id
        self.allowed_domains = frozenset({source.official_domain, f"www.{source.official_domain}"})
        self._result: tuple[FixtureResearchEvidence, ...] | None = None
        self._failure: str | None = None

    def search(
        self, query: str, *, source: AssociationSource | None, as_of_time: datetime
    ) -> tuple[FixtureResearchEvidence, ...]:
        """Keep query local and cache pages across multilingual search attempts."""
        del query
        if source is not None and source.association_id != self.source.association:
            return ()
        if self._result is not None:
            return self._result
        if self._failure is not None:
            raise ValueError(self._failure)
        try:
            self._result = self._read(as_of_time)
        except ValueError as error:
            self._failure = str(error)
            raise
        except (OSError, httpx.TransportError) as error:
            if network_policy_blocked(error):
                self._failure = "NETWORK_POLICY_BLOCKED"
                raise OSError("NETWORK_POLICY_BLOCKED") from error
            self._failure = "OFFICIAL_SOURCE_UNREACHABLE"
            raise OSError("OFFICIAL_SOURCE_UNREACHABLE") from error
        return self._result

    def _fetch(self, url: str, as_of_time: datetime):
        base = self.source.base_url
        if base is None:
            raise ValueError("OFFICIAL_FIXTURE_ENTRYPOINT_NOT_CONFIGURED")
        parsed = urlparse(url)
        if (parsed.scheme != "https" or parsed.hostname != urlparse(base).hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("OFFICIAL_PAGE_DOMAIN_MISMATCH")
        definition = SourceDefinition(
            source_id=self.provider_id, display_name=self.source.official_name,
            category="OFFICIAL_FIXTURE", base_url=base, endpoints={"page": url},
            schema_version="OFFICIAL_FIXTURE_PAGE_V1",
        )
        client = CoreNetworkClient(ExternalSourceRegistry((definition,)))
        try:
            return client.fetch_text(self.provider_id, "page", prediction_time=as_of_time)
        finally:
            client.close()

    def _read(self, as_of_time: datetime) -> tuple[FixtureResearchEvidence, ...]:
        entrypoint = self.source.fixture_entrypoint or self.source.schedule_entrypoint
        if entrypoint is None:
            raise ValueError("OFFICIAL_FIXTURE_ENTRYPOINT_NOT_CONFIGURED")
        page = self._fetch(entrypoint, as_of_time)
        if self.source.adapter == "UEFA_FIXTURES_V1":
            parser = _JsonLdScripts()
            parser.feed(str(page.body))
            found: list[FixtureResearchEvidence] = []
            import json
            for script in parser.scripts:
                try:
                    payload = json.loads(script)
                except json.JSONDecodeError:
                    continue
                for event in _events(payload):
                    item = _extract_event(event, self.source, page.final_url or entrypoint,
                                          page.retrieved_at, as_of_time, self.home, self.away)
                    if item is not None:
                        found.append(item)
            if not found and 'data-name="matches-calendar"' in str(page.body):
                raise ValueError("OFFICIAL_PAGE_CLIENT_RENDERED_NO_FIXTURE_ROWS")
            return tuple(found)
        if self.source.adapter == "UEFA_FIXTURES_ARTICLE_V1":
            return self._uefa_article(page, as_of_time)
        if self.source.adapter == "KNVB_FIXTURES_V1":
            return self._knvb(page, as_of_time)
        if self.source.adapter == "DFB_NATIONAL_SCHEDULE_V1":
            return self._dfb(page)
        raise ValueError("OFFICIAL_ADAPTER_NOT_CONFIGURED")

    def _knvb(self, directory, as_of_time: datetime) -> tuple[FixtureResearchEvidence, ...]:
        soup = BeautifulSoup(str(directory.body), "html.parser")
        team = self.home if self.home.country == "Netherlands" else self.away
        if team.country != "Netherlands":
            return ()
        age = team.age_group[1:] if team.age_group.startswith("U") else None
        if age is None:
            return ()
        group_label = "Jeugd vrouwen" if team.gender == "WOMEN" else "Jeugd mannen"
        links = []
        for link in soup.select('a[href^="/teams/"][href$="/overzicht"]'):
            group = link.find_parent(class_="OverviewList-group")
            if group is None:
                continue
            if (re.search(rf"\bOnder\s+{re.escape(age)}\b", str(link.get("title", "")), re.IGNORECASE)
                    and group_label.casefold() in group.get_text(" ", strip=True).casefold()):
                links.append(link)
        if len(links) != 1:
            return ()
        overview_url = urljoin(self.source.base_url or "", str(links[0].get("href", "")))
        overview = self._fetch(overview_url, as_of_time)
        overview_soup = BeautifulSoup(str(overview.body), "html.parser")
        schedule_links = overview_soup.select('a.DossierMenu-itemlink[href$="/programma"]')
        if len(schedule_links) != 1:
            return ()
        schedule = urljoin(self.source.base_url or "", str(schedule_links[0].get("href", "")))
        page = self._fetch(schedule, as_of_time)
        rows = BeautifulSoup(str(page.body), "html.parser").select(".Matchblock")
        found: list[FixtureResearchEvidence] = []
        for row in rows:
            home = row.select_one(".Matchblock-team--home .Matchblock-teamname")
            away = row.select_one(".Matchblock-team--away .Matchblock-teamname")
            match_link = row.select_one('a.Matchblock-wrapper[href*="/programma/wedstrijd/"]')
            competition = row.select_one(".Matchblock-tournament")
            date = row.select_one(".Matchblock-metadata--date")
            time = row.select_one(".Matchblock-metadata--time")
            if any(item is None for item in (home, away, match_link, competition, date, time)):
                continue
            assert home is not None and away is not None and match_link is not None
            assert competition is not None and date is not None and time is not None
            home_name = home.get_text(" ", strip=True)
            away_name = away.get_text(" ", strip=True)
            if not (_site_team_match(home_name, self.home, "nl")
                    and _site_team_match(away_name, self.away, "nl")):
                continue
            kickoff_local = _knvb_kickoff(date.get_text(" ", strip=True),
                                          time.get_text(" ", strip=True),
                                          self.source.timezone_policy)
            if kickoff_local is None:
                continue
            found.append(FixtureResearchEvidence(
                source_id=self.provider_id, source_url=page.final_url or schedule,
                source_name=self.source.official_name, source_tier=3,
                retrieved_at=page.retrieved_at, home_entity_id=self.home.team_id,
                away_entity_id=self.away.team_id,
                competition_name=competition.get_text(" ", strip=True), confidence="HIGH",
                kickoff_local=kickoff_local.isoformat(),
                kickoff_timezone=self.source.timezone_policy,
                fixture_id=str(match_link.get("href")),
                source_content_hash=page.content_hash,
            ))
        return tuple(found)

    def _dfb(self, page) -> tuple[FixtureResearchEvidence, ...]:
        soup = BeautifulSoup(str(page.body), "html.parser")
        found: list[FixtureResearchEvidence] = []
        for row in soup.select(".c-MatchTable-body .c-MatchTable-row"):
            home = row.select_one(".c-MatchTable-team--home[data-team-kind='national']")
            away = row.select_one(".c-MatchTable-team--away[data-team-kind='national']")
            info = row.select_one(".c-MatchTable-info--home .c-MatchTable-description")
            link = row.select_one(".c-MatchTable-score a[href]")
            if any(item is None for item in (home, away, info, link)):
                continue
            assert home is not None and away is not None and info is not None and link is not None
            if not (_site_team_match(home.get_text(" ", strip=True), self.home, "de")
                    and _site_team_match(away.get_text(" ", strip=True), self.away, "de")):
                continue
            paragraphs = info.select("p")
            if len(paragraphs) < 2:
                continue
            kickoff_local = _dfb_kickoff(
                paragraphs[1].get_text(" ", strip=True), self.source.timezone_policy
            )
            if kickoff_local is None:
                continue
            found.append(FixtureResearchEvidence(
                source_id=self.provider_id, source_url=page.final_url or self.source.fixture_entrypoint or "",
                source_name=self.source.official_name, source_tier=3,
                retrieved_at=page.retrieved_at, home_entity_id=self.home.team_id,
                away_entity_id=self.away.team_id,
                competition_name=paragraphs[0].get_text(" ", strip=True), confidence="HIGH",
                kickoff_local=kickoff_local.isoformat(),
                kickoff_timezone=self.source.timezone_policy,
                fixture_id=str(link.get("href")),
                source_content_hash=page.content_hash,
            ))
        return tuple(found)

    def _uefa_article(self, article, as_of_time: datetime) -> tuple[FixtureResearchEvidence, ...]:
        """Follow an exact team-pair link from UEFA's public competition schedule."""
        soup = BeautifulSoup(str(article.body), "html.parser")
        matches: dict[str, FixtureResearchEvidence] = {}
        for anchor in soup.find_all("a", href=True):
            label = anchor.get_text(" ", strip=True)
            pair = re.split(r"\s+(?:vs?\.?|v\.?|[-–])\s+", label, maxsplit=1,
                            flags=re.IGNORECASE)
            if len(pair) != 2 or not (
                _site_team_match(pair[0], self.home, "en")
                and _site_team_match(pair[1], self.away, "en")
            ):
                continue
            event_url = urljoin(self.source.base_url or "", str(anchor.get("href")))
            if not re.search(r"/match/[^/]+/?$", urlparse(event_url).path):
                continue
            event_page = self._fetch(event_url, as_of_time)
            parser = _JsonLdScripts()
            parser.feed(str(event_page.body))
            import json
            for script in parser.scripts:
                try:
                    payload = json.loads(script)
                except json.JSONDecodeError:
                    continue
                event = _sports_event(payload)
                if event is None:
                    continue
                event = {**event, "superEvent": {"name": "UEFA Nations League"}}
                evidence = _extract_event(
                    event, self.source, event_page.final_url or event_url,
                    event_page.retrieved_at, as_of_time, self.home, self.away,
                )
                if evidence is not None:
                    evidence = replace(evidence, source_content_hash=event_page.content_hash)
                    key = evidence.fixture_id or event_url
                    matches[key] = evidence
        return tuple(matches.values())


def _site_team_match(label: str, team: TeamIdentity, language: str) -> bool:
    """Match exact country/age/gender identity, never fuzzy-select a side."""
    countries = CountryAliasRegistry()
    country = countries.resolve(team.country)
    if country is None:
        return False
    association = AssociationSourceRegistry().get(country.iso3)
    names = {team.country, country.name, *country.aliases, *team.aliases}
    if language == "nl":
        names.update(_DUTCH_COUNTRY_LABELS.get(team.country.casefold(), ()))
    if association:
        names.update(alias for lang, alias in association.country_aliases if lang == language)
    age = team.age_group[1:] if team.age_group.startswith("U") else ""
    normalized = re.sub(r"\s+", " ", label).strip().casefold()
    for name in names:
        if not name:
            continue
        candidates = {name.casefold()}
        if age:
            candidates.update({f"{name} U{age}".casefold(), f"{name} O{age}".casefold(),
                               f"{name} U {age}".casefold(), f"{name} O {age}".casefold(),
                               f"{name} U {age} (m)".casefold()})
        if normalized in candidates:
            return True
    return False


_DUTCH_COUNTRY_LABELS: dict[str, tuple[str, ...]] = {
    "austria": ("Oostenrijk",),
    "belgium": ("België",),
    "bulgaria": ("Bulgarije",),
    "croatia": ("Kroatië",),
    "czechia": ("Tsjechië", "Tsjechische Republiek"),
    "czech republic": ("Tsjechië", "Tsjechische Republiek"),
    "denmark": ("Denemarken",),
    "england": ("Engeland",),
    "finland": ("Finland",),
    "france": ("Frankrijk",),
    "germany": ("Duitsland",),
    "greece": ("Griekenland",),
    "hungary": ("Hongarije",),
    "iceland": ("IJsland",),
    "ireland": ("Ierland", "Republiek Ierland"),
    "israel": ("Israël",),
    "italy": ("Italië",),
    "luxembourg": ("Luxemburg",),
    "moldova": ("Moldavië",),
    "netherlands": ("Nederland",),
    "norway": ("Noorwegen",),
    "poland": ("Polen",),
    "portugal": ("Portugal",),
    "romania": ("Roemenië",),
    "scotland": ("Schotland",),
    "serbia": ("Servië",),
    "slovakia": ("Slowakije",),
    "slovenia": ("Slovenië",),
    "spain": ("Spanje",),
    "sweden": ("Zweden",),
    "switzerland": ("Zwitserland",),
    "turkiye": ("Turkije",),
    "ukraine": ("Oekraïne",),
    "wales": ("Wales",),
}


def _sports_event(payload: object) -> dict[str, object] | None:
    """Find an explicitly marked SportsEvent nested in an official JSON-LD object."""
    if isinstance(payload, dict):
        event_type = payload.get("@type")
        if event_type == "SportsEvent" or (
            isinstance(event_type, list) and "SportsEvent" in event_type
        ):
            return payload
        for value in payload.values():
            if found := _sports_event(value):
                return found
    elif isinstance(payload, list):
        for value in payload:
            if found := _sports_event(value):
                return found
    return None


def _knvb_kickoff(date_text: str, time_text: str, timezone_name: str) -> datetime | None:
    """Parse a dated KNVB local kickoff; undetermined schedule times stay absent."""
    months = {"jan": 1, "feb": 2, "mrt": 3, "apr": 4, "mei": 5, "jun": 6,
              "jul": 7, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "dec": 12}
    date_match = re.search(
        r"\b(\d{1,2})\s+([a-z]{3})\s+(\d{4})\b", date_text, re.IGNORECASE
    )
    time_match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", time_text)
    if date_match is None or time_match is None:
        return None
    month = months.get(date_match.group(2).casefold())
    if month is None:
        return None
    try:
        local = datetime.fromisoformat(
            f"{int(date_match.group(3)):04d}-{month:02d}-{int(date_match.group(1)):02d}"
            f"T{int(time_match.group(1)):02d}:{int(time_match.group(2)):02d}"
        )
        return _localize_kickoff(local, timezone_name)
    except ValueError:
        return None


def _dfb_kickoff(value: str, timezone_name: str) -> datetime | None:
    """Parse the DFB schedule's explicit local date and time."""
    match = re.search(r"\b(\d{2})\.(\d{2})\.(\d{4})\s+(\d{1,2}):(\d{2})\b", value)
    if match is None:
        return None
    try:
        local = datetime.fromisoformat(
            f"{int(match.group(3)):04d}-{int(match.group(2)):02d}-{int(match.group(1)):02d}"
            f"T{int(match.group(4)):02d}:{int(match.group(5)):02d}"
        )
        return _localize_kickoff(local, timezone_name)
    except ValueError:
        return None


def _localize_kickoff(local: datetime, timezone_name: str) -> datetime | None:
    """Apply the registered source timezone and reject DST gaps or ambiguity."""
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return None
    first = local.replace(tzinfo=zone, fold=0)
    second = local.replace(tzinfo=zone, fold=1)
    if first.utcoffset() != second.utcoffset():
        return None
    if first.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != local:
        return None
    return first
