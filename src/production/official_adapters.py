"""Site-level JSON-LD fixture extraction from explicitly registered official pages."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.network.client import CoreNetworkClient
from production.association_sources import AssociationSource
from production.global_fixture_research import FixtureResearchEvidence
from production.official_source_registry import OfficialFootballSource


class _JsonLdScripts(HTMLParser):
    """Collect structured-data scripts without evaluating page JavaScript."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._active = False
        self._buffer: list[str] = []
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "script":
            return
        values = {key.casefold(): (value or "") for key, value in attrs}
        self._active = values.get("type", "").casefold() == "application/ld+json"
        if self._active:
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._active:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "script" and self._active:
            self.scripts.append("".join(self._buffer))
            self._active = False
            self._buffer = []


class OfficialWebsiteAdapter:
    """Read only configured pages, then extract exact official Event records."""

    def __init__(
        self,
        *,
        source: OfficialFootballSource,
        endpoint_ids: tuple[str, ...],
        client: CoreNetworkClient,
        home_team: TeamIdentity,
        away_team: TeamIdentity,
    ) -> None:
        if not endpoint_ids:
            raise ValueError("OFFICIAL_FIXTURE_ENDPOINTS_REQUIRED")
        self.source = source
        self.provider_id = source.source_id
        self.source_tier = 3
        self.allowed_domains = frozenset({source.official_domain})
        self.endpoint_ids = endpoint_ids
        self.client = client
        self.home_team = home_team
        self.away_team = away_team

    def search(
        self, query: str, *, source: AssociationSource | None, as_of_time: datetime
    ) -> tuple[FixtureResearchEvidence, ...]:
        """Filter official site events locally; ``query`` is never sent over HTTP."""
        del query
        if source is not None and source.association_id != self.source.association:
            return ()
        requested_host = self.source.official_domain.casefold()
        result: list[FixtureResearchEvidence] = []
        seen: set[tuple[str, str, datetime | None]] = set()
        for endpoint_id in self.endpoint_ids:
            page = self.client.fetch_text(
                self.source.source_id, endpoint_id, prediction_time=as_of_time
            )
            parsed_url = urlparse(page.final_url or "")
            if (
                parsed_url.hostname is None
                or parsed_url.hostname.casefold() != requested_host
            ):
                raise ValueError("OFFICIAL_PAGE_DOMAIN_MISMATCH")
            parser = _JsonLdScripts()
            parser.feed(str(page.body))
            for script in parser.scripts:
                try:
                    payload = json.loads(script)
                except json.JSONDecodeError:
                    continue
                for event in _events(payload):
                    item = _extract_event(
                        event,
                        self.source,
                        page.final_url or "",
                        page.retrieved_at,
                        as_of_time,
                        self.home_team,
                        self.away_team,
                    )
                    if item is None:
                        continue
                    identity = (
                        item.home_entity_id,
                        item.away_entity_id,
                        item.kickoff_utc,
                    )
                    if identity not in seen:
                        seen.add(identity)
                        result.append(item)
        return tuple(result)


def _events(value: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(value, list):
        return tuple(event for item in value for event in _events(item))
    if not isinstance(value, dict):
        return ()
    graph = value.get("@graph")
    if isinstance(graph, list):
        return tuple(event for item in graph for event in _events(item))
    event_type = value.get("@type")
    types = event_type if isinstance(event_type, list) else [event_type]
    if any(
        str(item).casefold() in {"event", "sportsEvent".casefold()} for item in types
    ):
        return (value,)
    return ()


def _extract_event(
    event: dict[str, Any],
    source: OfficialFootballSource,
    url: str,
    retrieved_at: datetime,
    request_time: datetime,
    home_identity: TeamIdentity,
    away_identity: TeamIdentity,
) -> FixtureResearchEvidence | None:
    home = _participant_name(event.get("homeTeam"))
    away = _participant_name(event.get("awayTeam"))
    kickoff_raw = event.get("startDate")
    competition = (
        _participant_name(event.get("superEvent"))
        or str(event.get("name") or "").strip()
    )
    if not home or not away or not isinstance(kickoff_raw, str) or not competition:
        return None
    if not _exact_team_match(home, home_identity) or not _exact_team_match(
        away, away_identity
    ):
        return None
    try:
        kickoff = datetime.fromisoformat(kickoff_raw)
    except ValueError:
        kickoff = None
    if kickoff is not None and (kickoff.tzinfo is None or kickoff.utcoffset() is None):
        kickoff = None
    if kickoff is not None:
        kickoff = kickoff.astimezone(UTC)
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        return None
    # Entity identity is assigned only for an exact official/canonical label match.
    # The provider adapter cannot invent aliases or select a fuzzy team candidate.
    return FixtureResearchEvidence(
        source_id=source.source_id,
        source_url=url,
        source_name=source.official_name,
        source_tier=3,
        retrieved_at=retrieved_at,
        home_entity_id=home_identity.team_id,
        away_entity_id=away_identity.team_id,
        competition_name=competition,
        confidence="HIGH",
        kickoff_utc=kickoff,
        kickoff_local=kickoff_raw,
        kickoff_timezone=(
            str(event.get("timeZone")) if event.get("timeZone") else None
        ),
        venue=_participant_name(event.get("location")),
        published_at=_date_time(event.get("datePublished")),
        fixture_id=str(event.get("identifier") or event.get("@id") or "") or None,
    )


def _participant_name(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("name")
    if not isinstance(value, str):
        return None
    normalized = " ".join(re.sub(r"\s+", " ", value).split())
    return normalized or None


def _date_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else None


def _exact_team_match(page_name: str, identity: TeamIdentity) -> bool:
    """Match against resolver-owned exact aliases only, never fuzzy-select."""
    candidates = {
        " ".join(value.casefold().split())
        for value in (identity.official_name, *identity.aliases)
        if value
    }
    return " ".join(page_name.casefold().split()) in candidates
