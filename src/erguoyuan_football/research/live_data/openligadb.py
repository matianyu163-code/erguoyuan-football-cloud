"""Documented OpenLigaDB JSON adapter through the shared audited HTTP client."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.config import RateLimitPolicy
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from erguoyuan_football.research.live_data.schemas import VerifiedFixture
from erguoyuan_football.research.provider_coverage_profile import (
    ProviderCoverageProfile,
)
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.sources.source_schema import SourceRecord
from erguoyuan_football.web_research.time_utils import parse_utc, utc_iso


@dataclass(frozen=True)
class TeamBinding:
    """Reviewed exact mapping from local team ID to one provider ID and name."""

    provider_team_id: int
    provider_name: str


@dataclass(frozen=True)
class OpenLigaDBConfig:
    """Explicit one-league source configuration, not an arbitrary user URL."""

    provider_id: str
    base_url: str
    league_shortcut: str
    league_season: int
    competition_id: str
    competition_name: str
    team_bindings: dict[str, TeamBinding]
    enabled: bool
    license: str
    license_url: str
    team_country: str = "UNKNOWN"
    team_federation: str = "UNKNOWN"
    competition_gender: str = "UNKNOWN"
    competition_age_group: str = "SENIOR"
    competition_entity_type: str = "CLUB"
    history_seasons: tuple[int, ...] = ()
    competition_type: str = "LEAGUE"
    competition_aliases: tuple[str, ...] = ()

    @classmethod
    def from_yaml(cls, path: Path | str) -> OpenLigaDBConfig:
        """Load a reviewed allowlisted provider contract from local configuration."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("INVALID_PROVIDER_CONFIG")
        if raw.get("source_tier") != 2 or raw.get("source_type") != "STRUCTURED":
            raise ValueError("OPENLIGADB_MUST_BE_STRUCTURED_TIER_B")
        url = str(raw.get("base_url", ""))
        parsed = urlparse(url)
        if (parsed.scheme != "https" or parsed.netloc != "api.openligadb.de"
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            raise ValueError("OPENLIGADB_HOST_NOT_ALLOWLISTED")
        bindings = raw.get("team_bindings")
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError("TEAM_BINDINGS_REQUIRED")
        mapped = {str(key): TeamBinding(int(value["provider_team_id"]),
                                        str(value["provider_name"]))
                  for key, value in bindings.items()}
        if any(binding.provider_team_id <= 0 or not binding.provider_name
               for binding in mapped.values()):
            raise ValueError("INVALID_TEAM_BINDING")
        season = int(raw["league_season"])
        league = str(raw["league_shortcut"])
        if not (2000 <= season <= 2100 and league.isalnum()):
            raise ValueError("INVALID_LEAGUE_SEASON")
        return cls(str(raw["provider_id"]), url.rstrip("/"), league, season,
                   str(raw["competition_id"]), str(raw["competition_name"]),
                   mapped, raw.get("enabled") is True,
                   str(raw.get("license", "")), str(raw.get("license_url", "")),
                   str(raw.get("team_country", "UNKNOWN")),
                   str(raw.get("team_federation", "UNKNOWN")),
                   str(raw.get("competition_gender", "UNKNOWN")),
                   str(raw.get("competition_age_group", "SENIOR")),
                   str(raw.get("competition_entity_type", "CLUB")),
                   tuple(int(value) for value in raw.get("history_seasons", ())),
                   str(raw.get("competition_type", "LEAGUE")),
                   tuple(str(value) for value in raw.get("competition_aliases", ())))


@dataclass(frozen=True)
class SeasonBatch:
    """One real response with actual retrieval time; no historical backdating."""

    matches: tuple[dict[str, Any], ...]
    retrieved_at: datetime
    source_url: str
    http_status: int
    latency_ms: float
    season: int | None = None


@dataclass(frozen=True)
class FixtureCheck:
    """Exact match or an explicit rejected/ambiguous candidate set."""

    status: str
    fixture: VerifiedFixture | None
    candidates: tuple[dict[str, Any], ...]


class OpenLigaDBProvider:
    """Read a documented season endpoint; parse only its verified field names."""

    def __init__(self, config: OpenLigaDBConfig,
                 *, client: CoreNetworkClient | None = None) -> None:
        self.config = config
        endpoint = (f"{config.base_url}/getmatchdata/"
                    f"{config.league_shortcut}/{config.league_season}")
        teams_endpoint = (f"{config.base_url}/getavailableteams/"
                          f"{config.league_shortcut}/{config.league_season}")
        endpoints = {"season_matches": endpoint, "available_teams": teams_endpoint}
        for season in config.history_seasons:
            if season == config.league_season or not 2000 <= season <= 2100:
                raise ValueError("INVALID_HISTORY_SEASON")
            endpoints[f"season_matches_{season}"] = (
                f"{config.base_url}/getmatchdata/{config.league_shortcut}/{season}")
        definition = SourceDefinition(
            source_id=config.provider_id, display_name="OpenLigaDB",
            category="WEB_RESEARCH", base_url=config.base_url,
            endpoints=endpoints,
            schema_version="OPENLIGADB_V1", supports_live=True,
            rate_limit_policy=RateLimitPolicy(requests_per_second=1,
                                              requests_per_minute=60, burst=1),
        )
        registry = ExternalSourceRegistry((definition,))
        self.sources = SourceRegistry(registry)
        self.source_record = SourceRecord(
            config.provider_id, "OpenLigaDB", "STRUCTURED", 2,
            config.base_url, config.enabled,
            allowed_domains=(urlparse(config.base_url).hostname or "",),
        )
        self.sources.add(self.source_record)
        self._owns_client = client is None
        self.client = client or CoreNetworkClient(registry)
        self.endpoint = endpoint
        self.teams_endpoint = teams_endpoint

    def close(self) -> None:
        """Close the owned pooled transport."""
        if self._owns_client:
            self.client.close()

    def coverage_profile(self) -> ProviderCoverageProfile:
        """Return scope declarations as UNVERIFIED until a caller validates live data."""
        config = self.config
        capabilities = frozenset({"FIXTURE", "RESULTS", "HISTORICAL_RESULTS",
                                  "TEAM_DIRECTORY", "STANDINGS"})
        return ProviderCoverageProfile(
            config.provider_id, 2, frozenset({"EUROPE"}),
            frozenset({config.team_country}), frozenset({config.team_federation}),
            frozenset({config.competition_gender}), frozenset({config.competition_age_group}),
            frozenset({config.competition_entity_type}), frozenset({"FIRST_TEAM"}),
            frozenset({"LEAGUE"}), capabilities, 20, True, False, True,
            f"{config.license}; community-maintained results; API rate limit applies",
        )

    def fetch_season(self, season: int | None = None, *, bypass_cache: bool = True) -> SeasonBatch:
        """Fetch once; reject non-list, non-football or mismatched-season bodies."""
        if not self.config.enabled:
            raise ValueError("PROVIDER_UNCONFIGURED")
        selected = self.config.league_season if season is None else season
        endpoint_id = ("season_matches" if selected == self.config.league_season
                       else f"season_matches_{selected}")
        if endpoint_id not in self.sources.network_registry.get(self.config.provider_id).endpoints:
            raise ValueError("SEASON_NOT_ALLOWLISTED")
        response = self.client.fetch_json(
            self.config.provider_id, endpoint_id, params={},
            prediction_time=None, bypass_cache=bypass_cache,
        )
        rows = response.body
        if (not isinstance(rows, list) or not rows
                or any(not isinstance(row, dict) for row in rows)):
            raise ValueError("INVALID_PROVIDER_RESPONSE:MATCH_LIST_REQUIRED")
        if any(row.get("leagueShortcut") != self.config.league_shortcut
               or str(row.get("leagueSeason")) != str(selected)
               for row in rows):
            raise ValueError("INVALID_PROVIDER_RESPONSE:LEAGUE_MISMATCH")
        self.sources.validate_result_url(self.config.provider_id,
                                         response.final_url or self.endpoint)
        return SeasonBatch(tuple(rows), response.retrieved_at,
                           response.final_url or self.endpoint,
                           response.status_code, response.latency_ms, selected)

    def verify_fixture(self, batch: SeasonBatch, home_team_id: str,
                       away_team_id: str, *, date_hint: date | None = None,
                       cutoff: datetime) -> FixtureCheck:
        """Require exact provider IDs, names, competition, UTC kickoff and orientation."""
        if batch.retrieved_at > cutoff:
            return FixtureCheck("FUTURE_EVIDENCE_REJECTED", None, ())
        home = self.config.team_bindings.get(home_team_id)
        away = self.config.team_bindings.get(away_team_id)
        if home is None or away is None:
            return FixtureCheck("TEAM_BINDING_UNAVAILABLE", None, ())
        exact: list[dict[str, Any]] = []
        reverse: list[dict[str, Any]] = []
        for row in batch.matches:
            team1, team2 = row.get("team1"), row.get("team2")
            if not isinstance(team1, dict) or not isinstance(team2, dict):
                continue
            kickoff_text = row.get("matchDateTimeUTC")
            if not isinstance(kickoff_text, str):
                continue
            try:
                kickoff = parse_utc(kickoff_text)
            except ValueError:
                continue
            if (date_hint is not None and kickoff.date() != date_hint) or kickoff <= cutoff:
                continue
            if row.get("leagueShortcut") != self.config.league_shortcut:
                continue
            if (team1.get("teamId") == home.provider_team_id
                    and team2.get("teamId") == away.provider_team_id
                    and team1.get("teamName") == home.provider_name
                    and team2.get("teamName") == away.provider_name):
                exact.append(row)
            elif (team1.get("teamId") == away.provider_team_id
                  and team2.get("teamId") == home.provider_team_id):
                reverse.append(row)
        if len(exact) > 1:
            return FixtureCheck("AMBIGUOUS", None, tuple(exact))
        if not exact:
            return FixtureCheck("REVERSE_FIXTURE" if reverse else "NOT_FOUND",
                                None, tuple(reverse))
        row = exact[0]
        match_id = row.get("matchID")
        if not isinstance(match_id, int) or match_id <= 0:
            return FixtureCheck("INVALID_PROVIDER_RESPONSE", None, tuple(exact))
        location = row.get("location")
        venue = location.get("locationStadium") if isinstance(location, dict) else None
        fixture = VerifiedFixture(
            str(match_id), home_team_id, away_team_id,
            self.config.competition_id, parse_utc(row["matchDateTimeUTC"]),
            venue if isinstance(venue, str) else None, batch.source_url,
            self.config.provider_id, batch.retrieved_at,
            "EXACT" if date_hint is not None else "HIGH", 2,
            row.get("neutralVenue") if isinstance(row.get("neutralVenue"), bool) else None,
        )
        return FixtureCheck("VERIFIED", fixture, tuple(exact))

    def historical_results(self, batch: SeasonBatch, fixture: VerifiedFixture,
                           cutoff: datetime) -> tuple[EvidenceRecord, ...]:
        """Store only completed, score-typed rows observed by the live cutoff."""
        if batch.retrieved_at > cutoff:
            return ()
        records: list[EvidenceRecord] = []
        for row in batch.matches:
            team1, team2 = row.get("team1"), row.get("team2")
            if not isinstance(team1, dict) or not isinstance(team2, dict):
                continue
            raw_kickoff = row.get("matchDateTimeUTC")
            try:
                kickoff = parse_utc(raw_kickoff) if isinstance(raw_kickoff, str) else None
            except ValueError:
                continue
            if (kickoff is None or kickoff >= cutoff or kickoff >= fixture.kickoff_at
                    or row.get("matchIsFinished") is not True):
                continue
            results = row.get("matchResults")
            if not isinstance(results, list):
                continue
            final = [result for result in results if isinstance(result, dict)
                     and result.get("resultTypeKind") == "After90Minutes"]
            if len(final) != 1:
                continue
            home_goals, away_goals = final[0].get("pointsTeam1"), final[0].get("pointsTeam2")
            match_id = row.get("matchID")
            if (not isinstance(match_id, int) or isinstance(home_goals, bool)
                    or isinstance(away_goals, bool)
                    or not isinstance(home_goals, int) or not isinstance(away_goals, int)
                    or min(home_goals, away_goals) < 0
                    or not isinstance(team1.get("teamId"), int)
                    or not isinstance(team2.get("teamId"), int)):
                continue
            value = {
                "provider_match_id": str(match_id), "team1_provider_id": team1["teamId"],
                "team2_provider_id": team2["teamId"], "home_goals": home_goals,
                "away_goals": away_goals, "kickoff_time": utc_iso(kickoff),
                "competition_id": self.config.competition_id,
            }
            digest = hashlib.sha256(f"{match_id}:{utc_iso(batch.retrieved_at)}".encode()).hexdigest()
            records.append(EvidenceRecord(
                f"OPENLIGADB_RESULT_{digest}", "HISTORICAL_RESULT", value,
                self.config.provider_id, None, utc_iso(batch.retrieved_at), "MEDIUM",
                batch.source_url, utc_iso(batch.retrieved_at),
                provider_id=self.config.provider_id, source_tier=2,
                observed_time=utc_iso(batch.retrieved_at),
                match_key=f"MATCH:{fixture.provider_match_id}",
            ))
        return tuple(records)
