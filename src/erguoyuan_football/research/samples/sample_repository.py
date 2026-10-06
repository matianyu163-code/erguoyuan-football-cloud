"""Read existing evidence as canonical historical samples; no second database."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from erguoyuan_football.knowledge.entities.entity_fingerprint import (
    canonical_entity_id,
    entity_fingerprint,
)
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBConfig,
    SeasonBatch,
)
from erguoyuan_football.research.match_package import MatchResearchPackage
from erguoyuan_football.research.samples.match_deduplicator import (
    HistoricalMatchSample,
)
from erguoyuan_football.web_research.policies.freshness_policy import (
    is_valid_for_cutoff,
)
from erguoyuan_football.web_research.time_utils import parse_utc


class SampleRepository:
    """PIT-filtered query surface over existing source evidence, not a new store."""

    def __init__(self, samples: tuple[HistoricalMatchSample, ...]) -> None:
        self.samples = samples

    def get_team_matches(self, team_id: str, cutoff: datetime
                         ) -> tuple[HistoricalMatchSample, ...]:
        """Direct means this exact canonical ID participated."""
        return tuple(row for row in self.samples if row.kickoff < cutoff
                     and row.fetched_at <= cutoff
                     and team_id in {row.home_team_id, row.away_team_id})

    def get_competition_matches(self, competition_id: str, cutoff: datetime
                                ) -> tuple[HistoricalMatchSample, ...]:
        """Return only exact competition ID rows before cutoff."""
        return tuple(row for row in self.samples if row.competition_id == competition_id
                     and row.kickoff < cutoff and row.fetched_at <= cutoff)

    def get_comparable_matches(self, *, federation: str, gender: str,
                               age_group: str, competition_type: str,
                               exclude_competition: str, cutoff: datetime
                               ) -> tuple[HistoricalMatchSample, ...]:
        """Comparison requires known exact federation/gender/age/type matches."""
        if "UNKNOWN" in {federation, gender, age_group, competition_type}:
            return ()
        return tuple(row for row in self.samples if row.competition_id != exclude_competition
                     and row.federation == federation and row.gender == gender
                     and row.age_group == age_group
                     and row.competition_type == competition_type
                     and row.kickoff < cutoff and row.fetched_at <= cutoff)

    def get_prior_matches(self, *, federation: str, gender: str,
                          age_group: str, cutoff: datetime
                          ) -> tuple[HistoricalMatchSample, ...]:
        """Only same-gender federation rows, never relabeled as direct."""
        if "UNKNOWN" in {federation, gender, age_group}:
            return ()
        return tuple(row for row in self.samples if row.federation == federation
                     and row.gender == gender and row.age_group != age_group
                     and row.kickoff < cutoff and row.fetched_at <= cutoff)

    @classmethod
    def from_openligadb(cls, batch: SeasonBatch, package: MatchResearchPackage,
                        config: OpenLigaDBConfig, cutoff: datetime, *,
                        provider_team_ids_by_source: Mapping[str, Mapping[int, str]] | None = None
                        ) -> SampleRepository:
        """Translate source IDs to reviewed canonical IDs; preserve evidence ancestry."""
        names: dict[int, str] = {}
        ambiguous_ids: set[int] = set()
        for row in batch.matches:
            for key in ("team1", "team2"):
                team = row.get(key)
                if isinstance(team, dict) and isinstance(team.get("teamId"), int) and (
                    isinstance(team.get("teamName"), str)):
                    existing = names.get(team["teamId"])
                    if existing is not None and existing != team["teamName"]:
                        ambiguous_ids.add(team["teamId"])
                    names[team["teamId"]] = team["teamName"]
        reviewed = {binding.provider_team_id: team_id
                    for team_id, binding in config.team_bindings.items()}

        source_maps = provider_team_ids_by_source or {}

        def canonical(provider_id: str, provider_team_id: int) -> str | None:
            source_map = source_maps.get(provider_id, {})
            if provider_team_id in source_map:
                return source_map[provider_team_id]
            if provider_id != config.provider_id:
                return None
            if provider_team_id in ambiguous_ids:
                return None
            if provider_team_id in reviewed:
                return reviewed[provider_team_id]
            name = names.get(provider_team_id)
            if name is None:
                return None
            fingerprint = entity_fingerprint(
                country=config.team_country, entity_type=config.competition_entity_type,
                gender=config.competition_gender, age_group=config.competition_age_group,
                squad_level="FIRST_TEAM", name=name,
                provider_id=provider_id,
                provider_team_id=str(provider_team_id),
            )
            return canonical_entity_id(
                country=config.team_country, federation=config.team_federation,
                entity_type=config.competition_entity_type,
                gender=config.competition_gender,
                age_group=config.competition_age_group,
                fingerprint=fingerprint, provider_id=provider_id,
                provider_team_id=str(provider_team_id),
            )

        samples = []
        for record in package.available_data.get("historical_results", []):
            if not is_valid_for_cutoff(record, cutoff) or not isinstance(record.value, dict):
                continue
            value = record.value
            provider_id = record.provider_id or record.source_id
            home_provider, away_provider = (value.get("team1_provider_id"),
                                             value.get("team2_provider_id"))
            if not isinstance(home_provider, int) or not isinstance(away_provider, int):
                continue
            if not isinstance(provider_id, str) or not provider_id:
                continue
            home_id = canonical(provider_id, home_provider)
            away_id = canonical(provider_id, away_provider)
            if home_id is None or away_id is None:
                continue
            kickoff = parse_utc(value["kickoff_time"])
            if kickoff >= cutoff:
                continue
            samples.append(HistoricalMatchSample(
                str(value["provider_match_id"]), home_id, away_id,
                str(value["competition_id"]), kickoff, int(value["home_goals"]),
                int(value["away_goals"]), (record.evidence_id,),
                (provider_id,), record.source_tier or 0,
                parse_utc(record.fetched_time), config.team_federation,
                config.competition_gender, config.competition_age_group,
                "LEAGUE",
            ))
        return cls(tuple(samples))
