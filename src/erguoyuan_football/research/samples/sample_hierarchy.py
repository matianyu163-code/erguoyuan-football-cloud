"""Count and full lineage for each noninterchangeable sample level."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SampleLineage:
    """One historical match with source and original evidence IDs."""

    match_id: str
    competition_id: str
    evidence_ids: tuple[str, ...]
    provider_ids: tuple[str, ...]
    kickoff: datetime
    source_tier: int
    home_team_id: str = ""
    away_team_id: str = ""


@dataclass(frozen=True)
class SampleQuality:
    """Quality diagnostics, not a model confidence or probability."""

    earliest_match: datetime | None
    latest_match: datetime | None
    span_days: int
    latest_age_days: int | None
    opponent_diversity_score: float
    home_matches: int
    away_matches: int
    neutral_matches: int
    min_source_tier: int | None
    min_training_team_matches: int
    min_multi_comp_training_team_matches: int
    roster_continuity_unknown: bool
    source_conflict_count: int


@dataclass(frozen=True)
class SampleHierarchy:
    """Direct, competition, comparable and prior sets remain distinct."""

    competition_id: str
    home_direct: tuple[SampleLineage, ...]
    away_direct: tuple[SampleLineage, ...]
    competition: tuple[SampleLineage, ...]
    comparable: tuple[SampleLineage, ...]
    federation_prior: tuple[SampleLineage, ...]
    age_group_prior: tuple[SampleLineage, ...]
    gender_prior: tuple[SampleLineage, ...]
    prior_sources: tuple[str, ...]
    generated_at: datetime
    quality: SampleQuality

    @property
    def home_team_direct_matches(self) -> int:
        return len(self.home_direct)

    @property
    def away_team_direct_matches(self) -> int:
        return len(self.away_direct)

    @property
    def competition_matches(self) -> int:
        return len(self.competition)

    @property
    def comparable_competition_matches(self) -> int:
        return len(self.comparable)

    @property
    def federation_prior_matches(self) -> int:
        return len(self.federation_prior)

    @property
    def age_group_prior_matches(self) -> int:
        return len(self.age_group_prior)

    @property
    def gender_prior_matches(self) -> int:
        return len(self.gender_prior)
