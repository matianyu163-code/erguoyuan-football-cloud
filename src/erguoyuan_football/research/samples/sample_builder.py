"""Build nonoverlapping source-backed sample categories and quality diagnostics."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.research.samples.match_deduplicator import (
    HistoricalMatchSample,
    deduplicate_matches,
)
from erguoyuan_football.research.samples.sample_hierarchy import (
    SampleHierarchy,
    SampleLineage,
    SampleQuality,
)
from erguoyuan_football.research.samples.sample_recency import latest_age_days
from erguoyuan_football.research.samples.sample_repository import SampleRepository


def _lineage(rows: tuple[HistoricalMatchSample, ...]) -> tuple[SampleLineage, ...]:
    return tuple(SampleLineage(row.match_id, row.competition_id,
                               row.evidence_ids, row.provider_ids,
                               row.kickoff, row.source_tier,
                               row.home_team_id, row.away_team_id) for row in rows)


class SampleBuilder:
    """Count direct matches only by exact canonical IDs and compatible dimensions."""

    def __init__(self, repository: SampleRepository) -> None:
        self.repository = repository

    def build(self, home: TeamIdentity, away: TeamIdentity,
              competition: CompetitionIdentity, *, cutoff: datetime,
              competition_type: str = "UNKNOWN") -> SampleHierarchy:
        """Filter PIT, deduplicate, exclude conflicting scores, then classify."""
        eligible = tuple(row for row in self.repository.samples
                         if row.kickoff < cutoff and row.fetched_at <= cutoff
                         and 1 <= row.source_tier <= 3 and row.evidence_ids)
        deduped = deduplicate_matches(eligible)
        rows = deduped.samples
        target_gender = home.gender if home.gender == away.gender else "UNKNOWN"
        target_age = home.age_group if home.age_group == away.age_group else "UNKNOWN"
        federation = home.federation if home.federation == away.federation else "UNKNOWN"

        def compatible(row: HistoricalMatchSample) -> bool:
            return (target_gender != "UNKNOWN" and row.gender == target_gender
                    and target_age != "UNKNOWN" and row.age_group == target_age)

        home_direct = tuple(row for row in rows if compatible(row) and home.team_id in
                            {row.home_team_id, row.away_team_id})
        away_direct = tuple(row for row in rows if compatible(row) and away.team_id in
                            {row.home_team_id, row.away_team_id})
        direct_keys = {row.fingerprint for row in (*home_direct, *away_direct)}
        competition_rows = tuple(row for row in rows if compatible(row)
                                 and row.competition_id == competition.competition_id
                                 and row.fingerprint not in direct_keys)
        comparable = tuple(row for row in rows if compatible(row)
                           and row.federation == federation
                           and competition_type != "UNKNOWN"
                           and row.competition_type == competition_type
                           and row.competition_id != competition.competition_id
                           and row.fingerprint not in direct_keys)
        used = direct_keys | {row.fingerprint for row in (*competition_rows, *comparable)}
        federation_prior = tuple(row for row in rows if row.fingerprint not in used
                                 and target_gender != "UNKNOWN" and row.gender == target_gender
                                 and federation != "UNKNOWN" and row.federation == federation
                                 and row.age_group == target_age)
        age_prior = tuple(row for row in rows if row.fingerprint not in used
                          and target_gender != "UNKNOWN" and row.gender == target_gender
                          and federation != "UNKNOWN" and row.federation == federation
                          and row.age_group != target_age)
        gender_prior = tuple(row for row in rows if row.fingerprint not in used
                             and row not in (*federation_prior, *age_prior)
                             and target_gender != "UNKNOWN"
                             and row.gender == target_gender and row.age_group == target_age)
        direct = tuple({row.fingerprint: row for row in
                        (*home_direct, *away_direct)}.values())
        times = sorted(row.kickoff for row in direct)
        opponents = {row.away_team_id if row.home_team_id in
                     {home.team_id, away.team_id} else row.home_team_id for row in direct}
        training_rows = tuple(row for row in rows
                              if row.competition_id == competition.competition_id
                              and compatible(row))
        training_counts = Counter(team_id for row in training_rows
                                  for team_id in (row.home_team_id, row.away_team_id))
        multi_counts = Counter(team_id for row in rows if compatible(row)
                               for team_id in (row.home_team_id, row.away_team_id))
        quality = SampleQuality(
            times[0] if times else None, times[-1] if times else None,
            (times[-1] - times[0]).days if len(times) > 1 else 0,
            latest_age_days(tuple(times), cutoff),
            len(opponents) / len(direct) if direct else 0.0,
            sum(row.home_team_id in {home.team_id, away.team_id} for row in direct),
            sum(row.away_team_id in {home.team_id, away.team_id} for row in direct),
            sum(row.neutral_venue for row in direct),
            min((row.source_tier for row in rows), default=None),
            min(training_counts.values(), default=0),
            min(multi_counts.values(), default=0),
            target_age != "SENIOR" and len(times) > 1
            and (times[-1] - times[0]).days > 365,
            len(deduped.conflicts),
        )
        return SampleHierarchy(
            competition.competition_id,
            _lineage(home_direct), _lineage(away_direct),
            _lineage(competition_rows), _lineage(comparable),
            _lineage(federation_prior), _lineage(age_prior), _lineage(gender_prior),
            tuple(sorted({provider for row in (*federation_prior, *age_prior, *gender_prior)
                          for provider in row.provider_ids})), cutoff, quality,
        )
