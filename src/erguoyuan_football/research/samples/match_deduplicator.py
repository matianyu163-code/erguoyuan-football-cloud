"""Provider-independent historical match identity and score-conflict rejection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime


@dataclass(frozen=True)
class HistoricalMatchSample:
    """Verified canonical IDs and source-backed completed score."""

    match_id: str
    home_team_id: str
    away_team_id: str
    competition_id: str
    kickoff: datetime
    home_goals: int
    away_goals: int
    evidence_ids: tuple[str, ...]
    provider_ids: tuple[str, ...]
    source_tier: int
    fetched_at: datetime
    federation: str
    gender: str
    age_group: str
    competition_type: str
    neutral_venue: bool = False

    def __post_init__(self) -> None:
        if self.kickoff.tzinfo is None or self.fetched_at.tzinfo is None:
            raise ValueError("UTC_TIMESTAMP_REQUIRED")
        if min(self.home_goals, self.away_goals) < 0 or not self.evidence_ids:
            raise ValueError("INVALID_HISTORICAL_SAMPLE")

    @property
    def fingerprint(self) -> tuple[str, str, datetime, str]:
        """Match key excludes provider so duplicate sources converge."""
        return (self.home_team_id, self.away_team_id, self.kickoff,
                self.competition_id)


@dataclass(frozen=True)
class DeduplicationResult:
    """Clean rows and conflicts retained for audit."""

    samples: tuple[HistoricalMatchSample, ...]
    conflicts: tuple[tuple[str, ...], ...]


def deduplicate_matches(rows: tuple[HistoricalMatchSample, ...]) -> DeduplicationResult:
    """Merge agreeing providers once; exclude every contradictory result."""
    groups: dict[tuple[str, str, datetime, str], list[HistoricalMatchSample]] = {}
    for row in rows:
        groups.setdefault(row.fingerprint, []).append(row)
    clean = []
    conflicts = []
    for group in groups.values():
        if len({(row.home_goals, row.away_goals) for row in group}) != 1:
            conflicts.append(tuple(sorted({evidence for row in group
                                           for evidence in row.evidence_ids})))
            continue
        first = group[0]
        clean.append(replace(
            first, evidence_ids=tuple(sorted({evidence for row in group
                                               for evidence in row.evidence_ids})),
            provider_ids=tuple(sorted({provider for row in group
                                        for provider in row.provider_ids})),
            # Tier numbers are ordered best-first (A=1, B=2, backup=3).
            source_tier=min(row.source_tier for row in group),
            fetched_at=max(row.fetched_at for row in group),
        ))
    return DeduplicationResult(tuple(sorted(clean, key=lambda row: row.kickoff)),
                               tuple(conflicts))
