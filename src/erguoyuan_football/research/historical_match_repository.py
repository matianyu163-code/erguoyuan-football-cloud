"""Point-in-time repository for normalized, evidence-backed final results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

FINAL_STATUSES = frozenset({"FINISHED", "FINAL", "CLOSED"})


@dataclass(frozen=True)
class HistoricalMatch:
    """One canonical final score with source and observation lineage."""

    canonical_match_id: str
    provider_match_ids: tuple[tuple[str, str], ...]
    home_team_id: str
    away_team_id: str
    competition_id: str | None
    kickoff_at: datetime
    home_score: int
    away_score: int
    halftime_home_score: int | None
    halftime_away_score: int | None
    venue_type: str
    status: str
    source_evidence_ids: tuple[str, ...]
    observed_at: datetime
    gender: str
    age_group: str
    entity_type: str
    squad_level: str
    season: str

    def __post_init__(self) -> None:
        for name in ("kickoff_at", "observed_at"):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("HISTORICAL_MATCH_TIMESTAMP_MUST_BE_AWARE")
        if self.home_score < 0 or self.away_score < 0:
            raise ValueError("INVALID_FINAL_SCORE")
        if self.status.upper() not in FINAL_STATUSES:
            raise ValueError("NON_FINAL_RESULT_REJECTED")
        if not self.source_evidence_ids or not self.provider_match_ids:
            raise ValueError("HISTORICAL_MATCH_EVIDENCE_REQUIRED")


class HistoricalMatchRepository:
    """Deterministic query methods over normalized rows; no hidden time source."""

    def __init__(self, matches: tuple[HistoricalMatch, ...] = ()) -> None:
        self.matches = matches

    def get_team_history(self, team_id: str, *, before: datetime,
                         competition_ids: tuple[str, ...] | None = None,
                         limit: int | None = None) -> tuple[HistoricalMatch, ...]:
        rows = tuple(row for row in self.matches
                     if row.kickoff_at < before and row.observed_at <= before
                     and team_id in {row.home_team_id, row.away_team_id}
                     and (competition_ids is None or row.competition_id in competition_ids))
        return rows[-limit:] if limit is not None else rows

    def get_competition_history(self, competition_id: str, *, before: datetime,
                                seasons: tuple[str, ...] | None = None
                                ) -> tuple[HistoricalMatch, ...]:
        return tuple(row for row in self.matches if row.competition_id == competition_id
                     and row.kickoff_at < before and row.observed_at <= before
                     and (seasons is None or row.season in seasons))

