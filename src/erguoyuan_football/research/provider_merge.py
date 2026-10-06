"""Cross-provider historical-result deduplication and fail-closed conflict records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ProviderMatchRecord:
    """Canonical provider result with original evidence lineage."""

    provider_match_id: str
    home_team_id: str
    away_team_id: str
    kickoff: datetime
    competition_id: str
    home_score: int | None
    away_score: int | None
    status: str
    source: str
    source_url: str
    observed_at: datetime
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kickoff.tzinfo is None or self.observed_at.tzinfo is None:
            raise ValueError("UTC_TIMESTAMPS_REQUIRED")
        if not self.home_team_id or not self.away_team_id or self.home_team_id == self.away_team_id:
            raise ValueError("INVALID_TEAM_ORIENTATION")
        if (self.home_score is None) != (self.away_score is None):
            raise ValueError("PARTIAL_SCORE_FORBIDDEN")
        if self.home_score is not None and min(self.home_score, self.away_score or 0) < 0:
            raise ValueError("NEGATIVE_SCORE_FORBIDDEN")


@dataclass(frozen=True)
class MergeConflict:
    """Competing source versions excluded from downstream samples."""

    conflict_type: str
    records: tuple[ProviderMatchRecord, ...]


@dataclass(frozen=True)
class ProviderMergeResult:
    """Unique agreed records and excluded conflicts with all evidence IDs."""

    records: tuple[ProviderMatchRecord, ...]
    conflicts: tuple[MergeConflict, ...]


def merge_historical_results(rows: tuple[ProviderMatchRecord, ...]) -> ProviderMergeResult:
    """Deduplicate exact fixtures and exclude score, orientation or time conflicts."""
    by_fixture: dict[tuple[str, str, datetime, str], list[ProviderMatchRecord]] = {}
    for row in rows:
        key = (row.home_team_id, row.away_team_id, row.kickoff, row.competition_id)
        by_fixture.setdefault(key, []).append(row)
    conflicts: list[MergeConflict] = []
    accepted: list[ProviderMatchRecord] = []
    conflicted: set[int] = set()
    groups = list(by_fixture.values())
    for group in groups:
        if len({(row.home_score, row.away_score, row.status) for row in group}) > 1:
            conflicts.append(MergeConflict("SCORE_OR_STATUS_CONFLICT", tuple(group)))
            conflicted.update(id(row) for row in group)
            continue
        first = group[0]
        accepted.append(ProviderMatchRecord(
            first.provider_match_id, first.home_team_id, first.away_team_id,
            first.kickoff, first.competition_id, first.home_score, first.away_score,
            first.status, first.source, first.source_url,
            max(row.observed_at for row in group),
            tuple(sorted({item for row in group for item in row.evidence_ids}))))
    # Same teams and competition near the same kickoff, but reversed or shifted, are not
    # silently treated as separate games. Exact repeat fixtures are already grouped above.
    for left_index, left in enumerate(rows):
        if id(left) in conflicted:
            continue
        for right in rows[left_index + 1:]:
            if id(right) in conflicted or left.competition_id != right.competition_id:
                continue
            same_pair = {left.home_team_id, left.away_team_id} == {
                right.home_team_id, right.away_team_id}
            if (same_pair and left.kickoff.date() == right.kickoff.date()
                    and (left.home_team_id != right.home_team_id
                         or left.away_team_id != right.away_team_id
                         or left.kickoff != right.kickoff)):
                conflicts.append(MergeConflict("ORIENTATION_OR_KICKOFF_CONFLICT", (left, right)))
                conflicted.update({id(left), id(right)})
    accepted = [row for row in accepted if not any(
        id(original) in conflicted and original.home_team_id == row.home_team_id
        and original.away_team_id == row.away_team_id and original.kickoff == row.kickoff
        and original.competition_id == row.competition_id for original in rows)]
    return ProviderMergeResult(tuple(sorted(accepted, key=lambda row: (row.kickoff,
                                                                       row.provider_match_id))),
                               tuple(conflicts))

