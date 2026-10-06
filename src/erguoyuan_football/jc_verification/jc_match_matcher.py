"""Identity-aware fixture matching with competition, date and kickoff checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, timedelta

from erguoyuan_football.jc_verification.jc_schema import JCMatch, JCMatchQuery
from erguoyuan_football.knowledge.competitions.competition_resolver import (
    CompetitionResolver,
)
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver


@dataclass(frozen=True)
class MatchCandidate:
    """One provider fixture with measured identity agreement."""

    match: JCMatch
    exact_competition: bool
    exact_kickoff: bool
    kickoff_delta: timedelta
    competition_id: str | None


class JCMatchMatcher:
    """Reject ambiguous, date-mismatched, or string-only fixture candidates."""

    def __init__(self, team_resolver: TeamResolver | None = None,
                 competition_resolver: CompetitionResolver | None = None,
                 *, max_kickoff_delta: timedelta = timedelta(minutes=30)) -> None:
        if max_kickoff_delta < timedelta(0):
            raise ValueError("NEGATIVE_JC_KICKOFF_WINDOW")
        self.teams = team_resolver or TeamResolver()
        self.competitions = competition_resolver or CompetitionResolver()
        self.max_kickoff_delta = max_kickoff_delta

    def match(self, query: JCMatchQuery, candidates: tuple[JCMatch, ...]
              ) -> MatchCandidate | None:
        """Return a unique candidate only when canonical teams and date agree."""
        if query.kickoff_time is None or not (
            query.competition_id or query.competition_name
        ):
            return None
        query_time = query.kickoff_time
        hits: list[MatchCandidate] = []
        for row in candidates:
            if row.kickoff_time.astimezone(UTC).date() != query_time.astimezone(UTC).date():
                continue
            home = self.teams.resolve(row.home_team)
            away = self.teams.resolve(row.away_team)
            if home is None or away is None or home.team_id != query.home_team_id \
                    or away.team_id != query.away_team_id:
                continue
            delta = abs(row.kickoff_time - query_time)
            if delta > self.max_kickoff_delta:
                continue
            provider_comp = self.competitions.resolve(row.competition)
            query_comp = (self.competitions.resolve(query.competition_name)
                          if query.competition_name else None)
            expected_competition_id = query.competition_id or (
                query_comp.competition_id if query_comp else None)
            exact_competition = bool(
                expected_competition_id
                and (row.competition_id or (provider_comp.competition_id
                                            if provider_comp else None))
                == expected_competition_id)
            if (query.competition_id or query.competition_name) and not exact_competition:
                continue
            canonical_competition_id = row.competition_id or (
                provider_comp.competition_id if provider_comp else None)
            hits.append(MatchCandidate(row, exact_competition, delta == timedelta(0),
                                       delta, canonical_competition_id))
        return hits[0] if len(hits) == 1 else None
