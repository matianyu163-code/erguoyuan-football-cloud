"""Point-in-time form windows, kept separate from long-term training history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from erguoyuan_football.research.samples.match_deduplicator import HistoricalMatchSample

WINDOW_POLICY_VERSION = "MATCH_WINDOWS_V1"
SHORT_FORM_WINDOW = 5
RECENT_WINDOW_MIN = 10
RECENT_WINDOW_TARGET = 20
STANDARD_FORM_WINDOW = 20


@dataclass(frozen=True)
class FormSummary:
    """Observed score form for one explicitly selected match window."""

    match_count: int
    goals_for: int
    goals_against: int
    wins: int
    draws: int
    losses: int
    home_matches: int
    away_matches: int


@dataclass(frozen=True)
class MatchWindowSet:
    """Latest N matches per target team plus the independent full history."""

    short_home: tuple[HistoricalMatchSample, ...]
    short_away: tuple[HistoricalMatchSample, ...]
    recent_home: tuple[HistoricalMatchSample, ...]
    recent_away: tuple[HistoricalMatchSample, ...]
    standard_home: tuple[HistoricalMatchSample, ...]
    standard_away: tuple[HistoricalMatchSample, ...]
    long_term_home: tuple[HistoricalMatchSample, ...]
    long_term_away: tuple[HistoricalMatchSample, ...]
    competition_history: tuple[HistoricalMatchSample, ...]
    home_team_id: str
    away_team_id: str
    cutoff: datetime
    policy_version: str = WINDOW_POLICY_VERSION

    @property
    def forms(self) -> dict[str, FormSummary]:
        """Provide score-derived last-5/10/20 aggregates without invented weights."""
        return {
            "home_last5": _summarize(self.short_home, self.home_team_id),
            "away_last5": _summarize(self.short_away, self.away_team_id),
            "home_last10": _summarize(self.recent_home, self.home_team_id),
            "away_last10": _summarize(self.recent_away, self.away_team_id),
            "home_last20": _summarize(self.standard_home, self.home_team_id),
            "away_last20": _summarize(self.standard_away, self.away_team_id),
        }


def build_match_windows(
    samples: tuple[HistoricalMatchSample, ...], home_team_id: str,
    away_team_id: str, competition_id: str, cutoff: datetime,
) -> MatchWindowSet:
    """Construct deterministic windows from completed, observed-before-cutoff data."""
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("UTC_CUTOFF_REQUIRED")
    eligible = tuple(row for row in samples
                     if row.kickoff < cutoff and row.fetched_at <= cutoff)
    ordered = tuple(sorted(eligible, key=lambda row: (row.kickoff, row.match_id), reverse=True))
    home = tuple(row for row in ordered if home_team_id in {row.home_team_id, row.away_team_id})
    away = tuple(row for row in ordered if away_team_id in {row.home_team_id, row.away_team_id})
    competition = tuple(row for row in ordered if row.competition_id == competition_id)
    return MatchWindowSet(home[:5], away[:5], home[:10], away[:10], home[:20], away[:20],
                          home, away, competition, home_team_id, away_team_id, cutoff)


def _summarize(rows: tuple[HistoricalMatchSample, ...], team_id: str) -> FormSummary:
    gf = ga = wins = draws = losses = home = away = 0
    for row in rows:
        is_home = row.home_team_id == team_id
        scored, conceded = ((row.home_goals, row.away_goals) if is_home
                            else (row.away_goals, row.home_goals))
        gf += scored
        ga += conceded
        wins += int(scored > conceded)
        draws += int(scored == conceded)
        losses += int(scored < conceded)
        home += int(is_home)
        away += int(not is_home)
    return FormSummary(len(rows), gf, ga, wins, draws, losses, home, away)
