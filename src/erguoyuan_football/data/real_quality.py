"""Measured real-data coverage and hard historical-PIT readiness gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from erguoyuan_football.data.real_bootstrap import RealDataWarehouse


@dataclass(frozen=True)
class HistoricalCoverageMatrix:
    competition: str
    season: str
    provider: str
    fixtures: int
    results: int
    events: int
    xg: int
    market: int
    coverage_pct: float
    quality_status: str
    exact_utc_kickoffs: int


@dataclass(frozen=True)
class DataQualityReport:
    row_count: int
    fixtures: int
    results: int
    date_coverage: tuple[str | None, str | None]
    competition_coverage: tuple[str, ...]
    missing_scores: int
    duplicate_matches: int
    unknown_teams: int
    quarantined: int
    score_conflicts: int
    invalid_scores: int
    future_results: int
    exact_utc_kickoffs: int
    unknown_zone_kickoffs: int
    date_only_kickoffs: int
    timezone_quality: str
    xg_missingness_pct: float
    market_missingness_pct: float
    provenance_quality: str
    real_oos_predictions: int
    strict_historical_oos_status: str
    reasons: tuple[str, ...]


def assess_real_data(path: str | Path) -> tuple[DataQualityReport, tuple[HistoricalCoverageMatrix, ...]]:
    """Count actual rows; missing source timestamps and market rows remain blockers."""
    with RealDataWarehouse(path) as warehouse:
        db = warehouse.connection
        counts = db.execute("""
            SELECT count(*), count(*) FILTER (WHERE status='FINISHED'),
              count(*) FILTER (WHERE timestamp_precision='EXACT_UTC'),
              count(*) FILTER (WHERE timestamp_precision='LOCAL_TIME_UNKNOWN_ZONE'),
              count(*) FILTER (WHERE timestamp_precision='DATE_ONLY')
            FROM real_canonical_matches
        """).fetchone()
        assert counts is not None
        fixtures, results, exact, unknown, date_only = counts
        def count(query: str) -> int:
            row = db.execute(query).fetchone()
            assert row is not None
            return int(row[0])

        quarantined = count("SELECT count(*) FROM real_data_quarantine")
        conflicts = count("SELECT count(*) FROM real_data_quarantine WHERE reason='SCORE_CONFLICT'")
        invalid = count("SELECT count(*) FROM real_data_quarantine WHERE reason='INVALID_SCORE'")
        future = count("SELECT count(*) FROM real_canonical_matches WHERE status='FINISHED' "
                       "AND match_date>current_date")
        oos_legacy = count("SELECT count(*) FROM real_oos_predictions WHERE data_origin='REAL' AND is_oos")
        has_phase8_1_oos = db.execute("SELECT 1 FROM information_schema.tables "
                                     "WHERE table_name='real_oos_predictions_v2'").fetchone() is not None
        oos_v2 = count("SELECT count(*) FROM real_oos_predictions_v2 "
                       "WHERE data_origin='REAL' AND is_oos") if has_phase8_1_oos else 0
        date_safe_oos = count("SELECT count(*) FROM real_oos_predictions_v2 "
                              "WHERE data_origin='REAL' AND is_oos "
                              "AND prediction_temporal_mode='DATE_SAFE_BATCH'") if has_phase8_1_oos else 0
        oos = oos_legacy + oos_v2
        reasons: list[str] = []
        if exact == 0:
            reasons.append("NO_VERIFIED_UTC_KICKOFFS")
        # Late retrieval does not invalidate immutable event facts under the
        # separate EVENT_IMMUTABLE policy; it still blocks PIT snapshots.
        historical_rows = count("SELECT count(*) FROM real_canonical_matches "
                                "WHERE retrieved_at::DATE>match_date")
        dates = db.execute("SELECT min(match_date), max(match_date) FROM real_canonical_matches").fetchone()
        assert dates is not None
        competitions = tuple(row[0] for row in db.execute(
            "SELECT DISTINCT competition_id FROM real_canonical_matches ORDER BY 1").fetchall())
        duplicates = count("SELECT count(*) FROM (SELECT match_id FROM real_canonical_matches "
                           "GROUP BY match_id HAVING count(*)>1)")
        unknown_teams = count("SELECT count(*) FROM real_canonical_matches m "
            "LEFT JOIN real_canonical_teams h ON m.home_team_id=h.team_id "
            "LEFT JOIN real_canonical_teams a ON m.away_team_id=a.team_id "
            "WHERE h.team_id IS NULL OR a.team_id IS NULL")
        if historical_rows and date_safe_oos == 0:
            reasons.append("HISTORICAL_RETRIEVAL_CANNOT_BE_BACKDATED")
        if future:
            reasons.append("FUTURE_DATED_RESULTS")
        matrix = []
        for row in warehouse.coverage():
            matrix.append(HistoricalCoverageMatrix(
                competition=row["competition_id"], season=row["season_id"], provider="OPENFOOTBALL",
                fixtures=row["fixtures"], results=row["results"], events=0, xg=0, market=0,
                coverage_pct=100 * row["results"] / row["fixtures"] if row["fixtures"] else 0,
                quality_status="BLOCKED_FOR_STRICT_OOS" if row["exact_utc"] == 0 else "REVIEW_REQUIRED",
                exact_utc_kickoffs=row["exact_utc"],
            ))
        report = DataQualityReport(
            row_count=fixtures,
            fixtures=fixtures, results=results, quarantined=quarantined,
            date_coverage=(dates[0].isoformat() if dates[0] else None,
                           dates[1].isoformat() if dates[1] else None),
            competition_coverage=competitions, missing_scores=fixtures - results,
            duplicate_matches=duplicates, unknown_teams=unknown_teams,
            score_conflicts=conflicts, invalid_scores=invalid, future_results=future,
            exact_utc_kickoffs=exact, unknown_zone_kickoffs=unknown,
            date_only_kickoffs=date_only, real_oos_predictions=oos,
            timezone_quality="UNVERIFIED" if exact == 0 else "MIXED",
            xg_missingness_pct=100.0, market_missingness_pct=100.0,
            provenance_quality="PINNED_SOURCE_AND_CONTENT_HASH" if fixtures else "UNAVAILABLE",
            strict_historical_oos_status=("DATE_SAFE_BATCH_ONLY" if date_safe_oos and not future else
                                          "BLOCKED" if reasons else "REVIEW_REQUIRED"),
            reasons=tuple(reasons),
        )
        return report, tuple(matrix)


def assert_oos_lineage(*, training_available_at: datetime, fixture_available_at: datetime,
                       prediction_time: datetime, kickoff_time: datetime) -> None:
    """Reject a historical prediction when either source was acquired later."""
    times = (training_available_at, fixture_available_at, prediction_time, kickoff_time)
    if any(value.tzinfo is None for value in times):
        raise ValueError("OOS_TIMESTAMPS_REQUIRE_TIMEZONE")
    known = tuple(value.astimezone(UTC) for value in times)
    if not (known[0] <= known[2] and known[1] <= known[2] < known[3]):
        raise ValueError("HISTORICAL_OOS_POINT_IN_TIME_VIOLATION")
