"""Optional lawful UTC kickoff enrichment through football-data.org."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.data.external_providers import (
    FootballDataOrgProvider,
    ProviderResult,
)
from erguoyuan_football.data.kickoff_enrichment import KickoffEnrichmentLayer

COMPETITION_MAP = {
    "PL": "EPL",
    "PD": "LALIGA",
    "BL1": "BUNDESLIGA",
    "SA": "SERIE_A",
    "FL1": "LIGUE_1",
}


@dataclass(frozen=True)
class KickoffImportReport:
    """Counts from actual provider rows; unavailable requests add no enrichments."""

    status: str
    source: str
    requests: int
    provider_unavailable: int
    linked: int
    ambiguous: int
    unmatched: int
    reason_counts: dict[str, int]


def _event_record(row: dict[str, Any], requested_competition: str) -> dict[str, Any] | None:
    competition = row.get("competition")
    season = row.get("season")
    home = row.get("homeTeam")
    away = row.get("awayTeam")
    if not all(isinstance(value, dict) for value in (competition, season, home, away)):
        return None
    start_date = season.get("startDate")
    event_id, utc_date = row.get("id"), row.get("utcDate")
    home_name, away_name = home.get("name"), away.get("name")
    if not all((event_id, utc_date, start_date, home_name, away_name)):
        return None
    try:
        year = int(str(start_date)[:4])
    except ValueError:
        return None
    return {
        "id": str(event_id),
        "competition": competition.get("code") or requested_competition,
        "season": year,
        "home": str(home_name),
        "away": str(away_name),
        "utcDate": str(utc_date),
    }


def import_verified_kickoffs(db_path: str | Path, *, from_season: int = 2021,
                             through_season: int = 2025,
                             competitions: tuple[str, ...] = tuple(COMPETITION_MAP),
                             provider: FootballDataOrgProvider | None = None) -> KickoffImportReport:
    """Fetch registered provider seasons and persist only unique verified links."""
    if from_season > through_season:
        raise ValueError("from_season must not exceed through_season")
    selected = tuple(code for code in competitions if code in COMPETITION_MAP)
    if len(selected) != len(competitions) or not selected:
        raise ValueError("UNREGISTERED_COMPETITION")
    client = provider or FootballDataOrgProvider()
    first = client.fetch_matches(selected[0], from_season)
    requests = 1
    if first.status != "AVAILABLE":
        return KickoffImportReport(first.status, first.source, requests, 1, 0, 0, 0,
                                   {first.reason or "UNKNOWN": 1})
    provider_results: list[tuple[str, ProviderResult]] = [(selected[0], first)]
    for competition in selected:
        start = from_season + (1 if competition == selected[0] else 0)
        for season in range(start, through_season + 1):
            result = client.fetch_matches(competition, season)
            requests += 1
            provider_results.append((competition, result))
    report_counts = {"linked": 0, "ambiguous": 0, "unmatched": 0, "unavailable": 0}
    reasons: dict[str, int] = {}
    with KickoffEnrichmentLayer(db_path) as layer:
        for requested_competition, result in provider_results:
            if result.status != "AVAILABLE":
                report_counts["unavailable"] += 1
                reason = result.reason or "UNKNOWN"
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            retrieved_at = datetime.fromisoformat(result.retrieved_at) if result.retrieved_at else None
            if retrieved_at is None:
                report_counts["unavailable"] += 1
                reasons["RETRIEVAL_TIMESTAMP_MISSING"] = reasons.get("RETRIEVAL_TIMESTAMP_MISSING", 0) + 1
                continue
            for raw in result.records:
                event = _event_record(raw, requested_competition)
                if event is None:
                    report_counts["unmatched"] += 1
                    reasons["INVALID_PROVIDER_EVENT"] = reasons.get("INVALID_PROVIDER_EVENT", 0) + 1
                    continue
                record = layer.resolve(event, provider_id=result.source, retrieved_at=retrieved_at,
                                       competition_map=COMPETITION_MAP)
                if record is not None:
                    report_counts["linked"] += 1
                else:
                    reason_row = layer.connection.execute(
                        "SELECT reason FROM kickoff_enrichment_review_queue WHERE provider_id=? AND source_event_id=?",
                        [result.source, event["id"]]).fetchone()
                    reason = reason_row[0] if reason_row else "UNMATCHED"
                    key = "ambiguous" if reason == "AMBIGUOUS_MATCH" else "unmatched"
                    report_counts[key] += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
    unavailable = report_counts.pop("unavailable")
    status = "AVAILABLE" if report_counts["linked"] else "UNAVAILABLE"
    return KickoffImportReport(status, "FOOTBALL_DATA_ORG", requests, unavailable,
                               report_counts["linked"], report_counts["ambiguous"],
                               report_counts["unmatched"], reasons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import verified football-data.org UTC kickoffs")
    parser.add_argument("--db", type=Path, default=Path("data/football.duckdb"))
    parser.add_argument("--from-season", type=int, default=2021)
    parser.add_argument("--through-season", type=int, default=2025)
    parser.add_argument("--competition", action="append", choices=tuple(COMPETITION_MAP))
    args = parser.parse_args(argv)
    result = import_verified_kickoffs(args.db, from_season=args.from_season,
        through_season=args.through_season,
        competitions=tuple(args.competition) if args.competition else tuple(COMPETITION_MAP))
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0 if result.status == "AVAILABLE" or result.reason_counts.get("AUTH_NOT_CONFIGURED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
