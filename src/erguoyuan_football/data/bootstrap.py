"""Phase 8 offline importer for a locally pinned OpenFootball repository."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

from erguoyuan_football.data.openfootball_provider import OpenFootballProvider
from erguoyuan_football.data.real_bootstrap import RealDataWarehouse

REPOSITORY = "https://github.com/openfootball/football.json"
LEAGUES = {"en.1": "EPL", "es.1": "LALIGA", "de.1": "BUNDESLIGA",
           "it.1": "SERIE_A", "fr.1": "LIGUE_1"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-data bootstrap; never generates historical OOS by backdating downloads")
    parser.add_argument("--provider", choices=("openfootball", "football_data_org", "statsbomb_open"))
    parser.add_argument("--all-open", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path("data/raw/openfootball_repo"))
    parser.add_argument("--db", type=Path, default=Path("data/football.duckdb"))
    parser.add_argument("--from-date", type=date.fromisoformat, default=date(2021, 1, 1))
    parser.add_argument("--to-date", type=date.fromisoformat, default=datetime.now(UTC).date())
    parser.add_argument("--competition", choices=tuple(LEAGUES.values()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.all_open and args.provider is None:
        parser.error("--provider or --all-open required")
    if args.provider in {"football_data_org", "statsbomb_open"}:
        print(json.dumps({"status": "UNAVAILABLE", "reason": "PROVIDER_NOT_CONFIGURED_OR_LICENSE_NOT_VERIFIED"}))
        return 2
    if args.to_date < args.from_date:
        parser.error("--to-date precedes --from-date")
    if not (args.repo / ".git").exists():
        print(json.dumps({"status": "UNAVAILABLE", "reason": "PINNED_LOCAL_REPOSITORY_MISSING"}))
        return 2
    provider = OpenFootballProvider(args.repo)
    commit = provider.commit_sha
    years = range(args.from_date.year - 1, args.to_date.year + 1)
    paths = [(season, code, args.repo / season / f"{code}.json")
             for year in years for season in (f"{year}-{(year + 1) % 100:02d}",)
             for code, competition in LEAGUES.items()
             if args.competition is None or competition == args.competition]
    paths = [(season, code, path) for season, code, path in paths if path.exists()]
    if not paths:
        print(json.dumps({"status": "UNAVAILABLE", "reason": "NO_LOCAL_SEASON_FILES"}))
        return 2
    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.db.exists() and not args.dry_run and not args.resume:
        backup = args.db.with_suffix(f".phase8-before-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.duckdb")
        shutil.copy2(args.db, backup)
        print(json.dumps({"backup": str(backup)}))
    summaries = []
    with RealDataWarehouse(args.db) as warehouse:
        for season, code, path in paths:
            source_file = provider.source_file(season, code)
            summary = warehouse.promote_openfootball(source_file.path, repository=REPOSITORY,
                source_commit=commit, competition_id=LEAGUES[code], season_id=season,
                from_date=args.from_date, to_date=args.to_date,
                dry_run=args.dry_run)
            summaries.append(summary.__dict__)
        coverage = warehouse.coverage()
    print(json.dumps({"status": "DRY_RUN" if args.dry_run else "IMPORTED",
                      "source_commit": commit,
                      "files": len(summaries),
                      "promoted": sum(item["promoted"] for item in summaries),
                      "quarantined": sum(item["quarantined"] for item in summaries),
                      "repeated": sum(item["repeated"] for item in summaries),
                      "coverage": coverage}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
