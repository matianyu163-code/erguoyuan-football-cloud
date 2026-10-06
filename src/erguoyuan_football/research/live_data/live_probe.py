"""Read-only live research CLI; it never starts a prediction pipeline."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from erguoyuan_football.research.live_data.model_requirements import (
    load_model_requirements,
)
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBConfig,
    OpenLigaDBProvider,
)
from erguoyuan_football.research.live_data.provider_coverage import build_coverage
from erguoyuan_football.research.live_data.provider_health import ProviderHealthChecker
from erguoyuan_football.research.live_data.readiness import (
    LiveDataReadinessGate,
    ReadinessConfig,
)
from erguoyuan_football.research.live_data.service import LiveResearchService
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.time_utils import parse_utc, utc_iso

ROOT = Path(__file__).resolve().parents[4]


def main(argv: list[str] | None = None) -> int:
    """Probe one explicitly named fixture; ambiguous pairs stay unresolved."""
    parser = argparse.ArgumentParser(description="Phase 13.5 research-only live probe")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--date", help="UTC match date, YYYY-MM-DD")
    parser.add_argument("--cutoff", help="UTC prediction cutoff, ISO-8601")
    args = parser.parse_args(argv)
    resolver = TeamResolver()
    home, away = resolver.resolve(args.home), resolver.resolve(args.away)
    print("MATCH RESEARCH LIVE PROBE")
    print(f"TEAM RESOLUTION: {home.team_id if home else 'NOT_FOUND'} / "
          f"{away.team_id if away else 'NOT_FOUND'}")
    if home is None or away is None or home.team_id == away.team_id:
        print("Prediction: NOT EXECUTED")
        return 2
    hint = (datetime.fromisoformat(args.date).replace(tzinfo=UTC)
            if args.date else None)
    config = OpenLigaDBConfig.from_yaml(ROOT / "config/phase13_5_provider.yaml")
    provider = OpenLigaDBProvider(config)
    store = EvidenceStore(":memory:", provider.sources)
    try:
        try:
            batch = provider.fetch_season()
        except (ValueError, OSError, RuntimeError) as error:
            health = ProviderHealthChecker.check(config.provider_id, config.enabled,
                                                 None, type(error).__name__)
            print(f"PROVIDER STATUS: {health.status}")
            print("Prediction: NOT EXECUTED")
            return 3
        cutoff = parse_utc(args.cutoff) if args.cutoff else datetime.now(UTC)
        if cutoff <= batch.retrieved_at:
            print("FIXTURE VERIFICATION: FUTURE_EVIDENCE_REJECTED")
            print("Prediction: NOT EXECUTED")
            return 2
        gate = LiveDataReadinessGate(
            ReadinessConfig.from_yaml(ROOT / "config/phase13_5_readiness.yaml"),
            load_model_requirements(ROOT / "config/model_registry.yaml"),
        )
        result = LiveResearchService(provider, store, gate).build(
            batch, home.team_id, away.team_id, cutoff=cutoff, date_hint=hint)
        health = ProviderHealthChecker.check(config.provider_id, config.enabled, batch)
        verified = frozenset({"FIXTURE", "RESULTS"} if result.package and
                             result.historical_count and result.recent_form_count else
                             {"FIXTURE"} if result.package else set())
        coverage = build_coverage(config.provider_id,
                                  declared=frozenset({"FIXTURE", "RESULTS"}),
                                  verified=verified)
        print(f"PROVIDER STATUS: {health.status} HTTP {batch.http_status}")
        print(f"FETCHED AT: {utc_iso(batch.retrieved_at)}")
        print(f"FIXTURE VERIFICATION: {result.fixture_check.status}")
        if result.fixture_check.fixture:
            fixture = result.fixture_check.fixture
            print(f"MATCH ID: {fixture.provider_match_id}")
            print(f"KICKOFF: {utc_iso(fixture.kickoff_at)}")
            print(f"SOURCE: {fixture.source_url}")
        print(f"HISTORICAL RESULTS: {result.historical_count}")
        print(f"RECENT FORM RECORDS: {result.recent_form_count}")
        print("DATA COVERAGE: " + ", ".join(
            f"{name}={status}" for name, status in coverage.statuses.items()))
        if result.readiness:
            report = result.readiness
            hierarchy = report.sample_hierarchy
            if hierarchy is not None:
                print(f"TEAM SAMPLE: home={hierarchy.home_team_direct_matches}, "
                      f"away={hierarchy.away_team_direct_matches}")
                print(f"COMPETITION SAMPLE: {hierarchy.competition_matches}")
                print(f"COMPARABLE SAMPLE: {hierarchy.comparable_competition_matches}")
                print("PRIOR SAMPLE: "
                      f"federation={hierarchy.federation_prior_matches}, "
                      f"age={hierarchy.age_group_prior_matches}, "
                      f"gender={hierarchy.gender_prior_matches}")
            print(f"READINESS: {report.overall_status}")
            print(f"UNCERTAINTY: {report.uncertainty_level}")
            print("MODEL GROUPS: " + ", ".join(
                f"{name}={state}" for name, state in (report.model_groups or {}).items()))
            print("USABLE MODELS: " + ", ".join(report.usable_models))
            print("BLOCKED MODELS: " + ", ".join(report.blocked_models))
            for model in report.model_readiness:
                if not model.ready:
                    print(f"BLOCKED {model.model_name}: {', '.join(model.reasons)}")
            print(f"RESEARCH PACKAGE: {'CREATED' if result.package else 'UNAVAILABLE'}")
        print("Prediction: NOT EXECUTED")
        return 0 if result.package else 2
    finally:
        store.close()
        provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
