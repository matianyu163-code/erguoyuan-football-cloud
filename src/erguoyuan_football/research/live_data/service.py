"""Adapter-backed research assembly from a validated OpenLigaDB batch."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime

from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)
from erguoyuan_football.knowledge.teams.team_database import TeamDatabase
from erguoyuan_football.research.data_quality import DataQualityEvaluator
from erguoyuan_football.research.live_data.openligadb import (
    FixtureCheck,
    OpenLigaDBProvider,
    SeasonBatch,
)
from erguoyuan_football.research.live_data.readiness import (
    LiveDataReadinessGate,
    LiveDataReadinessReport,
)
from erguoyuan_football.research.match_package import MatchResearchPackage
from erguoyuan_football.research.research_status import ResearchStatus
from erguoyuan_football.research.samples.sample_builder import SampleBuilder
from erguoyuan_football.research.samples.sample_repository import SampleRepository
from erguoyuan_football.research.validators import validate_package
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.time_utils import utc_iso


@dataclass(frozen=True)
class LiveResearchResult:
    """A real fixture verdict plus optional persisted research and readiness."""

    fixture_check: FixtureCheck
    package: MatchResearchPackage | None
    readiness: LiveDataReadinessReport | None
    historical_count: int
    recent_form_count: int
    prediction_executed: bool = False


def _record(kind: str, value: dict[str, object], batch: SeasonBatch,
            match_key: str, *, confidence: str = "MEDIUM",
            source_id: str = "OPENLIGADB") -> EvidenceRecord:
    digest = hashlib.sha256(f"{kind}:{match_key}:{value}:{utc_iso(batch.retrieved_at)}"
                            .encode()).hexdigest()
    stamp = utc_iso(batch.retrieved_at)
    return EvidenceRecord(
        f"{kind}_{digest}", kind, value, source_id, None, stamp, confidence,
        batch.source_url, stamp, provider_id=source_id, source_tier=2,
        observed_time=stamp, match_key=match_key,
    )


class LiveResearchService:
    """Build a PIT evidence package; this class has no model dependency."""

    def __init__(self, provider: OpenLigaDBProvider, store: EvidenceStore,
                 gate: LiveDataReadinessGate,
                 *, team_database: TeamDatabase | None = None) -> None:
        self.provider = provider
        self.store = store
        self.gate = gate
        self.team_database = team_database or TeamDatabase()

    def build(self, batch: SeasonBatch, home_team_id: str, away_team_id: str,
              *, cutoff: datetime, date_hint: datetime | None = None) -> LiveResearchResult:
        """Require exact identity and save only evidence available by cutoff."""
        check = self.provider.verify_fixture(
            batch, home_team_id, away_team_id,
            date_hint=date_hint.date() if date_hint else None, cutoff=cutoff,
        )
        fixture = check.fixture
        if fixture is None:
            return LiveResearchResult(check, None, None, 0, 0)
        home = self.team_database.get(home_team_id)
        away = self.team_database.get(away_team_id)
        if home is None or away is None:
            raise ValueError("TEAM_IDENTITY_NOT_IN_CATALOG")
        match_key = f"MATCH:{fixture.provider_match_id}"
        fixture_record = _record("FIXTURE", {
            "provider_match_id": fixture.provider_match_id,
            "home_team_id": home_team_id, "away_team_id": away_team_id,
            "competition_id": fixture.competition_id,
            "kickoff_at": utc_iso(fixture.kickoff_at), "venue": fixture.venue,
        }, batch, match_key, confidence="HIGH", source_id=self.provider.config.provider_id)
        results = self.provider.historical_results(batch, fixture, cutoff)
        for record in (fixture_record, *results):
            self.store.save(record)
        available = {"fixture": [fixture_record]}
        if results:
            available["historical_results"] = list(results)
        form_records: list[EvidenceRecord] = []
        for team_id in (home_team_id, away_team_id):
            binding = self.provider.config.team_bindings[team_id]
            played = [record for record in results if binding.provider_team_id in
                      (record.value["team1_provider_id"], record.value["team2_provider_id"])]
            played.sort(key=lambda record: record.value["kickoff_time"], reverse=True)
            recent = played[:5]
            if not recent:
                continue
            wins = draws = losses = goals_for = goals_against = 0
            for record in recent:
                value = record.value
                is_home = value["team1_provider_id"] == binding.provider_team_id
                scored = value["home_goals"] if is_home else value["away_goals"]
                allowed = value["away_goals"] if is_home else value["home_goals"]
                goals_for += scored
                goals_against += allowed
                wins += int(scored > allowed)
                draws += int(scored == allowed)
                losses += int(scored < allowed)
            form = _record("RECENT_FORM", {
                "team_id": team_id, "matches": len(recent), "wins": wins,
                "draws": draws, "losses": losses, "goals_for": goals_for,
                "goals_against": goals_against, "window": "LAST_5_COMPLETED",
                "data_origin": "LOCAL_DERIVED",
                "derived_from": [record.evidence_id for record in recent],
            }, batch, match_key, source_id=self.provider.config.provider_id)
            self.store.save(form)
            form_records.append(form)
        if form_records:
            available["recent_form"] = form_records
        competition = CompetitionIdentity(fixture.competition_id,
                                          self.provider.config.competition_name,
                                          self.provider.config.team_federation,
                                          self.provider.config.team_country, "CLUB")
        package = MatchResearchPackage(
            fixture.provider_match_id, home, away, competition,
            available_data=available,
            missing_data=(([] if results else ["historical_results"]) +
                          ["odds", "xg", "xga", "news", "injury", "lineup"]),
            sources=[self.provider.source_record], status=ResearchStatus.PARTIAL,
            as_of_time=utc_iso(cutoff), network_status="AVAILABLE",
            live_fetch_attempted=True, live_fetch_completed=True,
            verified_fixture=fixture, live_data_status="VERIFIED",
        )
        package.quality_score = DataQualityEvaluator().evaluate(package)
        validate_package(package)
        config = self.provider.config
        sampled_home = replace(home, entity_type=config.competition_entity_type,
                               gender=config.competition_gender,
                               age_group=config.competition_age_group,
                               federation=config.team_federation)
        sampled_away = replace(away, entity_type=config.competition_entity_type,
                               gender=config.competition_gender,
                               age_group=config.competition_age_group,
                               federation=config.team_federation)
        hierarchy = SampleBuilder(SampleRepository.from_openligadb(
            batch, package, config, cutoff)).build(
                sampled_home, sampled_away, competition,
                cutoff=cutoff, competition_type="LEAGUE")
        bindings = self.provider.config.team_bindings
        report = self.gate.evaluate(
            package, fixture=fixture, cutoff=cutoff,
            team_provider_ids=(bindings[home_team_id].provider_team_id,
                               bindings[away_team_id].provider_team_id),
            sample_hierarchy=hierarchy, entity_resolution_status="VERIFIED",
        )
        package.readiness_report = report
        return LiveResearchResult(check, package, report, len(results), len(form_records))
