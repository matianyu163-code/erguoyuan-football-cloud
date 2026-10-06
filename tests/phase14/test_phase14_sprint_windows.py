"""Regression for model-specific history and point-in-time form windows."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from scipy.stats import poisson

from erguoyuan_football.models.hierarchical_bayes import CoreHierarchicalBayesianModel
from erguoyuan_football.models.score_matrix import ScoreMatrix
from erguoyuan_football.research.as_of_evidence_loader import (
    AsOfEvidenceLoader,
    ReplayEvidenceUnavailable,
)
from erguoyuan_football.research.historical_result_cache import HistoricalResultCache
from erguoyuan_football.research.live_data.competition_provider_factory import (
    bind_verified_competition,
)
from erguoyuan_football.research.live_data.model_readiness import (
    ModelReadinessEvaluator,
)
from erguoyuan_football.research.live_data.model_requirements import (
    ModelDataRequirement,
    ModelSamplePolicy,
    RequirementLevel,
)
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBConfig,
    OpenLigaDBProvider,
    SeasonBatch,
)
from erguoyuan_football.research.live_data.openligadb_directory import (
    DirectoryFetch,
    DirectoryTeam,
    OpenLigaDBDirectoryProvider,
)
from erguoyuan_football.research.live_data.readiness import (
    DataAvailability,
    DataReadinessItem,
)
from erguoyuan_football.research.match_universe import MatchUniverse
from erguoyuan_football.research.provider_coverage_profile import CoverageStatus
from erguoyuan_football.research.samples.bayesian_prior import BayesianPrior
from erguoyuan_football.research.samples.sample_builder import SampleBuilder
from erguoyuan_football.research.samples.sample_repository import SampleRepository
from erguoyuan_football.research.samples.sample_windows import build_match_windows
from erguoyuan_football.research.venue_context import (
    VenueContext,
    resolve_venue_context,
)
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from tests.phase14.conftest import AWAY, COMPETITION, HOME, START, make_samples


def test_model_specific_training_requirement_does_not_apply_global_40_gate() -> None:
    rows = make_samples(count=39)
    cutoff = START + timedelta(days=90)
    hierarchy = SampleBuilder(SampleRepository(rows)).build(
        HOME, AWAY, COMPETITION, cutoff=cutoff, competition_type="LEAGUE")
    items = {
        "HISTORICAL_RESULTS": DataReadinessItem("HISTORICAL_RESULTS",
            DataAvailability.AVAILABLE, evidence_ids=("SYNTHETIC_TEST_HISTORY",)),
        "THREE_WAY_MAPPER_TRAINING": DataReadinessItem(
            "THREE_WAY_MAPPER_TRAINING", DataAvailability.AVAILABLE,
            evidence_ids=("SYNTHETIC_TEST_OUTCOMES",)),
    }
    elo = ModelDataRequirement("ELO_V1",
        {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED},
        ModelSamplePolicy(12, 0, 0, False, require_three_outcomes=True,
                          training_min_matches=12, preferred_direct_each=10))
    result = ModelReadinessEvaluator().evaluate(
        elo, items, hierarchy, fixture_verified=True)
    assert len(rows) == 39
    assert result.ready
    assert result.degraded
    assert not any(reason.startswith("TRAINING_SAMPLE_LOW") for reason in result.reasons)


def test_windows_are_latest_first_and_strictly_point_in_time() -> None:
    cutoff = START + timedelta(days=90)
    rows = make_samples(count=80)
    windows = build_match_windows(rows, HOME.team_id, AWAY.team_id,
                                  COMPETITION.competition_id, cutoff)
    assert len(windows.long_term_home) > len(windows.standard_home)
    assert len(windows.short_home) <= 5
    assert len(windows.recent_home) <= 10
    assert len(windows.standard_home) <= 20
    assert all(row.kickoff < cutoff and row.fetched_at <= cutoff
               for row in windows.long_term_home + windows.long_term_away)
    assert tuple(sorted((row.kickoff for row in windows.recent_home), reverse=True)) == tuple(
        row.kickoff for row in windows.recent_home)
    assert all(form.match_count <= 20 for key, form in windows.forms.items()
               if "last20" in key)


def test_match_universe_values_are_explicit() -> None:
    assert MatchUniverse.JC_PRODUCTION.value == "JC_PRODUCTION"
    assert MatchUniverse.GLOBAL_RESEARCH.value == "GLOBAL_RESEARCH"


def test_historical_cache_preserves_observation_time_and_rejects_future(tmp_path) -> None:
    cache = HistoricalResultCache(tmp_path / "history.sqlite")
    observed = datetime(2025, 6, 1, tzinfo=UTC)
    batch = SeasonBatch(({"matchID": 7, "matchIsFinished": True},), observed,
                        "https://api.openligadb.de/getmatchdata/bl1/2025", 200, 4.0, 2025)
    try:
        cache.put("OPENLIGADB", "TEST_LEAGUE", batch)
        available = cache.get("OPENLIGADB", "TEST_LEAGUE", 2025,
            cutoff=datetime(2025, 6, 2, tzinfo=UTC),
            now=datetime(2025, 6, 2, tzinfo=UTC))
        unavailable = cache.get("OPENLIGADB", "TEST_LEAGUE", 2025,
            cutoff=datetime(2025, 5, 31, tzinfo=UTC),
            now=datetime(2025, 6, 2, tzinfo=UTC))
        assert available is not None and available.retrieved_at == observed
        assert unavailable is None
    finally:
        cache.close()


def test_bayesian_prior_influence_decreases_as_direct_evidence_grows() -> None:
    prior = BayesianPrior("SAME_COMPETITION", 0.0, 0.3, 0.0, 0.3,
        0.1, 0.1, 1.4, 1.0, 8.0, 31, ("PRIOR_EVIDENCE",), 2, 0.9, 1.0)
    pmf_home = poisson.pmf(np.arange(6), 3.2)
    pmf_away = poisson.pmf(np.arange(6), 0.8)
    initial_matrix = ScoreMatrix.from_raw(np.outer(pmf_home, pmf_away))

    def pooled(direct_count: int) -> float:
        model = CoreHierarchicalBayesianModel()
        result = model._apply_empirical_prior({
            "lambda_home": 3.2, "lambda_away": 0.8,
            "expected_home_goals": 3.2, "expected_away_goals": 0.8,
            "score_matrix": initial_matrix.values,
            "metadata": {"score_matrix": initial_matrix.model_dump(mode="json")},
            "p_home": 0.3, "p_draw": 0.3, "p_away": 0.4,
        }, prior, direct_count)
        return abs(result["lambda_home"] - prior.home_goal_mean)

    assert pooled(3) < pooled(50)


def test_as_of_loader_only_returns_persisted_cutoff_valid_evidence(tmp_path) -> None:
    provider = OpenLigaDBProvider(OpenLigaDBConfig.from_yaml(
        "config/phase13_5_provider.yaml"))
    store = EvidenceStore(tmp_path / "evidence.sqlite", provider.sources)
    cutoff = datetime(2025, 6, 2, tzinfo=UTC)
    try:
        for evidence_id, kind, observed in (
                ("fixture", "FIXTURE", datetime(2025, 6, 1, tzinfo=UTC)),
                ("history", "HISTORICAL_RESULT", datetime(2025, 6, 1, tzinfo=UTC)),
                ("future", "HISTORICAL_RESULT", datetime(2025, 6, 3, tzinfo=UTC))):
            stamp = observed.isoformat().replace("+00:00", "Z")
            store.save(EvidenceRecord(evidence_id, kind, {}, "OPENLIGADB", None,
                stamp, "HIGH", provider.endpoint, stamp, provider_id="OPENLIGADB",
                source_tier=2, observed_time=stamp, match_key="MATCH:123"))
        bundle = AsOfEvidenceLoader(store).load_as_of("123", cutoff)
        assert tuple(row.evidence_id for row in bundle.fixture) == ("fixture",)
        assert tuple(row.evidence_id for row in bundle.historical_results) == ("history",)
        with pytest.raises(ReplayEvidenceUnavailable, match="REPLAY_EVIDENCE_UNAVAILABLE"):
            AsOfEvidenceLoader(store).load_as_of("missing", cutoff)
    finally:
        store.close()
        provider.close()


def test_domestic_league_can_resolve_missing_neutral_flag_safely() -> None:
    league = resolve_venue_context(neutral_venue=None, fixture_verified=True,
        competition_type="LEAGUE", domestic_competition=True)
    cup = resolve_venue_context(neutral_venue=None, fixture_verified=True,
        competition_type="CUP", domestic_competition=True)
    assert league.context == VenueContext.HOME_AWAY
    assert cup.context == VenueContext.UNKNOWN


def test_verified_directory_creates_exact_second_division_team_bindings() -> None:
    directory = OpenLigaDBDirectoryProvider("config/phase13_7_openligadb.yaml")
    scope_id = "germany_men_second_division_2026"
    scope = directory.scopes[scope_id]
    retrieved = datetime(2026, 10, 1, tzinfo=UTC)
    records = tuple(DirectoryTeam(100 + index, name, scope, f"EVIDENCE_{index}",
        retrieved, "https://api.openligadb.de/getavailableteams/bl2/2026")
        for index, name in enumerate(("Example FC A", "Example FC B")))
    response = DirectoryFetch(CoverageStatus.VERIFIED, records, retrieved,
                              tuple(row.evidence_id for row in records))
    bound = bind_verified_competition(directory, scope_id, response)
    try:
        assert bound.provider.config.competition_id == scope.competition_id
        assert len(bound.provider.config.team_bindings) == 2
        resolved = bound.team_resolver.resolve("Example FC A")
        assert resolved is not None
        assert bound.provider.config.team_bindings[resolved.team_id].provider_team_id == 100
        assert resolved.identity_status == "PROVIDER_VERIFIED"
    finally:
        bound.provider.close()
        directory.close()
