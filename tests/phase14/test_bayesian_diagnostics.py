"""Phase 14.4 stabilization policy and source-routing tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from erguoyuan_football.models.bayesian_diagnostics import (
    BayesianDiagnosticStatus,
    diagnose_sampler,
)
from erguoyuan_football.models.bayesian_sampler_manager import BayesianSamplerManager
from erguoyuan_football.research.data_priority_scheduler import (
    DataFetchTask,
    DataKind,
    DataPriorityScheduler,
)
from erguoyuan_football.research.historical_data_acquirer import (
    HistoricalDataAcquirer,
    HistoricalProviderUnavailable,
)
from erguoyuan_football.research.historical_provider_router import (
    HistoricalCapability,
    HistoricalProviderOption,
    HistoricalProviderRouter,
    HistoricalProviderTier,
)
from erguoyuan_football.research.live_data.openligadb import SeasonBatch
from erguoyuan_football.research.match_universe import MatchUniverse
from erguoyuan_football.research.samples.prior_influence_report import (
    build_prior_influence_report,
)


def _diagnostic(rhat: float, ess: float):
    return diagnose_sampler(
        diagnostics={"r_hat": [rhat], "ess": [ess]}, posterior=[[0.1, 0.2]],
        max_r_hat=1.05, min_ess=200, execution_mode="FULL_MCMC",
        iterations=1000, warmup=500, chains=4,
    )


def test_bayesian_diagnostic_rejects_bad_rhat_and_nonfinite_posterior() -> None:
    assert _diagnostic(1.2, 500).status == BayesianDiagnosticStatus.R_HAT_FAILED
    failed = diagnose_sampler(
        diagnostics={"r_hat": [1.0], "ess": [500]}, posterior=[[float("nan")]],
        max_r_hat=1.05, min_ess=200, execution_mode="FULL_MCMC",
        iterations=1000, warmup=500, chains=4,
    )
    assert failed.status == BayesianDiagnosticStatus.POSTERIOR_INVALID


def test_bayesian_retry_keeps_map_out_of_production() -> None:
    calls: list[str] = []

    def fail() -> object:
        return _diagnostic(1.2, 50)

    outcome = BayesianSamplerManager().run(
        full_mcmc=lambda: (calls.append("full") or fail()),
        simplified_mcmc=lambda: (calls.append("simplified") or fail()),
        map_estimation=lambda: (calls.append("map") or ((0.1, 0.2), None)),
    )
    assert calls == ["full", "simplified", "map"]
    assert outcome.production_eligible is False
    assert outcome.diagnostic.status == BayesianDiagnosticStatus.MAP_INTERNAL_ONLY
    assert outcome.map_parameters == (0.1, 0.2)
    assert len(outcome.attempts) == 3


def test_prior_influence_reports_dominance_and_unused_prior() -> None:
    dominated = build_prior_influence_report(
        direct_sample_count=2, prior_sample_count=100, prior_strength=10,
        direct_posterior_parameters=(1.8, 1.2), posterior_parameters=(1.0, 1.0),
    )
    unused = build_prior_influence_report(
        direct_sample_count=50, prior_sample_count=100, prior_strength=10,
        direct_posterior_parameters=(1.2, 0.9), posterior_parameters=(1.2, 0.9),
    )
    assert dominated.status == "PRIOR_DOMINATED"
    assert dominated.posterior_shift > 0
    assert unused.status == "PRIOR_UNUSED"


def _provider_option(provider_id: str, tier: HistoricalProviderTier,
                     season: int, *, competition: str = "COMP") -> HistoricalProviderOption:
    retrieved = datetime(2026, 1, 1, tzinfo=UTC)
    return HistoricalProviderOption(
        provider_id=provider_id, tier=tier, competition_id=competition, country="DEU",
        seasons=frozenset({season}), capabilities=frozenset({"HISTORICAL_RESULTS"}),
        coverage_status="VERIFIED",
        fetch=lambda requested: SeasonBatch((), retrieved, "https://valid.example/data",
                                             200, 1.0, requested),
        historical_results=lambda batch, fixture, cutoff: (),
        canonical_team_map_complete=True, canonical_team_ids={1: "TEAM_1"},
    )


def test_history_fallback_uses_exact_verified_tiered_coverage() -> None:
    router = HistoricalProviderRouter()
    selected = router.resolve(
        competition="COMP", country="DEU", season=2025,
        capability=HistoricalCapability.RESULTS,
        providers=(
            _provider_option("TIER_B", HistoricalProviderTier.TIER_B, 2025),
            _provider_option("BACKUP", HistoricalProviderTier.BACKUP, 2025),
            _provider_option("WRONG_COMP", HistoricalProviderTier.TIER_A, 2025,
                             competition="OTHER"),
        ),
    )
    assert [item.provider_id for item in selected] == ["TIER_B", "BACKUP"]

    class Config:
        competition_id = "COMP"
        team_country = "DEU"

    class Primary:
        config = Config()

    from erguoyuan_football.research.historical_result_cache import (
        HistoricalResultCache,
    )

    acquirer = HistoricalDataAcquirer(cache=HistoricalResultCache(":memory:"),
                                      fallback_providers=selected)
    try:
        fallback_rows = acquirer._fetch_fallback(
            Primary(), 2025, datetime(2026, 2, 1, tzinfo=UTC), [],
            RuntimeError("primary unavailable"),
        )
        batch, source = fallback_rows[0]
        assert batch.season == 2025
        assert source == "TIER_B"
        assert [provider_id for _, provider_id in fallback_rows] == ["TIER_B", "BACKUP"]
    finally:
        acquirer.close()


def test_future_fallback_evidence_is_rejected() -> None:
    late = _provider_option("LATE", HistoricalProviderTier.TIER_A, 2025)
    late = HistoricalProviderOption(
        **{**late.__dict__, "fetch": lambda season: SeasonBatch(
            (), datetime(2026, 3, 1, tzinfo=UTC), "https://valid.example/data", 200, 1.0,
            season)}
    )

    class Config:
        competition_id = "COMP"
        team_country = "DEU"

    class Primary:
        config = Config()

    from erguoyuan_football.research.historical_result_cache import (
        HistoricalResultCache,
    )

    acquirer = HistoricalDataAcquirer(cache=HistoricalResultCache(":memory:"),
                                      fallback_providers=(late,))
    try:
        with pytest.raises(HistoricalProviderUnavailable, match="HISTORY_PROVIDER_UNAVAILABLE"):
            acquirer._fetch_fallback(Primary(), 2025,
                datetime(2026, 2, 1, tzinfo=UTC), [], RuntimeError("primary unavailable"))
    finally:
        acquirer.close()


def test_jc_universe_runs_ahead_of_global_research_within_fetch_priorities() -> None:
    tasks = (
        DataFetchTask("g-history", MatchUniverse.GLOBAL_RESEARCH, DataKind.HISTORICAL, 0),
        DataFetchTask("jc-history", MatchUniverse.JC_PRODUCTION, DataKind.HISTORICAL, 1),
        DataFetchTask("jc-fixture", MatchUniverse.JC_PRODUCTION, DataKind.FIXTURE, 2),
        DataFetchTask("g-fixture", MatchUniverse.GLOBAL_RESEARCH, DataKind.FIXTURE, 3),
    )
    assert [item.match_id for item in DataPriorityScheduler().order(tasks)] == [
        "jc-fixture", "jc-history", "g-history", "g-fixture",
    ]
