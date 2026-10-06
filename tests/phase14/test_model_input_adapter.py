"""Model-specific PIT inputs keep direct, competition and prior samples isolated."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from erguoyuan_football.prediction.model_input_adapter import (
    DixonColesInputAdapter,
    EloInputAdapter,
    HierarchicalBayesianInputAdapter,
    ModelInputError,
    PriorSupport,
    default_model_input_adapters,
)
from erguoyuan_football.research.samples.sample_builder import SampleBuilder
from erguoyuan_football.research.samples.sample_repository import SampleRepository
from tests.phase14.conftest import AWAY, COMPETITION, HOME, make_samples


def test_registry_has_all_model_specific_input_adapters() -> None:
    adapters = default_model_input_adapters()
    assert len(adapters) == 11
    assert adapters["DIXON_COLES_V1"].prior_support == PriorSupport.DIRECT_PLUS_COMPETITION
    assert adapters["CORE_OPTA_XG_ELO_LIKE_V1"].prior_support == PriorSupport.FEATURE_BASED


def test_goal_and_elo_adapters_build_real_training_datasets(sample_context) -> None:
    repository, hierarchy, cutoff = sample_context
    dc = DixonColesInputAdapter().build_input(repository, hierarchy, cutoff)
    elo = EloInputAdapter().build_input(repository, hierarchy, cutoff)
    assert len(dc.training_dataset.matches) == 48
    assert len(elo.training_dataset.matches) == 48
    assert dc.training_dataset.dataset_kind == "SYNTHETIC_TEST"
    assert all(row.kickoff_time < cutoff for row in dc.training_dataset.matches)
    assert dc.evidence_ids == tuple(sorted({evidence for row in dc.direct_samples
                                            for evidence in row.evidence_ids}
                                           | {evidence for row in dc.competition_samples
                                              for evidence in row.evidence_ids}))
    assert all(row.source == "SYNTHETIC_TEST" for row in dc.training_dataset.matches)


def test_hierarchical_adapter_fits_direct_rows_and_consumes_disjoint_prior(
        sample_context) -> None:
    repository, _hierarchy, cutoff = sample_context
    priors = make_samples(count=12, age_group="U17", competition_id="SYN_U17",
                          start_index=120)
    combined = SampleRepository(repository.samples + priors)
    hierarchy_with_prior = SampleBuilder(combined).build(
        replace(HOME, age_group="U16"), replace(AWAY, age_group="U16"),
        COMPETITION, cutoff=cutoff + timedelta(days=60), competition_type="LEAGUE")
    adapter = HierarchicalBayesianInputAdapter()
    bundle = adapter.build_input(combined, hierarchy_with_prior,
                                 cutoff + timedelta(days=60))
    assert bundle.prior_samples
    prior_ids = {row.match_id for row in bundle.prior_samples}
    assert prior_ids.isdisjoint(row.match_id for row in bundle.training_dataset.matches)
    assert adapter.supports_prior_execution() is True
    assert bundle.bayesian_prior is not None
    assert bundle.bayesian_prior.sample_count >= 8
    assert bundle.prior_parameter_source == "EMPIRICAL_PIT_HIERARCHICAL_RATE_PRIOR"
    assert len(bundle.training_dataset.matches) == len(bundle.direct_samples)
    assert bundle.bayesian_prior.evidence_ids


def test_future_sample_lineage_is_rejected(sample_context) -> None:
    repository, hierarchy, cutoff = sample_context
    future = make_samples(count=1, start_index=110)[0]
    future_repo = SampleRepository(repository.samples + (future,))
    bad_hierarchy = replace(hierarchy, competition=hierarchy.competition + (
        replace(hierarchy.competition[0], match_id=future.match_id,
                kickoff=future.kickoff, evidence_ids=future.evidence_ids),))
    with pytest.raises(ModelInputError, match="FUTURE_EVIDENCE_REJECTED"):
        DixonColesInputAdapter().build_input(future_repo, bad_hierarchy, cutoff)


def test_direct_sample_counts_do_not_include_priors(sample_context) -> None:
    repository, hierarchy, cutoff = sample_context
    bundle = DixonColesInputAdapter().build_input(repository, hierarchy, cutoff)
    direct_ids = {row.match_id for row in bundle.direct_samples}
    prior_ids = {row.match_id for row in bundle.prior_samples}
    assert direct_ids.isdisjoint(prior_ids)
    assert bundle.sample_counts["prior"] == 0
