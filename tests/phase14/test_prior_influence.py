"""Bayesian prior influence is visible and gated."""

from erguoyuan_football.research.samples.prior_influence_report import (
    build_prior_influence_report,
)


def test_prior_effect_ratio_and_dominance_are_reported() -> None:
    report = build_prior_influence_report(
        direct_sample_count=2, prior_sample_count=80, prior_strength=8,
        direct_posterior_parameters=(2.0, 0.7), posterior_parameters=(1.0, 0.9),
    )
    assert report.prior_effect_ratio == 0.8
    assert report.posterior_shift > 0
    assert report.status == "PRIOR_DOMINATED"


def test_direct_posterior_unchanged_by_prior_is_prior_unused() -> None:
    report = build_prior_influence_report(
        direct_sample_count=40, prior_sample_count=100, prior_strength=5,
        direct_posterior_parameters=(1.1, 0.9), posterior_parameters=(1.1, 0.9),
    )
    assert report.status == "PRIOR_UNUSED"
