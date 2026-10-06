"""Deterministic confidence assignment from provider and identity evidence."""

from __future__ import annotations

from erguoyuan_football.jc_verification.jc_provider import ProviderTier


def confidence_for(tier: ProviderTier, *, exact_kickoff: bool,
                   exact_competition: bool) -> float:
    """Return conservative confidence for a candidate inside the match window."""
    if not exact_competition:
        return 0.0
    exact = {
        ProviderTier.OFFICIAL: 0.99,
        ProviderTier.LICENSED_STABLE: 0.92,
        ProviderTier.MANUAL_IMPORT: 0.80,
    }[tier]
    approximate = {
        ProviderTier.OFFICIAL: 0.90,
        ProviderTier.LICENSED_STABLE: 0.82,
        ProviderTier.MANUAL_IMPORT: 0.70,
    }[tier]
    return exact if exact_kickoff else approximate
