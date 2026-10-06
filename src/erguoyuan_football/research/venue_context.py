"""Explicit venue classification with narrowly-scoped domestic league inference."""

from dataclasses import dataclass
from enum import StrEnum


class VenueContext(StrEnum):
    """Whether the fixture uses ordinary home advantage, a neutral venue, or unknown."""

    HOME_AWAY = "HOME_AWAY"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class VenueResolution:
    """Resolved venue context and the evidence or rule that supports it."""

    context: VenueContext
    basis: str


def resolve_venue_context(*, neutral_venue: bool | None, fixture_verified: bool,
                          competition_type: str, domestic_competition: bool) -> VenueResolution:
    """Infer home/away only for a verified domestic league fixture without neutral evidence."""
    if neutral_venue is True:
        return VenueResolution(VenueContext.NEUTRAL, "PROVIDER_EXPLICIT_NEUTRAL")
    if neutral_venue is False:
        return VenueResolution(VenueContext.HOME_AWAY, "PROVIDER_EXPLICIT_NON_NEUTRAL")
    if fixture_verified and domestic_competition and competition_type == "LEAGUE":
        return VenueResolution(VenueContext.HOME_AWAY,
                               "VERIFIED_DOMESTIC_LEAGUE_HOME_AWAY_INFERENCE")
    return VenueResolution(VenueContext.UNKNOWN, "VENUE_EVIDENCE_INSUFFICIENT")
