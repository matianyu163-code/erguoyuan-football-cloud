"""Explicit provider capability vocabulary."""

from __future__ import annotations

from enum import StrEnum


class ProviderCapability(StrEnum):
    """A provider declares only schemas it actually implements."""

    FIXTURE = "FIXTURE"
    RESULT = "RESULT"
    TEAM_STATS = "TEAM_STATS"
    STANDINGS = "STANDINGS"
    ELO = "ELO"
    XG = "XG"
    INJURY = "INJURY"
    LINEUP = "LINEUP"
    ODDS = "ODDS"
    NEWS = "NEWS"
    TOURNAMENT_CONTEXT = "TOURNAMENT_CONTEXT"
    TEAM_DISCOVERY = "TEAM_DISCOVERY"
