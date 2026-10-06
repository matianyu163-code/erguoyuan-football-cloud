"""Evidence-gated data/model resource planning, without running models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from erguoyuan_football.jc_verification.jc_schema import JCStatus
from erguoyuan_football.research.match_universe import MatchUniverse


class ResourcePriority(StrEnum):
    """Competition profile priority; supplied only by reviewed configuration."""

    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass(frozen=True)
class CompetitionProfile:
    """Offline planning profile; it never establishes JC status."""

    competition_id: str
    priority: ResourcePriority


@dataclass(frozen=True)
class ResourcePlan:
    """Data and model priorities; no model execution is performed."""

    mode: str
    universe: MatchUniverse
    priority: ResourcePriority
    requested_data: tuple[str, ...]
    ordered_models: tuple[str, ...]
    reason: str


_FULL_DATA = ("historical_results", "odds", "lineup", "injury")
_FULL_MODELS = ("ELO_V1", "DIXON_COLES_V1", "BAYESIAN_HIERARCHICAL_V1")


class JCResourceScheduler:
    """Turn verification and an optional profile into a conservative plan."""

    def plan(self, status: JCStatus,
             profile: CompetitionProfile | None = None) -> ResourcePlan:
        """Schedule work based on evidence; UNKNOWN remains usable in basic mode."""
        priority = profile.priority if profile else ResourcePriority.D
        if status == JCStatus.NOT_JC:
            return ResourcePlan("LIMITED", MatchUniverse.GLOBAL_RESEARCH,
                ResourcePriority.D, ("historical_results",), ("ELO_V1",), "EXPLICIT_NOT_JC")
        if status == JCStatus.UNKNOWN:
            return ResourcePlan("BASIC", MatchUniverse.GLOBAL_RESEARCH,
                priority, ("historical_results",), ("ELO_V1",), "JC_STATUS_UNKNOWN")
        if status == JCStatus.USER_CONFIRMED and profile is None:
            return ResourcePlan("BASIC", MatchUniverse.JC_PRODUCTION,
                ResourcePriority.D, ("historical_results",), ("ELO_V1",),
                "USER_DECLARED_JC_WITHOUT_COMPETITION_PROFILE")
        if status == JCStatus.LIKELY:
            return ResourcePlan("STANDARD", MatchUniverse.JC_PRODUCTION,
                priority, ("historical_results", "odds"),
                ("ELO_V1", "DIXON_COLES_V1"), "JC_LIKELY")
        modes = {ResourcePriority.S: ("FULL", _FULL_DATA, _FULL_MODELS),
                 ResourcePriority.A: ("STANDARD", ("historical_results", "odds", "lineup"),
                                      ("ELO_V1", "DIXON_COLES_V1")),
                 ResourcePriority.B: ("ENHANCED_BASELINE", ("historical_results", "odds"),
                                      ("ELO_V1", "DIXON_COLES_V1")),
                 ResourcePriority.C: ("RESEARCH", ("historical_results",), ("ELO_V1",)),
                 ResourcePriority.D: ("RESEARCH", ("historical_results",), ("ELO_V1",))}
        mode, data, models = modes[priority]
        reason = ("USER_CONFIRMED_COMPETITION_PROFILE"
                  if status == JCStatus.USER_CONFIRMED else "JC_CONFIRMED")
        return ResourcePlan(mode, MatchUniverse.JC_PRODUCTION, priority, data, models,
                            reason)
