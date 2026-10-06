"""Transparent evidence-completeness uncertainty accounting (not a probability)."""

from __future__ import annotations

from collections.abc import Mapping

from erguoyuan_football.contracts.common import Contract


class ContextUncertainty(Contract):
    score: float
    method: str
    assessed_components: int
    unavailable_components: int
    component_availability: dict[str, bool]
    reason_codes: tuple[str, ...]


class ContextUncertaintyEngine:
    """Compute the missing-component fraction and preserve its exact rationale."""

    METHOD = "MISSING_COMPONENT_FRACTION_V1"

    def evaluate(self, availability: Mapping[str, bool]) -> ContextUncertainty:
        if not availability:
            raise ValueError("CONTEXT_UNCERTAINTY_REQUIRES_COMPONENTS")
        missing = tuple(sorted(name for name, available in availability.items() if not available))
        score = len(missing) / len(availability)
        return ContextUncertainty(
            score=score, method=self.METHOD, assessed_components=len(availability),
            unavailable_components=len(missing), component_availability=dict(availability),
            reason_codes=tuple(f"MISSING:{name}" for name in missing),
        )
