"""Prior influence audit for posterior goal-rate pooling."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

PriorInfluenceStatus = Literal["ACCEPTED", "PRIOR_DOMINATED", "PRIOR_UNUSED"]


@dataclass(frozen=True)
class PriorInfluenceReport:
    """Quantify the prior's share and measured parameter shift."""

    direct_sample_count: int
    prior_sample_count: int
    prior_strength: float
    posterior_shift: float
    prior_effect_ratio: float
    status: PriorInfluenceStatus

    def to_dict(self) -> dict[str, int | float | str]:
        """Provide a JSON-ready representation for prediction metadata."""
        return asdict(self)


def build_prior_influence_report(*, direct_sample_count: int,
                                 prior_sample_count: int, prior_strength: float,
                                 direct_posterior_parameters: tuple[float, float],
                                 posterior_parameters: tuple[float, float],
                                 unused_tolerance: float = 1e-8
                                 ) -> PriorInfluenceReport:
    """Classify prior dominance/use without changing any fitted parameter."""
    if direct_sample_count < 0 or prior_sample_count < 0 or prior_strength < 0:
        raise ValueError("PRIOR_INFLUENCE_COUNTS_MUST_BE_NONNEGATIVE")
    if not math.isfinite(prior_strength) or unused_tolerance < 0:
        raise ValueError("PRIOR_INFLUENCE_SETTINGS_INVALID")
    if len(direct_posterior_parameters) != 2 or len(posterior_parameters) != 2:
        raise ValueError("PRIOR_INFLUENCE_EXPECTS_TWO_GOAL_RATES")
    if not all(math.isfinite(value) for value in (*direct_posterior_parameters,
                                                   *posterior_parameters)):
        raise ValueError("PRIOR_INFLUENCE_PARAMETERS_NONFINITE")
    shift = math.dist(direct_posterior_parameters, posterior_parameters)
    denominator = direct_sample_count + prior_strength
    ratio = prior_strength / denominator if denominator else 0.0
    status: PriorInfluenceStatus = (
        "PRIOR_UNUSED" if shift <= unused_tolerance or prior_strength == 0 else
        "PRIOR_DOMINATED" if ratio >= 0.5 else "ACCEPTED")
    return PriorInfluenceReport(direct_sample_count, prior_sample_count,
        prior_strength, shift, ratio, status)
