"""One score convention: row=home goals, column=away goals, inclusive max_goals."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import Contract, Probability
from erguoyuan_football.contracts.predictions import ProbabilityVector


class ScoreMatrix(Contract):
    """Normalized finite support, with explicitly measured discarded tail mass."""

    values: tuple[tuple[Probability, ...], ...]
    max_goals: int = Field(ge=0)
    retained_mass: Probability
    tail_mass: Probability
    tail_policy: Literal["CONDITIONAL_RENORMALIZATION"] = "CONDITIONAL_RENORMALIZATION"

    @model_validator(mode="after")
    def validate_mass(self) -> ScoreMatrix:
        """Reject malformed support, non-finite cells and inconsistent tail accounting."""
        if len(self.values) != self.max_goals + 1 or any(len(r) != len(self.values) for r in self.values):
            raise ValueError("invalid square score support")
        if not math.isclose(sum(map(sum, self.values)), 1.0, abs_tol=1e-8, rel_tol=0):
            raise ValueError("score mass must equal one")
        if self.retained_mass <= 0 or not math.isclose(self.retained_mass + self.tail_mass, 1, abs_tol=1e-8):
            raise ValueError("invalid retained/tail probability")
        return self

    @classmethod
    def from_raw(cls, raw: NDArray[np.float64]) -> ScoreMatrix:
        """Condition on the retained rectangle; never silently drop probability mass."""
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != values.shape[1] or not np.isfinite(values).all():
            raise ValueError("invalid probability matrix")
        if (values < 0).any():
            raise ValueError("negative score probability")
        mass = float(values.sum())
        if not 0 < mass <= 1 + 1e-8:
            raise ValueError("invalid raw score probability mass")
        normalized = values / mass
        return cls(values=tuple(tuple(float(v) for v in row) for row in normalized),
                   max_goals=values.shape[0] - 1, retained_mass=min(mass, 1), tail_mass=max(0, 1 - mass))

    def outcome(self) -> ProbabilityVector:
        """Derive H/D/A exclusively by summing the same normalized cells."""
        grid = np.asarray(self.values)
        return ProbabilityVector(p_home=float(np.tril(grid, -1).sum()), p_draw=float(np.trace(grid)),
                                 p_away=float(np.triu(grid, 1).sum()))
