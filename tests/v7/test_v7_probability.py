"""V7 probability vectors are finite, bounded, and sum to one."""

import pytest
from pydantic import ValidationError

from erguoyuan_football.contracts.v7 import V7Probability


def test_v7_probability_accepts_valid_vector() -> None:
    assert V7Probability(home=0.4, draw=0.3, away=0.3).home == 0.4


@pytest.mark.parametrize("values", [
    {"home": 0.4, "draw": 0.3, "away": 0.2},
    {"home": 1.1, "draw": 0.0, "away": -0.1},
    {"home": float("nan"), "draw": 0.5, "away": 0.5},
])
def test_v7_probability_rejects_invalid_vector(values) -> None:
    with pytest.raises(ValidationError):
        V7Probability(**values)
