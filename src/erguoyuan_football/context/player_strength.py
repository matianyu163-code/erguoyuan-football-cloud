"""Provider interface for trained, time-bounded player strength evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from erguoyuan_football.context.schemas import PlayerStrengthEvidence


class PlayerStrengthProvider(Protocol):
    """A real implementation returns versioned historical strength evidence."""

    def get_strength(self, player_id: str, *, as_of_time: datetime
                     ) -> PlayerStrengthEvidence | None: ...


class UnavailablePlayerStrengthProvider:
    """No player names or reputation are translated into numeric values."""

    def get_strength(self, player_id: str, *, as_of_time: datetime
                     ) -> PlayerStrengthEvidence | None:
        return None


def calculate_lineup_strength_delta(*, home_lineup: Sequence[str], away_lineup: Sequence[str],
                                    player_strength: Mapping[str, PlayerStrengthEvidence],
                                    baseline_home_strength: float | None,
                                    baseline_away_strength: float | None
                                    ) -> tuple[float | None, str]:
    """Compare evidenced lineup totals with model-versioned team baselines."""
    if not home_lineup or not away_lineup:
        return None, "EXPECTED_OR_CONFIRMED_LINEUP_UNAVAILABLE"
    if baseline_home_strength is None or baseline_away_strength is None:
        return None, "BASELINE_TEAM_STRENGTH_UNAVAILABLE"
    required = set(home_lineup) | set(away_lineup)
    if any(player_id not in player_strength or not player_strength[player_id].available
           for player_id in required):
        return None, "PLAYER_STRENGTH_EVIDENCE_UNAVAILABLE"
    home_deviation = sum(player_strength[player_id].value for player_id in home_lineup) - baseline_home_strength
    away_deviation = sum(player_strength[player_id].value for player_id in away_lineup) - baseline_away_strength
    return home_deviation - away_deviation, "AVAILABLE"
