"""Constantinou–Fenton style dynamic Pi rating with separate home/away strengths."""

from __future__ import annotations

import math
from datetime import datetime
from typing import cast

from erguoyuan_football.contracts.common import utc
from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.rating_probability import RatingProbabilityMapper
from erguoyuan_football.models.training import TrainingDataset


class CorePiRatingModel(BaseFootballModel):
    """Updates two strengths per team using score discrepancy, then fits its own mapper."""

    model_id = "PI_RATING_V1"
    model_name = "Pi Rating"
    model_version = "1.0.0"
    # Chronological multi-tournament fitting is limited to senior national teams.
    allows_senior_national_multi_competition = True
    required_data = ("historical_goals",)
    optional_data = ("historical_results",)
    supports_score_matrix = False
    supports_expected_goals = False

    def __init__(self) -> None:
        super().__init__()
        self.home_strength: dict[str, float] = {}
        self.away_strength: dict[str, float] = {}
        self.pi_history: list[dict[str, object]] = []
        self.mapper = RatingProbabilityMapper(self.model_id)

    def _fit(self, data: TrainingDataset) -> None:
        if self.trained_until is None:
            raise RuntimeError("TRAINING_CUTOFF_NOT_SET")
        self.home_strength = {team: 0.0 for team in self.teams}
        self.away_strength = {team: 0.0 for team in self.teams}
        examples: list[tuple[float, float, int]] = []
        for row in sorted(data.matches, key=lambda item: (item.kickoff_time, item.match_id)):
            h = self.home_strength[row.home_team_id]
            a = self.away_strength[row.away_team_id]
            difference = h - a if not row.neutral_venue else (
                (h + self.away_strength[row.home_team_id]) / 2
                - (self.home_strength[row.away_team_id] + a) / 2)
            outcome = 0 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 2
            examples.append((difference, float(row.neutral_venue), outcome))
            self.pi_history.extend(({"team_id": row.home_team_id, "as_of_time": row.kickoff_time,
                                     "home_strength": h, "away_strength": self.away_strength[row.home_team_id],
                                     "phase": "PRE_MATCH", "match_id": row.match_id},
                                    {"team_id": row.away_team_id, "as_of_time": row.kickoff_time,
                                     "home_strength": self.home_strength[row.away_team_id],
                                     "away_strength": a, "phase": "PRE_MATCH", "match_id": row.match_id}))
            error = (row.home_goals - row.away_goals) - difference
            psi = math.copysign(3.0 * math.log10(1.0 + abs(error)), error) if error else 0.0
            alpha = self.config.pi_learning_rate
            beta = self.config.pi_coupling
            self.home_strength[row.home_team_id] += alpha * psi
            self.away_strength[row.home_team_id] += beta * psi
            self.home_strength[row.away_team_id] -= beta * psi
            self.away_strength[row.away_team_id] -= alpha * psi
        self.mapper.fit(examples, self.trained_until, c=self.config.mapper_c)
        self.metadata.update({"rating_method": "pi_score_discrepancy_log10", "psi_c": 3.0,
                              "alpha": self.config.pi_learning_rate, "beta": self.config.pi_coupling,
                              **self.mapper.metadata()})

    def pre_match_rating(self, team_id: str, as_of_time: datetime | None = None) -> dict[str, float]:
        """Return the two pre-match strengths at an optional UTC time."""
        if as_of_time is None:
            return {"home": self.home_strength[team_id], "away": self.away_strength[team_id]}
        cutoff = utc(as_of_time)
        rows = [row for row in self.pi_history
                if row["team_id"] == team_id and utc(cast(datetime, row["as_of_time"])) <= cutoff]
        if not rows:
            return {"home": 0.0, "away": 0.0}
        row = rows[-1]
        return {"home": float(cast(float, row["home_strength"])), "away": float(cast(float, row["away_strength"]))}

    def _predict_values(self, match):
        if match.neutral_venue:
            home = (self.home_strength[match.home_team_id] + self.away_strength[match.home_team_id]) / 2
            away = (self.home_strength[match.away_team_id] + self.away_strength[match.away_team_id]) / 2
        else:
            home = self.home_strength[match.home_team_id]
            away = self.away_strength[match.away_team_id]
        difference = home - away
        vector = self.mapper.predict(difference, bool(match.neutral_venue))
        return {"p_home": vector.p_home, "p_draw": vector.p_draw, "p_away": vector.p_away,
                "metadata": {"pi_home": home, "pi_away": away, "rating_difference": difference,
                              **self.mapper.metadata()}}
