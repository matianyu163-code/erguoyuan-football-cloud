"""Independent football Elo with pre-match history and a fitted 1X2 mapper."""

from __future__ import annotations

import math
from datetime import datetime
from typing import cast

from erguoyuan_football.contracts.common import utc
from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.rating_probability import RatingProbabilityMapper
from erguoyuan_football.models.training import TrainingDataset


class CoreEloModel(BaseFootballModel):
    """Elo rating updates and mapper are trained strictly in chronological order."""

    model_id = "ELO_V1"
    model_name = "Elo"
    model_version = "1.0.0"
    # Chronological multi-tournament fitting is limited to senior national teams.
    allows_senior_national_multi_competition = True
    required_data = ("historical_results",)
    optional_data = ("historical_goals",)
    supports_score_matrix = False
    supports_expected_goals = False

    def __init__(self) -> None:
        super().__init__()
        self.ratings: dict[str, float] = {}
        self.elo_history: list[dict[str, object]] = []
        self.mapper = RatingProbabilityMapper(self.model_id)

    def _binary_expected(self, home: float, away: float, neutral: bool) -> float:
        advantage = 0 if neutral else self.config.home_advantage
        return 1.0 / (1.0 + 10.0 ** ((away - (home + advantage)) / self.config.elo_scale))

    def _fit(self, data: TrainingDataset) -> None:
        if self.trained_until is None:
            raise RuntimeError("TRAINING_CUTOFF_NOT_SET")
        self.ratings = {team: self.config.initial_rating for team in self.teams}
        examples: list[tuple[float, float, int]] = []
        for row in sorted(data.matches, key=lambda item: (item.kickoff_time, item.match_id)):
            home = self.ratings[row.home_team_id]
            away = self.ratings[row.away_team_id]
            advantage = 0.0 if row.neutral_venue else self.config.home_advantage
            difference = home + advantage - away
            outcome = 0 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 2
            examples.append((difference, float(row.neutral_venue), outcome))
            expected = self._binary_expected(home, away, row.neutral_venue)
            actual = 1.0 if outcome == 0 else 0.5 if outcome == 1 else 0.0
            margin = max(1, abs(row.home_goals - row.away_goals))
            multiplier = math.log1p(margin) if self.config.goal_difference_adjustment else 1.0
            change = self.config.k_factor * multiplier * (actual - expected)
            self.elo_history.append({"team_id": row.home_team_id, "as_of_time": row.kickoff_time,
                                     "rating": home, "phase": "PRE_MATCH", "match_id": row.match_id})
            self.elo_history.append({"team_id": row.away_team_id, "as_of_time": row.kickoff_time,
                                     "rating": away, "phase": "PRE_MATCH", "match_id": row.match_id})
            self.ratings[row.home_team_id] = home + change
            self.ratings[row.away_team_id] = away - change
        self.mapper.fit(examples, self.trained_until, c=self.config.mapper_c)
        self.metadata.update({"rating_method": "logistic_elo_expected_score", **self.mapper.metadata()})

    def pre_match_rating(self, team_id: str, as_of_time: datetime | None = None) -> float:
        """Read the latest recorded pre-match rating at an optional UTC time."""
        if as_of_time is None:
            return float(self.ratings[team_id])
        cutoff = utc(as_of_time)
        rows = [row for row in self.elo_history
                if row["team_id"] == team_id and utc(cast(datetime, row["as_of_time"])) <= cutoff]
        return float(cast(float, rows[-1]["rating"]) if rows else self.config.initial_rating)

    def _predict_values(self, match):
        home = self.ratings[match.home_team_id]
        away = self.ratings[match.away_team_id]
        difference = (0 if match.neutral_venue else self.config.home_advantage) + home - away
        vector = self.mapper.predict(difference, bool(match.neutral_venue))
        return {"p_home": vector.p_home, "p_draw": vector.p_draw, "p_away": vector.p_away,
                "metadata": {"rating_home": home, "rating_away": away, "rating_difference": difference,
                              **self.mapper.metadata()}}
