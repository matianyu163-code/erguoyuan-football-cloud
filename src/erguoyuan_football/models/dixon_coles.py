"""Dixon–Coles V1 through the penaltyblog adapter."""

from penaltyblog.models import DixonColesGoalModel

from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.penaltyblog_adapter import PenaltyblogAdapter
from erguoyuan_football.models.training import TrainingDataset


class CoreDixonColesModel(BaseFootballModel):
    """Attack/defence, home advantage, rho correction and exponential time decay."""

    model_id = "DIXON_COLES_V1"
    model_name = "Dixon-Coles"
    model_version = "1.0.0"
    # Multi-tournament fitting is enabled only for an all-senior-national corpus.
    allows_senior_national_multi_competition = True
    required_data = ("historical_goals",)
    optional_data = ("market_odds", "xg")

    def __init__(self) -> None:
        super().__init__()
        self.adapter = PenaltyblogAdapter(DixonColesGoalModel)

    def _fit(self, data: TrainingDataset) -> None:
        self.adapter.fit(data, self.config)
        backend = self.adapter.backend
        if backend is None:
            raise RuntimeError("DIXON_COLES_BACKEND_NOT_FITTED")
        params = backend.get_params()
        self.metadata.update({"rho": float(params["rho"]), "home_advantage_log": float(params["home_advantage"])})

    def _predict_values(self, match):
        return self.adapter.predict(match.home_team_id, match.away_team_id,
                                    max_goals=self.config.max_goals, neutral_venue=bool(match.neutral_venue))
