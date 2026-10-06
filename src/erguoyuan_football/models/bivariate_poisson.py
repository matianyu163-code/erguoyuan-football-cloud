"""Karlis–Ntzoufras bivariate Poisson V1 with a shared lambda3 component."""

from penaltyblog.models import BivariatePoissonGoalModel

from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.penaltyblog_adapter import PenaltyblogAdapter
from erguoyuan_football.models.training import TrainingDataset


class CoreBivariatePoissonModel(BaseFootballModel):
    """Joint score model X=W1+W3, Y=W2+W3; lambda3 is fitted, not asserted."""

    model_id = "BIVARIATE_POISSON_V1"
    model_name = "Bivariate Poisson"
    model_version = "1.0.0"
    # Multi-tournament fitting is enabled only for an all-senior-national corpus.
    allows_senior_national_multi_competition = True
    required_data = ("historical_goals",)

    def __init__(self) -> None:
        super().__init__()
        self.adapter = PenaltyblogAdapter(BivariatePoissonGoalModel)

    def _fit(self, data: TrainingDataset) -> None:
        self.adapter.fit(data, self.config)
        backend = self.adapter.backend
        if backend is None:
            raise RuntimeError("BIVARIATE_BACKEND_NOT_FITTED")
        params = backend.get_params()
        self.metadata.update({"correlation_log": float(params["correlation_log"]),
                              "lambda3": float(params["lambda3"])})

    def _predict_values(self, match):
        return self.adapter.predict(match.home_team_id, match.away_team_id,
                                    max_goals=self.config.max_goals, neutral_venue=bool(match.neutral_venue))
