"""Time-isolated, fitted rating-to-1X2 probability mapping."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from erguoyuan_football.contracts.predictions import ProbabilityVector
from erguoyuan_football.models.training import InsufficientData


class RatingProbabilityMapper:
    """Multinomial logistic mapping fit only on pre-update ratings."""

    mapper_version = "1.0.0"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model: LogisticRegression | None = None
        self.trained_until: datetime | None = None
        self.training_sample_count = 0

    def fit(self, examples: list[tuple[float, float, int]], trained_until: datetime, c: float = 1.0) -> None:
        """Fit three-way probabilities from rating states before each result."""
        if len(examples) < 12 or len({label for _, _, label in examples}) != 3:
            raise InsufficientData("RATING_MAPPER_REQUIRES_12_ROWS_AND_3_OUTCOMES")
        x = np.asarray([[difference, neutral] for difference, neutral, _ in examples], dtype=float)
        y = np.asarray([label for _, _, label in examples], dtype=int)
        # Recent scikit-learn versions select multinomial loss automatically for
        # lbfgs when three classes are present; keeping the solver explicit also
        # avoids the removed ``multi_class`` constructor argument.
        self.model = LogisticRegression(C=c, solver="lbfgs", max_iter=1000)
        self.model.fit(x, y)
        self.trained_until = trained_until
        self.training_sample_count = len(examples)

    def predict(self, rating_difference: float, neutral_venue: bool) -> ProbabilityVector:
        """Return a normalized vector from the fitted classifier only."""
        if self.model is None:
            raise InsufficientData("RATING_MAPPER_NOT_FITTED")
        probabilities = self.model.predict_proba([[rating_difference, float(neutral_venue)]])[0]
        by_label = dict(zip(self.model.classes_, probabilities, strict=True))
        return ProbabilityVector(p_home=float(by_label.get(0, 0)), p_draw=float(by_label.get(1, 0)),
                                 p_away=float(by_label.get(2, 0)))

    def metadata(self) -> dict[str, Any]:
        """Return auditable mapper provenance."""
        return {"mapper_version": self.mapper_version, "mapper_id": self.model_id,
                "mapper_trained_until": self.trained_until.isoformat() if self.trained_until else None,
                "mapper_training_sample_count": self.training_sample_count}
