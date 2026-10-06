"""Production Bayesian execution must reject explicitly synthetic fixtures."""

from datetime import timedelta

import pytest

from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.hierarchical_bayes import CoreHierarchicalBayesianModel
from erguoyuan_football.models.training import InsufficientData
from erguoyuan_football.prediction.model_input_adapter import DixonColesInputAdapter


def test_synthetic_test_data_is_blocked_before_bayesian_sampling(sample_context) -> None:
    repository, hierarchy, cutoff = sample_context
    bundle = DixonColesInputAdapter().build_input(repository, hierarchy, cutoff)
    assert bundle.training_dataset.dataset_kind == "SYNTHETIC_TEST"
    model = CoreHierarchicalBayesianModel()
    config = ModelConfig(min_matches=10, min_team_matches=1, training_window=None)
    with pytest.raises(InsufficientData, match="SYNTHETIC_DATA_FORBIDDEN_IN_PRODUCTION"):
        model.fit(bundle.training_dataset, cutoff + timedelta(hours=1), config)
    assert model.fitted is False
