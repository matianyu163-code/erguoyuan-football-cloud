"""Release binding and Brazil frozen-golden checks."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from erguoyuan_football.blind_test_r3.release_registry import load_verified_release
from erguoyuan_football.data.schemas import Fixture

ROOT = Path(__file__).resolve().parents[2]


def test_verified_release_loads_all_four_models() -> None:
    """All registered artifacts are hash-bound and independently loadable."""
    release, _ = load_verified_release(ROOT)
    assert release["production_ready"] is False
    assert release["blind_test_ready"] is True
    assert set(release["loaded_models"]) == {
        "DIXON_COLES_V1", "CLUB_ELO_BRAZIL_V1", "CLUB_DIXON_COLES_BRAZIL_V1",
        "CLUB_BIVARIATE_POISSON_BRAZIL_V1",
    }


def test_brazil_frozen_golden_001_and_002() -> None:
    """Existing model probabilities reproduce both registered Brazil references."""
    release, _ = load_verified_release(ROOT)
    manifest = json.loads((ROOT / release["brazil_snapshot_manifest"]).read_text(
        encoding="utf-8"))
    brazil_models = {key: value for key, value in release["loaded_models"].items()
                     if key.startswith("CLUB_")}
    assert len(manifest["golden_references"]) >= 2
    for golden in manifest["golden_references"][:2]:
        kickoff = datetime.fromisoformat(golden["match_date"]).replace(tzinfo=UTC)
        fixture = Fixture(
            match_id=golden["fixture_id"], competition_id=golden["competition_id"],
            home_team_id=golden["home_team_id"], away_team_id=golden["away_team_id"],
            kickoff_time=kickoff, source="GOLDEN_REFERENCE", retrieved_at=kickoff,
            as_of_time=kickoff, data_version="GOLDEN_REFERENCE_V1",
            neutral_venue=golden["neutral_venue"],
        )
        for model_id, model in brazil_models.items():
            assert model.supports_fixture(fixture)
            values = model._predict_values(fixture)
            actual = [float(values[key]) for key in ("p_home", "p_draw", "p_away")]
            expected = golden["model_probabilities"][model_id]
            assert all(math.isclose(value, reference, abs_tol=1e-12)
                       for value, reference in zip(actual, expected, strict=True))
