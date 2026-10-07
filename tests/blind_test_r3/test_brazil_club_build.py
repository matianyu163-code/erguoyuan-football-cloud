"""Brazil Serie A frozen model snapshot acceptance checks."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from erguoyuan_football.blind_test_r3.brazil_club_build import (
    MODEL_CLASSES,
    PROJECT_ROOT,
    TARGET_CUTOFF,
    load_mapping,
    parse_matches,
)
from erguoyuan_football.blind_test_r3.club_brazil import (
    BRAZIL_SERIE_A,
    BrazilBivariatePoissonModel,
    BrazilDixonColesModel,
    BrazilEloModel,
)
from erguoyuan_football.data.schemas import Fixture


def test_source_rows_are_date_safe_and_aliases_are_exact() -> None:
    rows = parse_matches()
    mapping = load_mapping()
    source_names = set(mapping)
    assert len(rows) == 3296
    assert all(row.kickoff_time.date() < TARGET_CUTOFF for row in rows)
    assert {row.home_team_id for row in rows} | {row.away_team_id for row in rows}
    assert all(row.source == "OPENFOOTBALL_CC0_1_0" for row in rows)
    assert {"Vitória", "Chapecoense", "Botafogo", "Vasco da Gama", "Remo", "Grêmio",
            "Red Bull Bragantino", "Mirassol", "Internacional", "Corinthians"} <= source_names
    assert mapping["SC Corinthians Paulista"] == mapping["Corinthians"]
    assert mapping["SC Internacional"] == mapping["Internacional"]
    assert mapping["RB Bragantino"] == mapping["Red Bull Bragantino"]


@pytest.mark.parametrize("model_id,model_type", list(MODEL_CLASSES.items()))
def test_frozen_brazil_artifact_loads_and_reproduces_golden(
    model_id: str, model_type: type[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = PROJECT_ROOT / "cloud_release" / "models" / "club_brazil"
    release_path = root / "manifest_R3-BRAZIL-07a4099b742f1d7e6f5b0873.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    assert release["release_status"] == "FROZEN_CANDIDATE_PENDING_ROUTER_BINDING_AND_CLOUD_INSTALL"
    record = release["models"][model_id]
    model_class: Any = model_type

    def reject_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("RUNTIME_FIT_FORBIDDEN")

    monkeypatch.setattr(model_class, "fit", reject_fit)
    model_path = root / record["artifact_path"]
    model = model_class.load(model_path)
    assert model.model_id == model_id
    assert model.fitted

    for golden in release["golden_references"]:
        kickoff = datetime.fromisoformat(golden["match_date"]).replace(tzinfo=UTC) + timedelta(days=1)
        fixture = Fixture(
            match_id=golden["fixture_id"],
            competition_id=BRAZIL_SERIE_A,
            home_team_id=golden["home_team_id"],
            away_team_id=golden["away_team_id"],
            kickoff_time=kickoff,
            source="GOLDEN_TEST",
            retrieved_at=datetime.now(UTC),
            as_of_time=datetime.now(UTC),
            data_version="golden-reference-v1",
            neutral_venue=golden["neutral_venue"],
        )
        assert model.supports_fixture(fixture)
        values = model._predict_values(fixture)
        expected = golden["model_probabilities"][model_id]
        assert (values["p_home"], values["p_draw"], values["p_away"]) == pytest.approx(
            expected, abs=1e-12
        )


def test_fixture_support_is_closed_to_other_domains_and_unknown_teams() -> None:
    root = PROJECT_ROOT / "cloud_release" / "models" / "club_brazil"
    release_path = root / "manifest_R3-BRAZIL-07a4099b742f1d7e6f5b0873.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    model_classes = {
        "CLUB_ELO_BRAZIL_V1": BrazilEloModel,
        "CLUB_DIXON_COLES_BRAZIL_V1": BrazilDixonColesModel,
        "CLUB_BIVARIATE_POISSON_BRAZIL_V1": BrazilBivariatePoissonModel,
    }
    for model_id, model_class in model_classes.items():
        model = model_class.load(root / release["models"][model_id]["artifact_path"])
        fixture = Fixture(
            match_id="unsupported-test-fixture",
            competition_id="FINLAND_VEIKKAUSLIIGA",
            home_team_id="BRAZIL_CLUB_VITORIA",
            away_team_id="BRAZIL_CLUB_CHAPECOENSE",
            kickoff_time=datetime(2026, 10, 8, tzinfo=UTC),
            source="SYNTHETIC_TEST",
            retrieved_at=datetime.now(UTC),
            as_of_time=datetime.now(UTC),
            data_version="synthetic-test",
            neutral_venue=False,
        )
        assert not model.supports_fixture(fixture)
