"""Golden and synthetic-only checks for the user-authoritative daily entry."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from erguoyuan_football.blind_test_r3.user_daily import UserDailySlate
from erguoyuan_football.blind_test_r3.user_daily_runner import (
    ROOT,
    _verify_release,
    init_state,
    run_daily,
)


def _synthetic_slate(tmp_path: Path) -> Path:
    """The future date is a test fixture, never a real market claim."""
    fixture = json.loads((ROOT / "inputs/daily_market/2026-10-06_luxembourg_bulgaria.json")
                         .read_text(encoding="utf-8"))
    fixture["slate_date"] = (datetime.now(UTC) + timedelta(days=7)).date().isoformat()
    row = fixture["fixtures"][0]
    row["kickoff"] = (datetime.now(UTC) + timedelta(days=7)).isoformat()
    row["screenshot_path"] = None
    row["metadata_sources"] = {"test": "SYNTHETIC_TEST"}
    path = tmp_path / "synthetic_daily.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    return path


def test_user_fixture_market_authoritative_without_reverification(tmp_path: Path) -> None:
    slate = UserDailySlate.model_validate(json.loads(_synthetic_slate(tmp_path).read_text(
        encoding="utf-8")))
    assert slate.source == "USER_AUTHORITATIVE"
    assert slate.fixtures[0].spf.home == 1.85
    assert slate.fixtures[0].rqspf.handicap == -1
    assert abs(sum(slate.fixtures[0].spf.no_vig().values()) - 1) < 1e-12
    assert abs(sum(slate.fixtures[0].rqspf.no_vig().values()) - 1) < 1e-12
    assert slate.fixtures[0].fixture_id(slate.slate_date).startswith("YYF-")


def test_strict_odds_and_complete_hda(tmp_path: Path) -> None:
    data = json.loads(_synthetic_slate(tmp_path).read_text(encoding="utf-8"))
    del data["fixtures"][0]["spf"]["away"]
    with pytest.raises(ValueError):
        UserDailySlate.model_validate(data)
    data["fixtures"][0]["spf"]["away"] = 1.0
    with pytest.raises(ValueError):
        UserDailySlate.model_validate(data)


def test_frozen_artifact_load_and_no_runtime_fit(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:
    model, release, _ = _verify_release(ROOT)
    assert model.fitted and release["source_snapshot_id"].startswith("R3-")

    def forbidden_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("RUNTIME_FIT_FORBIDDEN")

    monkeypatch.setattr(type(model), "fit", forbidden_fit)
    result = run_daily(_synthetic_slate(tmp_path), state_root=tmp_path / "state")
    row = result["results"][0]
    assert row["model_status"] == "AVAILABLE"
    assert row["r3"]["one_x_two"]["probabilities"]["HOME"] > 0
    assert row["r5"]["SPF"]["probabilities"]["HOME"] > 0
    assert row["fusion"]["model_weight"] == 0.7
    assert row["handicap_fusion"]["model_weight"] == 0.7
    assert abs(sum(row["handicap_fusion"]["probabilities"].values()) - 1) < 1e-9
    assert (tmp_path / "state/locks" / f"{row['lock_id']}.json").exists()
    prediction = (tmp_path / "state/predictions" / f"{row['prediction_id']}.json").read_bytes()
    lock = json.loads((tmp_path / "state/locks" / f"{row['lock_id']}.json")
                      .read_text(encoding="utf-8"))
    assert lock["prediction_sha256"] == hashlib.sha256(prediction).hexdigest()
    assert (tmp_path / "state/locks" /
            f"{result['final_output_lock_id']}.json").exists()
    assert result["selection_engine"]["total_goals_duplex"]["status"] == "AVAILABLE"


def test_market_and_lock_append_only(tmp_path: Path) -> None:
    init_state(tmp_path)
    from erguoyuan_football.blind_test_r3.store import write_once

    target = tmp_path / "locks" / "one.json"
    write_once(target, b"first")
    with pytest.raises(FileExistsError):
        write_once(target, b"second")
    assert target.read_bytes() == b"first"


def test_golden_reference_offline() -> None:
    reference = json.loads((ROOT / "tests/reference/GOLDEN_REFERENCE_001.json")
                           .read_text(encoding="utf-8"))
    assert reference["fusion_version"] == "MODEL_MARKET_FUSION_V1"
    assert abs(sum(reference["model_probabilities"].values()) - 1) < 1e-9
    assert abs(sum(reference["jc_no_vig"].values()) - 1) < 1e-9
