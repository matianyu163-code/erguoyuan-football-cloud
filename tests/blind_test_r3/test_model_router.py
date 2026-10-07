"""Synthetic-only tests for domain routing and frozen-model fallbacks."""

from __future__ import annotations

from erguoyuan_football.blind_test_r3.capability import (
    CLUB,
    NATIONAL_TEAM,
    UNIVERSAL,
    UNKNOWN,
    ModelCapabilityRouter,
    ModelRegistration,
)


def _result(home: float, draw: float, away: float) -> dict[str, float]:
    """Return a synthetic normalized distribution for router tests only."""
    return {"HOME": home, "DRAW": draw, "AWAY": away}


def test_club_fixture_does_not_use_only_national_snapshot() -> None:
    router = ModelCapabilityRouter([ModelRegistration(
        "SYNTHETIC_TEST_NATIONAL_MODEL", NATIONAL_TEAM,
        lambda: _result(0.5, 0.3, 0.2))])

    result = router.execute(CLUB)

    assert result["available_models"] == []
    assert result["model_coverage"] == "UNAVAILABLE"
    assert result["unavailable_models"]["SYNTHETIC_TEST_NATIONAL_MODEL"] == (
        "DOMAIN_MISMATCH")
    assert "NO_CLUB_MODEL_REGISTERED" in result["unavailable_models"]


def test_national_model_failure_does_not_block_club_model() -> None:
    def fail() -> dict[str, float]:
        raise RuntimeError("SYNTHETIC_TEST_MODEL_FAILURE")

    router = ModelCapabilityRouter([
        ModelRegistration("SYNTHETIC_TEST_NATIONAL_MODEL", NATIONAL_TEAM, fail),
        ModelRegistration("SYNTHETIC_TEST_CLUB_MODEL", CLUB,
                          lambda: _result(0.45, 0.3, 0.25)),
    ])

    result = router.execute(CLUB)

    assert result["available_models"] == ["SYNTHETIC_TEST_CLUB_MODEL"]
    assert result["available_model_ensemble"] == _result(0.45, 0.3, 0.25)
    assert result["unavailable_models"]["SYNTHETIC_TEST_NATIONAL_MODEL"] == (
        "DOMAIN_MISMATCH")


def test_one_available_model_is_enough() -> None:
    router = ModelCapabilityRouter([ModelRegistration(
        "SYNTHETIC_TEST_CLUB_MODEL", CLUB,
        lambda: _result(0.4, 0.35, 0.25))], minimum_available_models=1)

    result = router.execute(CLUB)

    assert result["available_model_count"] == 1
    assert result["model_coverage"] == "FULL"
    assert result["available_model_ensemble"] == _result(0.4, 0.35, 0.25)


def test_available_models_are_ensembled_equally() -> None:
    router = ModelCapabilityRouter([
        ModelRegistration("SYNTHETIC_TEST_CLUB_A", CLUB,
                          lambda: _result(0.6, 0.25, 0.15)),
        ModelRegistration("SYNTHETIC_TEST_CLUB_B", CLUB,
                          lambda: _result(0.4, 0.35, 0.25)),
    ])

    result = router.execute(CLUB)

    assert result["available_model_count"] == 2
    assert result["available_model_ensemble"] == _result(0.5, 0.3, 0.2)
    assert result["ensemble_method"] == "EQUAL_WEIGHT_AVAILABLE_MODELS_V1"


def test_unknown_domain_uses_only_a_registered_universal_model() -> None:
    router = ModelCapabilityRouter([
        ModelRegistration("SYNTHETIC_TEST_NATIONAL_MODEL", NATIONAL_TEAM,
                          lambda: _result(0.5, 0.3, 0.2)),
        ModelRegistration("SYNTHETIC_TEST_GLOBAL_MODEL", UNIVERSAL,
                          lambda: _result(0.4, 0.3, 0.3)),
    ])

    result = router.execute(UNKNOWN)

    assert result["available_models"] == ["SYNTHETIC_TEST_GLOBAL_MODEL"]
    assert result["unavailable_models"]["SYNTHETIC_TEST_NATIONAL_MODEL"] == (
        "DOMAIN_MISMATCH")
