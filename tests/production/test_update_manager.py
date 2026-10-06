"""Update manager tests verify version immutability using test-only manifests."""

import json

import pytest

from production.update_manager import UpdateManager


def test_update_manager_appends_version_history(tmp_path) -> None:
    history = tmp_path / "update_history.json"
    manager = UpdateManager(history)
    manifest = manager.register_update(
        category="models", version="MODEL_V1", change="SYNTHETIC_TEST initial",
        operator="SYNTHETIC_TEST",
    )
    record = json.loads(history.read_text(encoding="utf-8"))[0]
    assert manifest.exists()
    assert record["version"] == "MODEL_V1"
    assert record["operator"] == "SYNTHETIC_TEST"
    with pytest.raises(FileExistsError):
        manager.register_update(
            category="models", version="MODEL_V1", change="overwrite",
            operator="SYNTHETIC_TEST",
        )


def test_update_manager_rejects_unknown_version(tmp_path) -> None:
    manager = UpdateManager(tmp_path / "update_history.json")
    with pytest.raises(ValueError, match="UPDATE_VERSION_INVALID"):
        manager.register_update(
            category="configs", version="MODEL_V2", change="SYNTHETIC_TEST",
            operator="SYNTHETIC_TEST",
        )


def test_update_manager_registers_provider_and_rules_versions(tmp_path) -> None:
    history = tmp_path / "update_history.json"
    manager = UpdateManager(history)
    provider = manager.register_update(
        category="providers", version="PROVIDER_V1",
        change="SYNTHETIC_TEST provider metadata", operator="SYNTHETIC_TEST",
    )
    rule = manager.register_update(
        category="rules", version="RULE_V1",
        change="SYNTHETIC_TEST rule metadata", operator="SYNTHETIC_TEST",
    )
    assert provider.exists() and rule.exists()
    assert [row["category"] for row in json.loads(history.read_text(encoding="utf-8"))] == [
        "providers", "rules",
    ]
