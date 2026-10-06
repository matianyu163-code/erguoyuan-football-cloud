"""Smoke checks only; these do not validate any prediction model."""

import tomllib
from pathlib import Path

import pytest

import erguoyuan_football


@pytest.mark.unit
def test_package_imports_from_project_source(project_root: Path) -> None:
    assert Path(erguoyuan_football.__file__).resolve() == (
        project_root / "src" / "erguoyuan_football" / "__init__.py"
    ).resolve()


@pytest.mark.unit
def test_project_configuration(project_root: Path) -> None:
    with (project_root / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    assert config["project"]["name"] == "erguoyuan-football"
    assert config["project"]["requires-python"] == ">=3.11,<3.12"
    assert config["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert set(config["tool"]["pytest"]["ini_options"]["testpaths"]) == {
        "tests/blind_test_r3", "tests/markets", "tests/models", "tests/pit", "tests/unit",
    }
