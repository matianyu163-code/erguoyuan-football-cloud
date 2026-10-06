"""Packaged and source launchers must pass the CORE project root to the bridge."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from erguoyuan_football.app import main as desktop_main
from production import launcher
from production.bridge import runner


def test_production_launcher_forwards_project_root(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    config_path = root / "config" / "production.yaml"
    config_path.write_text("", encoding="utf-8")
    received: list[str] = []

    def fake_bridge_main(args: list[str]) -> int:
        received.extend(args)
        return 0

    monkeypatch.setattr(runner, "main", fake_bridge_main)
    code = launcher.main([
        "--config", str(config_path), "--bridge-input", "packet.json",
    ])

    assert code == 0
    assert received[received.index("--project-root") + 1] == str(root)


def test_desktop_launcher_forwards_app_project_root(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "portable-root"
    received: list[str] = []

    def fake_bridge_main(args: list[str]) -> int:
        received.extend(args)
        return 0

    monkeypatch.setattr(desktop_main, "bridge_main", fake_bridge_main)
    monkeypatch.setattr(desktop_main.AppConfig, "load",
                        lambda _path: SimpleNamespace(root=root))
    code = desktop_main.main(["--bridge-input", "packet.json"])

    assert code == 0
    assert received[received.index("--project-root") + 1] == str(root)
