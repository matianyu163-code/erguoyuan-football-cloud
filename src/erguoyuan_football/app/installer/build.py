"""Reproducible Windows onedir build; external data remains outside the executable."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from erguoyuan_football.app.installer.version import (
    SMART_BRIDGE_VERSION,
    build_id_from_fingerprints,
    source_fingerprints,
)
from production.bridge import BRIDGE_VERSION


def build(root: Path, *, output_name: str = "COREFootballEngine") -> Path:
    """Build the desktop executable and copy only its small config file."""
    root = root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_name):
        raise ValueError("INVALID_BUILD_OUTPUT_NAME")
    workspace = root / ".workspace"
    if not workspace.is_relative_to(root):
        raise ValueError("BUILD_WORKSPACE_OUTSIDE_PROJECT")
    dist = workspace / "dist"
    work = workspace / "build" / output_name
    dist.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    target = dist / output_name
    stale_work = work
    for stale in (target, stale_work):
        resolved = stale.resolve()
        if not resolved.is_relative_to(workspace.resolve()) or resolved == workspace.resolve():
            raise ValueError("BUILD_CLEANUP_PATH_OUTSIDE_WORKSPACE")
        if stale.is_symlink():
            raise ValueError("BUILD_CLEANUP_REFUSES_SYMLINK")
        if stale.exists():
            shutil.rmtree(stale)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    fingerprints = source_fingerprints(root)
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "UNKNOWN"
    registry_hash = fingerprints["country_data"]
    aggregate_hash = hashlib.sha256(
        "".join(fingerprints[key] for key in sorted(fingerprints)).encode()
    ).hexdigest()
    build_info_path = workspace / "build_info.json"
    build_info_path.write_text(json.dumps({
        "app_version": "CORE V1.0 Trial",
        "build_id": build_id_from_fingerprints(fingerprints, stamp=stamp),
        "git_commit": git_commit or "UNKNOWN",
        "build_time": datetime.now(UTC).isoformat(timespec="seconds"),
        "resolver_version": "UNIVERSAL_ENTITY_RESOLVER_V1",
        "resolver_source_hash": fingerprints["resolver"],
        "resolver_bundle_hash": aggregate_hash,
        "source_hash": aggregate_hash,
        "smart_bridge_version": SMART_BRIDGE_VERSION,
        "bridge_version": BRIDGE_VERSION,
        "country_registry_version": "ICU_CLDR_ISO_REGIONS_V1",
        "country_registry_hash": registry_hash,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm",
        "--clean", "--onedir", "--windowed", "--name", output_name,
        "--distpath", str(dist), "--workpath", str(work),
        "--specpath", str(work), "--paths", str(root / "src"),
        "--collect-all", "pytz",
        "--add-data", (str(root / "src/erguoyuan_football/knowledge/entities/country_aliases.json")
                        + os.pathsep + "erguoyuan_football/knowledge/entities"),
        "--add-data", (str(root / "src/erguoyuan_football/app/input/competition_prefixes.json")
                        + os.pathsep + "erguoyuan_football/app/input"),
        "--add-data", str(build_info_path) + os.pathsep + ".",
        str(root / "src/erguoyuan_football/app/launcher.py")]
    environment = os.environ.copy()
    environment["PYINSTALLER_CONFIG_DIR"] = str(workspace / "pyinstaller_cache")
    environment["TEMP"] = environment["TMP"] = str(workspace / "temp")
    environment["MPLCONFIGDIR"] = str(workspace / "cache")
    (workspace / "temp").mkdir(exist_ok=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
    (target / "config").mkdir(exist_ok=True)
    with (root / "config/application.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["project_root"] = str(root)
    (target / "config/application.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    shutil.copy2(root / "config/production.yaml", target / "config/production.yaml")
    return target / f"{output_name}.exe"


if __name__ == "__main__":
    selected_name = sys.argv[1] if len(sys.argv) > 1 else "COREFootballEngine"
    print(build(Path(__file__).resolve().parents[4], output_name=selected_name))
