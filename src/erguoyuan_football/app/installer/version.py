"""Build provenance visible in the desktop application."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from erguoyuan_football.knowledge.entities.country_alias_registry import (
    COUNTRY_REGISTRY_VERSION,
)
from erguoyuan_football.knowledge.entities.universal_team_resolver import (
    UniversalTeamResolver,
)
from production.bridge import BRIDGE_VERSION

SMART_BRIDGE_VERSION = "SMART_BRIDGE_V1_2"
DEFAULT_DESKTOP_MODE = "SMART_BRIDGE"


def source_fingerprints(root: Path) -> dict[str, str]:
    """Hash the exact resolver and registry resources that affect team resolution."""
    paths = {
        "resolver": root / "src/erguoyuan_football/knowledge/entities/universal_team_resolver.py",
        "team_parser": root / "src/erguoyuan_football/knowledge/entities/team_entity_parser.py",
        "country_registry": root / "src/erguoyuan_football/knowledge/entities/country_alias_registry.py",
        "country_data": root / "src/erguoyuan_football/knowledge/entities/country_aliases.json",
        "input_parser": root / "src/erguoyuan_football/app/input/match_input.py",
        "competition_data": root / "src/erguoyuan_football/app/input/competition_prefixes.json",
        "identity_resolver": root / "src/erguoyuan_football/knowledge/match_identity.py",
        "production_runner": root / "src/production/runner.py",
        "jc_metadata_parser": root / "src/production/jc_metadata.py",
        "fixture_router": root / "src/production/fixture_discovery.py",
        "global_fixture_research": root / "src/production/global_fixture_research.py",
        "association_sources": root / "src/production/association_sources.py",
        "v7_diagnostic": root / "src/production/v7_renderer.py",
        "desktop_dispatch": root / "src/erguoyuan_football/app/ui/window.py",
        "desktop_entry": root / "src/erguoyuan_football/app/main.py",
        "source_classifier": root / "src/erguoyuan_football/match_source/match_source_classifier.py",
        "production_config": root / "config/production.yaml",
        "bridge_runner": root / "src/production/bridge/runner.py",
        "bridge_validator": root / "src/production/bridge/validator.py",
        "bridge_contracts": root / "src/production/bridge/contracts.py",
        "bridge_importer": root / "src/production/bridge/importer.py",
        "bridge_intake": root / "src/production/bridge/intake.py",
        "bridge_version": root / "src/production/bridge/__init__.py",
    }
    result: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            result[name] = "MISSING"
            continue
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def build_id_from_fingerprints(fingerprints: dict[str, str], *, stamp: str) -> str:
    """Create a build ID that changes when bundled resolution code or data changes."""
    digest = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    return f"CORE-{stamp}-{digest[:12]}"


def runtime_build_info() -> dict[str, str]:
    """Load packaged build provenance or describe the active source checkout."""
    if getattr(sys, "frozen", False):
        bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        info_path = bundle_root / "build_info.json"
        if info_path.is_file():
            data: Any = json.loads(info_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(key): str(value) for key, value in data.items()}
    root = Path(__file__).resolve().parents[4]
    fingerprints = source_fingerprints(root)
    resolver_hash = hashlib.sha256(
        "".join(fingerprints[key] for key in sorted(fingerprints)).encode()
    ).hexdigest()
    country_hash = fingerprints.get("country_data", "MISSING")
    return {
        "app_version": "CORE V1.0 Trial",
        "build_id": f"SOURCE-{resolver_hash[:12]}",
        "git_commit": "UNKNOWN",
        "build_time": "SOURCE CHECKOUT",
        "resolver_version": UniversalTeamResolver.version,
        "resolver_source_hash": fingerprints.get("resolver", "MISSING"),
        "country_registry_version": COUNTRY_REGISTRY_VERSION,
        "country_registry_hash": country_hash,
        "source_hash": resolver_hash,
        "smart_bridge_version": SMART_BRIDGE_VERSION,
        "bridge_version": BRIDGE_VERSION,
    }


def format_build_info(info: dict[str, str]) -> str:
    """Format read-only provenance for the desktop status area."""
    return (
        f"{info.get('app_version', 'CORE V1.0 Trial')}  |  "
        f"Build ID: {info.get('build_id', 'UNKNOWN')}  |  "
        f"Commit: {info.get('git_commit', 'UNKNOWN')}  |  "
        f"Build Time: {info.get('build_time', 'UNKNOWN')}\n"
        f"Resolver: {info.get('resolver_version', UniversalTeamResolver.version)}  |  "
        f"Country Registry: {info.get('country_registry_version', COUNTRY_REGISTRY_VERSION)}  |  "
        f"Data SHA256: {info.get('country_registry_hash', 'UNKNOWN')[:12]}  |  "
        f"Source SHA256: {info.get('source_hash', 'UNKNOWN')[:12]}"
        f"  |  Smart Bridge: {info.get('smart_bridge_version', 'UNKNOWN')}"
        f"  |  Bridge: {info.get('bridge_version', 'UNKNOWN')}"
    )


def version_check() -> dict[str, str]:
    """Do not contact a remote server or auto-update in Phase 12."""
    info = runtime_build_info()
    return {"status": "NOT_IMPLEMENTED_OFFLINE", **info}
