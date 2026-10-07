"""Install and smoke-test the R3 release from a clean portable project copy."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _copy_required_files(destination: Path) -> None:
    """Copy only source, release artifacts, tests, and runtime configuration."""
    for name in ("src", "cloud_release", "tests/blind_test_r3"):
        shutil.copytree(ROOT / name, destination / name)
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, destination / name)
    (destination / "config").mkdir()
    shutil.copy2(ROOT / "config/r3_market_intelligence_v1.yaml",
                 destination / "config/r3_market_intelligence_v1.yaml")


def main() -> int:
    """Verify local install, all release models, Goldens, and a synthetic sample."""
    with tempfile.TemporaryDirectory(prefix="yycore-r3-cloud-portable-") as temp_dir:
        temporary = Path(temp_dir)
        project = temporary / "project"
        installed = temporary / "installed"
        project.mkdir()
        installed.mkdir()
        _copy_required_files(project)
        shutil.copytree(project / "src/erguoyuan_football",
                        installed / "erguoyuan_football")
        dependency_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"], check=False,
            capture_output=True, text=True)
        if dependency_check.returncode:
            raise RuntimeError(f"CLOUD_DEPENDENCY_CHECK_FAILED:{dependency_check.stdout[-1200:]}")
        print("CLEAN_COPY_SOURCE_INSTALL_PASS", flush=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(installed)
        smoke_code = """
import json, runpy, sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from erguoyuan_football.blind_test_r3 import release_registry
from erguoyuan_football.blind_test_r3.user_daily_runner import run_daily
root = Path(sys.argv[1])
assert Path(release_registry.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
tests = runpy.run_path(str(root / 'tests/blind_test_r3/test_release_registry.py'))
tests['test_verified_release_loads_all_four_models']()
tests['test_brazil_frozen_golden_001_and_002']()
input_path = root / 'synthetic_brazil_sample.json'
input_path.write_text(json.dumps({
  'slate_date': datetime.now(UTC).date().isoformat(), 'source': 'USER_AUTHORITATIVE',
  'fixtures': [{'jc_match_number': 'SYNTHETIC_TEST_BRAZIL_001',
    'competition': 'Campeonato Brasileiro Série A', 'home_team': 'Botafogo',
    'away_team': 'Vasco da Gama',
    'kickoff': (datetime.now(UTC) + timedelta(days=7)).isoformat(),
    'neutral_venue': False, 'spf': {'home': 2.0, 'draw': 3.0, 'away': 4.0},
    'rqspf': {'home': 3.0, 'draw': 3.2, 'away': 2.1, 'handicap': -1},
    'screenshot_path': None, 'market_observed_at': datetime.now(UTC).isoformat(),
    'metadata_sources': {'fixture': 'SYNTHETIC_TEST'},
    'external_research_warnings': []}]}, ensure_ascii=False), encoding='utf-8')
result = run_daily(input_path, root=root, state_root=root / 'cloud_state')
assert result['official_r3_predictions'] == 1, result
assert result['model_execution_failures'] == 0, result
assert result['results'][0]['model_coverage'] == 'FULL', result
print(json.dumps({'official_r3_predictions': result['official_r3_predictions'],
                  'model_execution_failures': result['model_execution_failures'],
                  'run_version': result['run_version']}, sort_keys=True))
"""
        smoke = subprocess.run(
            [sys.executable, "-c", smoke_code, str(project), str(installed)],
            cwd=project, env=environment, check=False, capture_output=True, text=True,
        )
        if smoke.returncode:
            raise RuntimeError(f"CLEAN_COPY_PREDICTION_FAILED:{smoke.stdout[-1800:]}{smoke.stderr[-1400:]}")
        print(json.dumps({"install": "PASS", "dependency_check": "PASS", "golden": "PASS",
                          "synthetic_brazil_sample": json.loads(smoke.stdout),
                          "project_source": "CLEAN_TEMP_COPY"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
