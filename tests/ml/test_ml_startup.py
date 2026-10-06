"""The existing daily program remains importable after Phase 7 registration."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


def test_program_startup_help(tmp_path) -> None:
    environment = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
                   "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    result = subprocess.run([sys.executable, "-m", "erguoyuan_football", "--help"],
                            capture_output=True, text=True, encoding="utf-8", env=environment,
                            timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert "--db" in result.stdout and "--text" in result.stdout
