"""CLI checks use synthetic fixture names and never assert invented probabilities."""

from pathlib import Path

from production.launcher import main


def test_launcher_runs_and_reports_warning_for_synthetic_match(
    production_config, capsys
) -> None:
    code = main([
        "--config", str(_config_path(production_config)),
        "--text", "SYNTHETIC_TEST_HOME vs SYNTHETIC_TEST_AWAY",
        "--competition", "SYNTHETIC_TEST_LEAGUE",
        "--jc-confirmed",
        "--simulation",
    ])
    output = capsys.readouterr().out
    assert code == 1
    assert "JC状态: USER_CONFIRMED" in output
    assert "RUN_STATUS: WARNING" in output
    assert "FINAL_V7_STATUS: NOT_READY" in output
    assert "RUN MODE: PROCESS_SIMULATION" in output
    assert "UNAVAILABLE" in output


def _config_path(config) -> Path:
    """Find the isolated test config from its project-local DB path."""
    root = config.record_database.parents[1]
    return root / "config" / "production.yaml"
