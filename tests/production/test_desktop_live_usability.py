"""Offline checks for the usable live desktop route and trial exports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from erguoyuan_football.app.export.reports import export_html, export_json
from erguoyuan_football.app.session import ApplicationResult
from erguoyuan_football.research.live_data.openligadb_directory import (
    DirectoryCompetition,
    DirectoryTeam,
)
from production.runner import ProductionRunner


def _scope(scope_id: str, name: str) -> DirectoryCompetition:
    return DirectoryCompetition(scope_id, "bl1" if name == "Bundesliga" else "bl2",
                                2026, scope_id, name, "GER", "UEFA", "MEN",
                                "SENIOR", "CLUB", "FIRST_TEAM", "LEAGUE")


def test_live_directory_routes_only_one_exact_shared_league(
    production_config, monkeypatch,
) -> None:
    """Two live directories must agree on a pair; no closest-name selection."""
    scopes = {
        "germany_men_bundesliga_2026": _scope("germany_men_bundesliga_2026", "Bundesliga"),
        "germany_men_second_division_2026": _scope(
            "germany_men_second_division_2026", "2. Bundesliga"
        ),
    }
    observed = datetime(2026, 10, 1, tzinfo=UTC)

    class FakeDirectory:
        def __init__(self, _config: Path) -> None:
            self.scopes = scopes

        def fetch_teams(self, scope_id: str) -> SimpleNamespace:
            names = (("Borussia Dortmund", "SV Werder Bremen")
                     if scope_id == "germany_men_bundesliga_2026" else
                     ("Hertha BSC", "FC Schalke 04"))
            rows = tuple(DirectoryTeam(index, name, scopes[scope_id],
                                       f"EVIDENCE_{index}", observed, "https://example.org")
                         for index, name in enumerate(names, 1))
            return SimpleNamespace(status=SimpleNamespace(value="VERIFIED"), rows=rows)

        def close(self) -> None:
            pass

    monkeypatch.setattr("production.runner.OpenLigaDBDirectoryProvider", FakeDirectory)
    runner = ProductionRunner(production_config)
    try:
        assert runner._infer_german_competition(
            "Borussia Dortmund VS SV Werder Bremen"
        ) == "Bundesliga"
        assert runner._infer_german_competition("Borussia Dortmund VS Hertha BSC") is None
        assert runner._infer_german_competition("Borussia Dortmund VS Unknown FC") is None
    finally:
        runner.close()


def test_trial_diagnostic_can_be_exported_without_core_report(tmp_path: Path) -> None:
    """The real desktop output is exportable even while final CORE is unavailable."""
    result = ApplicationResult("PRODUCTION_TRIAL", (), None,
                               "BASE_MODELS_EXECUTED_WITH_WARNINGS\nPREDICTION_ID: audit-1\n",
                               "hash", {}, ())
    json_path = export_json(result, tmp_path / "trial.json")
    html_path = export_html(result, tmp_path / "trial.html")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["report_scope"] == "REAL_PRODUCTION_TRIAL"
    assert "audit-1" in payload["report_text"]
    assert "历史开发阶段说明" not in html_path.read_text(encoding="utf-8")
