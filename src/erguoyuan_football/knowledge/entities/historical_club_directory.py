"""Read verified historical club IDs as local aliases, never as live fixtures."""

from __future__ import annotations

from pathlib import Path

import duckdb

from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity


def load_historical_clubs(database: Path) -> tuple[TeamIdentity, ...]:
    """Expose exact historical IDs with source provenance; no youth inference."""
    if not database.is_file():
        return ()
    with duckdb.connect(str(database), read_only=True) as connection:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        if "real_canonical_teams" not in tables:
            return ()
        rows = connection.execute(
            "SELECT team_id,team_name,competition_id,source FROM real_canonical_teams"
        ).fetchall()
    countries = {
        "EPL": ("England", "UEFA"), "LALIGA": ("Spain", "UEFA"),
        "BUNDESLIGA": ("Germany", "UEFA"), "SERIE_A": ("Italy", "UEFA"),
        "LIGUE_1": ("France", "UEFA"),
    }
    result = []
    for team_id, name, competition, source in rows:
        if not all(isinstance(value, str) and value for value in
                   (team_id, name, competition, source)):
            continue
        country, federation = countries.get(competition, ("UNKNOWN", "UNKNOWN"))
        result.append(TeamIdentity(team_id, name, country, federation, [name],
                                   entity_type="CLUB", identity_status=f"HISTORICAL_{source}"))
    return tuple(result)
