"""Shared offline test fixtures for the project scaffold."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def at():
    return datetime(2026, 9, 29, 12, tzinfo=UTC)


@pytest.fixture
def store(tmp_path, project_root, at):
    from datetime import timedelta

    from erguoyuan_football.data.schemas import (
        Competition,
        CompetitionAlias,
        Fixture,
        Team,
        TeamAlias,
    )
    from erguoyuan_football.data.store import Store
    catalog = json.loads((project_root / "tests/fixtures/catalog.json").read_text(encoding="utf-8"))
    with Store(tmp_path / "synthetic.duckdb") as repository:
        for team in catalog["teams"]:
            repository.add_team(Team(team_id=team["team_id"], team_name=team["team_name"]))
            for alias in team["aliases"]:
                repository.add_team_alias(TeamAlias(alias=alias, team_id=team["team_id"], language="und",
                    source=catalog["source"], confidence=1, created_at=at - timedelta(days=30)))
        for competition in catalog["competitions"]:
            repository.add_competition(Competition(competition_id=competition["competition_id"],
                                                   competition_name=competition["competition_name"], tier=1))
            for alias in competition["aliases"]:
                repository.add_competition_alias(CompetitionAlias(alias=alias, competition_id=competition["competition_id"],
                    language="und", source=catalog["source"], confidence=1, created_at=at - timedelta(days=30)))
        for fixture in catalog["fixtures"]:
            repository.add_fixture(Fixture(**fixture, kickoff_time=at + timedelta(hours=8), source=catalog["source"],
                retrieved_at=at - timedelta(days=2), as_of_time=at - timedelta(days=2), data_version="test_v1"))
        yield repository


@pytest.fixture
def quote_factory(at):
    from erguoyuan_football.data.schemas import OddsSnapshot
    def factory(**changes):
        return OddsSnapshot(**{"match_id": "test_match_1", "source": "SYNTHETIC_TEST", "retrieved_at": at,
            "as_of_time": at, "data_version": "test_v1", "bookmaker": "TEST_BOOK", "market_type": "1X2",
            "phase": "CURRENT", "selection": "HOME", "odds": 2.0, **changes})
    return factory
