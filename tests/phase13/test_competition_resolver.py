"""Phase 13.1 competition identity tests; custom fixtures are SYNTHETIC_TEST."""

import pytest

from erguoyuan_football.knowledge.competitions.competition_database import (
    CompetitionDatabase,
)
from erguoyuan_football.knowledge.competitions.competition_resolver import (
    CompetitionResolver,
)
from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)


def test_phase13_competition_seed() -> None:
    """The three specified competition identities are present and distinct."""
    database = CompetitionDatabase()
    assert {item.competition_id for item in database.all()} == {
        "FIFA_WORLD_CUP", "UEFA_CL", "OFC_NATIONS"
    }
    assert database.get("UEFA_CL") == CompetitionIdentity(
        "UEFA_CL", "UEFA Champions League", "UEFA", "EUROPE", "CLUB"
    )


def test_phase13_world_cup_year_resolution() -> None:
    """A year suffix is permitted only after an exact registered competition name."""
    resolver = CompetitionResolver()
    competition = resolver.resolve("FIFA World Cup 2026")
    assert competition is not None and competition.competition_id == "FIFA_WORLD_CUP"
    assert resolver.resolve("FIFA World Cup 2026 Qualifying") is None


def test_phase13_competition_aliases_and_unknown() -> None:
    """Only registered exact names resolve; fuzzy queries do not."""
    resolver = CompetitionResolver()
    for name, expected in (("欧冠", "UEFA_CL"),
                           ("UEFA-Champions League", "UEFA_CL"),
                           ("OFC Nations Cup", "OFC_NATIONS")):
        competition = resolver.resolve(name)
        assert competition is not None and competition.competition_id == expected
    assert resolver.resolve("World Cup Qualifiers") is None
    assert resolver.resolve("") is None


def test_phase13_competition_collision_is_not_selected() -> None:
    """SYNTHETIC_TEST: a shared name must remain unresolved."""
    competitions = (
        CompetitionIdentity("TEST_A", "Alpha Cup", "TEST", "TEST", "CLUB"),
        CompetitionIdentity("TEST_B", "Beta Cup", "TEST", "TEST", "CLUB"),
    )
    database = CompetitionDatabase(
        competitions, {"TEST_A": ("Shared Cup",), "TEST_B": ("Shared-Cup",)}
    )
    assert CompetitionResolver(database).resolve("Shared Cup") is None


def test_phase13_invalid_competition_catalog_rejected() -> None:
    """SYNTHETIC_TEST: duplicate IDs and dangling aliases cannot enter the catalog."""
    competition = CompetitionIdentity("TEST", "Test Cup", "TEST", "TEST", "CLUB")
    with pytest.raises(ValueError, match="DUPLICATE_COMPETITION_ID"):
        CompetitionDatabase((competition, competition))
    with pytest.raises(ValueError, match="UNKNOWN_COMPETITION_ALIAS_TARGET"):
        CompetitionDatabase((competition,), {"MISSING": ("Other",)})
