"""Official-source scope routing and site-level fixture parsing stay fail-closed."""

from datetime import UTC, datetime

from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from production.official_adapters import _extract_event
from production.official_source_registry import OfficialFootballSourceRegistry


def _team(team_id: str, official_name: str, country: str) -> TeamIdentity:
    return TeamIdentity(
        team_id,
        official_name,
        country,
        "UEFA",
        [official_name],
        "NATIONAL",
        "MEN",
        "U18",
        "YOUTH",
    )


def test_registry_routes_national_sides_to_associations_and_continental_sources() -> (
    None
):
    registry = OfficialFootballSourceRegistry()
    routed = registry.for_national_teams("Ireland", "Netherlands")
    assert {item.source_id for item in routed} == {"FAI", "KNVB"}
    assert registry.get("UEFA") is not None
    assert {item.source_id for item in registry.for_competition("UEFA Nations League")} == {"UEFA"}
    assert registry.validate_domain("onsoranje.nl")
    assert not registry.validate_domain("search.example")


def test_jsonld_fixture_requires_exact_resolver_aliases_and_offset() -> None:
    source = OfficialFootballSourceRegistry().get("KNVB")
    assert source is not None
    home = _team("NATIONAL_IRL_M_U18", "Republic of Ireland U18", "Ireland")
    away = _team("NATIONAL_NLD_M_U18", "Netherlands U18", "Netherlands")
    event = {
        "@type": "SportsEvent",
        "name": "Ireland U18 v Netherlands U18",
        "homeTeam": {"name": "Republic of Ireland U18"},
        "awayTeam": {"name": "Netherlands U18"},
        "startDate": "2026-10-03T16:00:00+02:00",
        "superEvent": {"name": "International Friendly"},
        "location": {"name": "Marbella"},
        "identifier": "official-fixture-1",
    }
    retrieved = datetime(2026, 10, 2, tzinfo=UTC)
    record = _extract_event(
        event, source, "https://onsoranje.nl/fixtures", retrieved, retrieved, home, away
    )
    assert record is not None
    assert record.home_entity_id == home.team_id
    assert record.away_entity_id == away.team_id
    assert record.kickoff_utc == datetime(2026, 10, 3, 14, tzinfo=UTC)
    assert record.competition_name == "International Friendly"
    assert record.venue == "Marbella"


def test_jsonld_fixture_does_not_fuzzy_map_teams_or_guess_timezone() -> None:
    source = OfficialFootballSourceRegistry().get("KNVB")
    assert source is not None
    home = _team("NATIONAL_IRL_M_U18", "Republic of Ireland U18", "Ireland")
    away = _team("NATIONAL_NLD_M_U18", "Netherlands U18", "Netherlands")
    event = {
        "@type": "SportsEvent",
        "name": "Friendly",
        "homeTeam": {"name": "Ireland U18"},
        "awayTeam": {"name": "Dutch U18"},
        "startDate": "2026-10-03T16:00:00",
        "superEvent": {"name": "International Friendly"},
    }
    assert (
        _extract_event(
            event,
            source,
            "https://onsoranje.nl/fixtures",
            datetime(2026, 10, 2, tzinfo=UTC),
            datetime(2026, 10, 2, tzinfo=UTC),
            home,
            away,
        )
        is None
    )
