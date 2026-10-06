"""Official source contracts and fail-closed HTML extraction."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from production.association_sources import AssociationSourceRegistry
from production.fixture_discovery import FixtureDiscoveryRequest
from production.global_fixture_research import GlobalFixtureResearch
from production.network_diagnostics import network_policy_blocked
from production.official_live_adapters import (
    OfficialLiveFixtureAdapter,
    _dfb_kickoff,
    _knvb_kickoff,
    _sports_event,
)
from production.official_source_registry import (
    OfficialFootballSource,
    OfficialFootballSourceRegistry,
)


def _team(name: str, country: str, age: str) -> TeamIdentity:
    return TeamIdentity(
        f"{country}_{age}", f"{name} {age}", country, "UEFA", [f"{name} {age}"],
        "NATIONAL", "MEN", age, "YOUTH",
    )


def test_fixture_capability_requires_site_level_entrypoint() -> None:
    with pytest.raises(ValueError, match="OFFICIAL_FIXTURE_ENTRYPOINT_NOT_CONFIGURED"):
        OfficialFootballSource(
            "MISSING", "Missing", "example.org", "ASSOCIATION", 20, ("en",),
            frozenset({"FIXTURES"}),
        )
    registry = OfficialFootballSourceRegistry()
    assert {row.source_id for row in registry.fixture_sources()} == {"UEFA", "KNVB", "DFB"}
    assert "FIXTURES" not in registry.get("FAI").capabilities  # type: ignore[union-attr]
    assert registry.get("DFB").schedule_entrypoint == (  # type: ignore[union-attr]
        "https://datencenter.dfb.de/datencenter/naechste-spiele"
    )
    assert registry.get("DFB").timezone_policy == "Europe/Berlin"  # type: ignore[union-attr]
    assert registry.get("UEFA").adapter == "UEFA_FIXTURES_ARTICLE_V1"  # type: ignore[union-attr]


def test_dfb_schedule_extracts_exact_pair_with_declared_source_timezone() -> None:
    source = OfficialFootballSourceRegistry().get("DFB")
    assert source is not None
    adapter = OfficialLiveFixtureAdapter(
        source, _team("Germany", "Germany", "U16"),
        _team("Austria", "Austria", "U16"),
    )
    body = """<div class="c-MatchTable-body"><div class="c-MatchTable-row">
    <div class="c-MatchTable-info--home"><div class="c-MatchTable-description">
    <p>U 16-Vier-Nationen-Turnier (m)</p><p>Sonntag, 04.10.2026 11:00 Uhr</p>
    </div></div><div class="c-MatchTable-team--home" data-team-kind="national">
    Deutschland U 16 (m)</div><div class="c-MatchTable-score">
    <a href="https://datencenter.dfb.de/datencenter/fixture-123">- : -</a></div>
    <div class="c-MatchTable-team--away" data-team-kind="national">Österreich U 16</div>
    </div></div>"""
    rows = adapter._dfb(SimpleNamespace(
        body=body, retrieved_at=datetime.now(UTC),
        final_url="https://datencenter.dfb.de/", content_hash="test-content-hash",
    ))
    assert len(rows) == 1
    assert rows[0].competition_name == "U 16-Vier-Nationen-Turnier (m)"
    assert rows[0].kickoff_local == "2026-10-04T11:00:00+02:00"
    assert rows[0].kickoff_utc is None
    assert rows[0].kickoff_timezone == "Europe/Berlin"


def test_official_schedule_kickoff_parsers_require_real_date_and_time() -> None:
    knvb = _knvb_kickoff("06 okt 2026", "13:00", "Europe/Amsterdam")
    assert knvb is not None and knvb.isoformat() == "2026-10-06T13:00:00+02:00"
    assert _knvb_kickoff("11 nov 2026", "n.n.b.", "Europe/Amsterdam") is None
    dfb = _dfb_kickoff("Sonntag, 04.10.2026 11:00 Uhr", "Europe/Berlin")
    assert dfb is not None and dfb.isoformat() == "2026-10-04T11:00:00+02:00"
    assert _dfb_kickoff("Termin offen", "Europe/Berlin") is None


def test_uefa_livesport_event_can_be_found_in_official_wrapper() -> None:
    event = {
        "@type": "SportsEvent", "name": "Kosovo vs Austria",
        "homeTeam": {"name": "Kosovo"}, "awayTeam": {"name": "Austria"},
    }
    assert _sports_event({"@type": "LiveBlogPosting", "mainEntity": event}) == event


def test_windows_permission_denial_is_distinct_from_fixture_absence() -> None:
    error = OSError("[WinError 10013] permission denied")
    wrapped = RuntimeError("transport failure")
    wrapped.__cause__ = error
    assert network_policy_blocked(wrapped)
    assert not network_policy_blocked(OSError("HTTP_ERROR:404"))


def test_network_policy_block_is_not_reported_as_fixture_not_found() -> None:
    class BlockedOfficial:
        provider_id = "KNVB"
        source_tier = 3
        allowed_domains = frozenset({"onsoranje.nl"})

        def search(self, query, *, source, as_of_time):
            raise OSError("NETWORK_POLICY_BLOCKED")

    home = _team("Ireland", "Ireland", "U18")
    away = _team("Netherlands", "Netherlands", "U18")
    research = GlobalFixtureResearch(
        associations=AssociationSourceRegistry(), providers=(BlockedOfficial(),)
    )
    result = research.discover(
        FixtureDiscoveryRequest(home, away, datetime.now(UTC)),
        structured_result="NO_COVERAGE",
    )
    assert result.status == "OFFICIAL_NETWORK_BLOCKED"
    assert result.reason == "NETWORK_POLICY_BLOCKED"


def test_registered_country_source_without_entrypoint_is_distinct() -> None:
    research = GlobalFixtureResearch(
        associations=AssociationSourceRegistry(), fixture_sources_enabled=True
    )
    result = research.discover(
        FixtureDiscoveryRequest(
            _team("France", "France", "U17"),
            _team("United States", "United States", "U17"),
            datetime.now(UTC),
        ),
        structured_result="NO_COVERAGE",
    )
    assert result.status == "OFFICIAL_FIXTURE_ENTRYPOINT_NOT_CONFIGURED"

