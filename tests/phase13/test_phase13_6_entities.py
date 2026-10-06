"""SYNTHETIC_TEST identity cases; no invented production team is registered."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from erguoyuan_football.knowledge.entities.candidate_scoring import score_candidate
from erguoyuan_football.knowledge.entities.competition_hint_parser import (
    CompetitionHintParser,
    split_competition_hint,
)
from erguoyuan_football.knowledge.entities.discovery import TeamDiscoveryCandidate
from erguoyuan_football.knowledge.entities.entity_fingerprint import entity_fingerprint
from erguoyuan_football.knowledge.entities.match_resolution import resolve_match_text
from erguoyuan_football.knowledge.entities.team_entity_parser import (
    UniversalTeamNameParser,
)
from erguoyuan_football.knowledge.entities.universal_team_resolver import (
    UniversalTeamResolver,
)
from erguoyuan_football.knowledge.entities.verified_entity_store import (
    VerifiedEntityStore,
)

AT = datetime(2026, 10, 1, tzinfo=UTC)


@pytest.mark.parametrize(("raw", "base", "gender", "age", "squad", "kind"), [
    ("德国U16", "德国", "UNKNOWN", "U16", "FIRST_TEAM", "NATIONAL"),
    ("希腊U16", "希腊", "UNKNOWN", "U16", "FIRST_TEAM", "NATIONAL"),
    ("德国女足U17", "德国", "WOMEN", "U17", "FIRST_TEAM", "NATIONAL"),
    ("Japan U23", "Japan", "UNKNOWN", "U23", "FIRST_TEAM", "NATIONAL"),
    ("Barcelona B", "Barcelona", "UNKNOWN", "SENIOR", "B_TEAM", "CLUB"),
    ("Bayern Munich Women", "Bayern Munich", "WOMEN", "SENIOR", "FIRST_TEAM", "CLUB"),
    ("Shakhtar Donetsk Women", "Shakhtar Donetsk", "WOMEN", "SENIOR", "FIRST_TEAM", "CLUB"),
    ("Real Madrid Castilla", "Real Madrid", "UNKNOWN", "SENIOR", "B_TEAM", "CLUB"),
])
def test_universal_parser(raw: str, base: str, gender: str, age: str,
                          squad: str, kind: str) -> None:
    hint = UniversalTeamNameParser().parse(raw)
    assert (hint.base_name, hint.gender_hint, hint.age_group_hint,
            hint.squad_level_hint, hint.entity_type_hint) == (base, gender, age, squad, kind)


def test_competition_hint_and_unknown_small_league() -> None:
    text, hint = split_competition_hint("Team A vs Team B | 女欧U17")
    assert text == "Team A vs Team B" and hint is not None
    assert (hint.federation, hint.gender, hint.age_group) == ("UEFA", "WOMEN", "U17")
    unknown = CompetitionHintParser().parse("印加尔联")
    assert unknown.raw_hint == "印加尔联"
    assert unknown.federation is None and unknown.country is None


def test_gender_age_and_b_team_fingerprint_separation() -> None:
    common = {"country": "UKR", "entity_type": "CLUB",
              "name": "Shakhtar Donetsk", "provider_id": "SYNTHETIC_TEST",
              "provider_team_id": "7"}
    men = entity_fingerprint(**common, gender="MEN", age_group="SENIOR",
                             squad_level="FIRST_TEAM")
    women = entity_fingerprint(**common, gender="WOMEN", age_group="SENIOR",
                               squad_level="FIRST_TEAM")
    youth = entity_fingerprint(**common, gender="MEN", age_group="U17",
                               squad_level="YOUTH")
    reserve = entity_fingerprint(**common, gender="MEN", age_group="SENIOR",
                                 squad_level="B_TEAM")
    assert len({men, women, youth, reserve}) == 4


class _SyntheticDiscovery:
    provider_id = "SYNTHETIC_TEST"
    supports_team_discovery = True

    def __init__(self, candidates: tuple[TeamDiscoveryCandidate, ...]) -> None:
        self.candidates = candidates
        self.calls = 0

    def can_cover(self, hint: object, competition: object = None) -> bool:
        return True

    def discover(self, hint: object, competition: object = None
                 ) -> tuple[TeamDiscoveryCandidate, ...]:
        self.calls += 1
        return self.candidates


def _candidate(team_id: str, name: str, *, gender: str = "MEN",
               age: str = "U16") -> TeamDiscoveryCandidate:
    return TeamDiscoveryCandidate("SYNTHETIC_TEST", team_id, name, "GER", "UEFA",
                                  "NATIONAL", gender, age, "FIRST_TEAM", (),
                                  "https://synthetic.invalid/teams", 2, AT, "HIGH",
                                  f"SYNTHETIC_TEST_{team_id}")


def test_country_name_defaults_to_mens_team_before_provider_discovery(tmp_path: Path) -> None:
    provider = _SyntheticDiscovery((_candidate("1", "Germany U16"),
                                    _candidate("2", "Germany Women U16", gender="WOMEN")))
    store = VerifiedEntityStore(tmp_path / "entities.sqlite")
    try:
        resolver = UniversalTeamResolver(store=store, discovery=(provider,))
        result = resolver.resolve("Germany U16", allow_discovery=True,
                                  as_of=AT + timedelta(seconds=1))
        assert result.status == "VERIFIED"
        assert result.identity is not None and result.identity.gender == "MEN"
        assert len(result.candidates) == 1
        assert result.candidates[0].gender == "MEN"
        assert store.find("Germany U16", as_of=AT + timedelta(seconds=2))
    finally:
        store.close()


def test_verified_store_survives_offline_restart(tmp_path: Path) -> None:
    path = tmp_path / "entities.sqlite"
    provider = _SyntheticDiscovery((_candidate("2", "Germany Women U16", gender="WOMEN"),))
    store = VerifiedEntityStore(path)
    try:
        resolver = UniversalTeamResolver(store=store, discovery=(provider,))
        result = resolver.resolve("Germany Women U16", allow_discovery=True,
                                  as_of=AT + timedelta(seconds=1))
        assert result.status == "VERIFIED"
        assert result.identity is not None
        assert result.identity.team_id == "UEFA_GER_W_U16"
    finally:
        store.close()
    restarted = VerifiedEntityStore(path)
    try:
        offline = UniversalTeamResolver(store=restarted).resolve(
            "Germany Women U16", as_of=AT + timedelta(days=1))
        assert offline.status == "VERIFIED" and offline.source == "DYNAMIC_STORE"
        stale = UniversalTeamResolver(store=restarted).resolve(
            "Germany Women U16", as_of=AT + timedelta(days=31))
        assert stale.status == "STALE_IDENTITY"
        assert provider.calls == 1
    finally:
        restarted.close()


def test_candidate_hard_conflict_precedes_score() -> None:
    hint = UniversalTeamNameParser().parse("Germany Women U16")
    assert score_candidate(hint, _candidate("1", "Germany U16", gender="MEN")) is None
    assert score_candidate(hint, _candidate("2", "Germany Women U16",
                                            gender="WOMEN")) is not None


def test_static_backward_compatibility_and_no_auto_network() -> None:
    provider = _SyntheticDiscovery(())
    resolver = UniversalTeamResolver(discovery=(provider,))
    assert resolver.resolve("Arsenal").identity is not None
    assert resolver.resolve("Liverpool").identity is not None
    assert resolver.resolve("Cook Islands").identity is not None
    assert resolver.resolve("Tahiti").identity is not None
    national = resolver.resolve("Germany U16")
    assert national.status == "VERIFIED"
    assert national.identity is not None
    assert national.identity.entity_type == "NATIONAL"
    assert national.identity.age_group == "U16"
    assert provider.calls == 0


def test_competition_gender_conflict_fails_closed() -> None:
    provider = _SyntheticDiscovery((_candidate("1", "Germany U17", age="U17"),))
    hint = CompetitionHintParser().parse("女欧U17")
    result = UniversalTeamResolver(discovery=(provider,)).resolve(
        "Germany Men U17", competition=hint, allow_discovery=True)
    assert result.status == "TEAM_IDENTITY_CONFLICT:GENDER"
    assert provider.calls == 0


def test_provider_id_cannot_be_reused_for_other_gender(tmp_path: Path) -> None:
    store = VerifiedEntityStore(tmp_path / "entities.sqlite")
    try:
        first = _SyntheticDiscovery((_candidate("5", "Germany Women U16",
                                                gender="WOMEN"),))
        assert UniversalTeamResolver(store=store, discovery=(first,)).resolve(
            "Germany Women U16", allow_discovery=True,
            as_of=AT + timedelta(seconds=1)).status == "VERIFIED"
        changed = _SyntheticDiscovery((_candidate("5", "Germany Men U16",
                                                  gender="MEN"),))
        verdict = UniversalTeamResolver(store=store, discovery=(changed,)).resolve(
            "Germany Men U16", allow_discovery=True,
            as_of=AT + timedelta(seconds=2))
        assert verdict.status == "TEAM_PROVIDER_ID_CONFLICT"
        assert verdict.identity is None
    finally:
        store.close()


def test_provider_verified_rename_preserves_canonical_id(tmp_path: Path) -> None:
    store = VerifiedEntityStore(tmp_path / "entities.sqlite")
    old_candidate = replace(_candidate("77", "Club Alpha", age="SENIOR"),
                            country="ENG", entity_type="CLUB")
    new_candidate = replace(old_candidate, official_name="Club Alpha FC",
                            evidence_id="SYNTHETIC_TEST_RENAME")
    try:
        old = UniversalTeamResolver(store=store, discovery=(
            _SyntheticDiscovery((old_candidate,)),)).resolve(
                "Club Alpha", allow_discovery=True, as_of=AT + timedelta(seconds=1))
        renamed = UniversalTeamResolver(store=store, discovery=(
            _SyntheticDiscovery((new_candidate,)),)).resolve(
                "Club Alpha FC", allow_discovery=True, as_of=AT + timedelta(seconds=2))
        assert old.identity is not None and renamed.identity is not None
        assert renamed.identity.team_id == old.identity.team_id
        assert "Club Alpha" in (renamed.identity.former_names or [])
    finally:
        store.close()


def test_discovery_provider_failure_is_explicit_and_second_provider_can_continue() -> None:
    class Failing(_SyntheticDiscovery):
        def discover(self, hint: object, competition: object = None
                     ) -> tuple[TeamDiscoveryCandidate, ...]:
            raise OSError("SYNTHETIC_TEST_UNREACHABLE")

    failing = Failing(())
    unavailable = UniversalTeamResolver(discovery=(failing,)).resolve(
        "Germany Women U16", allow_discovery=True, as_of=AT)
    assert unavailable.status == "VERIFIED"
    assert unavailable.source == "COUNTRY_REGISTRY"
    assert unavailable.identity is not None
    assert unavailable.identity.gender == "WOMEN"
    assert unavailable.reasons == ("OSError",)
    succeeding = _SyntheticDiscovery((_candidate("2", "Germany Women U16",
                                                gender="WOMEN"),))
    recovered = UniversalTeamResolver(discovery=(failing, succeeding)).resolve(
        "Germany Women U16", allow_discovery=True,
        as_of=AT + timedelta(seconds=1))
    assert recovered.status == "VERIFIED" and recovered.identity is not None


def test_match_parser_v2_competition_hint_identity_flow_stops_before_fixture() -> None:
    result = resolve_match_text("德国U16VS希腊U16 | 国际友谊赛",
                                UniversalTeamResolver())
    assert result.request.validation_status == "VALID"
    assert result.competition_hint is not None
    assert result.competition_hint.competition_type == "FRIENDLY"
    assert result.home is not None and result.home.parsed.country_hint == "DEU"
    assert result.away is not None and result.away.parsed.country_hint == "GRC"
    assert result.status == "IDENTITIES_RESOLVED_NO_FIXTURE"
    assert result.fixture_verified is False and result.prediction_executed is False
