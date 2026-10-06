"""SYNTHETIC_TEST linked evidence only; no real HTTP or prediction path."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from erguoyuan_football.app.input.match_input import MatchInputParserV2, MatchRequest
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from erguoyuan_football.research.orchestrator import MatchResearchOrchestrator
from erguoyuan_football.research.research_status import ResearchStatus
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.search.query_builder import QueryBuilder
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.sources.source_schema import SourceRecord

_AS_OF = datetime(2025, 1, 2, 12, tzinfo=UTC)
_MATCH_ID = "SYNTHETIC_TEST_ARS_LIV_FIXTURE"
_FETCHED = "2025-01-01T11:00:00Z"
_PUBLISHED = "2025-01-01T10:00:00Z"
_TYPE_SOURCE = {"ODDS": "TEST_STRUCTURED", "TEAM_STATS": "TEST_STRUCTURED",
                "NEWS": "TEST_MEDIA"}


def _sources() -> SourceRegistry:
    definitions = tuple(SourceDefinition(
        source_id=source_id, display_name=source_id, category="research",
        base_url=f"https://{host}.example",
        endpoints={"data": f"https://{host}.example/data"},
        schema_version="SYNTHETIC_TEST_V1",
    ) for source_id, host in (("TEST_OFFICIAL", "official"),
                              ("TEST_STRUCTURED", "structured"),
                              ("TEST_MEDIA", "media")))
    registry = SourceRegistry(ExternalSourceRegistry(definitions))
    for source_id, host, source_type, tier in (
        ("TEST_OFFICIAL", "official", "OFFICIAL", 3),
        ("TEST_STRUCTURED", "structured", "STRUCTURED", 2),
        ("TEST_MEDIA", "media", "MEDIA", 1),
    ):
        registry.add(SourceRecord(source_id, source_id, source_type, tier,
                                  f"https://{host}.example", True))
    return registry


def _seed(
    path: Path,
    *,
    omit: str | None = None,
    delayed_link: str | None = None,
) -> tuple[SourceRegistry, EvidenceStore]:
    registry = _sources()
    link_clock = {"now": datetime(2025, 1, 1, 12, tzinfo=UTC)}
    store = EvidenceStore(path, registry, clock=lambda: link_clock["now"])
    tasks = QueryBuilder.build("Arsenal FC", "Liverpool FC", "UEFA Champions League",
                               fixture_key=_MATCH_ID)
    for task in tasks:
        if task.task_type == omit:
            continue
        source_id = _TYPE_SOURCE.get(task.task_type, "TEST_OFFICIAL")
        host = source_id.removeprefix("TEST_").lower()
        evidence = EvidenceRecord(
            f"SYNTHETIC_TEST_{task.task_type}", task.task_type,
            {"claim": "SYNTHETIC_TEST_ONLY"}, source_id,
            _PUBLISHED, _FETCHED, "HIGH", f"https://{host}.example/data",
            _PUBLISHED,
        )
        store.save(evidence)
        link_clock["now"] = (_AS_OF + timedelta(days=1)
                             if task.task_type == delayed_link
                             else datetime(2025, 1, 1, 12, tzinfo=UTC))
        store.link_to_task(evidence.evidence_id, task.task_id)
    return registry, store


def _request() -> MatchRequest:
    request = MatchInputParserV2().parse("Arsenal VS Liverpool")
    return replace(request, competition="UEFA Champions League", match_id=_MATCH_ID,
                   validation_status="RESOLVED")


def test_complete_match_research_package(tmp_path: Path) -> None:
    """All six sourced categories yield FOUND and full coverage, not probabilities."""
    sources, store = _seed(tmp_path / "research.sqlite")
    package = MatchResearchOrchestrator(sources=sources, evidence_store=store).build(
        _request(), as_of_time=_AS_OF
    )
    assert package.status == ResearchStatus.FOUND
    assert package.match_id == _MATCH_ID
    assert package.quality_score == 1.0
    assert package.missing_data == []
    assert set(package.available_data) == {"fixture", "injury", "lineup", "odds", "xg", "news"}
    assert {source.source_id for source in package.sources} == {
        "TEST_OFFICIAL", "TEST_STRUCTURED", "TEST_MEDIA"
    }
    store.close()


def test_missing_xg_is_partial(tmp_path: Path) -> None:
    """Missing xG remains missing; the evaluator cannot invent it."""
    sources, store = _seed(tmp_path / "research.sqlite", omit="TEAM_STATS")
    package = MatchResearchOrchestrator(sources=sources, evidence_store=store).build(
        _request(), as_of_time=_AS_OF
    )
    assert package.status == ResearchStatus.PARTIAL
    assert package.missing_data == ["xg"]
    assert package.quality_score == 0.8
    store.close()


def test_unknown_team_failed_without_guess() -> None:
    """Unknown identity does not produce research tasks or an invented team."""
    request = MatchInputParserV2().parse("Unknown FC VS Liverpool")
    package = MatchResearchOrchestrator().build(request, as_of_time=_AS_OF)
    assert package.status == ResearchStatus.FAILED
    assert package.error_code == "TEAM_NOT_FOUND"
    assert package.research_tasks == [] and package.available_data == {}


def test_late_task_link_not_visible_historically(tmp_path: Path) -> None:
    """An evidence link created later cannot enter an earlier research view."""
    sources, store = _seed(tmp_path / "research.sqlite", delayed_link="ODDS")
    orchestrator = MatchResearchOrchestrator(sources=sources, evidence_store=store)
    earlier = orchestrator.build(_request(), as_of_time=_AS_OF)
    later = orchestrator.build(_request(), as_of_time=_AS_OF + timedelta(days=2))
    assert earlier.status == ResearchStatus.PARTIAL and "odds" in earlier.missing_data
    assert later.status == ResearchStatus.FOUND
    store.close()


def test_unlinked_other_match_evidence_excluded(tmp_path: Path) -> None:
    """Unlinked records cannot bleed into this match's research package."""
    sources, store = _seed(tmp_path / "research.sqlite", omit="TEAM_STATS")
    store.save(EvidenceRecord("SYNTHETIC_TEST_OTHER_XG", "TEAM_STATS", {"xg": 1.0},
                              "TEST_STRUCTURED", _PUBLISHED, _FETCHED, "HIGH",
                              "https://structured.example/data", _PUBLISHED))
    package = MatchResearchOrchestrator(sources=sources, evidence_store=store).build(
        _request(), as_of_time=_AS_OF
    )
    assert "xg" in package.missing_data and "xg" not in package.available_data
    store.close()


def test_wrong_source_tier_fails_closed(tmp_path: Path) -> None:
    """SYNTHETIC_TEST: a media claim cannot satisfy a structured odds task."""
    sources, store = _seed(tmp_path / "research.sqlite", omit="ODDS")
    odds_task = next(task for task in QueryBuilder.build(
        "Arsenal FC", "Liverpool FC", "UEFA Champions League",
        fixture_key=_MATCH_ID,
    ) if task.task_type == "ODDS")
    evidence = EvidenceRecord(
        "SYNTHETIC_TEST_MEDIA_ODDS", "ODDS", {"claim": "not bookmaker data"},
        "TEST_MEDIA", _PUBLISHED, _FETCHED, "LOW",
        "https://media.example/data", _PUBLISHED,
    )
    store.save(evidence)
    store.link_to_task(evidence.evidence_id, odds_task.task_id)
    package = MatchResearchOrchestrator(sources=sources, evidence_store=store).build(
        _request(), as_of_time=_AS_OF
    )
    assert package.status == ResearchStatus.FAILED
    assert package.error_code == "EVIDENCE_VALIDATION_FAILED"
    store.close()


def test_unverified_text_does_not_inherit_fixture_evidence(tmp_path: Path) -> None:
    """Fixture-specific evidence cannot be attached to an undated text request."""
    sources, store = _seed(tmp_path / "research.sqlite")
    unverified = replace(_request(), match_id=None, validation_status="VALID")
    package = MatchResearchOrchestrator(sources=sources, evidence_store=store).build(
        unverified, as_of_time=_AS_OF
    )
    assert package.match_id is None
    assert package.status == ResearchStatus.MISSING
    assert package.available_data == {}
    store.close()


def test_resolved_request_without_match_id_fails() -> None:
    """A RESOLVED flag alone cannot establish a fixture identity."""
    request = replace(_request(), match_id=None)
    package = MatchResearchOrchestrator().build(request, as_of_time=_AS_OF)
    assert package.status == ResearchStatus.FAILED
    assert package.error_code == "RESOLVED_MATCH_ID_REQUIRED"
