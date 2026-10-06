"""SYNTHETIC_TEST source, evidence and tier contracts; no live endpoint is used."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.policies.source_policy import tier_for
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.sources.source_schema import SourceRecord


def _official_registry() -> SourceRegistry:
    network = ExternalSourceRegistry((SourceDefinition(
        source_id="TEST_CLUB", display_name="Test Club", category="news",
        base_url="https://club.example", endpoints={"news": "https://club.example/news"},
        schema_version="SYNTHETIC_TEST_V1"
    ),))
    sources = SourceRegistry(network)
    sources.add(SourceRecord("TEST_CLUB", "Test Club", "OFFICIAL", 3,
                             "https://club.example", True))
    return sources


def test_register_official_source_and_tiers() -> None:
    """Only explicitly allowlisted HTTPS sources can be registered."""
    registry = _official_registry()
    assert registry.get("TEST_CLUB").tier == 3
    assert (tier_for("OFFICIAL"), tier_for("STRUCTURED"), tier_for("MEDIA")) == (3, 2, 1)
    with pytest.raises(ValueError, match="SOURCE_TYPE_NOT_ALLOWED"):
        tier_for("BLOG")
    with pytest.raises(ValueError, match="SOURCE_TIER_INVALID"):
        SourceRecord("BAD", "Blog", "BLOG", 0, "https://blog.example", True)


def test_source_url_gate_and_disabled_source() -> None:
    """An arbitrary website cannot masquerade as a registered official source."""
    registry = _official_registry()
    with pytest.raises(ValueError, match="RESULT_URL_NOT_ALLOWLISTED"):
        registry.validate_result_url("TEST_CLUB", "https://club.example/unregistered")
    with pytest.raises(KeyError, match="SOURCE_NOT_CONFIGURED"):
        registry.add(SourceRecord("UNKNOWN", "Unknown", "OFFICIAL", 3,
                                  "https://unknown.example", True))


def test_evidence_saved_and_pit_filtered(tmp_path: Path) -> None:
    """Injury evidence keeps source, publication, retrieval and as-of times."""
    store = EvidenceStore(tmp_path / "research.sqlite", _official_registry())
    evidence = EvidenceRecord(
        "SYNTHETIC_TEST_INJURY_1", "INJURY", "Saka doubtful", "TEST_CLUB",
        "2026-10-01T10:00:00Z", "2026-10-01T12:00:00Z", "HIGH",
        "https://club.example/news", "2026-10-01T10:00:00Z"
    )
    store.save(evidence)
    assert store.get(evidence.evidence_id) == evidence
    assert store.available_at(datetime(2026, 10, 1, 11, tzinfo=UTC)) == ()
    assert store.available_at(datetime(2026, 10, 1, 13, tzinfo=UTC)) == (evidence,)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        store.save(evidence)
    store.close()


def test_evidence_rejects_unregistered_endpoint(tmp_path: Path) -> None:
    """SYNTHETIC_TEST: evidence cannot bypass the network source registry."""
    store = EvidenceStore(tmp_path / "research.sqlite", _official_registry())
    evidence = EvidenceRecord(
        "SYNTHETIC_TEST_BAD", "INJURY", "claim", "TEST_CLUB",
        "2026-10-01T10:00:00Z", "2026-10-01T12:00:00Z", "LOW",
        "https://untrusted.example/news", "2026-10-01T10:00:00Z"
    )
    with pytest.raises(ValueError, match="RESULT_URL_NOT_ALLOWLISTED"):
        store.save(evidence)
    store.close()
