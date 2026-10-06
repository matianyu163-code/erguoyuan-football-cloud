"""Opt-in real team discovery plus honest out-of-scope identity cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from erguoyuan_football.knowledge.entities.openligadb_discovery import (
    OpenLigaDBTeamDiscovery,
)
from erguoyuan_football.knowledge.entities.universal_team_resolver import (
    UniversalTeamResolver,
)
from erguoyuan_football.knowledge.entities.verified_entity_store import (
    VerifiedEntityStore,
)
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBConfig,
    OpenLigaDBProvider,
)
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.live_network
def test_real_club_discovery_and_uncovered_youth_women_reserve(
        request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """Real PL directory verifies one team; other segments remain unverified."""
    if "live_network" not in request.config.option.markexpr:
        pytest.skip("explicit -m live_network required")
    provider = OpenLigaDBProvider(OpenLigaDBConfig.from_yaml(
        ROOT / "config/phase13_5_provider.yaml"))
    evidence = EvidenceStore(":memory:", provider.sources)
    store = VerifiedEntityStore(tmp_path / "entities.sqlite")
    try:
        discovery = OpenLigaDBTeamDiscovery(provider, evidence)
        resolver = UniversalTeamResolver(store=store, discovery=(discovery,))
        live = resolver.resolve("Aston Villa", allow_discovery=True)
        assert live.status == "VERIFIED" and live.identity is not None
        assert live.identity.provider_ids == {"OPENLIGADB": "438"}
        assert live.identity.verification_evidence_ids
        assert evidence.get(live.identity.verification_evidence_ids[0]) is not None
        assert provider.client.audit.records()
        assert resolver.resolve("Aston Villa").source == "DYNAMIC_STORE"
        for name in ("Germany U16", "Greece U16",
                     "Shakhtar Donetsk Women", "Barcelona B"):
            unresolved = resolver.resolve(name, allow_discovery=True)
            assert unresolved.status == "PROVIDER_COVERAGE_MISSING"
            assert unresolved.identity is None
        # Team verification is not fixture verification or prediction execution.
        assert live.identity.identity_status == "PROVIDER_VERIFIED"
    finally:
        store.close()
        evidence.close()
        provider.close()
