"""Synthetic offline contract checks; these are not claims of real provider coverage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from erguoyuan_football.research.live_data.model_requirements import (
    ModelDataRequirement,
    ModelSamplePolicy,
    RequirementLevel,
    load_model_requirements,
)
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBConfig,
    OpenLigaDBProvider,
    SeasonBatch,
)
from erguoyuan_football.research.live_data.readiness import (
    DataAvailability,
    LiveDataReadinessGate,
    ReadinessConfig,
)
from erguoyuan_football.research.live_data.service import LiveResearchService
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.time_utils import utc_iso

ROOT = Path(__file__).resolve().parents[2]
AT = datetime(2026, 10, 1, tzinfo=UTC)


def _row(match_id: int, home: int, away: int, kickoff: datetime,
         *, finished: bool = False) -> dict[str, object]:
    names = {2617: "FC Arsenal", 370: "FC Liverpool"}
    score = ((2, 1), (0, 0), (0, 2))[match_id % 3]
    return {"matchID": match_id, "leagueShortcut": "pl", "leagueSeason": 2026,
            "team1": {"teamId": home, "teamName": names[home]},
            "team2": {"teamId": away, "teamName": names[away]},
            "matchDateTimeUTC": utc_iso(kickoff), "matchIsFinished": finished,
            "matchResults": ([{"resultTypeKind": "After90Minutes",
                               "pointsTeam1": score[0], "pointsTeam2": score[1]}]
                             if finished else [])}


def _setup(history: int = 15) -> tuple[OpenLigaDBProvider, EvidenceStore, SeasonBatch,
                                       LiveDataReadinessGate]:
    config = OpenLigaDBConfig.from_yaml(ROOT / "config/phase13_5_provider.yaml")
    provider = OpenLigaDBProvider(config)
    store = EvidenceStore(":memory:", provider.sources)
    rows = [_row(index + 1, 2617 if index % 2 else 370,
                 370 if index % 2 else 2617, AT - timedelta(days=index + 1),
                 finished=True) for index in range(history)]
    rows.append(_row(99, 2617, 370, AT + timedelta(days=1)))
    batch = SeasonBatch(tuple(rows), AT, provider.endpoint, 200, 1.0)
    policy = ReadinessConfig(1, ("ELO_V1",),
                             ("FIXTURE", "HISTORICAL_RESULTS"), True,
                             {"BASE_STRENGTH": ("ELO_V1",)}, ("BASE_STRENGTH",), 1)
    requirements = (
        ModelDataRequirement("ELO_V1", {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED},
                             # This focused two-team synthetic fixture explicitly
                             # tests its configured readiness behavior; production
                             # model policies retain the three-team coverage floor.
                             ModelSamplePolicy(12, 1, 0, False, True,
                                              min_training_team_matches=2)),
        ModelDataRequirement("HISTORICAL_MARKET_BAYESIAN_POISSON_V1", {
            "ODDS": RequirementLevel.REQUIRED}, ModelSamplePolicy(12, 1, 0, False)),
    )
    return provider, store, batch, LiveDataReadinessGate(policy, requirements)


def test_fixture_history_readiness_and_lineage() -> None:
    """Actual algorithm runs on synthetic test inputs with explicit provenance."""
    provider, store, batch, gate = _setup()
    try:
        result = LiveResearchService(provider, store, gate).build(
            batch, "ENG_ARS", "ENG_LIV", cutoff=AT + timedelta(seconds=1))
        assert result.fixture_check.status == "VERIFIED"
        assert result.package is not None and result.readiness is not None
        assert result.historical_count == 15 and result.recent_form_count == 2
        assert result.readiness.overall_status == "DEGRADED"
        assert "ELO_V1" in result.readiness.usable_models
        assert "HISTORICAL_MARKET_BAYESIAN_POISSON_V1" in result.readiness.blocked_models
        assert result.readiness.item("ODDS").status == DataAvailability.MISSING
        assert result.readiness.item("RECENT_FORM").data_origin == "LOCAL_DERIVED"
        assert result.readiness.item("RECENT_FORM").derived_from
        assert result.readiness.item("HISTORICAL_RESULTS").evidence_ids
        assert all(store.get(evidence_id) is not None for evidence_id in
                   result.readiness.item("HISTORICAL_RESULTS").evidence_ids)
        assert result.package.prediction_executed is False
    finally:
        store.close()
        provider.close()


def test_ambiguous_fixture_fails_closed() -> None:
    provider, store, batch, gate = _setup()
    try:
        second = _row(100, 2617, 370, AT + timedelta(days=2))
        ambiguous = SeasonBatch((*batch.matches, second), AT, batch.source_url, 200, 1.0)
        result = LiveResearchService(provider, store, gate).build(
            ambiguous, "ENG_ARS", "ENG_LIV", cutoff=AT + timedelta(seconds=1))
        assert result.fixture_check.status == "AMBIGUOUS"
        assert result.package is None and result.prediction_executed is False
    finally:
        store.close()
        provider.close()


def test_low_sample_not_ready_and_future_evidence_rejected() -> None:
    provider, store, batch, gate = _setup(history=0)
    try:
        service = LiveResearchService(provider, store, gate)
        result = service.build(batch, "ENG_ARS", "ENG_LIV",
                               cutoff=AT + timedelta(seconds=1))
        assert result.readiness is not None
        assert result.readiness.overall_status == "NOT_READY"
        future = service.build(batch, "ENG_ARS", "ENG_LIV",
                               cutoff=AT - timedelta(seconds=1))
        assert future.fixture_check.status == "FUTURE_EVIDENCE_REJECTED"
        assert future.package is None
    finally:
        store.close()
        provider.close()


def test_stale_odds_and_conflict_block_market() -> None:
    provider, store, batch, gate = _setup()
    try:
        cutoff = AT + timedelta(hours=2)
        result = LiveResearchService(provider, store, gate).build(
            batch, "ENG_ARS", "ENG_LIV", cutoff=cutoff)
        assert result.package is not None
        old = EvidenceRecord("ODDS_1", "ODDS", {"home": 2.1},
                             provider.config.provider_id, None, utc_iso(AT),
                             "MEDIUM", batch.source_url, utc_iso(AT),
                             provider_id=provider.config.provider_id, source_tier=2)
        result.package.available_data["odds"] = [old]
        report = gate.evaluate(result.package, fixture=result.fixture_check.fixture,
                               cutoff=cutoff, team_provider_ids=(2617, 370))
        assert report.item("ODDS").status == DataAvailability.STALE
        assert "HISTORICAL_MARKET_BAYESIAN_POISSON_V1" in report.blocked_models
        second = EvidenceRecord("ODDS_2", "ODDS", {"home": 3.1},
                                provider.config.provider_id, None, utc_iso(AT),
                                "MEDIUM", batch.source_url, utc_iso(AT),
                                provider_id=provider.config.provider_id, source_tier=2)
        result.package.available_data["odds"] = [old, second]
        report = gate.evaluate(result.package, fixture=result.fixture_check.fixture,
                               cutoff=cutoff, team_provider_ids=(2617, 370))
        assert report.item("ODDS").status == DataAvailability.CONFLICT
    finally:
        store.close()
        provider.close()


def test_requirements_match_actual_registry() -> None:
    requirements = load_model_requirements(ROOT / "config/model_registry.yaml")
    assert len(requirements) == 11
    assert len({item.model_id for item in requirements}) == 11


def test_secret_redaction_for_authenticated_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """An auth header is sent privately; response and audit omit its value."""
    secret = "sentinel-private-phase13-5-token"
    monkeypatch.setenv("CORE_PHASE13_5_TEST_TOKEN", secret)

    class Transport:
        def get(self, url: str, *, params: dict[str, object],
                headers: dict[str, str]) -> object:
            assert headers["Authorization"] == f"Bearer {secret}"

            class Response:
                status_code = 200

                def __init__(self) -> None:
                    self.headers: dict[str, str] = {}

                def json(self) -> dict[str, str]:
                    return {"status": "ok"}

            return Response()

    source = SourceDefinition(
        source_id="PHASE13_5_TEST", display_name="Synthetic credential test",
        category="WEB_RESEARCH", base_url="https://source.example.test",
        endpoints={"health": "https://source.example.test/health"},
        schema_version="TEST_V1", requires_auth=True,
        auth_env_var="CORE_PHASE13_5_TEST_TOKEN",
    )
    client = CoreNetworkClient(ExternalSourceRegistry((source,)), http_client=Transport())
    response = client.fetch_json("PHASE13_5_TEST", "health", params={}, bypass_cache=True)
    assert secret not in repr(response)
    assert secret not in repr(client.audit.records())
