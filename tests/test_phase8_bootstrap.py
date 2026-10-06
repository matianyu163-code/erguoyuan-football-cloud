"""Offline contracts for real-data bootstrap; test fixtures never become REAL data."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from erguoyuan_football.data.external_providers import (
    FootballDataOrgProvider,
    StatsBombOpenDataProvider,
)
from erguoyuan_football.data.real_bootstrap import RealDataWarehouse
from erguoyuan_football.data.real_quality import assert_oos_lineage, assess_real_data
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.gates.phase8_readiness import assess_phase9_gates
from erguoyuan_football.models.runner import BaseModelPredictionBundle
from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from erguoyuan_football.output_contract.builder import CanonicalPredictionResultBuilder
from erguoyuan_football.output_contract.schemas import (
    AvailabilityStatus,
    BetAdvice,
    CanonicalPredictionResult,
    ProbabilityStage,
)

COMMIT = "e6744429ee395bc86f247348c6184bb08d4eb361"
REPO = "https://github.com/openfootball/football.json"


def _file(path: Path, matches: list[dict]) -> Path:
    path.write_text(json.dumps({"name": "Synthetic Test League", "matches": matches}), encoding="utf-8")
    return path


def _record(date: str = "2024-01-01", score: list[int] | None = None) -> dict:
    return {"date": date, "time": "14:30", "team1": "Test Home", "team2": "Test Away",
            "score": {"ft": score or [2, 1]}}


def test_openfootball_idempotent_and_unknown_timezone(tmp_path: Path) -> None:
    source = _file(tmp_path / "season.json", [_record()])
    with RealDataWarehouse(tmp_path / "warehouse.duckdb") as warehouse:
        first = warehouse.promote_openfootball(source, repository=REPO, source_commit=COMMIT,
                                               competition_id="TEST", season_id="2023-24")
        second = warehouse.promote_openfootball(source, repository=REPO, source_commit=COMMIT,
                                                competition_id="TEST", season_id="2023-24")
        row = warehouse.connection.execute("SELECT kickoff_time_utc,timestamp_precision,retrieved_at "
            "FROM real_canonical_matches").fetchone()
        assert first.promoted == 1 and second.repeated == 1
        assert row[0] is None and row[1] == "LOCAL_TIME_UNKNOWN_ZONE"
        assert row[2].tzinfo is not None
        assert warehouse.coverage()[0]["results"] == 1


def test_invalid_and_score_conflict_quarantined(tmp_path: Path) -> None:
    source = _file(tmp_path / "season.json", [_record(), _record(score=[1, 1]),
                                                    _record(date="2024-99-99")])
    with RealDataWarehouse(tmp_path / "warehouse.duckdb") as warehouse:
        summary = warehouse.promote_openfootball(source, repository=REPO, source_commit=COMMIT,
                                                 competition_id="TEST", season_id="2023-24")
        reasons = [row[0] for row in warehouse.connection.execute(
            "SELECT reason FROM real_data_quarantine ORDER BY reason").fetchall()]
        assert summary.promoted == 1 and summary.quarantined == 2
        assert reasons == ["INVALID_DATE", "SCORE_CONFLICT"]


def test_dry_run_rolls_back(tmp_path: Path) -> None:
    source = _file(tmp_path / "season.json", [_record()])
    with RealDataWarehouse(tmp_path / "warehouse.duckdb") as warehouse:
        result = warehouse.promote_openfootball(source, repository=REPO, source_commit=COMMIT,
            competition_id="TEST", season_id="2023-24", dry_run=True)
        assert result.status == "DRY_RUN"
        assert warehouse.connection.execute("SELECT count(*) FROM real_canonical_matches").fetchone()[0] == 0


def test_openfootball_direct_score_array_is_real_result(tmp_path: Path) -> None:
    item = _record()
    item["score"] = [0, 0]
    source = _file(tmp_path / "season.json", [item])
    with RealDataWarehouse(tmp_path / "warehouse.duckdb") as warehouse:
        summary = warehouse.promote_openfootball(source, repository=REPO, source_commit=COMMIT,
            competition_id="TEST", season_id="2023-24")
        assert summary.promoted == 1 and summary.quarantined == 0
        assert warehouse.coverage()[0]["results"] == 1


def test_unknown_license_rejected(tmp_path: Path) -> None:
    source = _file(tmp_path / "season.json", [_record()])
    with RealDataWarehouse(tmp_path / "warehouse.duckdb") as warehouse, pytest.raises(ValueError, match="LICENSE"):
        warehouse.promote_openfootball(source, repository=REPO, source_commit=COMMIT,
            competition_id="TEST", season_id="2023-24", license_class="UNVERIFIED")


def _base_result(**overrides: object) -> dict:
    values = {"prediction_id": "p", "prediction_snapshot_id": "s", "match_id": "m",
        "competition_id": "epl", "kickoff_time": datetime(2024, 1, 2, tzinfo=UTC),
        "prediction_time": datetime(2024, 1, 1, tzinfo=UTC), "prediction_horizon": "T-24H",
        "data_origin": "REAL",
        "status": AvailabilityStatus.AVAILABLE, "data_quality_status": "PASS", "pit_status": "PASS",
        "probability_stage": ProbabilityStage.BASE_MODEL, "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
        "models_used": ("ELO_V1",), "created_at": datetime(2024, 1, 1, tzinfo=UTC)}
    values.update(overrides)
    return values


def test_canonical_intermediate_probability_only() -> None:
    result = CanonicalPredictionResult(**_base_result())
    assert result.probability_stage == ProbabilityStage.BASE_MODEL
    assert result.calibration_status == AvailabilityStatus.NOT_IMPLEMENTED
    with pytest.raises(ValueError, match="sum to 1"):
        CanonicalPredictionResult(**_base_result(p_away=0.4))
    with pytest.raises(ValueError, match="unavailable output"):
        CanonicalPredictionResult(**_base_result(status="UNAVAILABLE"))
    with pytest.raises(ValueError, match="precede kickoff"):
        CanonicalPredictionResult(**_base_result(prediction_time=datetime(2024, 1, 3, tzinfo=UTC)))
    with pytest.raises(ValueError, match="REAL data origin"):
        CanonicalPredictionResult(**_base_result(data_origin="SYNTHETIC"))


def test_no_bet_zero_stake_contract() -> None:
    assert BetAdvice(decision="NO_BET", stake=0, reason="threshold").stake == 0
    with pytest.raises(ValueError, match="zero stake"):
        BetAdvice(decision="NO_BET", stake=10, reason="threshold")


def test_historical_retrieval_is_not_oos_evidence(tmp_path: Path) -> None:
    source = _file(tmp_path / "season.json", [_record()])
    path = tmp_path / "warehouse.duckdb"
    with RealDataWarehouse(path) as warehouse:
        warehouse.promote_openfootball(source, repository=REPO, source_commit=COMMIT,
            competition_id="TEST", season_id="2023-24")
    report, matrix = assess_real_data(path)
    assert report.strict_historical_oos_status == "BLOCKED"
    assert "NO_VERIFIED_UTC_KICKOFFS" in report.reasons
    assert matrix[0].market == 0 and matrix[0].results == 1
    gates = assess_phase9_gates(path)
    assert gates.no_market_meta == gates.full_market_meta == gates.output_contract == "BLOCKED"
    old = datetime(2024, 1, 1, tzinfo=UTC)
    new = datetime(2026, 9, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="POINT_IN_TIME"):
        assert_oos_lineage(training_available_at=new, fixture_available_at=new,
            prediction_time=old, kickoff_time=datetime(2024, 1, 2, tzinfo=UTC))


def test_external_providers_fail_closed_without_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOOTBALL_DATA_ORG_TOKEN", raising=False)
    assert FootballDataOrgProvider().fetch_matches("PL", 2024).reason == "AUTH_NOT_CONFIGURED"
    assert StatsBombOpenDataProvider(tmp_path).available().reason == "OFFICIAL_OPEN_DATA_NOT_PRESENT"


def test_candidate_and_unavailable_advice_are_distinct() -> None:
    from erguoyuan_football.output_contract.schemas import RankedCandidate

    candidate = RankedCandidate(candidate_id="c", match_id="m", prediction_snapshot_id="s",
                                selection="HOME", rank=1, status="AVAILABLE", probability=0.5)
    advice = BetAdvice(candidate_id=candidate.candidate_id, decision="NO_BET", stake=0, reason="threshold")
    assert candidate.status == AvailabilityStatus.AVAILABLE and advice.decision == "NO_BET"
    assert BetAdvice(decision="UNAVAILABLE", stake=0, reason="missing data").decision == "UNAVAILABLE"


def test_builder_never_invents_a_probability() -> None:
    fixture = Fixture(match_id="m", competition_id="epl", home_team_id="h", away_team_id="a",
        kickoff_time=datetime(2024, 1, 2, tzinfo=UTC), source="SYNTHETIC_TEST",
        retrieved_at=datetime(2024, 1, 1, tzinfo=UTC), as_of_time=datetime(2024, 1, 1, tzinfo=UTC),
        data_version="test")
    result = CanonicalPredictionResultBuilder().build(fixture, BaseModelPredictionBundle(()),
        prediction_snapshot_id="s", prediction_time=datetime(2024, 1, 1, tzinfo=UTC))
    assert result.status == AvailabilityStatus.UNAVAILABLE
    assert (result.p_home, result.p_draw, result.p_away) == (None, None, None)


def test_football_data_org_auth_header_is_not_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import timedelta

    from erguoyuan_football.contracts.common import now

    class Response:
        status_code = 200
        headers: dict[str, str] | None = None

        def json(self) -> dict:
            return {"matches": [{"id": 1}]}

    class FakeClient:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def get(self, _url: str, *, params: dict, headers: dict) -> Response:
            self.headers = headers
            return Response()

    monkeypatch.setenv("FOOTBALL_DATA_ORG_TOKEN", "secret-for-test")
    fake = FakeClient()
    source = SourceDefinition(source_id="FD", display_name="FD", category="HISTORICAL",
        base_url="https://api.football-data.org", endpoints={"matches": "https://api.football-data.org/v4/matches"},
        requires_auth=True, auth_env_var="FOOTBALL_DATA_ORG_TOKEN", auth_header_name="X-Auth-Token",
        schema_version="v4")
    client = CoreNetworkClient(ExternalSourceRegistry((source,)), http_client=fake)
    response = client.fetch_json("FD", "matches", params={}, prediction_time=now() + timedelta(minutes=5))
    assert response.status_code == 200 and fake.headers["X-Auth-Token"] == "secret-for-test"
    assert "secret-for-test" not in str(client.audit)
