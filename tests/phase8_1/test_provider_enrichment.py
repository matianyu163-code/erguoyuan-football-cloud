"""Optional authorized providers fail closed and retain network failure categories."""

import pytest

import erguoyuan_football.data.external_providers as provider_module
from erguoyuan_football.data.external_providers import FootballDataOrgProvider
from erguoyuan_football.data.kickoff_provider_import import (
    _event_record,
    import_verified_kickoffs,
)
from erguoyuan_football.network.errors import RetryExhausted


def test_provider_missing_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOOTBALL_DATA_ORG_TOKEN", raising=False)
    result = FootballDataOrgProvider().fetch_matches("PL", 2024)
    assert result.status == "UNAVAILABLE" and result.reason == "AUTH_NOT_CONFIGURED"


def test_provider_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def fetch_json(self, *_args, **_kwargs):
            raise RetryExhausted("RATE_LIMITED: retry budget exhausted")

        def close(self):
            pass

    monkeypatch.setenv("FOOTBALL_DATA_ORG_TOKEN", "test-only-secret")
    monkeypatch.setattr(provider_module, "CoreNetworkClient", FakeClient)
    result = FootballDataOrgProvider().fetch_matches("PL", 2024)
    assert result.status == "UNAVAILABLE" and result.reason == "RATE_LIMITED"
    assert "test-only-secret" not in str(result)


def test_provider_event_normalization_uses_actual_fields() -> None:
    result = _event_record({
        "id": 7,
        "competition": {"code": "PL"},
        "season": {"startDate": "2025-08-01"},
        "utcDate": "2025-10-01T19:00:00Z",
        "homeTeam": {"name": "Manchester United FC"},
        "awayTeam": {"name": "Arsenal FC"},
    }, "PL")
    assert result == {"id": "7", "competition": "PL", "season": 2025,
                      "home": "Manchester United FC", "away": "Arsenal FC",
                      "utcDate": "2025-10-01T19:00:00Z"}


def test_provider_missing_auth_does_not_migrate_database(tmp_path, monkeypatch) -> None:
    path = tmp_path / "unavailable.duckdb"
    import duckdb
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE preserved(value INTEGER)")
        connection.execute("INSERT INTO preserved VALUES (1)")
    monkeypatch.delenv("FOOTBALL_DATA_ORG_TOKEN", raising=False)
    result = import_verified_kickoffs(path, from_season=2024, through_season=2024,
                                      competitions=("PL",))
    assert result.reason_counts == {"AUTH_NOT_CONFIGURED": 1}
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute("SELECT value FROM preserved").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM information_schema.tables "
                                  "WHERE table_name='phase8_1_migrations'").fetchone()[0] == 0
