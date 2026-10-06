"""SQLite JC cache TTL and read-only history behavior."""

from datetime import UTC, date, datetime, timedelta

from erguoyuan_football.jc_verification.jc_cache import JCMatchCache
from erguoyuan_football.jc_verification.jc_schema import JCMatch


def _row() -> JCMatch:
    return JCMatch("jc-1", "Arsenal", "Liverpool", "Premier League",
        datetime(2026, 10, 2, 18, tzinfo=UTC), ("1X2",), "synthetic-test",
        "synthetic-evidence")


def test_same_day_cache_is_fresh_then_expires(tmp_path) -> None:
    cache = JCMatchCache(tmp_path / "jc.sqlite", ttl=timedelta(minutes=15))
    checked = datetime(2026, 10, 2, 10, tzinfo=UTC)
    cache.put(date(2026, 10, 2), (_row(),), checked_at=checked)
    assert cache.get(date(2026, 10, 2), now=checked + timedelta(minutes=14))
    assert cache.get(date(2026, 10, 2), now=checked + timedelta(minutes=16)) is None


def test_historical_cache_is_read_only_and_retained(tmp_path) -> None:
    cache = JCMatchCache(tmp_path / "jc.sqlite", ttl=timedelta(minutes=1))
    checked = datetime(2026, 10, 1, 10, tzinfo=UTC)
    row = JCMatch("jc-old", "Arsenal", "Liverpool", "Premier League",
        datetime(2026, 10, 1, 18, tzinfo=UTC), (), "synthetic-test", "old-evidence")
    cache.put(date(2026, 10, 1), (row,), checked_at=checked)
    assert cache.get(date(2026, 10, 1), now=datetime(2026, 10, 2, tzinfo=UTC))


def test_cache_rejects_match_on_wrong_utc_date(tmp_path) -> None:
    cache = JCMatchCache(tmp_path / "jc.sqlite")
    try:
        cache.put(date(2026, 10, 3), (_row(),), checked_at=datetime(2026, 10, 2, tzinfo=UTC))
    except ValueError as error:
        assert str(error) == "JC_CACHE_MATCH_DATE_MISMATCH"
    else:
        raise AssertionError("date mismatch accepted")


def test_empty_complete_daily_response_is_cached(tmp_path) -> None:
    cache = JCMatchCache(tmp_path / "jc.sqlite")
    checked = datetime(2026, 10, 2, 10, tzinfo=UTC)
    cache.put(date(2026, 10, 2), (), checked_at=checked,
              source="synthetic-test", complete_coverage=True)
    assert cache.get(date(2026, 10, 2), source="synthetic-test", now=checked) == ()
