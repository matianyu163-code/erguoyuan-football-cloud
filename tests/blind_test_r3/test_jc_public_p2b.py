"""Offline fixtures for the public-page and user-confirmed P2B paths."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from erguoyuan_football.blind_test_r3 import jc_public_lock
from erguoyuan_football.blind_test_r3.jc_public_import import load_user_confirmed
from erguoyuan_football.blind_test_r3.jc_public_intelligence import (
    fixed_bonus_market,
    match_jc_fixture,
    movement,
    select_market_sources,
)
from erguoyuan_football.blind_test_r3.jc_public_parser import (
    parse_schedule,
    parse_spf_rqspf,
)
from erguoyuan_football.blind_test_r3.jc_public_store import JCPublicStore
from erguoyuan_football.blind_test_r3.market_intelligence import load_r3_market_config
from erguoyuan_football.blind_test_r3.runner import PROJECT_ROOT
from erguoyuan_football.blind_test_r3.store import R3Store, sha256


def _html(*, bonus: bool = True, women: bool = False) -> str:
    home = "塞浦路斯女足" if women else "塞浦路斯"
    prices = ("<i data-market='JC_SPF' data-selection='HOME' data-bonus='1.80'/>"
              "<i data-market='JC_SPF' data-selection='DRAW' data-bonus='3.30'/>"
              "<i data-market='JC_SPF' data-selection='AWAY' data-bonus='4.50'/>"
              "<i data-market='JC_RQSPF' data-selection='HOME' data-bonus='3.10'/>"
              "<i data-market='JC_RQSPF' data-selection='DRAW' data-bonus='3.45'/>"
              "<i data-market='JC_RQSPF' data-selection='AWAY' data-bonus='1.95'/>") if bonus else ""
    return ("<div data-bonus-updated-at='2026-10-05T10:00:00+00:00'></div>"
            "<table id='mainTbl' data-jc-schedule='true'><tr data-jc-match-number='周一001' "
            f"data-competition='欧国联' data-home-team='{home}' data-away-team='拉脱维亚' "
            "data-kickoff-utc='2026-10-05T16:00:00+00:00' data-spf-sale-status='ON_SALE' "
            f"data-rqspf-sale-status='ON_SALE' data-handicap='-1'>{prices}</tr></table>")


def _publication() -> dict[str, object]:
    return {"home_team": "塞浦路斯", "away_team": "拉脱维亚",
            "kickoff_time": "2026-10-05T16:00:00+00:00",
            "input_snapshot": {"fixture_evidence": {"jc_code": "周一001"}}}


def test_schedule_spf_rqspf_and_fixture_identity() -> None:
    schedule = parse_schedule(_html())
    bonus = parse_spf_rqspf(_html())
    assert schedule.status == bonus.status == "AVAILABLE"
    assert bonus.matches[0].spf_bonus == {"HOME": 1.8, "DRAW": 3.3, "AWAY": 4.5}
    assert bonus.matches[0].rqspf_bonus == {"HOME": 3.1, "DRAW": 3.45, "AWAY": 1.95}
    assert bonus.matches[0].handicap == -1
    assert match_jc_fixture(bonus.matches, publication=_publication()) is not None
    assert match_jc_fixture(parse_spf_rqspf(_html(women=True)).matches,
                            publication=_publication()) is None
    assert match_jc_fixture(parse_spf_rqspf(_html().replace("塞浦路斯", "塞浦路斯U21")).matches,
                            publication=_publication()) is None


def test_fail_closed_missing_bonus_and_no_sale() -> None:
    assert parse_spf_rqspf(_html(bonus=False)).status == "PARSE_FAILED"
    html = _html(bonus=False).replace("ON_SALE", "NOT_ON_SALE")
    parsed = parse_spf_rqspf(html)
    assert parsed.status == "AVAILABLE" and parsed.matches[0].spf_bonus is None
    assert parse_spf_rqspf("<table id='mainTbl'></table>").status == (
        "PUBLIC_PAGE_AUTOMATION_UNAVAILABLE")


def test_jc_no_vig_fusion_and_handicap_movement() -> None:
    spf = fixed_bonus_market({"HOME": 1.8, "DRAW": 3.3, "AWAY": 4.5})
    assert spf is not None
    assert sum(spf["no_vig"].values()) == pytest.approx(1)
    rq = fixed_bonus_market({"HOME": 3.1, "DRAW": 3.45, "AWAY": 1.95}, handicap=-1)
    assert rq is not None and rq["handicap"] == -1
    config = load_r3_market_config(PROJECT_ROOT / "config/r3_market_intelligence_v1.yaml")
    model = {"HOME": .6, "DRAW": .25, "AWAY": .15}
    jc = spf["no_vig"]
    global_market = {"HOME": .5, "DRAW": .25, "AWAY": .25}
    assert select_market_sources(model, jc, None, config)["fusion_market_source"] == "JC_ONLY"
    assert select_market_sources(model, None, global_market, config)["fusion_market_source"] == "GLOBAL_ONLY"
    dual = select_market_sources(model, jc, global_market, config)
    assert dual["fusion_market_source"] == "DUAL_MARKET_AVAILABLE"
    assert dual["fusion"]["probabilities"]["HOME"] == pytest.approx(.7 * .6 + .3 * jc["HOME"])
    prior = {"snapshot_id": "old", "payload": {"spf": spf, "rqspf": rq}}
    changed = movement(prior, {"spf": spf, "rqspf": {**rq, "handicap": -2}})
    assert changed["rqspf"]["status"] == "HANDICAP_CHANGED"


def test_page_hash_cache_and_append_only(tmp_path: Path) -> None:
    store = JCPublicStore(tmp_path / "jc.sqlite", tmp_path / "raw")
    try:
        at = datetime.now(UTC)
        page = store.append_page(endpoint="spf", url="https://www.sporttery.cn/jc/jsq/zqspf/index.html",
            fetched_at=at, http_status=200, body="<table id='mainTbl'></table>",
            transport_sha256="abc", parser_version="JC_PUBLIC_PARSER_V1")
        assert store.fresh_page("spf", at=at + timedelta(seconds=59), ttl_seconds=60)
        assert store.fresh_page("spf", at=at + timedelta(seconds=60), ttl_seconds=60) is None
        assert sha256(Path(page["html_path"]).read_bytes()) == page["body_sha256"]
        with pytest.raises(FileExistsError):
            store.append_page(endpoint="spf", url=page["url"], fetched_at=at,
                http_status=200, body=page["body"], transport_sha256="abc",
                parser_version="JC_PUBLIC_PARSER_V1")
    finally:
        store.close()


def test_user_confirmed_json(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    path = tmp_path / "confirmed.json"
    record = {"source": "USER_CONFIRMED_MARKET", "confirmation_status": "CONFIRMED",
        "confirmed_by": "test-user", "evidence_description": "Visible public page checked",
        "jc_match_number": "周一001", "competition": "欧国联", "home_team": "塞浦路斯",
        "away_team": "拉脱维亚", "captured_at": (now - timedelta(minutes=2)).isoformat(),
        "confirmed_at": (now - timedelta(minutes=1)).isoformat(),
        "kickoff_time": (now + timedelta(hours=3)).isoformat(),
        "spf": {"home": 1.8, "draw": 3.3, "away": 4.5},
        "rqspf": {"handicap": -1, "home": 3.1, "draw": 3.45, "away": 1.95}}
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    parsed, evidence = load_user_confirmed(path)
    assert parsed.jc_match_number == "周一001"
    assert evidence["source_quality"] == "USER_CONFIRMED"


def test_supplement_lock_preserves_original(monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
    now = datetime.now(UTC)
    kickoff = now + timedelta(hours=3)
    publication = {**_publication(), "prediction_id": "synthetic-test", "fixture_id": "fixture-test",
        "kickoff_time": kickoff.isoformat(), "prediction_time": (now - timedelta(hours=1)).isoformat(),
        "model_snapshot_id": "test-frozen", "model_probabilities":
            {"HOME": .6, "DRAW": .25, "AWAY": .15}}
    monkeypatch.setattr(jc_public_lock, "_verified_publication", lambda _store, _id: publication)

    class FakeFetcher:
        def __init__(self, _database: object, *, http_client: object = None) -> None:
            pass

        def fetch(self, endpoint: str) -> dict[str, object]:
            body = _html().replace("2026-10-05T10:00:00+00:00", now.isoformat())
            body = body.replace("2026-10-05T16:00:00+00:00", kickoff.isoformat())
            return {"body": body, "fetched_at": now, "body_sha256": sha256(body.encode()),
                    "http_status": 200, "url": "https://www.sporttery.cn/" + endpoint}

        def close(self) -> None:
            pass

    monkeypatch.setattr(jc_public_lock, "JCPublicPageFetcher", FakeFetcher)
    store = R3Store(tmp_path)
    base = tmp_path / "official_locks/synthetic-test/prediction_lock.json"
    base.parent.mkdir(parents=True)
    base.write_text("original", encoding="utf-8")
    official = tmp_path / "official/synthetic-test.json"
    official.parent.mkdir(parents=True)
    official.write_text("publication", encoding="utf-8")
    original_hash = sha256(base.read_bytes())
    path, lock = jc_public_lock.append_jc_market_lock(store, "synthetic-test")
    assert lock["market"]["status"] == "AVAILABLE"
    assert lock["intelligence"]["fusion_market_source"] == "JC_ONLY"
    assert lock["jc_market_snapshot_id"] is not None
    assert path.exists() and sha256(base.read_bytes()) == original_hash
    assert lock["market_verified"] is True


@pytest.mark.skipif(os.environ.get("YYCORE_RUN_JC_PUBLIC_INTEGRATION") != "1",
                    reason="explicit live public-page integration only")
def test_integration_jc_public(tmp_path: Path) -> None:
    from erguoyuan_football.blind_test_r3.jc_public_fetcher import JCPublicPageFetcher

    store = JCPublicStore(tmp_path / "live.sqlite", tmp_path / "raw")
    fetcher = JCPublicPageFetcher(store)
    try:
        page = fetcher.fetch("spf")
        assert page["http_status"] == 200
        assert page["body_sha256"] == sha256(Path(page["html_path"]).read_bytes())
    finally:
        fetcher.close()
        store.close()
