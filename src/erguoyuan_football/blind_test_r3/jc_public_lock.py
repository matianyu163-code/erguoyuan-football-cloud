"""Append a separate P2B market evidence lock against an immutable R3 prediction."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.jc_public_fetcher import JCPublicPageFetcher
from erguoyuan_football.blind_test_r3.jc_public_import import load_user_confirmed
from erguoyuan_football.blind_test_r3.jc_public_intelligence import (
    fixed_bonus_market,
    match_jc_fixture,
    movement,
    select_market_sources,
    snapshot_id,
)
from erguoyuan_football.blind_test_r3.jc_public_parser import (
    parse_schedule,
    parse_spf_rqspf,
)
from erguoyuan_football.blind_test_r3.jc_public_store import JCPublicStore
from erguoyuan_football.blind_test_r3.market_intelligence import load_r3_market_config
from erguoyuan_football.blind_test_r3.runner import PROJECT_ROOT, R3_ROOT
from erguoyuan_football.blind_test_r3.settlement import _verified_publication
from erguoyuan_football.blind_test_r3.store import (
    R3Store,
    canonical_bytes,
    sha256,
    utc_time,
    write_once,
)

FUSION_CONFIG = PROJECT_ROOT / "config/r3_market_intelligence_v1.yaml"
RELEASE_PATH = R3_ROOT / "manifest/R3_JC_OFFICIAL_PUBLIC_P2B_V1.json"


def verify_jc_release() -> dict[str, Any]:
    """Pin the supplementary provider code and configs before recording a lock."""
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    if release.get("version") != "R3_JC_OFFICIAL_PUBLIC_P2B_V1":
        raise ValueError("R3_JC_RELEASE_INVALID")
    paths = {"config_sha256": PROJECT_ROOT / "config/r3_jc_public_p2b.yaml",
        "fusion_config_sha256": FUSION_CONFIG,
        "schema_sha256": PROJECT_ROOT / "src/erguoyuan_football/markets/schemas.py"}
    for name in ("jc_public_parser", "jc_public_store", "jc_public_fetcher",
                 "jc_public_import", "jc_public_intelligence", "jc_public_lock"):
        paths[f"{name}_sha256"] = Path(__file__).with_name(f"{name}.py")
    for key, path in paths.items():
        if sha256(path.read_bytes()) != release.get(key):
            raise ValueError(f"R3_JC_RELEASE_CHANGED:{key}")
    return release


def append_jc_market_lock(store: R3Store, prediction_id: str, *,
                          user_confirmed_path: Path | None = None,
                          http_client: Any | None = None) -> tuple[Path, dict[str, Any]]:
    """Read frozen model probabilities; append evidence, interpretation and lock."""
    publication = _verified_publication(store, prediction_id)
    release = verify_jc_release()
    now = datetime.now(UTC)
    kickoff = utc_time(publication["kickoff_time"])
    if now >= kickoff or now < utc_time(publication["prediction_time"]):
        raise ValueError("JC_SUPPLEMENT_NOT_PREMATCH")
    database = JCPublicStore(store.root / "market/jc_public/jc_public.sqlite",
                             store.root / "market/jc_public/raw")
    pages: dict[str, dict[str, Any]] = {}
    page_status: dict[str, str] = {}
    source_quality = "OFFICIAL_PUBLIC_PAGE"
    source_type = "JC_OFFICIAL_PUBLIC"
    parsed = None
    evidence: dict[str, Any] = {}
    try:
        if user_confirmed_path is not None:
            parsed, evidence = load_user_confirmed(user_confirmed_path)
            source_quality = "USER_CONFIRMED"
            source_type = "USER_CONFIRMED_MARKET"
            page_status = {"schedule": "USER_CONFIRMED", "spf": "USER_CONFIRMED"}
        else:
            fetcher = JCPublicPageFetcher(database, http_client=http_client)
            try:
                for endpoint in ("schedule", "spf"):
                    try:
                        pages[endpoint] = fetcher.fetch(endpoint)
                    except (OSError, ValueError, TimeoutError) as error:
                        page_status[endpoint] = ("ACCESS_RESTRICTED" if
                            "403" in str(error) or "429" in str(error) else
                            f"UNAVAILABLE:{type(error).__name__}:{error}")
            finally:
                fetcher.close()
            schedule = parse_schedule(pages["schedule"]["body"]) if "schedule" in pages else None
            bonuses = parse_spf_rqspf(pages["spf"]["body"]) if "spf" in pages else None
            if schedule is not None:
                page_status["schedule"] = schedule.status
            if bonuses is not None:
                page_status["spf"] = bonuses.status
            if bonuses is not None and bonuses.matches:
                found = match_jc_fixture(bonuses.matches, publication=publication)
                parsed = found[0] if found is not None else None
                if found is not None:
                    evidence["match_method"] = found[1]
            evidence["pages"] = {key: {field: (value.isoformat() if isinstance(value, datetime)
                else value) for field, value in item.items() if field != "body"}
                for key, item in pages.items()}
        captured = datetime.now(UTC)
        if captured >= kickoff:
            raise ValueError("JC_SUPPLEMENT_NOT_PREMATCH")
        if parsed is not None:
            found = match_jc_fixture((parsed,), publication=publication)
            if found is None:
                parsed = None
                page_status["fixture"] = "UNAVAILABLE:JC_FIXTURE_MISMATCH"
            else:
                evidence["match_method"] = found[1]
        spf = fixed_bonus_market(parsed.spf_bonus) if parsed is not None else None
        rqspf = (fixed_bonus_market(parsed.rqspf_bonus, handicap=parsed.handicap)
                 if parsed is not None else None)
        if parsed is not None and parsed.bonus_updated_at is not None and (
            parsed.bonus_updated_at > captured or parsed.bonus_updated_at >= kickoff
        ):
            raise ValueError("JC_BONUS_TIMESTAMP_NOT_PREMATCH")
        market_status = ("AVAILABLE" if spf is not None and
            (parsed is None or parsed.rqspf_sale_status != "ON_SALE" or rqspf is not None)
            else "PARTIAL" if parsed is not None else "UNAVAILABLE")
        freshness_time = (parsed.bonus_updated_at if parsed and parsed.bonus_updated_at else
                          pages.get("spf", {}).get("fetched_at", captured))
        latest_market = (spf is not None and isinstance(freshness_time, datetime) and
                         0 <= (captured - freshness_time).total_seconds() <= 600)
        if spf is not None and not latest_market:
            market_status = "JC_MARKET_STALE"
        market = {"prediction_id": prediction_id, "fixture_id": publication["fixture_id"],
            "jc_match_number": parsed.jc_match_number if parsed else None,
            "match_method": evidence.get("match_method"),
            "captured_at": captured.isoformat(), "kickoff_time": kickoff.isoformat(),
            "source_type": source_type, "source_quality": source_quality,
            "status": market_status, "spf": spf, "rqspf": rqspf,
            "page_bonus_updated_at": parsed.bonus_updated_at.isoformat() if parsed and
                parsed.bonus_updated_at else None,
            "page_status": page_status, "page_evidence": evidence,
            "parser_version": "JC_PUBLIC_PARSER_V1"}
        market["snapshot_id"] = snapshot_id(market)
        prior = database.latest_snapshot(publication["fixture_id"], before=captured)
        market["movement"] = movement(prior, market)
        market["jc_latest_market"] = latest_market
        global_market = None  # P2A is unconfigured; never promote screenshot odds.
        intelligence = select_market_sources(publication["model_probabilities"],
            spf["no_vig"] if spf is not None and latest_market else None, global_market,
            load_r3_market_config(FUSION_CONFIG))
        base_lock = store.root / "official_locks" / prediction_id / "prediction_lock.json"
        official = store.root / "official" / f"{prediction_id}.json"
        lock_id = "R3JC-" + sha256(canonical_bytes({"prediction_id": prediction_id,
            "snapshot_id": market["snapshot_id"], "base_lock_sha256": sha256(base_lock.read_bytes())}))[:32]
        lock_path = base_lock.parent / f"prediction_lock.market_jc_v1.{lock_id}.json"
        if lock_path.exists():
            raise FileExistsError("JC_MARKET_LOCK_ALREADY_EXISTS")
        if parsed is not None:
            database.append_market(market)
        lock = {"event": "PREDICTION_LOCK_MARKET_JC_V1", "market_lock_id": lock_id,
            "prediction_id": prediction_id, "model_snapshot_id": publication["model_snapshot_id"],
            "base_official_lock_sha256": sha256(base_lock.read_bytes()),
            "official_publication_sha256": sha256(official.read_bytes()),
            "jc_market_snapshot_id": market["snapshot_id"] if parsed is not None else None,
            "market": market, "intelligence": intelligence,
            "market_verified": spf is not None and latest_market and
                source_quality == "OFFICIAL_PUBLIC_PAGE",
            "market_release_version": release["version"],
            "market_release_sha256": sha256(RELEASE_PATH.read_bytes()),
            "fusion_version": "MODEL_MARKET_FUSION_V1", "locked_at": datetime.now(UTC).isoformat()}
        write_once(lock_path, canonical_bytes(lock))
        return lock_path, lock
    finally:
        database.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_id")
    parser.add_argument("--root", type=Path, default=R3_ROOT)
    parser.add_argument("--user-confirmed", type=Path)
    arguments = parser.parse_args(argv)
    path, lock = append_jc_market_lock(R3Store(arguments.root), arguments.prediction_id,
                                       user_confirmed_path=arguments.user_confirmed)
    print(json.dumps({"path": str(path), **lock}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
