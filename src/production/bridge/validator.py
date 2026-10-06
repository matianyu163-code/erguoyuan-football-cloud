"""Validate external citations, exact timestamps and pre-match eligibility."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from production.bridge.contracts import CoreDataPacketV1, HistoryMatch, SourceEvidence


class BridgeRejected(ValueError):
    """Structured refusal that is safe to expose to callers."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code + (f":{detail}" if detail else ""))
        self.code = code
        self.detail = detail


_DOMAINS: dict[str, tuple[str, ...]] = {
    "A": ("fifa.com", "uefa.com", "dfb.de", "knvb.nl", "onsoranje.nl",
          "thefa.com", "ffm.mk", "fifa.com", "oefb.at", "fshf.org"),
    "B": ("openligadb.de", "football-data.org", "soccerway.com"),
    "C": ("worldfootball.net", "11v11.com", "transfermarkt.com"),
}


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise BridgeRejected("BRIDGE_INPUT_INVALID", f"{label}_TIMEZONE_REQUIRED")
    return value.astimezone(UTC)


def _source(url: str, tier: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold().rstrip(".")
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or parsed.port not in (None, 443)
            or not any(host == domain or host.endswith("." + domain)
                       for domain in _DOMAINS[tier])):
        raise BridgeRejected("BRIDGE_SOURCE_REJECTED", f"{tier}:{host or 'NO_HOST'}")


def _check_evidence(row: SourceEvidence, cutoff: datetime) -> None:
    _source(row.source_url, row.source_tier)
    fetched = _utc(row.fetched_at, "FETCHED_AT")
    if fetched > cutoff:
        raise BridgeRejected("PIT_REJECTED", "FUTURE_FIXTURE_EVIDENCE")
    if row.published_at is not None and _utc(row.published_at, "PUBLISHED_AT") > cutoff:
        raise BridgeRejected("PIT_REJECTED", "FUTURE_FIXTURE_PUBLICATION")
    if not row.source.strip() or row.evidence_type not in {"FIXTURE", "SCHEDULE"}:
        raise BridgeRejected("BRIDGE_INPUT_INVALID", "FIXTURE_SOURCE_INVALID")


def _check_history(row: HistoryMatch, cutoff: datetime) -> None:
    _source(row.source_url, row.source_tier)
    if not row.source.strip():
        raise BridgeRejected("BRIDGE_INPUT_INVALID", "HISTORY_SOURCE_MISSING")
    if row.kickoff is None:
        raise BridgeRejected("BRIDGE_INPUT_INVALID", "EXACT_HISTORY_KICKOFF_REQUIRED")
    try:
        date.fromisoformat(row.date)
    except ValueError as error:
        raise BridgeRejected("BRIDGE_INPUT_INVALID", "HISTORY_DATE_INVALID") from error
    kickoff = _utc(row.kickoff, "HISTORY_KICKOFF")
    fetched = _utc(row.fetched_at, "FETCHED_AT")
    published = (_utc(row.published_at, "PUBLISHED_AT")
                 if row.published_at is not None else None)
    if kickoff >= cutoff or fetched > cutoff or (published is not None and published > cutoff):
        raise BridgeRejected("PIT_REJECTED", "FUTURE_HISTORY")
    if fetched <= kickoff or (published is not None and published <= kickoff):
        raise BridgeRejected("BRIDGE_INPUT_INVALID", "RESULT_BEFORE_KICKOFF")
    if published is not None and published > fetched:
        raise BridgeRejected("BRIDGE_INPUT_INVALID", "PUBLICATION_AFTER_FETCH")
    if row.home.strip() == row.away.strip() or not row.competition.strip():
        raise BridgeRejected("BRIDGE_INPUT_INVALID", "HISTORY_IDENTITY_INVALID")


def packet_hash(packet: CoreDataPacketV1) -> str:
    """Stable content hash for immutable request and audit linkage."""
    payload = json.dumps(packet.model_dump(mode="json"), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_packet(packet: CoreDataPacketV1, *, now: datetime | None = None) -> None:
    """CORE independently checks all research and result observation times."""
    present = _utc(now or datetime.now(UTC), "NOW")
    cutoff = _utc(packet.research_as_of, "RESEARCH_AS_OF")
    kickoff = _utc(packet.match.kickoff_utc, "TARGET_KICKOFF")
    if _utc(packet.created_at, "CREATED_AT") > cutoff or cutoff > present:
        raise BridgeRejected("PIT_REJECTED", "PACKET_TIME_IN_FUTURE")
    if cutoff >= kickoff:
        raise BridgeRejected("PIT_REJECTED", "TARGET_ALREADY_STARTED")
    try:
        local = datetime.fromisoformat(packet.match.kickoff_original)
        zone = ZoneInfo(packet.match.kickoff_timezone)
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise BridgeRejected("BRIDGE_INPUT_INVALID", "KICKOFF_TIMEZONE_INVALID") from error
    if local.tzinfo is None:
        local = local.replace(tzinfo=zone)
    if local.astimezone(UTC) != kickoff:
        raise BridgeRejected("BRIDGE_INPUT_INVALID", "KICKOFF_TIME_CONFLICT")
    if packet.match.home.casefold().strip() == packet.match.away.casefold().strip():
        raise BridgeRejected("BRIDGE_INPUT_INVALID", "SAME_TEAM")
    for evidence in packet.fixture_evidence.sources:
        _check_evidence(evidence, cutoff)
    for row in packet.history.all_matches():
        _check_history(row, cutoff)
    # Optional observations are archived, but cannot enter a model without a
    # separate typed, source-backed validator. Their timestamps still may not
    # assert future availability.
    for collection in (packet.optional.odds, packet.optional.xg,
                       packet.optional.lineup, packet.optional.injuries,
                       packet.optional.rankings, packet.optional.news):
        for item in collection:
            if not {"source", "source_url", "source_tier", "fetched_at"} <= item.keys():
                raise BridgeRejected("BRIDGE_INPUT_INVALID", "OPTIONAL_PROVENANCE_MISSING")
            tier = str(item["source_tier"])
            if tier not in _DOMAINS:
                raise BridgeRejected("BRIDGE_SOURCE_REJECTED", "OPTIONAL_TIER")
            _source(str(item["source_url"]), tier)
            try:
                stamp = datetime.fromisoformat(str(item["fetched_at"]))
            except ValueError as error:
                raise BridgeRejected("BRIDGE_INPUT_INVALID", "OPTIONAL_TIME_INVALID") from error
            if _utc(stamp, "OPTIONAL_FETCHED_AT") > cutoff:
                raise BridgeRejected("PIT_REJECTED", "FUTURE_OPTIONAL_EVIDENCE")
