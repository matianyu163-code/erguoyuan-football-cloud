"""Canonicalize, merge and deduplicate external history without score overrides."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)
from erguoyuan_football.knowledge.match_identity import GlobalKnowledgeResolver
from erguoyuan_football.research.samples.match_deduplicator import (
    HistoricalMatchSample,
    deduplicate_matches,
)
from erguoyuan_football.research.samples.sample_repository import SampleRepository
from production.bridge.contracts import CoreDataPacketV1
from production.bridge.validator import BridgeRejected


@dataclass(frozen=True)
class ImportedHistory:
    """PIT-eligible canonical observations and immutable provenance hashes."""

    repository: SampleRepository
    source_hashes: tuple[str, ...]
    imported_count: int
    deduplicated_count: int


_TIERS = {"A": 1, "B": 2, "C": 3}


def bridge_competition(name: str, federation: str,
                       entity_type: str) -> CompetitionIdentity:
    """Create a bridge-scoped ID from a cited name, never a global alias claim."""
    normalized = " ".join(name.casefold().split())
    if not normalized:
        raise BridgeRejected("ENTITY_RESOLUTION_FAILED", "COMPETITION_MISSING")
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:20]
    return CompetitionIdentity(f"BRIDGE_COMP_{digest}", name, federation,
                               "WORLD", "CLUB" if entity_type == "CLUB"
                               else "INTERNATIONAL")


def import_history(packet: CoreDataPacketV1, resolver: GlobalKnowledgeResolver,
                   *, local_history: tuple[HistoricalMatchSample, ...] = ()) -> ImportedHistory:
    """Resolve every team/competition, merge local rows and reject result conflicts."""
    cutoff = packet.research_as_of.astimezone(UTC)
    rows: list[HistoricalMatchSample] = []
    seen_ids: dict[str, tuple[str, str, datetime, int, int]] = {}
    hashes: set[str] = set()
    for item in packet.history.all_matches():
        assert item.kickoff is not None  # validate_packet ran before import.
        identity = resolver.resolve_names(item.home, item.away, item.competition,
                                          allow_discovery=False)
        home, away = identity.home_team, identity.away_team
        if home is None or away is None:
            raise BridgeRejected("ENTITY_RESOLUTION_FAILED", f"HISTORY:{item.home}:{item.away}")
        competition = bridge_competition(item.competition, home.federation,
                                         home.entity_type)
        if ((item.home_entity is not None and item.home_entity != home.team_id)
                or (item.away_entity is not None and item.away_entity != away.team_id)):
            raise BridgeRejected("ENTITY_RESOLUTION_FAILED", "HISTORY_ENTITY_ID_CONFLICT")
        kickoff = item.kickoff.astimezone(UTC)
        event_id = item.match_id or hashlib.sha256(
            f"{home.team_id}|{away.team_id}|{kickoff.isoformat()}|"
            f"{competition.competition_id}".encode()).hexdigest()[:24]
        identity_key = (home.team_id, away.team_id, kickoff,
                        item.home_goals, item.away_goals)
        if event_id in seen_ids and seen_ids[event_id] != identity_key:
            raise BridgeRejected("HISTORY_RESULT_CONFLICT", event_id)
        seen_ids[event_id] = identity_key
        evidence_hash = hashlib.sha256(
            f"{item.source}|{item.source_url}|{item.fetched_at.isoformat()}|"
            f"{item.home_goals}:{item.away_goals}".encode()).hexdigest()
        hashes.add(evidence_hash)
        rows.append(HistoricalMatchSample(
            event_id, home.team_id, away.team_id, competition.competition_id,
            kickoff, item.home_goals, item.away_goals, (evidence_hash,),
            (item.source,), _TIERS[item.source_tier],
            item.fetched_at.astimezone(UTC), home.federation,
            home.gender, home.age_group,
            "LEAGUE" if home.entity_type == "CLUB" else "INTERNATIONAL",
            item.neutral_venue,
        ))
    for local in local_history:
        if local.fetched_at > cutoff or local.kickoff >= cutoff:
            raise BridgeRejected("PIT_REJECTED", "LOCAL_HISTORY_FUTURE")
        existing = seen_ids.get(local.match_id)
        key = (local.home_team_id, local.away_team_id, local.kickoff,
               local.home_goals, local.away_goals)
        if existing is not None and existing != key:
            raise BridgeRejected("HISTORY_RESULT_CONFLICT", local.match_id)
        seen_ids[local.match_id] = key
        hashes.update(local.evidence_ids)
        rows.append(local)
    deduped = deduplicate_matches(tuple(rows))
    if deduped.conflicts:
        raise BridgeRejected("HISTORY_RESULT_CONFLICT", str(len(deduped.conflicts)))
    return ImportedHistory(SampleRepository(deduped.samples), tuple(sorted(hashes)),
                           len(rows), len(deduped.samples))
