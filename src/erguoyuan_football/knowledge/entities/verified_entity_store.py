"""Append/reverify exact source-backed team identities, separate from match data."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from erguoyuan_football.knowledge.teams.alias_matcher import normalize_alias
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.web_research.time_utils import parse_utc, utc_iso


@dataclass(frozen=True)
class StoredEntity:
    """A verified entity with configurable identity freshness."""

    identity: TeamIdentity
    first_seen_at: datetime
    last_verified_at: datetime
    status: str


class VerifiedEntityStore:
    """SQLite identity catalogue; aliases never override provider-ID conflicts."""

    def __init__(self, path: Path | str, *, ttl_days: int = 30) -> None:
        if ttl_days < 1:
            raise ValueError("INVALID_ENTITY_TTL")
        self.ttl = timedelta(days=ttl_days)
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("""CREATE TABLE IF NOT EXISTS verified_entities (
            canonical_team_id TEXT PRIMARY KEY,
            provider_id TEXT NOT NULL,
            provider_team_id TEXT NOT NULL,
            identity_json TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_verified_at TEXT NOT NULL,
            status TEXT NOT NULL,
            UNIQUE(provider_id, provider_team_id)
        )""")
        self.connection.commit()

    def close(self) -> None:
        """Close this store without touching research/model databases."""
        self.connection.close()

    def save(self, identity: TeamIdentity, *, provider_id: str,
             provider_team_id: str, verified_at: datetime) -> None:
        """Persist only source-backed identities, preserving first observation."""
        if not identity.verification_evidence_ids or not identity.provider_ids:
            raise ValueError("ENTITY_VERIFICATION_EVIDENCE_REQUIRED")
        if identity.provider_ids.get(provider_id) != provider_team_id:
            raise ValueError("TEAM_PROVIDER_ID_CONFLICT")
        stamp = utc_iso(verified_at)
        previous = self.connection.execute(
            "SELECT first_seen_at,identity_json FROM verified_entities WHERE canonical_team_id=?",
            (identity.team_id,),
        ).fetchone()
        if previous is not None:
            old = TeamIdentity(**json.loads(previous[1]))
            if old.provider_ids != identity.provider_ids or old.gender != identity.gender or (
                old.age_group != identity.age_group or old.squad_level != identity.squad_level):
                raise ValueError("TEAM_IDENTITY_CONFLICT")
        first_seen = previous[0] if previous else stamp
        try:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO verified_entities VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(canonical_team_id) DO UPDATE SET
                      identity_json=excluded.identity_json,
                      last_verified_at=excluded.last_verified_at,
                      status=excluded.status""",
                    (identity.team_id, provider_id, provider_team_id,
                     json.dumps(asdict(identity), ensure_ascii=False, sort_keys=True),
                     first_seen, stamp, "VERIFIED"),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("TEAM_PROVIDER_ID_CONFLICT") from error

    def find(self, name: str, *, as_of: datetime) -> tuple[StoredEntity, ...]:
        """Return all exact alias matches, including stale candidates."""
        rows = self.connection.execute(
            "SELECT identity_json,first_seen_at,last_verified_at,status FROM verified_entities"
        ).fetchall()
        key = normalize_alias(name)
        matches = []
        for identity_json, first, last, status in rows:
            identity = TeamIdentity(**json.loads(identity_json))
            if key not in {normalize_alias(alias) for alias in
                           (identity.official_name, *identity.aliases,
                            *(identity.former_names or []))}:
                continue
            verified = parse_utc(last)
            state = "STALE_IDENTITY" if as_of - verified > self.ttl else status
            matches.append(StoredEntity(identity, parse_utc(first), verified, state))
        return tuple(matches)

    def by_provider(self, provider_id: str, provider_team_id: str,
                    *, as_of: datetime) -> StoredEntity | None:
        """Find a previously verified provider ID before accepting a renamed club."""
        row = self.connection.execute(
            """SELECT identity_json,first_seen_at,last_verified_at,status
               FROM verified_entities WHERE provider_id=? AND provider_team_id=?""",
            (provider_id, provider_team_id),
        ).fetchone()
        if row is None:
            return None
        identity = TeamIdentity(**json.loads(row[0]))
        verified = parse_utc(row[2])
        state = "STALE_IDENTITY" if as_of - verified > self.ttl else row[3]
        return StoredEntity(identity, parse_utc(row[1]), verified, state)
