"""Pinned, reusable men's senior national-team historical result provider."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from erguoyuan_football.knowledge.entities.country_alias_registry import (
    CountryAliasRegistry,
)
from erguoyuan_football.models.training import TrainingDataset, TrainingMatch
from erguoyuan_football.network import CoreNetworkClient, ExternalSourceRegistry
from erguoyuan_football.network.config import NetworkConfig, TimeoutPolicy
from erguoyuan_football.network.schemas import SourceDefinition

SOURCE_ID = "MARTJ42_INTERNATIONAL_RESULTS_GITHUB_V1"
SOURCE_REPOSITORY = "https://github.com/martj42/international_results"
PINNED_COMMIT = "394fe81893b062fbc2cf6257e988ac7cc4c039a1"
REQUIRED_COLUMNS = {"date", "home_team", "away_team", "home_score",
                    "away_score", "tournament", "neutral"}
# Explicit source spellings resolved to existing ISO canonical country records.
# Unlisted names stay unavailable; this table applies to the full data source.
SOURCE_COUNTRY_ALIASES = {
    "DR Congo": "COD", "Ivory Coast": "CIV", "Trinidad and Tobago": "TTO",
    "Turkey": "TUR", "Bosnia and Herzegovina": "BIH", "Palestine": "PSE",
    "Hong Kong": "HKG", "Myanmar": "MMR",
    "Saint Vincent and the Grenadines": "VCT", "Saint Kitts and Nevis": "KNA",
    "Republic of Ireland": "IRL", "Saint Lucia": "LCA",
    "Antigua and Barbuda": "ATG", "United States Virgin Islands": "VIR",
    "São Tomé and Príncipe": "STP", "Congo": "COG",
    "Turks and Caicos Islands": "TCA", "Macau": "MAC",
}
TRAINING_TOURNAMENT_TYPES = frozenset({
    "WORLD_CUP_QUALIFIER", "CONTINENTAL_QUALIFIER", "NATIONS_LEAGUE",
})


@dataclass(frozen=True)
class NationalHistoryRow:
    """One date-precision result; kickoff is intentionally unavailable."""

    match_id: str
    competition: str
    match_date: str
    kickoff_time: None
    home_team: str
    away_team: str
    home_team_id: str
    away_team_id: str
    home_score: int
    away_score: int
    neutral_venue: bool
    tournament_type: str
    source: str
    source_timestamp: str


def tournament_type(name: str) -> str:
    """Classify all sources with one fixed rule, independent of target fixtures."""
    value = name.casefold()
    if "friendly" in value:
        return "FRIENDLY"
    if "world cup" in value and "qualification" in value:
        return "WORLD_CUP_QUALIFIER"
    if "world cup" in value:
        return "WORLD_CUP"
    if "nations league" in value:
        return "NATIONS_LEAGUE"
    if "qualification" in value or "qualifier" in value:
        return "CONTINENTAL_QUALIFIER"
    if any(token in value for token in ("cup", "championship", "gold cup")):
        return "CONTINENTAL_CUP"
    return "OTHER"


class NationalTeamHistoryProvider:
    """Import an exact GitHub Git blob, map teams, and enforce date-safe as-of."""

    def __init__(self, csv_path: Path, metadata_path: Path) -> None:
        self.csv_path = csv_path
        self.metadata_path = metadata_path
        self.countries = CountryAliasRegistry()
        countries_by_iso = {country.iso3: country for country in self.countries.all()}
        self.aliases = {name: countries_by_iso[iso] for name, iso in
                        SOURCE_COUNTRY_ALIASES.items() if iso in countries_by_iso}

    @staticmethod
    def _client(endpoint_id: str, endpoint: str) -> CoreNetworkClient:
        source = SourceDefinition(
            source_id=SOURCE_ID, display_name="International results Git snapshot",
            category="HISTORICAL_RESULTS", base_url="https://api.github.com",
            endpoints={endpoint_id: endpoint}, supports_history=True,
            schema_version="GITHUB_GIT_BLOB_V1",
        )
        return CoreNetworkClient(ExternalSourceRegistry((source,)),
            NetworkConfig(timeout=TimeoutPolicy(connect_timeout=15,
                read_timeout=120, total_timeout=180)))

    @classmethod
    def acquire(cls, root: Path) -> NationalTeamHistoryProvider:
        """Download only a pinned Git blob through the registered CORE client."""
        root.mkdir(parents=True, exist_ok=True)
        csv_path = root / f"results-{PINNED_COMMIT}.csv"
        metadata_path = root / f"results-{PINNED_COMMIT}.json"
        if csv_path.exists() or metadata_path.exists():
            if not csv_path.exists() or not metadata_path.exists():
                raise ValueError("NATIONAL_HISTORY_PARTIAL_SNAPSHOT")
            provider = cls(csv_path, metadata_path)
            provider.verify()
            return provider
        contents_url = ("https://api.github.com/repos/martj42/international_results/"
                        f"contents/results.csv?ref={PINNED_COMMIT}")
        client = cls._client("CONTENTS", contents_url)
        try:
            contents = client.fetch_json(SOURCE_ID, "CONTENTS", params={}, bypass_cache=True)
        finally:
            client.close()
        info = contents.body
        if not isinstance(info, dict) or not isinstance(info.get("sha"), str):
            raise TypeError("NATIONAL_HISTORY_CONTENTS_INVALID")
        blob_url = ("https://api.github.com/repos/martj42/international_results/"
                    f"git/blobs/{info['sha']}")
        client = cls._client("BLOB", blob_url)
        try:
            blob = client.fetch_json(SOURCE_ID, "BLOB", params={}, bypass_cache=True)
        finally:
            client.close()
        body = blob.body
        if not isinstance(body, dict) or body.get("encoding") != "base64":
            raise ValueError("NATIONAL_HISTORY_BLOB_INVALID")
        raw = base64.b64decode(str(body["content"]), validate=False)
        git_blob_hash = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if git_blob_hash != info["sha"]:
            raise ValueError("NATIONAL_HISTORY_GIT_BLOB_HASH_MISMATCH")
        source_time = blob.retrieved_at.astimezone(UTC).isoformat()
        metadata = {"source": SOURCE_ID, "repository": SOURCE_REPOSITORY,
            "commit": PINNED_COMMIT, "git_blob_sha1": git_blob_hash,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source_timestamp": source_time, "retrieved_at": source_time,
            "precision": "DATE_ONLY", "score_semantics": "INCLUDING_EXTRA_TIME_WHEN_PLAYED"}
        with csv_path.open("xb") as output:
            output.write(raw)
        with metadata_path.open("x", encoding="utf-8") as output:
            json.dump(metadata, output, ensure_ascii=False, indent=2)
        return cls(csv_path, metadata_path)

    def verify(self) -> dict[str, Any]:
        """Verify local bytes and fixed commit before parsing any rows."""
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        raw = self.csv_path.read_bytes()
        if (metadata.get("commit") != PINNED_COMMIT or
                hashlib.sha256(raw).hexdigest() != metadata.get("sha256")):
            raise ValueError("NATIONAL_HISTORY_SNAPSHOT_CHANGED")
        return metadata

    def rows_as_of(self, cutoff: datetime, *, train_start: date,
                   include_friendlies: bool = False) -> tuple[NationalHistoryRow, ...]:
        """Use only prior calendar dates whose snapshot was retrieved by cutoff."""
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("UTC_CUTOFF_REQUIRED")
        metadata = self.verify()
        source_time = datetime.fromisoformat(metadata["source_timestamp"])
        if source_time > cutoff:
            raise ValueError("NATIONAL_HISTORY_RETRIEVED_AFTER_CUTOFF")
        text = self.csv_path.read_text(encoding="utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        if not REQUIRED_COLUMNS <= set(reader.fieldnames or ()):
            raise ValueError("NATIONAL_HISTORY_SCHEMA_INVALID")
        selected: list[NationalHistoryRow] = []
        seen: set[str] = set()
        for line in reader:
            match_date = date.fromisoformat(line["date"])
            if match_date < train_start or match_date >= cutoff.date():
                continue
            category = tournament_type(line["tournament"])
            if category == "FRIENDLY" and not include_friendlies:
                continue
            home = self.countries.resolve(line["home_team"]) or self.aliases.get(line["home_team"])
            away = self.countries.resolve(line["away_team"]) or self.aliases.get(line["away_team"])
            if home is None or away is None or home.iso3 == away.iso3:
                continue
            neutral_text = line["neutral"].strip().upper()
            if neutral_text not in {"TRUE", "FALSE"}:
                continue
            home_score, away_score = int(line["home_score"]), int(line["away_score"])
            if min(home_score, away_score) < 0:
                continue
            identity = "|".join((line["date"], home.iso3, away.iso3,
                                 line["tournament"], str(home_score), str(away_score)))
            match_id = "NATIONAL_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            if match_id in seen:
                raise ValueError(f"NATIONAL_HISTORY_DUPLICATE:{match_id}")
            seen.add(match_id)
            selected.append(NationalHistoryRow(match_id, line["tournament"],
                match_date.isoformat(), None, line["home_team"], line["away_team"],
                f"NATIONAL_{home.iso3}_M_SENIOR", f"NATIONAL_{away.iso3}_M_SENIOR",
                home_score, away_score, neutral_text == "TRUE", category,
                SOURCE_ID, metadata["source_timestamp"]))
        return tuple(sorted(selected, key=lambda row: (row.match_date, row.match_id)))

    def training_dataset(self, cutoff: datetime, *, train_start: date,
                         include_friendlies: bool = False) -> TrainingDataset:
        """Return native DATE_SAFE_BATCH training rows with conservative result time."""
        rows = tuple(row for row in self.rows_as_of(cutoff, train_start=train_start,
            include_friendlies=include_friendlies) if
            row.tournament_type in TRAINING_TOURNAMENT_TYPES or
            (include_friendlies and row.tournament_type == "FRIENDLY"))
        metadata = self.verify()
        source_time = datetime.fromisoformat(metadata["source_timestamp"])
        matches = []
        for row in rows:
            day = date.fromisoformat(row.match_date)
            kickoff_placeholder = datetime.combine(day, datetime.min.time(), UTC)
            result_available = kickoff_placeholder + timedelta(days=1)
            matches.append(TrainingMatch(match_id=row.match_id,
                competition_id="SENIOR_MENS_INTERNATIONAL",
                season=str(day.year), kickoff_time=kickoff_placeholder,
                home_team_id=row.home_team_id, away_team_id=row.away_team_id,
                home_goals=row.home_score, away_goals=row.away_score,
                neutral_venue=row.neutral_venue, source=SOURCE_ID,
                completed_at=result_available, as_of_time=max(result_available, source_time),
                retrieved_at=source_time, data_version=metadata["sha256"]))
        teams = frozenset(team for match in matches for team in
                          (match.home_team_id, match.away_team_id))
        return TrainingDataset(matches=tuple(matches), known_team_ids=teams,
            dataset_kind="REAL", temporal_mode="DATE_SAFE_BATCH",
            assumptions=("DATE_ONLY_NO_KICKOFF_IN_SOURCE",
                         "CUP_AND_UNCLASSIFIED_EXCLUDED_TO_AVOID_EXTRA_TIME_SCORE_MISMATCH",
                         "NATIONS_LEAGUE_FINALS_EXTRA_TIME_NOT_INDEPENDENTLY_IDENTIFIED",
                         "FRIENDLIES_EXCLUDED" if not include_friendlies
                         else "FRIENDLIES_INCLUDED_UNWEIGHTED"))

    def summary(self, cutoff: datetime, train_start: date, *,
                training_only: bool = False) -> dict[str, Any]:
        """Return inspectable coverage for the requested common training window."""
        rows = self.rows_as_of(cutoff, train_start=train_start)
        if training_only:
            rows = tuple(row for row in rows if
                         row.tournament_type in TRAINING_TOURNAMENT_TYPES)
        return {"match_count": len(rows), "team_count": len({x for row in rows for x in
            (row.home_team_id, row.away_team_id)}),
            "competition_count": len({row.competition for row in rows}),
            "train_start": rows[0].match_date if rows else None,
            "train_end": rows[-1].match_date if rows else None,
            "data_as_of": self.verify()["source_timestamp"],
            "source": SOURCE_ID, "tournament_type_counts": {kind: sum(
                row.tournament_type == kind for row in rows) for kind in sorted(
                {row.tournament_type for row in rows})}}
