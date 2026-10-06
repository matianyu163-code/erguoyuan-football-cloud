"""Lawful provider access with explicit availability; no guessed coverage or xG."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from erguoyuan_football.contracts.common import now
from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.errors import NetworkError
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry


@dataclass(frozen=True)
class ProviderResult:
    status: str
    reason: str | None
    records: tuple[dict[str, Any], ...]
    source: str
    retrieved_at: str | None
    as_of_time: str | None


class FootballDataOrgProvider:
    """Use only the documented v4 endpoint and the account's actual response."""

    COMPETITIONS = frozenset({"PL", "PD", "BL1", "SA", "FL1", "CL", "EL"})

    def fetch_matches(self, competition: str, season: int) -> ProviderResult:
        if competition not in self.COMPETITIONS or season < 2000 or season > 2100:
            raise ValueError("UNREGISTERED_COMPETITION_OR_SEASON")
        if not os.getenv("FOOTBALL_DATA_ORG_TOKEN"):
            return ProviderResult("UNAVAILABLE", "AUTH_NOT_CONFIGURED", (), "FOOTBALL_DATA_ORG", None, None)
        endpoint = f"https://api.football-data.org/v4/competitions/{competition}/matches"
        source = SourceDefinition(source_id="FOOTBALL_DATA_ORG", display_name="football-data.org",
            category="HISTORICAL", base_url="https://api.football-data.org",
            endpoints={"matches": endpoint}, requires_auth=True,
            auth_env_var="FOOTBALL_DATA_ORG_TOKEN", auth_header_name="X-Auth-Token",
            schema_version="v4", supports_history=True)
        client = CoreNetworkClient(ExternalSourceRegistry((source,)))
        try:
            response = client.fetch_json("FOOTBALL_DATA_ORG", "matches", params={"season": season},
                                         prediction_time=now() + timedelta(minutes=5))
        except (ValueError, TimeoutError, OSError, NetworkError) as error:
            message = str(error)
            reason = ("RATE_LIMITED" if "RATE_LIMITED" in message else
                      "COVERAGE_UNAVAILABLE" if "403" in message or "404" in message else
                      error.code if isinstance(error, NetworkError) else type(error).__name__)
            return ProviderResult("UNAVAILABLE", reason, (), "FOOTBALL_DATA_ORG", None, None)
        finally:
            client.close()
        body = response.body
        matches = body.get("matches") if isinstance(body, dict) else None
        if not isinstance(matches, list):
            return ProviderResult("UNAVAILABLE", "INVALID_SCHEMA", (), "FOOTBALL_DATA_ORG", None, None)
        # Returning records here does not promote them into canonical tables.
        return ProviderResult("AVAILABLE", None, tuple(matches), "FOOTBALL_DATA_ORG",
            response.retrieved_at.isoformat(), response.as_of_time.isoformat() if response.as_of_time else None)


class StatsBombOpenDataProvider:
    """Read only a local official open-data checkout and actual published fields."""

    def __init__(self, repository: Path):
        self.repository = repository

    def available(self) -> ProviderResult:
        competitions = self.repository / "data" / "competitions.json"
        readme = self.repository / "README.md"
        manifest_path = self.repository / "source_manifest.json"
        if not competitions.is_file() or not readme.is_file():
            return ProviderResult("UNAVAILABLE", "OFFICIAL_OPEN_DATA_NOT_PRESENT", (),
                                  "STATSBOMB_OPEN", None, None)
        if not manifest_path.is_file():
            return ProviderResult("UNAVAILABLE", "PINNED_SOURCE_MANIFEST_MISSING", (),
                                  "STATSBOMB_OPEN", None, None)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ProviderResult("UNAVAILABLE", "INVALID_SOURCE_MANIFEST", (),
                                  "STATSBOMB_OPEN", None, None)
        if not isinstance(manifest, dict):
            return ProviderResult("UNAVAILABLE", "INVALID_SOURCE_MANIFEST", (),
                                  "STATSBOMB_OPEN", None, None)
        if (manifest.get("repository") != "https://github.com/hudl/open-data" or
                manifest.get("license_class") != "OPEN_DATA_WITH_TERMS" or
                not isinstance(manifest.get("commit_sha"), str) or
                not re.fullmatch(r"[0-9a-f]{40}", manifest["commit_sha"]) or
                not manifest.get("retrieved_at") or
                manifest.get("competitions_sha256") != hashlib.sha256(competitions.read_bytes()).hexdigest()):
            return ProviderResult("UNAVAILABLE", "UNVERIFIED_SOURCE_MANIFEST", (),
                                  "STATSBOMB_OPEN", None, None)
        try:
            retrieved = datetime.fromisoformat(manifest["retrieved_at"])
        except (AttributeError, ValueError):
            return ProviderResult("UNAVAILABLE", "INVALID_RETRIEVAL_TIMESTAMP", (),
                                  "STATSBOMB_OPEN", None, None)
        if retrieved.tzinfo is None:
            return ProviderResult("UNAVAILABLE", "INVALID_RETRIEVAL_TIMESTAMP", (),
                                  "STATSBOMB_OPEN", None, None)
        if not (self.repository / ".git").exists():
            return ProviderResult("UNAVAILABLE", "PINNED_GIT_REPOSITORY_MISSING", (),
                                  "STATSBOMB_OPEN", None, None)
        repo = self.repository.resolve()
        command = ["git", "-c", f"safe.directory={repo.as_posix()}"]
        try:
            head = subprocess.run([*command, "rev-parse", "HEAD"], cwd=repo,
                                  capture_output=True, text=True, check=True).stdout.strip()
            origin = subprocess.run([*command, "remote", "get-url", "origin"], cwd=repo,
                                    capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return ProviderResult("UNAVAILABLE", "PINNED_GIT_SOURCE_MISMATCH", (),
                                  "STATSBOMB_OPEN", None, None)
        if head != manifest["commit_sha"] or origin.rstrip("/").removesuffix(".git") != manifest["repository"]:
            return ProviderResult("UNAVAILABLE", "PINNED_GIT_SOURCE_MISMATCH", (),
                                  "STATSBOMB_OPEN", None, None)
        try:
            rows = json.loads(competitions.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ProviderResult("UNAVAILABLE", "INVALID_COMPETITIONS_SCHEMA", (),
                                  "STATSBOMB_OPEN", None, None)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            return ProviderResult("UNAVAILABLE", "INVALID_COMPETITIONS_SCHEMA", (),
                                  "STATSBOMB_OPEN", None, None)
        # An open catalogue is not evidence of event, lineup, 360 or xG coverage.
        return ProviderResult("AVAILABLE", None, tuple(rows), "STATSBOMB_OPEN",
                              manifest["retrieved_at"], manifest["retrieved_at"])
