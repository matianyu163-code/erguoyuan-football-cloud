"""The Odds API V4 transport through the registered CoreNetworkClient."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from erguoyuan_football.blind_test_r3.odds_api_cache import OddsApiResponseCache
from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.config import NetworkConfig, RetryPolicy, TimeoutPolicy
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry

SPORT_KEY = re.compile(r"^[a-z0-9_]+$")


def load_odds_api_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("version") != "R3_THE_ODDS_API_P2A_V1" or (
        config.get("provider_name") != "THE_ODDS_API_V4" or
        config.get("api_key_env") != "YYCORE_THE_ODDS_API_KEY" or
        config.get("base_url") != "https://api.the-odds-api.com/v4" or
        config.get("market") != "h2h" or config.get("odds_format") != "decimal" or
        config.get("date_format") != "iso" or
        config.get("regions") != ["eu", "uk"]
    ):
        raise ValueError("THE_ODDS_API_CONFIG_INVALID")
    for field in ("catalog_cache_seconds", "odds_cache_seconds",
                  "maximum_market_age_seconds", "quota_warning_threshold"):
        if int(config[field]) < 0:
            raise ValueError(f"THE_ODDS_API_CONFIG_INVALID:{field}")
    return config


def load_sport_map(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("version") != "THE_ODDS_API_SPORT_MAP_V1":
        raise ValueError("THE_ODDS_API_SPORT_MAP_VERSION_INVALID")
    for item in [*data.get("mappings", {}).values(), *data.get("candidates", [])]:
        if not SPORT_KEY.fullmatch(str(item["sport_key"])):
            raise ValueError("THE_ODDS_API_SPORT_KEY_INVALID")
    return data


class TheOddsApiV4Client:
    """Query only explicitly registered V4 paths; retain quota and raw hashes."""

    provider_name = "THE_ODDS_API_V4"

    def __init__(self, config: dict[str, Any], sport_map: dict[str, Any],
                 cache: OddsApiResponseCache, *, http_client: Any | None = None) -> None:
        self.config = config
        self.sport_map = sport_map
        self.cache = cache
        self.api_key_env = str(config["api_key_env"])
        base = str(config["base_url"])
        sport_keys = {str(item["sport_key"]) for item in [
            *sport_map.get("mappings", {}).values(), *sport_map.get("candidates", [])]}
        endpoints = {"sports": f"{base}/sports/"}
        endpoints.update({f"odds_{key}": f"{base}/sports/{key}/odds/"
                          for key in sport_keys})
        source = SourceDefinition(source_id=self.provider_name,
            display_name="The Odds API V4", category="GLOBAL_ODDS_PROVIDER",
            base_url=base, endpoints=endpoints, requires_auth=True,
            auth_env_var=self.api_key_env, auth_query_name="apiKey",
            cache_policy={"ttl_seconds": 0}, allow_empty_json=True,
            retry_policy=RetryPolicy(max_attempts=1),
            schema_version="THE_ODDS_API_V4_JSON_V1")
        self.network = CoreNetworkClient(ExternalSourceRegistry((source,)),
            NetworkConfig(timeout=TimeoutPolicy(read_timeout=float(config["timeout_seconds"]),
                total_timeout=float(config["timeout_seconds"]) + 10)),
            http_client=http_client)

    @property
    def configured(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def close(self) -> None:
        self.network.close()

    def _fetch(self, endpoint: str, sport_key: str | None, *, at: datetime,
               ttl_seconds: int, params: dict[str, Any], regions: str | None) -> dict[str, Any]:
        if not self.configured:
            raise ValueError("YYCORE_THE_ODDS_API_KEY_MISSING")
        fresh = self.cache.fresh(endpoint=endpoint, sport_key=sport_key,
                                 regions=regions, at=at)
        if fresh is not None:
            return {**fresh, "cache_hit": True, "http_status": None}
        response = self.network.fetch_json(self.provider_name, endpoint,
            params=params, bypass_cache=True)
        saved = self.cache.put(endpoint=endpoint, sport_key=sport_key,
            regions=regions, fetched_at=response.retrieved_at,
            ttl_seconds=ttl_seconds, payload=response.body,
            quota_remaining=response.quota_remaining,
            quota_used=response.quota_used, quota_last_cost=response.quota_last_cost)
        return {**saved, "cache_hit": False, "http_status": response.status_code}

    def refresh_sport_catalog(self, *, at: datetime | None = None) -> dict[str, Any]:
        instant = at or datetime.now(UTC)
        return self._fetch("sports", None, at=instant,
            ttl_seconds=int(self.config["catalog_cache_seconds"]),
            params={"all": "false"}, regions=None)

    def get_h2h(self, sport_key: str, *, at: datetime | None = None) -> dict[str, Any]:
        if not SPORT_KEY.fullmatch(sport_key) or (
            f"odds_{sport_key}" not in self.network.registry.get(self.provider_name).endpoints
        ):
            raise ValueError("SPORT_KEY_NOT_REGISTERED")
        regions = ",".join(self.config["regions"])
        instant = at or datetime.now(UTC)
        return self._fetch(f"odds_{sport_key}", sport_key, at=instant,
            ttl_seconds=int(self.config["odds_cache_seconds"]),
            params={"regions": regions, "markets": self.config["market"],
                    "oddsFormat": self.config["odds_format"],
                    "dateFormat": self.config["date_format"]}, regions=regions)
