"""Exact ISO country and Chinese display-name registry for team resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files

from erguoyuan_football.knowledge.teams.alias_matcher import normalize_alias

COUNTRY_REGISTRY_VERSION = "ICU_CLDR_ISO_REGIONS_V1"


@dataclass(frozen=True)
class Country:
    """Canonical country identity and football confederation when known."""

    iso3: str
    name: str
    federation: str
    aliases: tuple[str, ...]
    iso2: str | None = None
    canonical_name_zh: str | None = None

    @property
    def country_id(self) -> str:
        """Stable ISO-3 key for national-team construction."""
        return self.iso3

    @property
    def canonical_name_en(self) -> str:
        """Contract-facing English label retained alongside legacy ``name``."""
        return self.name


_SEED = (
    Country("GER", "Germany", "UEFA", ("德国", "GER")),
    Country("GRE", "Greece", "UEFA", ("希腊", "GRE")),
    Country("ENG", "England", "UEFA", ("英格兰", "ENG")),
    Country("FRA", "France", "UEFA", ("法国", "FRA")),
    Country("ESP", "Spain", "UEFA", ("西班牙", "ESP")),
    Country("ITA", "Italy", "UEFA", ("意大利", "ITA")),
    Country("POR", "Portugal", "UEFA", ("葡萄牙", "POR")),
    Country("JPN", "Japan", "AFC", ("日本", "JPN")),
    Country("KOR", "South Korea", "AFC", ("韩国", "KOR", "Korea Republic")),
    Country("CHN", "China", "AFC", ("中国", "CHN", "PR China")),
    Country("UKR", "Ukraine", "UEFA", ("乌克兰", "UKR")),
    Country("SCO", "Scotland", "UEFA", ("苏格兰", "SCO")),
    Country("WAL", "Wales", "UEFA", ("威尔士", "WAL")),
    Country("NIR", "Northern Ireland", "UEFA", ("北爱尔兰", "NIR")),
    Country("SAU", "Saudi Arabia", "AFC", ("沙特", "沙特阿拉伯", "KSA")),
    Country("TAH", "Tahiti", "OFC", ("塔西提", "TAH")),
)


def _default_countries() -> tuple[Country, ...]:
    """Load packaged ICU/CLDR ISO region names and preserve football codes."""
    resource = files(__package__).joinpath("country_aliases.json")
    rows = json.loads(resource.read_text(encoding="utf-8"))
    seeds_by_name = {normalize_alias(item.name): item for item in _SEED}
    merged: dict[str, Country] = {}
    matched_seed_codes: set[str] = set()
    for row in rows:
        iso3 = str(row["iso3"])
        name = str(row["english"])
        extra_aliases = row.get("aliases", [])
        if not isinstance(extra_aliases, list) or not all(
            isinstance(alias, str) for alias in extra_aliases
        ):
            raise ValueError(f"INVALID_COUNTRY_ALIAS_DATA:{iso3}")
        aliases = tuple(dict.fromkeys((str(row["iso2"]), iso3,
                                      str(row["zh_cn"]), name, *extra_aliases)))
        seed = seeds_by_name.get(normalize_alias(name))
        if seed is None:
            merged[iso3] = Country(iso3, name, "UNKNOWN", aliases,
                                   str(row["iso2"]), str(row["zh_cn"]))
        else:
            matched_seed_codes.add(seed.iso3)
            merged[iso3] = Country(iso3, name, seed.federation,
                tuple(dict.fromkeys((*seed.aliases, *aliases))),
                str(row["iso2"]), str(row["zh_cn"]))
    for seed in _SEED:
        if seed.iso3 not in matched_seed_codes:
            merged.setdefault(seed.iso3, seed)
    return tuple(merged.values())


class CountryAliasRegistry:
    """Resolve exact ISO/Chinese country aliases; ambiguous aliases never guess."""

    def __init__(self, countries: tuple[Country, ...] | None = None) -> None:
        records = countries if countries is not None else _default_countries()
        self._records = records
        by_key: dict[str, dict[str, Country]] = {}
        for country in records:
            for alias in (country.name, *country.aliases):
                key = normalize_alias(alias)
                if key:
                    by_key.setdefault(key, {})[country.iso3] = country
        self._index = by_key

    def all(self) -> tuple[Country, ...]:
        """Return source-backed country records for coverage audits."""
        return self._records

    def resolve(self, name: str) -> Country | None:
        """Return a uniquely identified country or ``None`` for absent/ambiguous."""
        matches = self._index.get(normalize_alias(name), {})
        return next(iter(matches.values())) if len(matches) == 1 else None
