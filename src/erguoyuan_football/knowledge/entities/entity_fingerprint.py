"""Dimension-separated canonical fingerprints."""

from __future__ import annotations

import hashlib

from erguoyuan_football.knowledge.teams.alias_matcher import normalize_alias


def entity_fingerprint(*, country: str, entity_type: str, gender: str,
                       age_group: str, squad_level: str, name: str,
                       provider_id: str, provider_team_id: str) -> str:
    """Never collapse women, youth or reserve sides by display-name similarity."""
    if not provider_id or not provider_team_id:
        raise ValueError("VERIFIED_PROVIDER_ID_REQUIRED")
    parts = (country, entity_type, gender, age_group, squad_level,
             normalize_alias(name), provider_id, provider_team_id)
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def canonical_entity_id(*, country: str, federation: str,
                        entity_type: str, gender: str, age_group: str,
                        fingerprint: str, provider_id: str,
                        provider_team_id: str) -> str:
    """Use verified country/program dimensions or provider ID, never name alone."""
    if entity_type == "NATIONAL" and country and federation:
        gender_code = {"MEN": "M", "WOMEN": "W"}.get(gender, "UNK")
        return f"{federation}_{country}_{gender_code}_{age_group}"
    return f"{country or 'UNK'}_{provider_id}_{provider_team_id}_{fingerprint[:12]}"
