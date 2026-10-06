"""Global team identity contract, separate from fixtures and model data."""

from __future__ import annotations

from dataclasses import dataclass

from erguoyuan_football.knowledge.entities.age_group import AgeGroup
from erguoyuan_football.knowledge.entities.entity_type import EntityType
from erguoyuan_football.knowledge.entities.gender import Gender
from erguoyuan_football.knowledge.entities.squad_level import SquadLevel


@dataclass(frozen=True)
class TeamIdentity:
    """User-supplied identity seed; aliases make no claim about a live fixture."""

    team_id: str
    official_name: str
    country: str
    federation: str
    aliases: list[str]
    entity_type: str = "UNKNOWN"
    gender: str = "UNKNOWN"
    age_group: str = "SENIOR"
    squad_level: str = "FIRST_TEAM"
    city: str | None = None
    provider_ids: dict[str, str] | None = None
    former_names: list[str] | None = None
    identity_status: str = "LOCAL_VERIFIED"
    verification_evidence_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if not all((self.team_id, self.official_name, self.country, self.federation)):
            raise ValueError("TEAM_IDENTITY_FIELDS_REQUIRED")
        if not self.aliases or not all(alias.strip() for alias in self.aliases):
            raise ValueError("TEAM_ALIASES_REQUIRED")
        if (self.entity_type not in {value.value for value in EntityType}
                or self.gender not in {value.value for value in Gender}
                or self.age_group not in {value.value for value in AgeGroup}
                or self.squad_level not in {value.value for value in SquadLevel}):
            raise ValueError("TEAM_DIMENSION_INVALID")
        if self.identity_status == "PROVIDER_VERIFIED" and (
            not self.provider_ids or not self.verification_evidence_ids):
            raise ValueError("TEAM_VERIFICATION_EVIDENCE_REQUIRED")
