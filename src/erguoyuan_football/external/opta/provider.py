"""Legal, explicit Opta boundary. Default provider has no data access."""

from typing import Literal, Protocol

from pydantic import model_validator

from erguoyuan_football.contracts.common import (
    Availability,
    Contract,
    Identifier,
    UTCTime,
    now,
)

OptaProduct = Literal["POWER_RANKING", "XG", "XGA", "TEAM_STATS", "PLAYER_STATS", "PREDICTION"]
PRODUCTS = ("POWER_RANKING", "XG", "XGA", "TEAM_STATS", "PLAYER_STATS", "PREDICTION")


class ExternalData(Contract):
    product: OptaProduct
    source: Identifier
    retrieved_at: UTCTime
    as_of_time: UTCTime | None
    availability: Availability
    payload: dict | None = None
    reason: str | None = None
    access_basis: Literal["PUBLIC_PAGE", "LICENSED_API"] | None = None

    @model_validator(mode="after")
    def evidence(self):
        if self.availability == Availability.UNAVAILABLE:
            if self.payload is not None or not self.reason:
                raise ValueError("unavailable external data must be empty with reason")
        elif self.as_of_time is None or not self.payload or not self.access_basis:
            raise ValueError("external data requires dated payload and lawful access evidence")
        return self


class OptaProvider(Protocol):
    def fetch(self, product: OptaProduct, match_id: str, prediction_time) -> ExternalData: ...

    def get_metadata(self) -> dict[str, str]: ...


class UnavailableOptaProvider:
    """Explicit no-credential provider that never fabricates Opta payloads."""

    def fetch(self, product: OptaProduct, match_id: str, prediction_time) -> ExternalData:
        return ExternalData(product=product, source="OPTA_NOT_CONFIGURED", retrieved_at=now(),
                            as_of_time=None, availability=Availability.UNAVAILABLE,
                            reason="NO_PUBLIC_OR_LICENSED_OPTA_PROVIDER")

    def get_metadata(self) -> dict[str, str]:
        return {"provider": "OPTA", "availability": "UNAVAILABLE",
                "reason": "NO_PUBLIC_OR_LICENSED_OPTA_PROVIDER"}
