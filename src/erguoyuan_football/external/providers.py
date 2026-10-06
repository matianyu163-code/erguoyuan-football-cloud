"""Shared external football provider interface routed through CoreNetworkClient."""

from __future__ import annotations

from typing import Protocol

from erguoyuan_football.external.opta.provider import ExternalData, OptaProduct


class ExternalFootballDataProvider(Protocol):
    """Provider contract for health and football data; no provider URL is implied."""

    def health_check(self): ...

    def fetch_match(self, match_id: str, prediction_time) -> ExternalData: ...

    def fetch_team_stats(self, team_id: str, prediction_time) -> ExternalData: ...

    def fetch_xg(self, match_id: str, prediction_time) -> ExternalData: ...

    def fetch_prediction(self, match_id: str, prediction_time) -> ExternalData: ...

    def get_metadata(self) -> dict[str, str]: ...


class OptaProvider(ExternalFootballDataProvider, Protocol):
    """Lawful Opta source contract; implementation requires authorized endpoints."""

    def fetch_product(self, product: OptaProduct, match_id: str, prediction_time) -> ExternalData: ...
