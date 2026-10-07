"""Brazilian Serie A wrappers around the repository's registered model implementations."""

from __future__ import annotations

from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.models.bivariate_poisson import CoreBivariatePoissonModel
from erguoyuan_football.models.dixon_coles import CoreDixonColesModel
from erguoyuan_football.models.elo import CoreEloModel

BRAZIL_SERIE_A = "BRAZIL_SERIE_A"


class BrazilClubModelMixin:
    """Limit an existing fitted implementation to the Brazil Serie A cohort."""

    def supports_fixture(self, fixture: Fixture) -> bool:
        """Return true only for a Brazil fixture with both exact fitted team IDs."""
        return (
            fixture.competition_id == BRAZIL_SERIE_A
            and fixture.neutral_venue is not None
            and fixture.home_team_id in self.teams
            and fixture.away_team_id in self.teams
        )


class BrazilEloModel(BrazilClubModelMixin, CoreEloModel):
    """Brazil Serie A specialization of the registered Elo V1 implementation."""

    model_id = "CLUB_ELO_BRAZIL_V1"
    model_name = "Brazil Serie A Elo"


class BrazilDixonColesModel(BrazilClubModelMixin, CoreDixonColesModel):
    """Brazil Serie A specialization of the registered Dixon-Coles V1 implementation."""

    model_id = "CLUB_DIXON_COLES_BRAZIL_V1"
    model_name = "Brazil Serie A Dixon-Coles"


class BrazilBivariatePoissonModel(BrazilClubModelMixin, CoreBivariatePoissonModel):
    """Brazil Serie A specialization of the registered Bivariate Poisson V1 implementation."""

    model_id = "CLUB_BIVARIATE_POISSON_BRAZIL_V1"
    model_name = "Brazil Serie A Bivariate Poisson"
