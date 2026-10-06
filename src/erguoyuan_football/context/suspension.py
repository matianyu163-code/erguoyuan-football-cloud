"""Suspension evidence remains unavailable without dated discipline and rule data."""

from erguoyuan_football.context.schemas import Availability


class SuspensionResolver:
    """Do not infer bans from names or unversioned card totals."""

    def resolve(self, *, discipline_events_available: bool,
                verified_suspension_rules_available: bool) -> Availability:
        if discipline_events_available and verified_suspension_rules_available:
            return Availability.AVAILABLE
        return Availability.UNAVAILABLE
