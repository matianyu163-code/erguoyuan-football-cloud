"""Resolve JC membership only from allow-listed, timestamped provider evidence."""

from __future__ import annotations

from datetime import UTC, datetime

from erguoyuan_football.jc_verification.jc_cache import JCMatchCache
from erguoyuan_football.jc_verification.jc_confidence import confidence_for
from erguoyuan_football.jc_verification.jc_match_matcher import JCMatchMatcher
from erguoyuan_football.jc_verification.jc_provider import (
    JCProviderRegistry,
    ProviderTier,
)
from erguoyuan_football.jc_verification.jc_schema import (
    JCMatchQuery,
    JCStatus,
    JCVerificationResult,
)


class JCMatchVerifier:
    """Query registered sources, use fresh cache, and fail closed to UNKNOWN."""

    def __init__(self, registry: JCProviderRegistry | None = None,
                 matcher: JCMatchMatcher | None = None,
                 cache: JCMatchCache | None = None) -> None:
        self.registry = registry or JCProviderRegistry()
        self.matcher = matcher or JCMatchMatcher()
        self.cache = cache

    def verify(self, query: JCMatchQuery, *, now: datetime | None = None
               ) -> JCVerificationResult:
        """Verify from provider rows; never infer non-membership from provider errors."""
        checked_at = _utc(now or datetime.now(UTC))
        if query.kickoff_time is None or query.match_date is None:
            return JCVerificationResult.unknown(checked_at, "MATCH_DATE_OR_KICKOFF_REQUIRED")
        match_date = query.match_date
        registrations = self.registry.list()
        if not registrations:
            return JCVerificationResult.unknown(checked_at, "SOURCE_UNAVAILABLE")
        historical_query = match_date < checked_at.date()
        errors: list[str] = []
        for registration in registrations:
            try:
                rows = (self.cache.get(match_date,
                                       source=registration.provider.provider_id,
                                       now=checked_at)
                        if self.cache is not None else None)
                if rows is None:
                    if historical_query:
                        errors.append("HISTORICAL_CACHE_MISS")
                        continue
                    fetched = tuple(registration.provider.get_daily_matches(match_date))
                    if self.cache is not None:
                        self.cache.put(match_date, fetched, checked_at=checked_at,
                                       source=registration.provider.provider_id,
                                       complete_coverage=registration.complete_daily_coverage)
                    rows = fetched
            except Exception as error:  # noqa: BLE001 - isolate a failed provider and continue.
                errors.append(type(error).__name__)
                continue
            candidate = self.matcher.match(query, rows)
            if candidate is not None:
                tier = ProviderTier(registration.provider.tier)
                confidence = confidence_for(tier,
                    exact_kickoff=candidate.exact_kickoff,
                    exact_competition=candidate.exact_competition)
                status = JCStatus.CONFIRMED if tier == ProviderTier.OFFICIAL \
                    and candidate.exact_kickoff \
                    else JCStatus.LIKELY
                return JCVerificationResult(status, confidence,
                    candidate.competition_id, candidate.match.competition,
                    candidate.match.kickoff_time, candidate.match.source,
                    candidate.match.evidence_id, checked_at,
                    "EXACT_PROVIDER_FIXTURE_MATCH" if candidate.exact_kickoff
                    else "PROVIDER_FIXTURE_MATCH_WITHIN_KICKOFF_WINDOW")
            # NOT_JC is allowed only with explicitly complete daily coverage and
            # a positively identified youth fixture; absence alone is not evidence.
            if registration.complete_daily_coverage:
                home_identity = self.matcher.teams.resolve(query.home_team_name)
                away_identity = self.matcher.teams.resolve(query.away_team_name)
                youth = (home_identity is not None and away_identity is not None
                         and home_identity.age_group not in {"SENIOR", "UNKNOWN"}
                         and away_identity.age_group not in {"SENIOR", "UNKNOWN"})
                if youth:
                    return JCVerificationResult(JCStatus.NOT_JC, 0.99,
                        query.competition_id, query.competition_name, query.kickoff_time,
                        registration.provider.provider_id, None, checked_at,
                        "EXPLICIT_YOUTH_TEAMS_OUTSIDE_JC_SCOPE")
        reason = ("HISTORICAL_CACHE_MISS_READ_ONLY" if historical_query and errors
                  else "SOURCE_UNAVAILABLE" if errors else "NO_VERIFIED_MATCH")
        return JCVerificationResult.unknown(checked_at, reason)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("JC_CHECK_TIMEZONE_REQUIRED")
    return value.astimezone(UTC)
