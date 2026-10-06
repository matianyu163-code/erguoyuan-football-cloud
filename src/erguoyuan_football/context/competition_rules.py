"""Explicit, season-versioned competition rule registry."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

import yaml

from erguoyuan_football.context.schemas import CompetitionRule


class CompetitionRuleRegistry:
    """Resolve only verified rules with a matching competition, season and date."""

    def __init__(self, rules: Iterable[CompetitionRule] = ()) -> None:
        self._rules = tuple(rules)
        keys: set[tuple[str, str, str]] = set()
        for rule in self._rules:
            key = (rule.competition_id, rule.season_id, rule.rule_version)
            if key in keys:
                raise ValueError("DUPLICATE_COMPETITION_RULE_VERSION")
            keys.add(key)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CompetitionRuleRegistry:
        """Load only explicitly declared, Pydantic-validated rule records."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        rules = [CompetitionRule.model_validate(item) for item in payload.get("rules", [])]
        return cls(rules)

    def resolve(self, competition_id: str, season_id: str,
                on_date: date, *, prediction_time: datetime | None = None) -> CompetitionRule | None:
        """Return a single verified rule; ambiguous and unverified entries fail closed."""
        candidates = [rule for rule in self._rules
                      if rule.competition_id == competition_id and rule.season_id == season_id
                      and rule.valid_from <= on_date <= rule.valid_to and rule.verified
                      and (prediction_time is None or rule.retrieved_at <= prediction_time)]
        if len(candidates) > 1:
            raise ValueError("AMBIGUOUS_COMPETITION_RULE_VERSION")
        return candidates[0] if candidates else None

    def available_rules(self) -> tuple[CompetitionRule, ...]:
        return tuple(rule for rule in self._rules if rule.verified)
