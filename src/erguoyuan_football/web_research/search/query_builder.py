"""Deterministic research task generation from explicitly named teams."""

from __future__ import annotations

import hashlib

from erguoyuan_football.web_research.search.search_task import SearchTask


class QueryBuilder:
    """Build plans only; no search, fetching, or model feature injection."""

    @staticmethod
    def build(
        home_team: str,
        away_team: str,
        competition: str | None = None,
        *,
        fixture_key: str | None = None,
    ) -> tuple[SearchTask, ...]:
        """Plan tasks; a verified fixture key isolates repeated team pairings."""
        home, away = " ".join(home_team.split()), " ".join(away_team.split())
        if not home or not away or home.casefold() == away.casefold():
            raise ValueError("TWO_DISTINCT_TEAMS_REQUIRED")
        if fixture_key is not None and not fixture_key.strip():
            raise ValueError("NONEMPTY_FIXTURE_KEY_REQUIRED")
        event = f"{home} vs {away}"
        if competition and competition.strip():
            event = f"{event} {competition.strip()}"
        specifications = (
            ("FIXTURE", f"{event} official fixture kickoff", ("OFFICIAL", "STRUCTURED")),
            ("INJURY", f"{home} {away} injury availability", ("OFFICIAL", "MEDIA")),
            ("LINEUP", f"{event} official lineup", ("OFFICIAL",)),
            ("ODDS", f"{event} bookmaker odds", ("STRUCTURED",)),
            ("TEAM_STATS", f"{home} {away} xG xGA statistics", ("STRUCTURED",)),
            ("NEWS", f"{event} team news press conference", ("OFFICIAL", "MEDIA")),
        )
        tasks: list[SearchTask] = []
        for task_type, query, required_sources in specifications:
            digest = hashlib.sha256(
                f"{task_type}\x1f{query}\x1f{','.join(required_sources)}"
                f"\x1f{fixture_key or ''}".encode()
            ).hexdigest()
            tasks.append(SearchTask(digest, task_type, query, required_sources))
        return tuple(tasks)
