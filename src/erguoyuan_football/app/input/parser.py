"""Parse and resolve requests only against real development-match identities."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from erguoyuan_football.app.config import AppConfig
from erguoyuan_football.app.input.match_input import (
    MatchInputParserV2,
    MatchRequest,
    normalize_team_name,
)
from erguoyuan_football.input.screenshot_parser import ScreenshotParser
from erguoyuan_football.knowledge.match_identity import GlobalKnowledgeResolver

_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*[|｜]\s*(.+)$")
_COMPETITIONS = {"英超": "EPL", "西甲": "LALIGA", "德甲": "BUNDESLIGA",
                 "意甲": "SERIE_A", "法甲": "LIGUE_1", "欧冠": "CHAMPIONS_LEAGUE"}


@dataclass(frozen=True)
class _CatalogRow:
    match_id: str
    competition: str
    match_date: date
    home: str
    away: str


def _norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", normalize_team_name(value)).casefold().split())


class MatchInputResolver:
    """Exact identity and ambiguity checks over development-only real rows."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.syntax = MatchInputParserV2()

    def _catalog(self) -> tuple[_CatalogRow, ...]:
        ids = []
        with self.config.development_predictions.open(encoding="utf-8") as handle:
            import json
            for line in handle:
                ids.append(str(json.loads(line)["match_id"]))
        with duckdb.connect(str(self.config.database), read_only=True) as db:
            rows = db.execute("""SELECT m.match_id,m.competition_id,m.match_date,
                h.team_name,a.team_name FROM real_canonical_matches m
                JOIN real_canonical_teams h ON h.team_id=m.home_team_id
                JOIN real_canonical_teams a ON a.team_id=m.away_team_id
                WHERE m.match_id IN (SELECT unnest(?)) AND m.match_date < DATE '2026-08-01'""",
                [ids]).fetchall()
        return tuple(_CatalogRow(*row) for row in rows)

    def parse_text(self, text: str) -> list[MatchRequest]:
        """Preserve every input line, with optional competition and date qualifiers."""
        catalog = self._catalog()
        result: list[MatchRequest] = []
        competition: str | None = None
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            if line.endswith(("：", ":")):
                name = line[:-1].strip()
                competition = _COMPETITIONS.get(name, name)
                continue
            day = None
            match = _DATE.match(line)
            body = line
            if match:
                try:
                    day = date.fromisoformat(match.group(1))
                except ValueError:
                    result.append(MatchRequest(None, None, None, competition, None,
                        "USER_TEXT", "INVALID", "INVALID_DATE", line))
                    continue
                body = match.group(2)
            if re.fullmatch(r"[a-f0-9]{24}", body):
                choices = [row for row in catalog if row.match_id == body]
                home = away = None
            else:
                syntax = self.syntax.parse(body)
                if syntax.validation_status != "VALID":
                    result.append(MatchRequest(None, None, None, competition, day,
                        "USER_TEXT", "INVALID", syntax.error_code, line))
                    continue
                home, away = syntax.home_team, syntax.away_team
                assert home is not None and away is not None
                choices = [row for row in catalog if _norm(row.home) == _norm(home)
                           and _norm(row.away) == _norm(away)]
            if competition:
                choices = [row for row in choices if _norm(row.competition) == _norm(competition)]
            if day:
                choices = [row for row in choices if row.match_date == day]
            if len(choices) == 1:
                row = choices[0]
                result.append(MatchRequest(row.match_id, row.home, row.away,
                    row.competition, row.match_date, "USER_TEXT", "RESOLVED", raw_text=line))
            else:
                known_teams = {_norm(team) for row in catalog for team in (row.home, row.away)}
                missing_team = (home is not None and away is not None and
                                (_norm(home) not in known_teams or _norm(away) not in known_teams))
                reason = "MATCH_NOT_FOUND"
                if missing_team and home is not None and away is not None:
                    identity = GlobalKnowledgeResolver().resolve_names(home, away, competition)
                    reason = ("FIXTURE_NOT_FOUND" if
                              identity.status == "IDENTITIES_RESOLVED_NO_FIXTURE"
                              else "TEAM_NOT_FOUND")
                result.append(MatchRequest(None, home, away, competition, day,
                    "USER_TEXT", "AMBIGUOUS" if choices else "NOT_FOUND",
                    "MULTIPLE_EXACT_MATCHES" if choices else
                    reason if missing_team else "MATCH_NOT_FOUND", line))
        if not result:
            result.append(MatchRequest(None, None, None, None, None,
                "USER_TEXT", "INVALID", "EMPTY_INPUT", text))
        return result

    @staticmethod
    def parse_image(path: Path) -> list[MatchRequest]:
        """Never treat unconfirmed OCR as prediction input."""
        request = ScreenshotParser().parse(str(path))[0]
        return [MatchRequest(None, request.home_team_name, request.away_team_name,
            request.competition_name, None, "USER_SCREENSHOT", "INPUT_REVIEW_REQUIRED",
            request.reason or "OCR_REQUIRES_HUMAN_CONFIRMATION", str(path))]
