"""Syntax only; fixture identity is resolved against a catalog later."""

import re
from typing import Any

from erguoyuan_football.input.schemas import InputType, MatchRequest, ResolutionStatus

NUMBER = re.compile(r"^(周[一二三四五六日天]\d{3})(?:\s*|$)")
VERSUS = re.compile(r"\s*(?:vs|对阵|对|ＶＳ)\s*", re.IGNORECASE)


class TextParser:
    def parse(self, text: str, *, input_type: InputType = InputType.TEXT,
              source: str = "USER_TEXT") -> list[MatchRequest]:
        value = text.strip()
        number = NUMBER.match(value)
        lottery_no = number.group(1) if number else None
        remainder = value[number.end():].strip() if number else value
        parts = VERSUS.split(remainder)
        fields: Any = {"input_type": input_type, "raw_text": text, "source": source,
                       "lottery_match_no": lottery_no}
        if not remainder:
            if lottery_no:
                return [MatchRequest(**fields)]
            return [MatchRequest(**fields, resolution_status=ResolutionStatus.INVALID,
                                 reason="EMPTY_INPUT")]
        if len(parts) == 2 and all(p.strip() for p in parts):
            return [MatchRequest(**fields, home_team_name=parts[0].strip(),
                                 away_team_name=parts[1].strip())]
        if len(parts) > 1:
            return [MatchRequest(**fields, resolution_status=ResolutionStatus.INVALID,
                                 reason="INVALID_MATCH_SYNTAX")]
        return [MatchRequest(**fields, team_query=remainder)]
