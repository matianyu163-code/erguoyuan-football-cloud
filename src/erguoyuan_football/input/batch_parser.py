"""One match per line, or a lottery number followed by its pairing."""

from erguoyuan_football.input.schemas import InputType, MatchRequest
from erguoyuan_football.input.text_parser import NUMBER, TextParser


class BatchParser:
    def parse(self, value: str | list[str]) -> list[MatchRequest]:
        lines = value.splitlines() if isinstance(value, str) else value
        lines = [line.strip() for line in lines if line.strip()]
        if not lines:
            return TextParser().parse("", input_type=InputType.BATCH)
        result = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if NUMBER.fullmatch(line) and index + 1 < len(lines) and not NUMBER.match(lines[index + 1]):
                index += 1
                line += "\n" + lines[index]
            result.extend(TextParser().parse(line, input_type=InputType.BATCH))
            index += 1
        return result
