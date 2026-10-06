"""Stable JSON renderer only; no probability, admission or stake computation."""

import json

from erguoyuan_football.output.v51.schema import V51Output
from erguoyuan_football.output.v51.validator import validate_output


def render(output: V51Output) -> str:
    value = output.model_dump(mode="json", by_alias=True)
    validate_output(value)
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
