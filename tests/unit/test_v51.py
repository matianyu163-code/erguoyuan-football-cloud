import json

import pytest

from erguoyuan_football.output.v51.renderer import render
from erguoyuan_football.output.v51.schema import (
    CATEGORIES,
    MatchRow,
    ParlayFields,
    ResultGroup,
    SummaryRow,
    V51Output,
)
from erguoyuan_football.output.v51.validator import validate_output


def test_v51_schema(at, project_root):
    expected = json.loads((project_root / "tests/golden/v51/columns.json").read_text(encoding="utf-8"))
    output = V51Output(freeze_time=at)
    row = MatchRow(match_code="SYNTHETIC_TEST", competition="TEST", matchup="TEST_A VS TEST_B", market="TEST", direction="TEST")
    assert list(output.model_dump(by_alias=True)) == expected["top"]
    assert list(SummaryRow(number="SYNTHETIC_TEST", match="TEST").model_dump(by_alias=True)) == expected["summary"]
    assert list(row.model_dump(by_alias=True)) == expected["match"]
    assert list(ParlayFields().model_dump(by_alias=True)) == expected["parlay"]


def test_v51_golden_output(at, project_root):
    value = V51Output(freeze_time=at, summary=(SummaryRow(number="SYNTHETIC_TEST_001", match="测试主队 VS 测试客队"),))
    expected = (project_root / "tests/golden/v51/no_bet.json").read_text(encoding="utf-8")
    assert render(value) == expected
    assert validate_output(json.loads(expected)) == value


@pytest.mark.parametrize("mutation", ["missing", "reordered", "renamed", "nested_missing", "nested_order"])
def test_v51_structure_changes_fail(at, mutation):
    value = V51Output(freeze_time=at, summary=(SummaryRow(number="TEST", match="TEST"),)).model_dump(mode="json", by_alias=True)
    if mutation == "missing":
        del value["半全场"]
    elif mutation == "reordered":
        value = dict(reversed(list(value.items())))
    elif mutation == "renamed":
        value["总进球"] = value.pop("总进球数")
    elif mutation == "nested_missing":
        del value["今日比赛筛选总表"][0]["MDI"]
    else:
        value["胜平负二串一"]["串关"] = dict(reversed(list(value["胜平负二串一"]["串关"].items())))
    with pytest.raises(ValueError):
        validate_output(value)


def test_no_bet_forbids_selections():
    row = MatchRow(match_code="SYNTHETIC_TEST", competition="TEST", matchup="A VS B", market="TEST", direction="TEST")
    with pytest.raises(ValueError):
        ResultGroup(matches=(row,))


def test_qualified_group_is_only_a_contract(at):
    rows = tuple(MatchRow(match_code=f"SYNTHETIC_TEST_{i}", competition="TEST", matchup="A VS B", market="TEST", direction="TEST") for i in range(2))
    group = ResultGroup(conclusion="QUALIFIED", matches=rows)
    value = V51Output(freeze_time=at, two_leg=group)
    assert len(validate_output(json.loads(render(value))).two_leg.matches) == 2
    with pytest.raises(ValueError):
        V51Output(freeze_time=at, three_leg=group)


def test_no_bet_stake_rejected(at):
    with pytest.raises(ValueError):
        V51Output(freeze_time=at, group_amounts={name: 10 for name in CATEGORIES})
