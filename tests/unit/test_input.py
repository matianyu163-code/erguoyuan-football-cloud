from datetime import timedelta

import pytest

from erguoyuan_football.data.schemas import Fixture, TeamAlias
from erguoyuan_football.input.batch_parser import BatchParser
from erguoyuan_football.input.match_resolver import MatchResolver
from erguoyuan_football.input.schemas import MatchRequest
from erguoyuan_football.input.screenshot_parser import ScreenshotParser, VisionMatch
from erguoyuan_football.input.text_parser import TextParser


@pytest.mark.parametrize("value", ["周二001\n阿森纳 VS 切尔西", "周二001 阿森纳VS切尔西", "阿森纳 切尔西", "Arsenal Chelsea"])
def test_text_match_parser(store, at, value):
    requests = TextParser().parse(value)
    assert len(requests) == 1
    resolved = MatchResolver(store).resolve(requests[0], prediction_time=at)
    assert resolved.resolution_status == "RESOLVED"
    assert resolved.match_id == "test_match_1"


def test_batch_match_parser(store, at):
    requests = BatchParser().parse("周二001 阿森纳VS切尔西\n周二002 皇马VS巴萨\n周二003 AC米兰VS国际米兰")
    assert len(requests) == 3
    assert [MatchResolver(store).resolve(r, prediction_time=at).match_id for r in requests] == [
        "test_match_1", "test_match_2", "test_match_3"]
    assert len(BatchParser().parse("周二001\n阿森纳 VS 切尔西\n\n周二002 皇马VS巴萨")) == 2


@pytest.mark.parametrize("name", ["Manchester United", "Man United", "Man Utd", "曼联"])
def test_team_alias_resolution(store, at, name):
    assert store.alias_ids(name, at) == ("test_united",)


@pytest.mark.parametrize("name", ["皇马", "皇家马德里", "Real Madrid"])
def test_madrid_alias(store, at, name):
    assert store.alias_ids(name, at) == ("test_madrid",)


@pytest.mark.parametrize("name", ["Champions League", "欧冠", "UEFA Champions League"])
def test_competition_alias(store, at, name):
    assert store.alias_ids(name, at, competition=True) == ("test_ucl",)


def test_ambiguous_team_rejected(store, at):
    for team in ("test_arsenal", "test_chelsea"):
        store.add_team_alias(TeamAlias(alias="同名球队", team_id=team, language="zh", source="SYNTHETIC_TEST",
                                     confidence=1, created_at=at))
    result = MatchResolver(store).resolve(TextParser().parse("同名球队")[0], prediction_time=at)
    assert result.resolution_status == "AMBIGUOUS"
    assert result.match_id is None


def test_match_resolution(store, at):
    resolver = MatchResolver(store)
    assert resolver.resolve(TextParser().parse("阿森纳")[0], prediction_time=at).match_id == "test_match_1"
    assert resolver.resolve(TextParser().parse("阿森那")[0], prediction_time=at).resolution_status == "NOT_FOUND"
    first = store.fixture_at("test_match_1", at)
    store.add_fixture(Fixture(**{**first.model_dump(), "match_id": "test_rematch", "lottery_match_no": "周三001",
                                "kickoff_time": at + timedelta(days=2)}))
    result = resolver.resolve(TextParser().parse("阿森纳")[0], prediction_time=at)
    assert result.resolution_status == "AMBIGUOUS"
    assert resolver.resolve(TextParser().parse("阿森纳")[0], prediction_time=at, match_date=at.date()).match_id == "test_match_1"


def test_future_alias_rejected(store, at):
    store.add_team_alias(TeamAlias(alias="枪手", team_id="test_arsenal", language="zh", source="SYNTHETIC_TEST",
                                 confidence=1, created_at=at + timedelta(seconds=1)))
    assert store.alias_ids("枪手", at) == ()


@pytest.mark.parametrize("value", ["", "   ", "阿森纳 VS", "阿森纳VS切尔西VS皇马"])
def test_invalid_inputs(value):
    assert TextParser().parse(value)[0].resolution_status == "INVALID"


def test_screenshot_unavailable(tmp_path):
    image = tmp_path / "synthetic.png"
    image.write_bytes(b"SYNTHETIC_TEST_PROVIDER_INPUT_NOT_REAL_IMAGE")
    result = ScreenshotParser().parse(str(image))[0]
    assert result.reason == "VISION_PROVIDER_UNAVAILABLE"
    assert result.match_id is None
    assert ScreenshotParser().parse(str(tmp_path / "missing.png"))[0].resolution_status == "INVALID"


def test_low_confidence_vision_rejected(store, at, tmp_path):
    class FakeVision:
        def extract(self, path):
            return (VisionMatch(home_team="阿森纳", away_team="切尔西", confidence=0.2),)
    image = tmp_path / "synthetic.png"
    image.touch()
    request = ScreenshotParser(FakeVision()).parse(str(image))[0]
    result = MatchResolver(store).resolve(request, prediction_time=at)
    assert result.resolution_status == "AMBIGUOUS"
    assert result.reason == "LOW_VISION_CONFIDENCE"


def test_screenshot_odds_provenance(store, at, tmp_path):
    class FakeVision:
        def extract(self, path):
            return (VisionMatch(lottery_match_no="周二001", home_team="阿森纳", away_team="切尔西",
                                confidence=1, as_of_time=at, home_odds=2, draw_odds=3, away_odds=4),)
    image = tmp_path / "synthetic.png"
    image.touch()
    parser = ScreenshotParser(FakeVision())
    requests, evidence = parser.parse_with_evidence(str(image))
    result = MatchResolver(store).resolve(requests[0], prediction_time=at)
    odds = parser.odds(result, evidence[result.request_id], retrieved_at=at)
    assert len(odds) == 3
    assert all(o.source == "USER_SCREENSHOT" and o.phase == "CURRENT" and o.as_of_time == at for o in odds)
    unknown_time = VisionMatch(**{**evidence[result.request_id].model_dump(), "as_of_time": None})
    assert parser.odds(result, unknown_time, retrieved_at=at) == ()


def test_request_requires_canonical_fields():
    with pytest.raises(ValueError):
        MatchRequest(input_type="MANUAL", source="USER_MANUAL", resolution_status="RESOLVED")
