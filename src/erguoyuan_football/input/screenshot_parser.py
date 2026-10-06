"""Pluggable vision boundary. No configured provider means no OCR claims."""

from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from erguoyuan_football.contracts.common import Contract, Probability, UTCTime, now
from erguoyuan_football.data.schemas import OddsSnapshot
from erguoyuan_football.input.schemas import InputType, MatchRequest, ResolutionStatus


class VisionMatch(Contract):
    lottery_match_no: str | None = None
    competition_name: str | None = None
    home_team: str
    away_team: str
    kickoff_time: UTCTime | None = None
    confidence: Probability
    # Provider must supply the weakest confidence across all extracted fields.
    as_of_time: UTCTime | None = None
    home_odds: float | None = Field(default=None, gt=1)
    draw_odds: float | None = Field(default=None, gt=1)
    away_odds: float | None = Field(default=None, gt=1)


class VisionProvider(Protocol):
    def extract(self, image_path: Path) -> tuple[VisionMatch, ...]: ...


class ScreenshotParser:
    def __init__(self, provider: VisionProvider | None = None, *, min_confidence: float = 1.0):
        if not 0 <= min_confidence <= 1:
            raise ValueError("confidence threshold must be within [0,1]")
        self.provider = provider
        self.min_confidence = min_confidence

    def parse(self, image_path: str) -> list[MatchRequest]:
        requests, _ = self.parse_with_evidence(image_path)
        return requests

    def parse_with_evidence(self, image_path: str):
        common: Any = {"input_type": InputType.SCREENSHOT, "image_path": image_path,
                       "source": "USER_SCREENSHOT"}
        if not Path(image_path).is_file():
            return [MatchRequest(**common, resolution_status=ResolutionStatus.INVALID, reason="IMAGE_NOT_FOUND")], {}
        if self.provider is None:
            return [MatchRequest(**common, reason="VISION_PROVIDER_UNAVAILABLE")], {}
        try:
            extracted = tuple(VisionMatch.model_validate(v.model_dump()) for v in self.provider.extract(Path(image_path)))
        except (OSError, RuntimeError, ValueError) as error:
            return [MatchRequest(**common, resolution_status=ResolutionStatus.INVALID,
                                 reason=f"VISION_PROVIDER_FAILED:{type(error).__name__}")], {}
        if not extracted:
            return [MatchRequest(**common, reason="NO_MATCHES_DETECTED")], {}
        requests, evidence = [], {}
        for item in extracted:
            low = item.confidence < self.min_confidence
            request = MatchRequest(**common, lottery_match_no=item.lottery_match_no,
                                   competition_name=item.competition_name, home_team_name=item.home_team,
                                   away_team_name=item.away_team, kickoff_time=item.kickoff_time,
                                   input_confidence=item.confidence,
                                   resolution_status=ResolutionStatus.AMBIGUOUS if low else ResolutionStatus.NOT_FOUND,
                                   reason="LOW_VISION_CONFIDENCE" if low else "VISION_VERIFIED")
            requests.append(request)
            evidence[request.request_id] = item
        return requests, evidence

    def odds(self, request: MatchRequest, item: VisionMatch, *, retrieved_at=None) -> tuple[OddsSnapshot, ...]:
        if (request.resolution_status != ResolutionStatus.RESOLVED or not request.match_id
                or item.confidence < self.min_confidence):
            return ()
        if item.as_of_time is None:
            return ()  # A screenshot without an observed timestamp is not dated market evidence.
        if item.confidence != request.input_confidence or item.lottery_match_no != request.lottery_match_no:
            raise ValueError("vision evidence does not match request")
        captured = retrieved_at or now()
        return tuple(OddsSnapshot(match_id=request.match_id, source="USER_SCREENSHOT",
                                  retrieved_at=captured, as_of_time=item.as_of_time,
                                  data_version=request.request_id, bookmaker="SPORTS_LOTTERY",
                                  market_type="SPORTS_LOTTERY", phase="CURRENT", selection=selection,
                                  odds=value)
                     for selection, value in (("HOME", item.home_odds), ("DRAW", item.draw_odds), ("AWAY", item.away_odds))
                     if value is not None)
