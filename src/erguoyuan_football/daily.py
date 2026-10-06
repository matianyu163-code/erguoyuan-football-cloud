"""Daily offline entry: parse -> resolve -> freeze -> report -> explicit stubs."""

from datetime import date

from pydantic import model_validator

from erguoyuan_football.contracts.common import Contract, UTCTime
from erguoyuan_football.contracts.predictions import CorePrediction
from erguoyuan_football.data.snapshots import PredictionSnapshot, SnapshotService
from erguoyuan_football.input.batch_parser import BatchParser
from erguoyuan_football.input.match_resolver import MatchResolver
from erguoyuan_football.input.schemas import InputType, MatchRequest, ResolutionStatus
from erguoyuan_football.input.screenshot_parser import ScreenshotParser
from erguoyuan_football.input.text_parser import TextParser
from erguoyuan_football.models.requirements import (
    REQUIREMENTS,
    RequirementDecision,
    check_requirements,
)
from erguoyuan_football.output.v51.schema import SummaryRow, V51Output
from erguoyuan_football.pipeline import (
    Calibration,
    MetaStacking,
    ModelPipeline,
    StageResult,
)


class DailyPredictionRequest(Contract):
    input_type: InputType
    prediction_time: UTCTime
    raw_text: str | None = None
    image_path: str | None = None
    matches: tuple[str, ...] = ()
    match_date: date | None = None  # Explicit UTC date filter; never infer today's date.

    @model_validator(mode="after")
    def valid_input(self):
        if self.input_type == InputType.SCREENSHOT:
            if not self.image_path or self.raw_text is not None or self.matches:
                raise ValueError("screenshot requires only image_path")
        elif self.image_path or (self.raw_text is None and not self.matches):
            raise ValueError("text/manual/batch requires text or matches")
        elif self.raw_text is not None and self.matches:
            raise ValueError("choose raw_text or matches")
        elif self.matches and self.input_type != InputType.BATCH:
            raise ValueError("matches list requires BATCH")
        return self


class DailyResult(Contract):
    requests: tuple[MatchRequest, ...]
    snapshots: tuple[PredictionSnapshot, ...]
    requirements: tuple[RequirementDecision, ...]
    stages: tuple[StageResult, ...]
    core_predictions: tuple[CorePrediction, ...]
    output: V51Output


class DailyService:
    def __init__(self, store, screenshot_parser: ScreenshotParser | None = None):
        self.store = store
        self.screenshot_parser = screenshot_parser or ScreenshotParser()

    def run(self, request: DailyPredictionRequest) -> DailyResult:
        evidence = {}
        if request.input_type == InputType.SCREENSHOT:
            parsed, evidence = self.screenshot_parser.parse_with_evidence(request.image_path)
        elif request.input_type == InputType.BATCH:
            parsed = BatchParser().parse(list(request.matches) if request.matches else request.raw_text)
        elif request.raw_text and len([line for line in request.raw_text.splitlines() if line.strip()]) > 1:
            parsed = BatchParser().parse(request.raw_text)
        else:
            parsed = TextParser().parse(request.raw_text or "", input_type=request.input_type)
        resolved, snapshots, requirements, core, summary = [], [], [], [], []
        seen = set()
        for item in parsed:
            item = MatchResolver(self.store, min_input_confidence=self.screenshot_parser.min_confidence).resolve(item, prediction_time=request.prediction_time,
                                                      match_date=request.match_date)
            self.store.save_request(item)
            resolved.append(item)
            if item.resolution_status != ResolutionStatus.RESOLVED or item.match_id in seen:
                continue
            seen.add(item.match_id)
            if item.request_id in evidence:
                for quote in self.screenshot_parser.odds(item, evidence[item.request_id]):
                    self.store.add_snapshot("odds_snapshots", quote)
            snapshot = SnapshotService(self.store).create(item.match_id, request.prediction_time)
            snapshots.append(snapshot)
            requirements.extend(check_requirements(model_id, snapshot.data_completeness) for model_id in REQUIREMENTS)
            core.append(CorePrediction(match_id=item.match_id, prediction_snapshot_id=snapshot.prediction_snapshot_id,
                                       prediction_time=snapshot.prediction_time))
            summary.append(SummaryRow(number=item.lottery_match_no or item.match_id,
                                      match=f"{item.home_team_name} VS {item.away_team_name}"))
        stages = (ModelPipeline().run(tuple(snapshots)), MetaStacking().run(None), Calibration().run(None))
        return DailyResult(requests=tuple(resolved), snapshots=tuple(snapshots), requirements=tuple(requirements),
                           stages=stages, core_predictions=tuple(core),
                           output=V51Output(freeze_time=request.prediction_time, summary=tuple(summary)))
