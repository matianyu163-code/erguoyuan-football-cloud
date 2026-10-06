"""Explicit future pipeline boundaries; never substitute random predictions."""

from typing import Literal

from erguoyuan_football.contracts.common import Contract


class StageResult(Contract):
    stage: str
    status: Literal["NOT_IMPLEMENTED"] = "NOT_IMPLEMENTED"


class ModelPipeline:
    def run(self, snapshot):
        return StageResult(stage="MODEL_PIPELINE")


class MetaStacking:
    def run(self, inputs):
        return StageResult(stage="META_STACKING")


class Calibration:
    def run(self, inputs):
        return StageResult(stage="CALIBRATION")
