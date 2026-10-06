"""Native library model files with checked, versioned CORE manifests."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import Contract, Identifier, UTCTime, now
from erguoyuan_football.ml.labels import CLASS_MAPPING
from erguoyuan_football.ml.schemas import FeatureDriftReport, FeatureSchema


def current_code_version() -> str:
    """Bind an artifact to Phase 7 source bytes when no release SHA is supplied."""
    explicit = os.environ.get("CORE_CODE_VERSION")
    if explicit:
        return explicit
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return "ML_SOURCE_SHA256:" + digest.hexdigest()


class MLArtifact(Contract):
    model_id: Identifier
    model_version: Identifier
    library_name: Identifier
    library_version: Identifier
    code_version: Identifier = Field(default_factory=current_code_version)
    feature_version: Identifier
    feature_schema_hash: Identifier
    feature_schema: FeatureSchema
    config_hash: Identifier
    config_payload: dict[str, Any]
    dataset_hash: Identifier
    trained_until: UTCTime
    class_mapping: dict[str, int]
    training_match_ids: tuple[Identifier, ...]
    training_metadata: dict[str, Any]
    drift_report: FeatureDriftReport
    created_at: UTCTime = Field(default_factory=now)
    artifact_path: Identifier
    model_file: Identifier
    payload_sha256: Identifier

    @model_validator(mode="after")
    def identity(self) -> MLArtifact:
        if self.class_mapping != CLASS_MAPPING or self.feature_schema_hash != self.feature_schema.schema_hash:
            raise ValueError("ML_ARTIFACT_SCHEMA_OR_CLASS_MISMATCH")
        if Path(self.model_file).name != self.model_file or self.model_file in {".", ".."}:
            raise ValueError("ML_ARTIFACT_MODEL_FILE_PATH_INVALID")
        return self


def file_sha256(path: Path) -> str:
    """Check bytes before delegating to a native model loader."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
