"""Feature-order-preserving missing-value conversion for native boosting libraries."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from erguoyuan_football.ml.feature_contract import validate_feature_values
from erguoyuan_football.ml.schemas import FeatureSchema, MLFeatureVector

MISSING_CATEGORY = "__MISSING__"


def feature_frame(vectors: tuple[MLFeatureVector, ...], schema: FeatureSchema) -> pd.DataFrame:
    """Unknown numeric values stay NaN; categorical missing is one explicit token."""
    if not vectors:
        raise ValueError("EMPTY_ML_FEATURE_BATCH")
    rows = []
    for vector in vectors:
        if vector.feature_schema_hash != schema.schema_hash or vector.model_family != schema.model_family:
            raise ValueError("FEATURE_SCHEMA_MISMATCH")
        validate_feature_values(vector.features, schema)
        row: dict[str, str | float] = {}
        for name in schema.feature_names:
            value = vector.features[name]
            if name in schema.categorical_features:
                row[name] = MISSING_CATEGORY if value is None else str(value)
            else:
                row[name] = math.nan if value is None else float(value)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(schema.feature_names))
    if tuple(frame.columns) != schema.feature_names:
        raise ValueError("FEATURE_SCHEMA_MISMATCH")
    if not np.isfinite(frame[list(schema.numeric_features)].to_numpy(dtype=float)[
            ~np.isnan(frame[list(schema.numeric_features)].to_numpy(dtype=float))]).all():
        raise ValueError("NONFINITE_ML_FEATURE")
    return frame
