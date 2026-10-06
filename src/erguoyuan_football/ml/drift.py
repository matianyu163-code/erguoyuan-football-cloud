"""Diagnostic distribution profiles; drift never changes probabilities in Phase 7."""

from __future__ import annotations

import numpy as np

from erguoyuan_football.ml.schemas import (
    FeatureDriftReport,
    FeatureSchema,
    MLFeatureVector,
)


def training_drift_profile(vectors: tuple[MLFeatureVector, ...], schema: FeatureSchema) -> FeatureDriftReport:
    numeric = {}
    for name in schema.numeric_features:
        values = np.asarray([float(value) for vector in vectors
                             if isinstance(value := vector.features[name], int | float)], dtype=float)
        numeric[name] = {"mean": float(values.mean()) if len(values) else None,
                         "std": float(values.std()) if len(values) else None,
                         "q01": float(np.quantile(values, .01)) if len(values) else None,
                         "q99": float(np.quantile(values, .99)) if len(values) else None,
                         "missing_rate": 1 - len(values) / len(vectors)}
    categories = {name: tuple(sorted({str(vector.features[name]) for vector in vectors
                                      if vector.features[name] is not None}))
                  for name in schema.categorical_features}
    return FeatureDriftReport(feature_schema_hash=schema.schema_hash,
        numeric_distribution=numeric, categorical_known_values=categories,
        sample_size=len(vectors))


def inference_drift_warnings(vector: MLFeatureVector, report: FeatureDriftReport) -> tuple[str, ...]:
    warnings = []
    for name, stats in report.numeric_distribution.items():
        value = vector.features[name]
        if value is None:
            continue
        mean, std = stats["mean"], stats["std"]
        if mean is not None and std is not None and std > 0 and abs(float(value) - mean) > 5 * std:
            warnings.append(f"DRIFT_WARNING:{name}")
    for name, known in report.categorical_known_values.items():
        value = vector.features[name]
        if value is not None and str(value) not in known:
            warnings.append(f"DRIFT_WARNING:UNKNOWN_CATEGORY:{name}")
    return tuple(warnings)
