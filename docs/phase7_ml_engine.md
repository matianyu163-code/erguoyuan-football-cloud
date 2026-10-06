# Phase 7 ML engine V1

`CORE_XGBOOST_V1` and `CORE_CATBOOST_V1` are independent official-library multiclass
1X2 wrappers. Their `ModelPrediction` probabilities are raw and uncalibrated. They do
not produce expected goals or a score matrix. The existing V5.1 output contract is
unchanged, and the daily production pipeline does not consume these models yet.

## Frozen input and labels

`MLFeatureBuilder` accepts only an existing `PredictionSnapshot`, audited base-model
prediction evidence and an explicit feature mode. It performs no network access.
`MLFeatureVector` contains the snapshot ID, input-data version, prediction time,
kickoff, ordered feature schema hash, content hash and per-source lineage. Both
`as_of_time` and `retrieved_at` must be no later than prediction time. A completed
target match may only enter the *label* side of a later training dataset; it may
never enter its own feature vector.

The class order is fixed: HOME=0, DRAW=1, AWAY=2. All model probabilities are
reordered by the estimator's actual `classes_` and checked for finite mass. A
float32 rounding correction is applied only after the unadjusted mass passes the
1e-6 check; this is not probability calibration.

Base-model prediction features require matching match/snapshot/data-version/time,
an OOS flag, target-match exclusion from training IDs, an earlier training cutoff
and a training-ID hash matching the prediction's own metadata. At fit time the
persisted evidence is checked again. New base-model outputs include this hash;
legacy outputs without it cannot be admitted as Phase 7 training features.

`NO_MARKET` excludes both raw market consensus and market-dependent base-model
probabilities. `WITH_MARKET` includes only a frozen, quote-backed 1X2 consensus;
its market quote source and retrieval times must pass the snapshot's PIT check.
Closing odds are never substituted for an earlier forecast. Missing numeric
values remain NaN for the native learner; missing CatBoost categories use the
explicit `__MISSING__` token. Team IDs are excluded from V1 features.

## Training and artifacts

`MLDatasetBuilder` creates one labeled row per match and refuses labels that were
not available by the supplied dataset cutoff. `MLWalkForwardSplit` makes expanding,
strictly chronological train/validation/test partitions. Train labels must be
available before validation prediction, and validation labels before test
prediction. Early stopping sees only the validation partition. The test partition
is frozen by content hash and is never passed to either official library's `fit`.
There is no parameter search or tuning against final-test results in V1.

`CoreXGBoostModel` uses `XGBClassifier` with `multi:softprob`; `CoreCatBoostModel`
uses `CatBoostClassifier` and native `Pool` categorical support. Both have named
development and production profiles in `config/xgboost.yaml` and
`config/catboost.yaml`. The production profile has a higher row requirement and
more iterations; this is a starting configuration, not performance-validated.
`MLArtifact` stores native model bytes, checksum, model/library/code versions,
feature schema, class mapping, config hash, dataset hash, training cutoff and
drift profile. Without an explicit release SHA, the code version is a SHA256 of
the Phase 7 Python source files. Loading checks this version, the manifest and
native-file checksum.

`MLModelRunner` isolates model failures. A missing vector returns `UNAVAILABLE`;
schema mismatch or corrupt artifacts return `FAILED`. A synthetic-test-trained
model cannot be used with `production=True`, even when external gates are marked
PASS. Real production also requires the existing network and market gates.

## Evaluation and limits

`MLWalkForwardBacktester` refits for every expanding window and scores only
successful OOS test predictions. It reports Log Loss, multiclass Brier, RPS,
top-class ECE, auxiliary Accuracy and draw diagnostics, plus competition, season,
prediction-horizon, market availability, base-feature availability and favorite
bucket breakdowns. `MLFeatureAblation` creates new immutable datasets for ratings,
ratings+goal models, ratings+goals+form/xG, full no-market and full with-market
experiments. `MLComparisonReport` shows XGBoost/CatBoost/base/market probabilities
and their disagreement as an internal diagnostic; it never blends them.
`compare_model_benchmarks` accepts only complete, same-snapshot OOS prediction
records for identical target matches. Missing baselines return `UNAVAILABLE`;
the scorer does not synthesize league frequencies or market probabilities.

The repository's `data/football.duckdb` has zero recorded matches, results and
markets as of Phase 7 review. Current executed tests use explicitly labeled
`SYNTHETIC_TEST` fixtures. They establish software behavior, including real
official-library training and OOS execution, but provide no real-league benchmark,
closing-market comparison or production-readiness evidence. See the Phase 7
completion report before interpreting any test metric as forecast performance.
