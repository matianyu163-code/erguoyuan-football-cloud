# Phase 3 base model engine

Phase 3 introduces five independently executable V1 models:

| model_id | implementation | score matrix | adapter / algorithm |
|---|---|---:|---|
| `DIXON_COLES_V1` | `REAL_IMPLEMENTATION` | yes | penaltyblog public Dixon–Coles adapter |
| `BIVARIATE_POISSON_V1` | `REAL_IMPLEMENTATION` | yes | penaltyblog public shared-component Bivariate Poisson adapter |
| `BAYESIAN_HIERARCHICAL_V1` | `REAL_IMPLEMENTATION` | yes | penaltyblog public hierarchical Bayesian MCMC adapter |
| `ELO_V1` | `REAL_IMPLEMENTATION` | no | chronological Elo plus fitted multinomial mapper |
| `PI_RATING_V1` | `REAL_IMPLEMENTATION` | no | separate home/away Pi-rating update plus fitted mapper |

The public API is `BaseFootballModel.fit`, `predict`, `predict_many`, `save`, `load`,
`get_metadata`, `validate_training_data`, and `validate_prediction_input`. `ModelRegistry`
only instantiates registered classes. `ModelRunner` checks the frozen
`DataAvailabilityReport`, fits each model independently, and emits a null-valued
`UNAVAILABLE` or `FAILED` record when that model cannot run.

Training rows are validated before fitting. A row must have a completed result,
known team IDs, a bounded score, and result availability no later than `trained_until`.
The repository's integration fixture is explicitly labelled `SYNTHETIC_TEST` and requires
`allow_test_data=True`; production configuration rejects that label.
Predictions require `trained_until <= prediction_time < kickoff_time` and reject a target
match present in the training IDs. The `PredictionSnapshot.input_data_version` and the
model artifact manifest provide the audit link from a result back to its frozen inputs.

Goal models return the common `ScoreMatrix` (home-goals rows, away-goals columns). A
truncated matrix is conditionally renormalized and records `retained_mass` and `tail_mass`;
all 1X2 and score-derived markets are summed from that same matrix. Ratings do not invent
goal rates or score matrices; their probabilities come from a separately fitted,
chronologically bounded multinomial mapper.

`backtesting.base_model_backtest` provides expanding walk-forward windows and evaluates
only records marked `is_oos=True`. It reports Log Loss, multiclass Brier, RPS and auxiliary
accuracy, together with sample size, competition, date range and model version. The two
benchmarks are league outcome frequency and a simple independent Poisson model.

Hierarchical Bayesian development settings are in `configs/models.yaml`; production
settings remain separate. Sampling diagnostics are recorded and a failed diagnostic
never degrades to random or fixed probabilities. The current V5.1 renderer remains a
frozen presentation contract and reports `NOT_READY_FOR_PRODUCTION` while later META,
calibration, market and external-data stages are unimplemented.
