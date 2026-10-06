# Dynamic Bayesian Poisson V1

## Scope and implementation

`DYNAMIC_BAYESIAN_POISSON_V1` is a separate real implementation in Phase 4. It
uses a sequential state-space Poisson model with damped Newton/Laplace posterior
updates. PyMC is not installed in the supported Python 3.11 environment; this
implementation does not claim MCMC sampling. The Poisson likelihood, prior
precision, local Hessian update, uncertainty propagation and posterior
predictive score averaging are executed by project code using NumPy/SciPy.
There is no fitted-form or synthetic-data fallback.

For home and away goals, the linear predictors are:

```text
log(lambda_home) = mu + home_advantage + attack(home, t) - defence(away, t)
log(lambda_away) = mu + attack(away, t) - defence(home, t)
```

Attack and defence parameters are latent log-rate strengths. Higher defence
means stronger prevention and therefore lowers the opposing team's goal rate.
The global intercept and home advantage are estimated from the training
window's observed goal rates, using non-neutral matches for home advantage;
neutral fixtures receive no home-advantage offset. Attack and defence states
update chronologically after a result is available.

## State process and priors

Each team starts from a league-level Normal prior configured by
`dynamic_league_*_mean` and `dynamic_league_*_sd`. An unseen or promoted team
uses that prior and carries its full uncertainty; the common base-model
validation permits unseen teams only for this model.

The configured transition is either `RANDOM_WALK` or `AR1`. Random walk retains
the state mean and adds process variance by the selected elapsed-time unit. AR1
decays the mean toward the league prior and propagates process variance.
`WEEKLY` measures process innovations in seven-day units; `MATCH_EVENT_TIME`
measures them in calendar-day units. Both use exact elapsed time rather than
assuming equal match spacing. Observations update only at result-availability
events. An additional configurable inactivity term increases variance with
elapsed calendar days.

At a season change, state means and variances shrink toward the league prior by
`dynamic_season_transition_weight`. The dynamic parameters are separate from
the static hierarchical sampler parameters. Development and production
posterior-predictive draw counts remain separate in `configs/models.yaml`.

## Point-in-time and history

Every training match is consumed in `(kickoff_time, match_id)` order after the
shared training validator has rejected future or unavailable rows. The
pre-match state is recorded immediately before kickoff. A posterior update
becomes queryable only at `TrainingMatch.available_at`, the maximum of result
completion, source `as_of_time`, and retrieval time. Thus a later kickoff on the
same day cannot see an earlier result before it was available. Prediction
requires `trained_until <= prediction_time < kickoff_time`; the historical
state query also filters by the stored availability timestamp.

## Prediction and diagnostics

Prediction samples attack/defence states from their Normal Laplace posterior
and averages the corresponding independent Poisson score matrices. Truncation
uses the shared `ScoreMatrix.from_raw` conditional-renormalization convention,
which records retained and tail mass. H/D/A probabilities are derived only by
summing that matrix. The record includes posterior means and standard
deviations, 90% expected-goals intervals, expected goals, uncertainty score,
state-change signal, posterior-predictive checks and diagnostics.

The current engine does not run MCMC chains, so `r_hat`, MCMC ESS and
divergences are recorded as null rather than fabricated. The prediction's
posterior draw count is recorded separately. Large posterior-predictive
discrepancy or extreme states produce `WARNING`; severe discrepancies or
extreme states produce `FAILED` with null probability fields. Numerical or
input failures return `FAILED` or `UNAVAILABLE` with null probability fields.

## Artifact and evaluation

The ordinary model artifact checksum and data/config hashes also apply here.
Its manifest records state-process version, time-index version, posterior
method, diagnostics and state-history count. The joblib payload contains the
model state needed to reproduce a query. Walk-forward evaluation uses the
shared time-ordered harness and records OOS status and training/prediction
cutoffs; in-sample predictions are excluded from formal metrics.

No authorized real competition dataset is bundled. Any synthetic fixture is
labelled `SYNTHETIC_TEST` and can only run with the explicit test-data flag, so
this document reports no production accuracy or performance claim.
