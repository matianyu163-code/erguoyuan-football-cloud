# YY-CORE R3 Cloud runtime

This repository contains a headless, frozen blind-test runner. The checked-in
multi-model release is `R3_RELEASE_V2_BRAZIL_20261007`; it keeps
`YY_CORE_PRODUCTION_READY = FALSE` and enables R3 only for registered model
domains. The release contains one national Dixon-Coles artifact and three frozen
Brazil Serie A artifacts. It does not promote the project to Production.

## Install and verify

Use Python 3.11. From a clean checkout, run:

```sh
python -m pip install -e ".[dev]"
python scripts/init_cloud_state.py
python scripts/check_cloud_readiness.py
```

The readiness command checks installed dependencies, the offline test suite,
national and Brazil Golden references, empty state initialization, source-path
portability, and a synthetic daily inference. It does not use real fixtures or
write into the persisted prediction ledger.

## Run a user slate

Save the user-authoritative JSON under `inputs/daily_market/`, then run:

```sh
python scripts/run_daily_prediction.py inputs/daily_market/today.json
```

The router loads only artifacts bound to
`cloud_release/verified_multi_model_release.json`. Brazil fixtures are routed
to every compatible frozen Brazil model; available H/D/A probabilities are
equally weighted. Dixon-Coles and Bivariate Poisson score matrices are averaged
separately for score, total-goal, and handicap derivations. Elo contributes only
its actual H/D/A output. No runtime fitting occurs. Unsupported competitions
and incomplete inputs stay as model failures with append-only audit locks.

User-provided JC prices remain unverified external-market data. They are saved
with their source and used only by the existing fixed 70/30 Fusion output. The
Finland fixture in the 2026-10-07 slate has no compatible frozen artifact and
must remain `MODEL_EXECUTION_FAILED`.

Successful predictions, market snapshots, and locks are appended under
`cloud_state/`. Failure audits and their locks are retained. Never edit or
delete a saved prediction or lock; report corrections as separate audit events.

## Cloud and phone access

`scripts/verify_r3_cloud_portability.py` verifies a clean temporary source copy,
the checked-in release, both Brazil Golden references, and a synthetic Brazil
prediction without importing project source from another checkout. The current
verification uses the available Python 3.11 environment for dependencies; it
does not reinstall all dependencies or test a Git clone.

No Codex Cloud environment, hosted prediction API, or Git remote is configured
in the current workspace. A phone cannot submit a slate to a remote runtime
until a hosted environment and endpoint are configured. This does not affect
local headless R3 execution. Formal Production remains disabled.
