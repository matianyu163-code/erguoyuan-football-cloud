# YY-CORE Cloud Clean Build

This package runs the independent frozen blind-test route. It does not mark
`config/production.yaml` as Production Ready.
The Desktop UI and export modules are omitted from this Cloud Copy. Shared
fixture input and version metadata modules remain for headless compatibility.

## First use

1. Clone the Cloud Copy Git repository.
2. Use Python 3.11 and run `pip install -e .`.
3. Run `python scripts/init_cloud_state.py` to create empty Cloud Era ledgers.
4. Run `pytest` and `python scripts/run_golden_reference.py`. The Cloud default
   suite covers blind-test, market, model, point-in-time and core unit tests.
   The wider development suite remains available in the archive; those tests
   need historical development files that are intentionally absent here.
   `python scripts/check_cloud_readiness.py` runs the full local Cloud gate.
5. Put a user-supplied daily JSON in `inputs/daily_market/`. The required fields
   are `slate_date`, `source: USER_AUTHORITATIVE`, and a nonempty `fixtures` list.
   Each fixture needs `jc_match_number`, `competition`, teams, timezone-aware
   `kickoff`, SPF H/D/A, RQSPF integer handicap and H/D/A. A known
   `neutral_venue` flag is needed for the frozen national model; an absent flag
   leaves that fixture's model route unavailable while the JC market route runs.
6. Run `python scripts/run_daily_prediction.py inputs/daily_market/today.json`.

The command validates **format** and point-in-time ordering, loads the pinned
national Dixon-Coles artifact, then writes user market snapshots, R3, R5,
fusion, deterministic plans, predictions and exclusive locks. It never fits
a model or rechecks user odds against a website. External research is optional.
For repeated or erroneous submissions, records remain append only. Corrections
must be separate events; existing locks must not be edited.

Model and market probabilities are separate. `MODEL_MARKET_FUSION_V1` uses fixed
70% model / 30% user JC no-vig for available SPF and RQSPF probabilities. This is a
blind-test output, not a Production approval or an EV-validated bet.

`GOLDEN_REFERENCE_001` replays one frozen pre-match example without writing to
the live prediction ledger. The daily command refuses an input after kickoff.

No secret or network account is needed for this route. If an optional provider
is added later, use environment variables or Cloud secrets; never commit keys.
