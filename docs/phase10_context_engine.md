# Phase 10 — Tournament State, Lineup, and Context Engine

## Scope and safety boundary

Phase 10 adds an evidence-first context path for canonical CORE V2 development predictions. It does not train a probability model, alter Phase 9 META or calibration artifacts, promote output, recommend bets, allocate stakes, or change V5.1. When there is no chronologically validated context residual artifact, tournament and lineup probabilities remain exact copies of the calibrated CORE probability.

The only historical reconstruction source currently wired is `real_canonical_matches`, treated as immutable result facts. DATE_SAFE batches include completed events whose UTC-normalized date is strictly earlier than the target match date. Same-day and future events are excluded for every target independently. Target rows are never used as historical context. The locked Phase 9 final-holdout window (2026-08-01 through 2027-06-30) is outside the configured development range and rejected by the CLI.

## Components

- `schemas.py`: immutable context, tournament, lineup, injury, player-strength, feature, probability-layer, and ledger contracts.
- `competition_rules.py` / `config/competition_rules.yaml`: versioned source-backed rule registry. A rule is eligible only if its effective period contains the target date and its recorded `retrieved_at` is no later than the prediction time. A rule without that PIT proof is unavailable.
- `standings.py`, `tie_state.py`, and `tournament_state.py`: prior-result standings and explicit two-leg aggregation. Tie inference requires a verified two-leg format, explicit tie ID, leg number, and exactly one eligible first-leg result. Approximate points-only rankings are not official tie-break rankings.
- `schedule_context.py` / `fatigue.py`: rest and congestion mathematics. A production caller must explicitly attest complete cross-competition schedule coverage. Current canonical history covers league fixtures only and lacks cup/other schedule completeness; the pipeline therefore leaves fatigue counts and rest days null and marks them unavailable.
- `lineup.py`, `injury.py`, `suspension.py`, and `player_strength.py`: source-time and retrieval-time guards plus provider contracts. No licensed historical lineup/injury/player-strength source is configured; no evidence is synthesized.
- `rotation.py`: returns unavailable until a real, point-in-time trained model and comparable historical lineup evidence exist.
- `incentive.py`: only mathematically derivable elimination states; no remaining-fixture snapshot means MAI and elimination claims remain null.
- `features.py` / `uncertainty.py`: versioned null-aware feature vector, evidence lineage, availability mask, source dates, and a named missing-component fraction. That fraction is evidence completeness, not calibrated probabilistic uncertainty.
- `adjustment.py` / `gates.py` / `audit.py`: centered-logit residual model interface, identity/no-fit path, independent context/lineup/tournament/promotion gates, and per-match invariant audit.
- `artifacts.py`: reads immutable real history and adds append-only/idempotent `context_*` DuckDB tables for states, feature vectors, assessments, accepted lineup/injury evidence, adjustment ledgers, and canonical outputs. It does not mutate Phase 9 tables.
- `cli.py` / `run_phase10.py`: DATE_SAFE development batch execution, dry-run, idempotent persistence, resume, structured diagnostics, and holdout-window refusal.

## Run

The CLI consumes existing, real `FINAL_CORE_CALIBRATED` Phase 9 development predictions. It does not manufacture base probabilities or accept an arbitrary unlabeled match as a forecast.

```powershell
.\.venv\Scripts\python.exe -m erguoyuan_football.context.run_phase10 `
  --db data/football.duckdb `
  --config config/context.yaml `
  --from-date 2026-05-01 `
  --to-date 2026-05-31 `
  --dry-run
```

Remove `--dry-run` to persist additive context records. `--resume` returns an already committed identical run using the hashes of the config, rules, real result dataset, and selected base prediction IDs. A request outside the development window is rejected.

## Current limitations

The real development fixture contains 167 calibrated CORE V2 rows from May 2026 and 9,158 canonical match records across five leagues. Those records do not establish exact UTC kickoff times, a complete all-competition schedule, historical rule publication snapshots for the May 2026 targets, licensed lineups/injuries, player strength, or a trained OOS context adjustment. Consequently the current true-data path validates integration and provenance while retaining base probabilities unchanged. It is not a live prediction path and remains `DEVELOPMENT_ONLY / NOT_PROMOTED`.

The implementation is not a betting strategy, does not produce V5.1, and does not change Phase 9's locked final holdout or partial status.
