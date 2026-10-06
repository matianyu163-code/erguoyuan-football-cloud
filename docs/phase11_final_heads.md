# Phase 11 — prediction heads and CORE REPORT V2

Phase 11 has four separate boundaries: effective CORE probability → mathematical play heads → ranked candidates → price-backed advice → capped experimental portfolio. The independent `CORE_REPORT_V2` JSON and Chinese renderer use six fixed sections. The existing V5.1 schema and golden output remain frozen.

The effective probability resolver accepts only a PIT-passing, validated calibrated CORE record or an actual verified `CONTEXT_ADJUSTED_CORE` record. The current Phase 10 context stage performs no adjustment, so the development report uses calibrated CORE probabilities. Every row retains `DEVELOPMENT_ONLY`, `NOT_PROMOTED`, and `DATE_SAFE_BATCH`.

The score-matrix source selector scores REAL OOS sources on completed **January–April 2026 development matches only**. It verifies reconstructed pre-match training IDs and hashes, compares exact-score log loss, goal-total log loss, result calibration ECE and support coverage, then chooses one source before the May target period. The target readiness audit verifies the exact prediction ID from the canonical source lineage, real OOS status, temporal mode, training cutoff, target exclusion, match-count and hash. It reads no August 2026–June 2027 final-holdout rows. Reference `artifact_id` is null where the source did not provide one.

Outcome-mass reconciliation scales the three score regions to the effective calibrated 1X2 probability. A positive target mass with zero source mass fails closed. The report preserves the reference and final matrices, hashes, scaling factors, masses and source identity. Score TOP2 and 0–6/7+ total goals come from this same final matrix. Official handicap 1X2 additionally requires a source-backed official line with PIT evidence, and half/full time requires independent REAL OOS half-time model lineage. Neither source is currently available.

Candidates remain visible when advice is `NO_BET` or `UNAVAILABLE`. Cross-match joint probabilities currently use multiplication with an explicit `INDEPENDENCE_ASSUMPTION`; same-match pairings are rejected. The 400 RMB account has four fixed slots and a reference cap of 400, while actual recommended stake requires valid purchasable market evidence. The current real market quote count is zero, so all real advice is `UNAVAILABLE`, actual stakes are zero, and 100/20 RMB experimental Value/Longshot accounts show `UNAVAILABLE_MARKET_DATA`. No odds, EV, official handicap or HTFT probabilities are inferred. Synthetic tests exercise market math but never enter real reports.

Use the real development runner from the project root:

```powershell
.\.venv\Scripts\python.exe -m erguoyuan_football.report.cli --count 20
```

It writes `reports/phase11_core_report_v2_20.json`, `.txt`, and `phase11_runtime_20.json`. `--count 1` and `--count 50` exercise the same batch path. These are historical development reconstructions, **not** daily production recommendations or evidence of positive returns. A legitimate final holdout, exact UTC kickoff history, real market prices, official handicap and independent HTFT data remain necessary before production promotion.
