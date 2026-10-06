# R3 Market Intelligence V1

This is an independent, frozen-provisional market interpretation layer for the
R3 blind test. It does not alter the national-team model, original prediction,
original prediction lock, formal Production gate, or R5 market-only route.

The registered providers are `JC_OFFICIAL_PROVIDER` and
`GLOBAL_ODDS_PROVIDER` (`NOT_CONFIGURED` until an authorized endpoint exists),
plus `MANUAL_MARKET_PROVIDER`. An API key by itself is insufficient to configure
an HTTP provider. Unconfirmed screenshots and OCR are not market evidence.

To supply a confirmed observation, place a JSON file under
`blind_test/r3/market/manual/`. Its `fixture_id`, teams and UTC kickoff must
exactly match a verified R3 fixture. The user must explicitly confirm the
source artifact and confirmation time before the match. Example *schema only*:

```json
{
  "fixture_id": "VERIFIED_FIXTURE_ID",
  "competition": "VERIFIED_COMPETITION",
  "home_team": "VERIFIED_HOME_TEAM",
  "away_team": "VERIFIED_AWAY_TEAM",
  "kickoff_utc": "YYYY-MM-DDTHH:MM:SS+00:00",
  "confirmation": {
    "status": "USER_CONFIRMED",
    "confirmation_id": "UNIQUE_CONFIRMATION_ID",
    "confirmed_at": "YYYY-MM-DDTHH:MM:SS+00:00",
    "source_artifact_sha256": "64_LOWERCASE_HEX_CHARACTERS"
  },
  "quotes": [{
    "bookmaker": "USER_VERIFIED_BOOKMAKER_NAME",
    "home_odds": 2.0,
    "draw_odds": 3.5,
    "away_odds": 4.0,
    "fetched_at": "YYYY-MM-DDTHH:MM:SS+00:00",
    "is_opening_confirmed": false
  }]
}
```

Only `MATCH_1X2` is supported in V1. Each bookmaker is normalized with the
existing proportional de-vig method, then current bookmaker probabilities are
combined by median. The fixed fusion is 70% frozen R3 model and 30% eligible
market consensus. If consensus is unavailable, `FUSION_MODE=MODEL_ONLY` and the
model probability is preserved exactly. R5 uses only market no-vig and remains
unavailable without a market observation. A confirmed opening quote is needed
for movement; the first observed quote is never called an opening quote.

Run the market supplement for an existing, still pre-match, officially locked
R3 prediction:

```powershell
.\.venv\Scripts\python.exe -m erguoyuan_football.blind_test_r3.market_lock PREDICTION_ID
```

The command reads the frozen market manifest, records an append-only snapshot
and quotes in `blind_test/r3/market/market.sqlite`, and writes a separate
`prediction_lock.market_v1.<market_id>.json` in the prediction's `official_locks`
directory. This supplement references the SHA256 of the original lock, which
cannot be changed retroactively. Later market captures produce new snapshots;
they never update the earlier lock. The supplement's capture timestamp is the
actual run time and must be within 30 seconds of the supplied timestamp.

After a verified result is appended through the existing R3 result path,
`append_market_evaluation` can score the original model, market no-vig (when
available), and frozen fusion separately. A single match records calibration
observations only; aggregate calibration requires multiple verified results.

The layer is not a formal Production release. `config/production.yaml` remains
`TRIAL`; Phase 9 remains `PARTIAL / AWAITING_FINAL_HOLDOUT`.
