# R3 The Odds API V4 Provider (P2A)

The P2A path adds a real HTTP provider to the existing R3 market pipeline.
The previously frozen `R3_MARKET_INTELLIGENCE_V1` implementation, its manifest,
the 70/30 `MODEL_MARKET_FUSION_V1` weights, original predictions, and original
locks are unchanged. The P2A provider has a separate config and release hash.

The documented API is `https://api.the-odds-api.com/v4`: `/sports/` supplies
active sport keys; `/sports/{sport_key}/odds/` supplies decimal `h2h` prices for
`eu,uk`. See https://the-odds-api.com/liveapi/guides/v4/ . Each request passes
through `CoreNetworkClient` with an allowlisted HTTPS endpoint. The client
inserts `apiKey` from `YYCORE_THE_ODDS_API_KEY` after auditing and cache-key
calculation; it does not log or persist the key or full query URL.

PowerShell example for the current shell only:

```powershell
$env:YYCORE_THE_ODDS_API_KEY="YOUR_KEY"
```

When this variable is absent the provider is `NOT_CONFIGURED`. The catalog
lookup and live integration tests are skipped; no market probability is
generated. `config/the_odds_api_sport_map.yaml` keeps unverified candidate
keys separate from endpoint-verified mappings. A candidate resolves only when
the current `/sports/` response contains its exact key and expected title.

For a still pre-match, already officially locked R3 prediction:

```powershell
.\.venv\Scripts\python.exe -m erguoyuan_football.blind_test_r3.market_p2a PREDICTION_ID
```

This fetches at most one current odds response per sport key per 60 seconds
and one sports catalog response per six hours. Both are stored by payload hash
in `blind_test/r3/market/the_odds_api_cache.sqlite` with true fetch time and
quota headers. A unique exact-orientation, kickoff-constrained fixture match
is required. Senior, youth, women, and B-team identities remain distinct.
Only complete, fresh, valid H/D/A odds from one bookmaker enter consensus.

The resulting canonical quotes, snapshot and movement rows are appended to
`blind_test/r3/market/market.sqlite`. Per-bookmaker proportional no-vig and
median consensus use the frozen R3 V1 market pipeline. The supplement is
`official_locks/<prediction_id>/prediction_lock.market_v1.<market_id>.json`.
The original `prediction_lock.json` is never rewritten. The first capture has
`movement=INSUFFICIENT_HISTORY`; later captures compare to the last genuine
stored pre-match consensus and are labeled `PREVIOUS_OBSERVED_TO_CURRENT`.
Neither is called a confirmed opening price.

R3 uses the preserved frozen model triple plus market validation and fixed
70/30 fusion when an eligible market exists. R5 remains market-only.
Unavailable or unconfigured market data leaves R3 in `MODEL_ONLY` mode.
This is a blind-test provider release, not Production promotion.
