# Phase 8 real data and CORE_OUTPUT_V2

This phase pins [OpenFootball football.json](https://github.com/openfootball/football.json)
at commit `e6744429ee395bc86f247348c6184bb08d4eb361`. The repository's
[license is CC0 1.0](https://github.com/openfootball/football.json/blob/master/LICENSE.md).
The import checks each local file against its tracked Git blob (allowing only
Windows line-ending conversion), records the raw bytes and SHA-256 in
`real_raw_sources`, validates rows in `real_staging_matches`, and promotes
validated rows transactionally into `real_canonical_*`. Conflicting scores and
invalid rows go to `real_data_quarantine`; no score is silently overwritten.
The first bulk import and the later date-window correction both have database
backup files in `data/`.

The canonical table deliberately keeps `kickoff_time_utc` null. OpenFootball
supplies dates and many local clock times, but no verified timezone for those
clock values. `LOCAL_TIME_UNKNOWN_ZONE` and `DATE_ONLY` are separate precision
states. Neither permits strict T-6H/T-60M/market backtests. The import's
`retrieved_at` is 2026-09-30; it is **not** backdated to historical match days.
Thus real historical results can form future training history once verified,
but they cannot prove a 2023/2024 pre-match data snapshot. The legacy `matches`
and `match_results` tables remain empty because no valid UTC fixture has yet
been promoted into the existing prediction pipeline.

`CanonicalPredictionResult` is a structured intermediate contract. A non-null
probability requires `data_origin=REAL`, a successful PIT check and a named
model. `probability_stage` is `BASE_MODEL`, `ML_MODEL` or `PRE_META`; Phase 8 has
no calibrated final CORE probability. Missing market, handicap, HTFT and score
matrix fields remain null with explicit statuses. `RankedCandidate` and
`BetAdvice` are distinct schemas; `NO_BET` requires stake zero, whereas
`UNAVAILABLE` represents missing or untrusted data. No selection, portfolio,
Chinese report or account allocation is implemented here. Legacy V5.1 remains
unchanged and is regression-tested.

`FootballDataOrgProvider` uses only the documented v4 competition-matches
endpoint through `CoreNetworkClient` with a token from
`FOOTBALL_DATA_ORG_TOKEN` and `X-Auth-Token`. No token is configured in this
environment, so its actual historical coverage is zero. The
`StatsBombOpenDataProvider` requires a local official open-data catalogue,
verified Git origin/commit and a manifest with license, retrieval time and
catalogue SHA-256;
none is installed. Neither provider currently promotes data into canonical
tables. OpenFootball is the only real source imported in this phase. Market,
xG, events, lineups, Opta and real OOS rows remain zero.

`assess_real_data` measures the imported coverage and reports timestamp and
provenance blockers. `assess_phase9_gates` returns blocked NO_MARKET,
FULL_MARKET and output-contract gates while real OOS is absent. A future OOS
writer must prove both fixture and training-result availability at the
prediction timestamp, as well as an exact UTC kickoff. It cannot use the
current bulk download to manufacture past predictions.
