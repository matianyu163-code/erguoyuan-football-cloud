# CORE SPI-like V1

This is a CORE implementation inspired by the public SPI model family. It is marked `LIKE_IMPLEMENTATION` and `NOT_OFFICIAL_FIVETHIRTYEIGHT_SPI`.

The V1 estimator keeps separate team attack and defense states. Attack performance after a match is `log((goals_for + smoothing) / baseline_goals) - opponent_defense`; defense performance is `log((goals_conceded + smoothing) / baseline_goals) - opponent_offense`. Both ratings are moving states with a configured learning rate. **A higher defensive rating means the team is expected to concede more goals, so it represents worse defense.**

The supported modes are `GOALS_ONLY` and `GOALS_XG`. xG mode is selected only when the configured training set has a PIT-valid, provenance-complete xG observation for every included match. Partial coverage falls back to the consistent goals-only feature mode and is recorded. No xG values are imputed. `FULL_AVAILABLE` currently resolves to the same complete goals+xG feature contract; market value, rest-day, and adjusted-goals inputs are not implemented.

Expected goals use independent Poisson rates from baseline goals, team attack/defense, optional observed CORE league strength, and home advantage. The score matrix accounts for omitted Poisson tail mass through the shared conditional-normalization contract. Overall rating is the modeled neutral expected points share against league-average attack/defense, scaled to 0–100; it is not a win probability.

`CORE_SPI_LEAGUE_STRENGTH_V1` updates relative league strengths only from cross-league fixtures with dated team hierarchy evidence. Sparse global shrinkage and preseason market-value inputs are not implemented. The default diagonal draw adjustment is 1.0 (no adjustment); tuning a non-identity value requires a separate train/validation estimate and is not enabled as a production default.

Season transitions apply configured carryover and mean reversion. The state journal distinguishes pre-match and post-match values. Training and prediction inputs remain subject to strict point-in-time checks.

## Difference and limits

CORE does not reproduce FiveThirtyEight's private adjusted-goal, non-shot xG, market-value regression, or original league-strength process. V1 uses a transparent CORE log-rate update and, when dated hierarchy is available, a CORE cross-league rating extension. No real historical xG or hierarchy source is currently configured in the repository.
