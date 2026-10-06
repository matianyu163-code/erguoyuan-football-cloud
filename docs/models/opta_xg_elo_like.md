# CORE Opta xG-Elo-like V1

`CORE_OPTA_XG_ELO_LIKE_V1` is a `LIKE_IMPLEMENTATION`, named `CORE_HIERARCHICAL_ELO` and `CORE_POWER_RATING`. It always declares `NOT_OFFICIAL_OPTA_MODEL`; the system has no official Opta endpoint or credentials configured.

Every match updates both participating team Elo components. If dated team-membership evidence is available at kickoff, exactly one non-team level receives a split zero-sum delta: teams in the same league update team only; different leagues in one country update league; different countries on one continent update country; cross-continent fixtures update continent. Membership records carry valid intervals, retrieval and as-of timestamps, source, and data version. Without memberships, no league, country, or continent value is guessed.

Traditional result Elo uses a logistic expected home result, configured home advantage, goal-difference multiplier, and configured K factor. When a match has a provenance-complete historical xG record, `CORE_XG_SCORE_DISTRIBUTION_V1` uses independent Poisson means equal to the observed home and away xG. Its scoreline mass is conditionally normalized with explicit tail accounting. The model computes a score-result Elo delta for each score cell and probability-weights those deltas. The reference blend is 80% result delta and 20% xG delta when xG exists; with no xG, it uses 100% result delta. These are CORE reference weights, not independently validated Opta parameters.

The probability mapper is a separate multinomial logistic regression fitted only on pre-match rating difference, home advantage, and cross-league status from the model's chronological training period. Raw hierarchical Elo is mapped separately to a CORE-estimated 0–100 min-max power scale; the 0–100 value is not interpreted as a win probability.

## Difference and limits

Opta's complete hierarchical update allocation, private priors, xG distribution, and power transformation parameters are not public. CORE's hierarchy allocation, independent-Poisson xG distribution, 80/20 reference blend, and power scale are explicit CORE assumptions. No source is configured to provide real xG or hierarchy history, so the default model path is result-only. Production-quality hierarchy ablations, external-source validation, and broad OOS performance evidence remain incomplete.
