"""Price-required Value and Longshot ranking with transparent lexicographic order."""

from __future__ import annotations

import json
from datetime import timedelta
from hashlib import sha256
from typing import Literal

from erguoyuan_football.recommendation.schemas import MarketPriceEvidence
from erguoyuan_football.selection.schemas import MatchHeads, RankedCandidate


class ValueSelectionEngine:
    """Construct priced single-match candidates; never infer value from hit rate alone."""

    def candidates(self, matches: tuple[MatchHeads, ...],
                   evidence: dict[tuple[str, str], MarketPriceEvidence], *,
                   objective: Literal["VALUE", "LONGSHOT"], minimum_longshot_odds: float = 5.0,
                   minimum_longshot_probability: float = 0.05,
                   allow_synthetic_test: bool = False,
                   max_quote_age_minutes: int = 120) -> tuple[RankedCandidate, ...]:
        if objective not in {"VALUE", "LONGSHOT"}:
            raise ValueError("VALUE_OBJECTIVE_REQUIRED")
        candidates = []
        for match in matches:
            if match.probability is None:
                continue
            for direction, p in (("HOME", match.probability.p_home),
                                 ("DRAW", match.probability.p_draw),
                                 ("AWAY", match.probability.p_away)):
                price = evidence.get((match.match_id, direction))
                if (price is None or not price.purchasable or
                    (price.synthetic_test_only and not allow_synthetic_test) or p <= 0 or
                    price.prediction_time - price.as_of_time > timedelta(minutes=max_quote_age_minutes)):
                    continue
                expected_id = sha256(json.dumps([objective, "MATCH_1X2", [match.match_id],
                                                 [direction], []],
                                               sort_keys=True).encode()).hexdigest()[:24]
                if price.candidate_id != expected_id:
                    continue
                ev = p * price.decimal_odds - 1
                if objective == "LONGSHOT" and (price.decimal_odds < minimum_longshot_odds or
                                                p < minimum_longshot_probability or
                                                ev <= 0 or p <= price.market_probability):
                    continue
                candidates.append(RankedCandidate(candidate_id=expected_id,
                    prediction_snapshot_ids=(match.prediction_snapshot_id,),
                    match_ids=(match.match_id,), objective=objective, play_type="MATCH_1X2",
                    selections=(direction,), component_probabilities=(p,),
                    joint_probability=p, coverage_probability=p,
                    probability_method="DIRECT_EFFECTIVE_CORE", market_probability=price.market_probability,
                    odds=price.decimal_odds, fair_odds=1 / p, expected_value=ev,
                    uncertainty=None, data_quality="REAL_OOS_DEVELOPMENT",
                    context_quality=match.context_status, correlation_status="NOT_APPLICABLE",
                    rank=1, validation_status=match.validation_status,
                    production_status=match.production_status, market_status="AVAILABLE",
                    market_quote_ids=price.quote_ids))
        ordered = sorted(candidates, key=lambda item: (-(item.expected_value or 0),
            item.uncertainty is None, item.uncertainty if item.uncertainty is not None else float("inf"),
            item.candidate_id))
        return tuple(item.model_copy(update={"rank": index}) for index, item in enumerate(ordered[:2], 1))
