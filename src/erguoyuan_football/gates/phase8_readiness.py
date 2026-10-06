"""Phase 9 entry gates from measured Phase 8 data, never inferred coverage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from erguoyuan_football.data.real_quality import assess_real_data


@dataclass(frozen=True)
class Phase9GateReport:
    no_market_meta: str
    full_market_meta: str
    output_contract: str
    reasons: tuple[str, ...]


def assess_phase9_gates(path: str | Path) -> Phase9GateReport:
    quality, _ = assess_real_data(path)
    reasons = list(quality.reasons)
    if quality.real_oos_predictions == 0:
        reasons.append("NO_REAL_OOS_PREDICTIONS")
    if quality.fixtures < 1000 or quality.results < 1000:
        reasons.append("INSUFFICIENT_REAL_HISTORICAL_COVERAGE")
    # Market rows in the existing Phase 6 store are not inferred from fixtures.
    with duckdb.connect(str(path), read_only=True) as connection:
        table = connection.execute("SELECT 1 FROM information_schema.tables "
                                   "WHERE table_name='odds_quotes'").fetchone()
        market_rows = connection.execute("SELECT count(*) FROM odds_quotes").fetchone() if table else (0,)
        oos_table = connection.execute("SELECT 1 FROM information_schema.tables "
                                       "WHERE table_name='real_oos_predictions_v2'").fetchone()
        base_coverage: dict[str, int] = {}
        if oos_table:
            base_coverage = dict(connection.execute("""SELECT model_id,count(*)
                FROM real_oos_predictions_v2 WHERE data_origin='REAL' AND is_oos
                  AND model_id IN ('DIXON_COLES_V1','ELO_V1','PI_RATING_V1')
                GROUP BY model_id""").fetchall())
    assert market_rows is not None
    if market_rows[0] == 0:
        reasons.append("NO_REAL_MARKET_QUOTES")
    date_safe_base_ready = all(base_coverage.get(model_id, 0) >= 100
                               for model_id in ("DIXON_COLES_V1", "ELO_V1", "PI_RATING_V1"))
    if not date_safe_base_ready:
        reasons.append("INSUFFICIENT_REAL_BASE_OOS_COVERAGE")
    no_market = ("PASS" if quality.real_oos_predictions > 0 and date_safe_base_ready
                 and quality.strict_historical_oos_status != "BLOCKED" else "BLOCKED")
    full_market = "PASS" if no_market == "PASS" and market_rows[0] > 0 else "BLOCKED"
    # Schema availability alone is insufficient for a real CORE output gate.
    output = "BLOCKED" if quality.real_oos_predictions == 0 else "REVIEW_REQUIRED"
    return Phase9GateReport(no_market, full_market, output, tuple(reasons))
