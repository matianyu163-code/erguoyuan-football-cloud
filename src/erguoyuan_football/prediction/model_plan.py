"""Read-only execution-plan CLI for Phase 14 model readiness."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from erguoyuan_football.prediction.model_execution_planner import (
    ModelExecutionPlanner,
)
from erguoyuan_football.research.live_data.readiness import (
    DataAvailability,
    DataReadinessItem,
    LiveDataReadinessReport,
    ModelReadiness,
)


def _empty_readiness() -> LiveDataReadinessReport:
    """Return fail-closed readiness when no real live evidence was supplied."""
    planner = ModelExecutionPlanner(
        Path("config/phase14_model_requirements.yaml"),
        Path("config/model_registry.yaml"),
    )
    names = {name for policy in planner.policies.values()
             for name in (*policy.training_inputs, *policy.inference_inputs)}
    items = tuple(DataReadinessItem(name, DataAvailability.MISSING,
                                    reasons=("NO_LIVE_EVIDENCE_SUPPLIED",))
                  for name in sorted(names))
    model_rows = tuple(ModelReadiness(
        model_id, False, False, ("NO_LIVE_EVIDENCE_SUPPLIED",), (), (), (),
        ("NO_LIVE_EVIDENCE_SUPPLIED",), "BLOCKED", "VERY_HIGH",
    ) for model_id in planner.policies)
    return LiveDataReadinessReport(
        False, "NOT_READY", items, model_rows, (), tuple(planner.policies), (),
        ("FIXTURE", "HISTORICAL_RESULTS"), datetime.now(UTC),
        entity_resolution_status="UNVERIFIED",
    )


def main() -> int:
    """Display blocked models without manufacturing data or probabilities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-time", required=True,
                        help="UTC ISO timestamp, e.g. 2026-10-02T12:00:00Z")
    parser.add_argument("--training-cutoff", required=True,
                        help="UTC ISO timestamp no later than prediction time")
    args = parser.parse_args()
    prediction_time = datetime.fromisoformat(args.prediction_time)
    training_cutoff = datetime.fromisoformat(args.training_cutoff)
    planner = ModelExecutionPlanner(
        Path("config/phase14_model_requirements.yaml"),
        Path("config/model_registry.yaml"),
    )
    plan = planner.plan(_empty_readiness(), prediction_time=prediction_time,
                        training_cutoff=training_cutoff,
                        neutral_venue_known=False)
    print(f"PLAN {plan.plan_id} | DRY_RUN | READINESS {plan.readiness_status}")
    for entry in plan.entries:
        print(f"{entry.model_id}: {entry.action} ({entry.mode})"
              + (f" | {'; '.join(entry.reasons)}" if entry.reasons else ""))
    print("PREDICTIONS: NOT_EXECUTED (no verified fixture or live evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
