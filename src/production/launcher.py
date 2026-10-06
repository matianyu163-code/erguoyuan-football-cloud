"""Unique interactive entry point for a V1.0 production trial."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from production.config import ProductionConfig, default_config_path
from production.runner import ProductionRunner


def main(argv: list[str] | None = None) -> int:
    """Parse one match, run all status gates, render a diagnostic, and save audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--network-diagnostics", type=Path)
    parser.add_argument("--bridge-input", type=Path,
                        help="Versioned CORE_DATA_PACKET_V1 JSON file")
    parser.add_argument("--text", help='e.g. "Arsenal vs Chelsea"')
    parser.add_argument("--competition", default=None)
    parser.add_argument("--jc-confirmed", action="store_true",
                        help="User explicitly confirms the input is a Sporttery fixture")
    parser.add_argument("--research-test", action="store_true",
                        help="Explicitly classify a real fixture as a research trial")
    parser.add_argument("--simulation", action="store_true",
                        help="Mark this as process-only test input; never a real prediction")
    args = parser.parse_args(argv)
    if args.network_diagnostics is not None:
        from production.network_diagnostics import write_diagnostics
        return write_diagnostics(args.network_diagnostics, process="PRODUCTION_LAUNCHER")
    if args.bridge_input is not None:
        from production.bridge.runner import main as bridge_main

        return bridge_main(["--input", str(args.bridge_input),
                            "--config", str(args.config),
                            "--project-root", str(Path(args.config).resolve().parents[1])])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    raw_text = args.text if args.text is not None else input("请输入比赛（主队 VS 客队）: ")
    try:
        config = ProductionConfig.load(Path(args.config))
        runner = ProductionRunner(config)
        try:
            result = runner.run(
                raw_text, competition=args.competition, jc_confirmed=args.jc_confirmed,
                simulation=args.simulation, research_test=args.research_test)
        finally:
            runner.close()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"PRODUCTION TRIAL FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(result.rendered_output, end="")
    saved = result.prediction_id is not None and any(
        stage.stage == "SAVE_RECORD" and stage.status == "READY" for stage in result.stages)
    executed = any(stage.stage == "MODEL_EXECUTION" and stage.status == "READY"
                   for stage in result.stages)
    accepted_with_warnings = (result.status == "WARNING" and saved and executed
        and any(stage.stage == "REAL_MATCH_SNAPSHOT" and stage.status == "READY"
                for stage in result.stages)
        and any(stage.stage == "FIXTURE_VERIFICATION" and stage.status == "READY"
                for stage in result.stages))
    final_status = ("PRODUCTION_TRIAL_READY_WITH_WARNINGS" if accepted_with_warnings
                    else result.status)
    print(f"RUN_STATUS: {final_status}")
    print(f"PREDICTION_ID: {result.prediction_id or 'NOT_SAVED'}")
    return 0 if result.status == "READY" else 1 if result.status == "WARNING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
