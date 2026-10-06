"""Real development-only Phase 11 command line entry point."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _workspace_temp() -> None:
    root = Path(__file__).resolve().parents[3]
    workspace = root / ".workspace"
    for name in ("temp", "duckdb", "matplotlib", "joblib"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TEMP", str(workspace / "temp"))
    os.environ.setdefault("TMP", str(workspace / "temp"))
    os.environ.setdefault("MPLCONFIGDIR", str(workspace / "matplotlib"))
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(workspace / "joblib"))
    os.environ.setdefault("DUCKDB_TEMP_DIRECTORY", str(workspace / "duckdb"))


_workspace_temp()

from erguoyuan_football.meta.runtime import peak_working_set_bytes
from erguoyuan_football.report.audit import audit_phase11
from erguoyuan_football.report.pipeline import run_development_batch
from erguoyuan_football.report.renderer import CoreReportV2Renderer


def main(argv: list[str] | None = None) -> int:
    """Run 1/20/50 real rows and write isolated CORE REPORT V2 artifacts."""
    parser = argparse.ArgumentParser(description="CORE REPORT V2 development run")
    parser.add_argument("--db", type=Path, default=Path("data/football.duckdb"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("reports/phase9_development_core_preview.jsonl"))
    parser.add_argument("--rules", type=Path, default=Path("config/competition_rules.yaml"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    try:
        report, timings = run_development_batch(db_path=args.db, predictions_path=args.predictions,
                                                rules_path=args.rules, count=args.count)
        checks = audit_phase11(report)
        if not all(checks.values()):
            raise ValueError("PHASE11_SELF_REVIEW_FAILED:" + ",".join(
                name for name, passed in checks.items() if not passed))
        started = time.perf_counter()
        encoded = report.model_dump_json(indent=2)
        timings["json_report_ms"] = round((time.perf_counter() - started) * 1000, 3)
        started = time.perf_counter()
        rendered = CoreReportV2Renderer().render(report)
        timings["chinese_renderer_ms"] = round((time.perf_counter() - started) * 1000, 3)
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"phase11_core_report_v2_{args.count}.json").write_text(
            encoded, encoding="utf-8")
        (output_dir / f"phase11_core_report_v2_{args.count}.txt").write_text(
            rendered, encoding="utf-8")
        summary = {"status": "DEVELOPMENT_ONLY_NOT_PROMOTED", "real_match_count": len(report.all_matches),
            "score_matrix_ready_rows": report.model_data_status["score_matrix_ready_rows"],
            "highest_hit_candidates": len(report.candidate_ledger.candidates),
            "market_rows": 0, "account_100": report.value_100.status,
            "account_20": report.longshot_20.status,
            "final_holdout_rows_read": 0, "timings": timings,
            "self_review_checks": checks,
            "peak_working_set_bytes": peak_working_set_bytes()}
        (output_dir / f"phase11_runtime_{args.count}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "FAILED", "reason": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
