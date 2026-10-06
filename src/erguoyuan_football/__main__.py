"""python -m erguoyuan_football --db ... --at ... --text ..."""

import argparse

from erguoyuan_football.daily import DailyPredictionRequest, DailyService
from erguoyuan_football.data.store import Store
from erguoyuan_football.output.v51.renderer import render


def main():
    parser = argparse.ArgumentParser(description="二果园足球预测系统 — PHASE 2 输入与快照")
    parser.add_argument("--db", required=True)
    parser.add_argument("--at", required=True, help="UTC or timezone-aware ISO prediction timestamp")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--image")
    parser.add_argument("--match-date", help="optional UTC kickoff date YYYY-MM-DD")
    args = parser.parse_args()
    request = DailyPredictionRequest(input_type="SCREENSHOT" if args.image else "BATCH",
                                     raw_text=args.text, image_path=args.image,
                                     prediction_time=args.at, match_date=args.match_date)
    with Store(args.db) as store:
        result = DailyService(store).run(request)
        # Full audit includes unresolved requests, not only the display-ready rows.
        print(result.model_dump_json(indent=2))
        print(render(result.output))


if __name__ == "__main__":
    main()
