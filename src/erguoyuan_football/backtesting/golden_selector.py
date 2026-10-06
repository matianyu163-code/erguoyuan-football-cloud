"""Fail-closed selection and PIT checks for the Phase 15 Golden OOS dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import duckdb

GoldenSourceType = Literal["USER_JC_CONFIRMED", "JC_VERIFIED"]
START_DATE = date(2026, 8, 1)
END_DATE = date(2027, 6, 30)
TARGET_QUOTAS: dict[str, int] = {
    "EPL": 20,
    "BUNDESLIGA": 15,
    "BUNDESLIGA_2": 10,
    "LALIGA": 15,
    "SERIE_A": 15,
    "UEFA_CL": 15,
    "J_LEAGUE": 10,
}
COMPETITION_LABELS: dict[str, str] = {
    "EPL": "英超",
    "BUNDESLIGA": "德甲",
    "BUNDESLIGA_2": "德乙",
    "LALIGA": "西甲",
    "SERIE_A": "意甲",
    "UEFA_CL": "欧冠",
    "J_LEAGUE": "日职",
}


@dataclass(frozen=True)
class JCSourceEvidence:
    """Auditable, time-bounded evidence that one historical fixture was JC."""

    match_id: str
    source_type: GoldenSourceType
    evidence_id: str
    evidence_time: datetime

    def __post_init__(self) -> None:
        if not self.match_id or not self.evidence_id:
            raise ValueError("JC_EVIDENCE_ID_REQUIRED")
        if self.source_type not in {"USER_JC_CONFIRMED", "JC_VERIFIED"}:
            raise ValueError("JC_EVIDENCE_SOURCE_TYPE_INVALID")
        if self.evidence_time.tzinfo is None:
            raise ValueError("JC_EVIDENCE_TIME_MUST_BE_TIMEZONE_AWARE")


@dataclass(frozen=True)
class GoldenCandidate:
    """A selected match with a verified time, result and JC provenance."""

    match_id: str
    competition: str
    competition_id: str
    home_team: str
    away_team: str
    kickoff_time: datetime
    result: tuple[int, int]
    source_type: GoldenSourceType
    evidence_id: str
    source_id: str


@dataclass(frozen=True)
class RejectedCandidate:
    """A real database match excluded with explicit reason codes."""

    match_id: str
    competition_id: str
    home_team: str
    away_team: str
    match_date: date
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GoldenSelection:
    """Deterministic Golden OOS candidate selection result."""

    created_at: datetime
    status: Literal["READY", "PARTIAL"]
    target_count: int
    candidates: tuple[GoldenCandidate, ...]
    rejected: tuple[RejectedCandidate, ...]
    quota_counts: Mapping[str, int]


def validate_oos_timestamps(
    *,
    training_end_time: datetime,
    prediction_time: datetime,
    kickoff_time: datetime,
    input_timestamps: tuple[datetime, ...] = (),
) -> None:
    """Reject training or inputs after prediction, or predictions at/after kickoff."""
    values = (training_end_time, prediction_time, kickoff_time, *input_timestamps)
    if any(value.tzinfo is None for value in values):
        raise ValueError("PIT_TIMESTAMPS_MUST_BE_TIMEZONE_AWARE")
    if training_end_time > prediction_time:
        raise ValueError("TRAINING_END_AFTER_PREDICTION_TIME")
    if prediction_time >= kickoff_time:
        raise ValueError("PREDICTION_NOT_BEFORE_KICKOFF")
    if any(value > prediction_time for value in input_timestamps):
        raise ValueError("FUTURE_DATA_DETECTED")


class GoldenCandidateSelector:
    """Select only real finished matches with verified kickoff and JC evidence."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def select(
        self,
        *,
        as_of: datetime,
        jc_evidence: Mapping[str, JCSourceEvidence] | None = None,
        target_count: int = 100,
    ) -> GoldenSelection:
        """Build a deterministic selection; never infer JC status or kickoff time."""
        if as_of.tzinfo is None:
            raise ValueError("AS_OF_MUST_BE_TIMEZONE_AWARE")
        if target_count <= 0:
            raise ValueError("TARGET_COUNT_MUST_BE_POSITIVE")
        as_of_utc = as_of.astimezone(UTC)
        evidence = jc_evidence or {}
        with duckdb.connect(str(self.database_path), read_only=True) as connection:
            rows = connection.execute(
                """
                SELECT m.match_id, m.competition_id, c.competition_name,
                       h.team_name, a.team_name, m.match_date,
                       m.kickoff_time_utc, e.enriched_kickoff_utc,
                       m.home_goals, m.away_goals, m.source_id, m.status
                FROM real_canonical_matches AS m
                LEFT JOIN real_canonical_competitions AS c USING (competition_id)
                LEFT JOIN real_canonical_teams AS h
                  ON h.team_id=m.home_team_id AND h.competition_id=m.competition_id
                LEFT JOIN real_canonical_teams AS a
                  ON a.team_id=m.away_team_id AND a.competition_id=m.competition_id
                LEFT JOIN (
                    SELECT match_id, min(enriched_kickoff_utc) AS enriched_kickoff_utc
                    FROM kickoff_enrichment_records
                    WHERE verified AND timestamp_precision='EXACT_UTC'
                    GROUP BY match_id
                    HAVING count(DISTINCT enriched_kickoff_utc)=1
                ) AS e USING (match_id)
                WHERE m.match_date BETWEEN ? AND ?
                ORDER BY m.match_date, m.match_id
                """,
                [START_DATE, min(END_DATE, as_of_utc.date())],
            ).fetchall()

        accepted: list[GoldenCandidate] = []
        rejected: list[RejectedCandidate] = []
        for row in rows:
            (match_id, competition_id, _competition_name, home, away, match_date,
             direct_kickoff, enriched_kickoff, home_goals, away_goals, source_id,
             match_status) = row
            reasons: list[str] = []
            kickoff = direct_kickoff or enriched_kickoff
            source = evidence.get(str(match_id))
            if match_status != "FINISHED":
                reasons.append("MATCH_NOT_FINISHED")
            if home_goals is None or away_goals is None:
                reasons.append("REAL_SCORE_MISSING")
            if not competition_id or competition_id not in TARGET_QUOTAS:
                reasons.append("COMPETITION_OUTSIDE_CONFIGURED_JC_UNIVERSE")
            if not home or not away:
                reasons.append("CANONICAL_TEAM_NAME_MISSING")
            if not kickoff:
                reasons.append("VERIFIED_UTC_KICKOFF_UNAVAILABLE")
            elif kickoff > as_of_utc:
                reasons.append("MATCH_IS_IN_FUTURE")
            if str(source_id).upper().startswith("SYNTHETIC"):
                reasons.append("SYNTHETIC_TEST_DATA")
            if source is None:
                reasons.append("JC_SOURCE_EVIDENCE_UNAVAILABLE")
            elif kickoff and source.evidence_time.astimezone(UTC) > kickoff:
                reasons.append("JC_EVIDENCE_AFTER_KICKOFF")

            if reasons:
                rejected.append(RejectedCandidate(
                    match_id=str(match_id), competition_id=str(competition_id or "UNKNOWN"),
                    home_team=str(home or "UNKNOWN"), away_team=str(away or "UNKNOWN"),
                    match_date=match_date, reasons=tuple(sorted(set(reasons))),
                ))
                continue
            assert kickoff is not None and source is not None
            accepted.append(GoldenCandidate(
                match_id=str(match_id),
                competition=COMPETITION_LABELS[str(competition_id)],
                competition_id=str(competition_id), home_team=str(home), away_team=str(away),
                kickoff_time=kickoff.astimezone(UTC),
                result=(int(home_goals), int(away_goals)),
                source_type=source.source_type, evidence_id=source.evidence_id,
                source_id=str(source_id),
            ))

        chosen = self._apply_quotas(accepted, target_count)
        quota_counts = {league: sum(row.competition_id == competition_id for row in chosen)
                        for competition_id, league in COMPETITION_LABELS.items()}
        return GoldenSelection(
            created_at=as_of_utc,
            status="READY" if len(chosen) == target_count else "PARTIAL",
            target_count=target_count,
            candidates=tuple(chosen), rejected=tuple(rejected),
            quota_counts=quota_counts,
        )

    @staticmethod
    def _apply_quotas(
        candidates: list[GoldenCandidate], target_count: int,
    ) -> list[GoldenCandidate]:
        """Fill fixed competition quotas first, then only configured league candidates."""
        ordered = sorted(candidates, key=lambda item: (item.kickoff_time, item.match_id))
        selected: list[GoldenCandidate] = []
        selected_ids: set[str] = set()
        for competition_id, quota in TARGET_QUOTAS.items():
            for item in (row for row in ordered if row.competition_id == competition_id):
                if sum(row.competition_id == competition_id for row in selected) >= quota:
                    break
                selected.append(item)
                selected_ids.add(item.match_id)
        for item in ordered:
            if len(selected) >= target_count:
                break
            if item.match_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.match_id)
        return sorted(selected, key=lambda item: (item.kickoff_time, item.match_id))


def write_candidate_file(selection: GoldenSelection, path: str | Path, *,
                         dataset_version: str = "GOLDEN_OOS_V1") -> Path:
    """Persist auditable candidate and rejection lists without locking a partial set."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_version": dataset_version,
        "status": selection.status,
        "locked": selection.status == "READY",
        "created_at": selection.created_at.isoformat(),
        "target_count": selection.target_count,
        "selected_count": len(selection.candidates),
        "quota_counts": dict(selection.quota_counts),
        "candidates": [
            {**{key: value for key, value in asdict(item).items() if key != "result"},
             "kickoff_time": item.kickoff_time.isoformat()}
            for item in selection.candidates
        ],
        "rejected_candidates": [
            {**asdict(item), "match_date": item.match_date.isoformat()}
            for item in selection.rejected
        ],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return target


REVIEW_CSV_FIELDS = (
    "candidate_id", "match_id", "competition", "home_team", "away_team",
    "kickoff_time", "data_quality", "source_available", "review_status",
)


def candidate_id_for(candidate: GoldenCandidate,
                     dataset_version: str = "GOLDEN_OOS_V1") -> str:
    """Return a deterministic review key bound to fixture identity and kickoff."""
    identity = (f"{dataset_version}|{candidate.match_id}|{candidate.competition_id}|"
                f"{candidate.home_team}|{candidate.away_team}|"
                f"{candidate.kickoff_time.isoformat()}")
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def write_review_csv(selection: GoldenSelection, path: str | Path, *,
                     dataset_version: str = "GOLDEN_OOS_V1") -> Path:
    """Export blind review rows; final scores are intentionally excluded."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_CSV_FIELDS,
                                extrasaction="raise")
        writer.writeheader()
        for item in selection.candidates:
            writer.writerow({
                "candidate_id": candidate_id_for(item, dataset_version),
                "match_id": item.match_id,
                "competition": item.competition,
                "home_team": item.home_team,
                "away_team": item.away_team,
                "kickoff_time": item.kickoff_time.isoformat(),
                "data_quality": "PASS: VERIFIED_KICKOFF; SOURCE_VERIFIED; FINISHED",
                "source_available": "AVAILABLE",
                "review_status": "PENDING",
            })
    return target


def write_selection_template(
    candidates_csv: str | Path,
    path: str | Path,
    *,
    dataset_version: str = "GOLDEN_OOS_V1",
) -> Path:
    """Create a user-editable JSON confirmation file without overwriting edits."""
    target = Path(path)
    if target.exists():
        current = json.loads(target.read_text(encoding="utf-8"))
        if current.get("dataset_version") != dataset_version:
            raise ValueError("SELECTION_FILE_VERSION_MISMATCH")
        if not isinstance(current.get("selected_matches"), list):
            raise ValueError("SELECTION_FILE_SCHEMA_INVALID")
        return target
    candidate_ids: list[str] = []
    with Path(candidates_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REVIEW_CSV_FIELDS:
            raise ValueError("CANDIDATE_REVIEW_SCHEMA_INVALID")
        for row in reader:
            candidate_ids.append(row["candidate_id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_version": dataset_version,
        "selected_matches": [
            {"candidate_id": candidate_id, "approved": False}
            for candidate_id in candidate_ids
        ],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return target


def main() -> int:
    """Generate an auditable candidate report from the configured local database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/football.duckdb")
    parser.add_argument("--as-of", required=True,
                        help="Timezone-aware ISO-8601 timestamp, for example 2026-10-02T00:00:00Z")
    parser.add_argument("--output", default="data/backtest/golden_candidates.json")
    parser.add_argument("--review-csv", default="data/backtest/golden_candidates_review.csv")
    parser.add_argument("--selection", default="data/backtest/golden_selection.json")
    parser.add_argument("--dataset-version", default="GOLDEN_OOS_V1")
    parser.add_argument("--target-count", type=int, default=100)
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of)
    selection = GoldenCandidateSelector(args.db).select(
        as_of=as_of, target_count=args.target_count)
    output = write_candidate_file(selection, args.output,
                                  dataset_version=args.dataset_version)
    review_csv = write_review_csv(selection, args.review_csv,
                                  dataset_version=args.dataset_version)
    selection_file = write_selection_template(review_csv, args.selection,
                                              dataset_version=args.dataset_version)
    print(json.dumps({
        "dataset_version": args.dataset_version, "status": selection.status,
        "locked": selection.status == "READY", "target_count": selection.target_count,
        "selected_count": len(selection.candidates), "rejected_count": len(selection.rejected),
        "output": str(output), "review_csv": str(review_csv),
        "selection_file": str(selection_file),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
