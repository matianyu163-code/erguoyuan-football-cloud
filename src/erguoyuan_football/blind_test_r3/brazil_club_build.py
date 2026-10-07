"""Development-only build and evaluation for the pinned Brazilian Serie A corpus."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.club_brazil import (
    BRAZIL_SERIE_A,
    BrazilBivariatePoissonModel,
    BrazilDixonColesModel,
    BrazilEloModel,
)
from erguoyuan_football.contracts.common import utc
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.training import TrainingDataset, TrainingMatch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "data" / "training" / "brazil_serie_a" / "openfootball" / "raw"
TEAM_MAP_PATH = PROJECT_ROOT / "data" / "training" / "brazil_serie_a" / "BRAZIL_CLUB_TEAM_MAP_V1.json"
SOURCE_COMMIT = "4371c453d0cb8882b85bbc665ab336561e60e125"
SOURCE_AS_OF = datetime(2026, 9, 9, 0, 0, tzinfo=UTC)
TRAINING_CUTOFF = date(2026, 6, 1)
VALIDATION_CUTOFF = date(2025, 1, 1)
TARGET_CUTOFF = date(2026, 10, 7)
TEMPORAL_ASSUMPTIONS = (
    "DATE_SAFE_BATCH: source provides local calendar dates but no timezone-qualified kickoff timestamps",
    "Domestic league home/away role is used; source does not identify neutral venues, so training rows encode neutral_venue=false",
    "Match-date batches are strictly excluded on and after each model cutoff date",
)
DATE_LINE = re.compile(r"^\s{2,}(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+\d{4})?\s*$")
SCORE_LINE = re.compile(
    r"^\s{4}(?:\d{1,2}:\d{2}\s+)?(.+?)\s+v\s+(.+?)\s+(\d{1,2})-(\d{1,2})(?:\s+\(\d+-\d+\))?\s*$"
)
MONTHS = {name: index for index, name in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1
)}
MODEL_CLASSES = {
    "CLUB_ELO_BRAZIL_V1": BrazilEloModel,
    "CLUB_DIXON_COLES_BRAZIL_V1": BrazilDixonColesModel,
    "CLUB_BIVARIATE_POISSON_BRAZIL_V1": BrazilBivariatePoissonModel,
}


def digest_file(path: Path) -> str:
    """Hash a file as raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_mapping() -> dict[str, str]:
    """Load exact aliases; never normalize with fuzzy matching."""
    value = json.loads(TEAM_MAP_PATH.read_text(encoding="utf-8"))
    if value["mapping_version"] != "BRAZIL_CLUB_TEAM_MAP_V1":
        raise ValueError("TEAM_MAPPING_VERSION_MISMATCH")
    aliases = value["aliases"]
    if len(aliases) != len(set(aliases)):
        raise ValueError("DUPLICATE_TEAM_ALIAS")
    return aliases


def parse_matches(*, retrieved_at: datetime | None = None) -> tuple[TrainingMatch, ...]:
    """Parse only completed top-tier season rows from the pinned raw snapshot."""
    aliases = load_mapping()
    retrieved = utc(retrieved_at or datetime.now(UTC))
    rows: list[TrainingMatch] = []
    for path in sorted(RAW_ROOT.glob("*_br1.txt")):
        season = int(path.name[:4])
        current_date: date | None = None
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            date_match = DATE_LINE.match(line)
            if date_match:
                year = int(path.name[:4])
                month = MONTHS[date_match.group(1)]
                day = int(date_match.group(2))
                if month == 1 and season > 2018:
                    previous_dates = [row.kickoff_time.date() for row in rows if row.season == str(season)]
                    if previous_dates and max(previous_dates).month >= 10:
                        year += 1
                current_date = date(year, month, day)
                continue
            if " v " not in line or re.search(r"\b\d+-\d+\b", line) is None:
                continue
            match = SCORE_LINE.match(line)
            if match is None:
                continue
            if current_date is None:
                raise ValueError(f"MATCH_WITHOUT_DATE:{path.name}:{line_number}")
            home_name, away_name = match.group(1).strip(), match.group(2).strip()
            if home_name not in aliases or away_name not in aliases:
                raise ValueError(f"UNMAPPED_TEAM_ALIAS:{home_name if home_name not in aliases else away_name}")
            if current_date >= TARGET_CUTOFF:
                continue
            kickoff = datetime.combine(current_date, time.min, UTC)
            completed = kickoff + timedelta(days=1) - timedelta(microseconds=1)
            match_id = hashlib.sha256(
                f"OPENFOOTBALL|{season}|{current_date.isoformat()}|{home_name}|{away_name}".encode()
            ).hexdigest()[:32]
            rows.append(TrainingMatch(
                match_id=match_id,
                competition_id=BRAZIL_SERIE_A,
                season=str(season),
                kickoff_time=kickoff,
                home_team_id=aliases[home_name],
                away_team_id=aliases[away_name],
                home_goals=int(match.group(3)),
                away_goals=int(match.group(4)),
                neutral_venue=False,
                source="OPENFOOTBALL_CC0_1_0",
                completed_at=completed,
                as_of_time=max(completed, SOURCE_AS_OF),
                retrieved_at=retrieved,
                data_version=SOURCE_COMMIT,
            ))
    return tuple(sorted(rows, key=lambda row: (row.kickoff_time, row.match_id)))


def dataset(rows: tuple[TrainingMatch, ...]) -> TrainingDataset:
    """Create a real, date-safe model dataset with exact canonical IDs."""
    teams = frozenset(team for row in rows for team in (row.home_team_id, row.away_team_id))
    return TrainingDataset(matches=rows, known_team_ids=teams, dataset_kind="REAL",
                           temporal_mode="DATE_SAFE_BATCH", assumptions=TEMPORAL_ASSUMPTIONS)


def _config() -> ModelConfig:
    """Return the predeclared development fit policy shared by all models."""
    return ModelConfig(
        time_decay=0.001,
        min_matches=40,
        min_team_matches=3,
        training_window=1095,
        max_goals=10,
        max_score=30,
        profile="development",
        allow_test_data=False,
    )


def _fit(model_type: type[Any], rows: tuple[TrainingMatch, ...], cutoff: date):
    cutoff_time = datetime.combine(cutoff, time.min, UTC)
    selected = tuple(row for row in rows if row.kickoff_time.date() < cutoff)
    if not selected:
        raise ValueError("EMPTY_TRAINING_WINDOW")
    model = model_type().fit(dataset(selected), cutoff_time, _config())
    return model


def _predict(model: Any, row: TrainingMatch) -> tuple[float, float, float]:
    """Call model inference on a historical fixture without using its score."""
    target = Fixture(
        match_id=row.match_id,
        competition_id=BRAZIL_SERIE_A,
        home_team_id=row.home_team_id,
        away_team_id=row.away_team_id,
        kickoff_time=row.kickoff_time + timedelta(days=1),
        source=row.source,
        retrieved_at=row.retrieved_at,
        as_of_time=row.as_of_time,
        data_version=row.data_version,
        season=row.season,
        neutral_venue=row.neutral_venue,
    )
    if not model.supports_fixture(target):
        raise ValueError("MODEL_DOES_NOT_SUPPORT_GOLDEN_FIXTURE")
    values = model._predict_values(target)
    probabilities = (float(values["p_home"]), float(values["p_draw"]), float(values["p_away"]))
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
        raise ValueError("INVALID_MODEL_PROBABILITY")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-7):
        raise ValueError("MODEL_PROBABILITIES_NOT_NORMALIZED")
    return probabilities


def metrics(predictions: list[tuple[tuple[float, float, float], int]]) -> dict[str, float | int]:
    """Compute multiclass log loss, Brier, RPS, accuracy and top-label ECE."""
    if not predictions:
        raise ValueError("EMPTY_EVALUATION_SET")
    log_loss = brier = rps = accuracy = 0.0
    bins: dict[int, list[tuple[float, bool]]] = {}
    for probabilities, outcome in predictions:
        log_loss -= math.log(max(probabilities[outcome], 1e-15))
        brier += sum((p - int(i == outcome)) ** 2 for i, p in enumerate(probabilities))
        rps += sum((sum(probabilities[:k + 1]) - int(outcome <= k)) ** 2 for k in range(2)) / 2
        chosen = max(range(3), key=probabilities.__getitem__)
        confidence = max(probabilities)
        bins.setdefault(min(9, int(confidence * 10)), []).append((confidence, chosen == outcome))
        accuracy += chosen == outcome
    count = len(predictions)
    ece = sum(len(items) / count * abs(
        sum(p for p, _ in items) / len(items) - sum(ok for _, ok in items) / len(items)
    ) for items in bins.values())
    return {
        "sample_size": count,
        "log_loss": log_loss / count,
        "brier": brier / count,
        "rps": rps / count,
        "accuracy": accuracy / count,
        "ece_top_label": ece,
    }


def evaluate_model(model_type: type[Any], rows: tuple[TrainingMatch, ...],
                   start: date, end: date, train_cutoff: date) -> dict[str, float | int]:
    """Fit before an evaluation period and score every available completed fixture."""
    model = _fit(model_type, rows, train_cutoff)
    eligible = [row for row in rows if start <= row.kickoff_time.date() < end
                and row.match_id not in model.training_ids]
    predictions = []
    unsupported = 0
    for row in eligible:
        if not model.supports_fixture(_fixture(row)):
            unsupported += 1
            continue
        p = _predict(model, row)
        outcome = 0 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 2
        predictions.append((p, outcome))
    return {**metrics(predictions), "unsupported_fixture_count": unsupported}


def _raw_data_manifest() -> dict[str, Any]:
    files = {path.name: digest_file(path) for path in sorted(RAW_ROOT.glob("*_br1.txt"))}
    if len(files) != 9:
        raise ValueError(f"EXPECTED_NINE_SEASON_FILES:{len(files)}")
    combined = hashlib.sha256("\n".join(f"{name}:{value}" for name, value in files.items()).encode()).hexdigest()
    return {"files": files, "raw_data_sha256": combined}


def build_release(*, retrieved_at: datetime | None = None) -> dict[str, Any]:
    """Evaluate, freeze, and Golden-check the fixed Brazil club model set."""
    raw = _raw_data_manifest()
    data_root = RAW_ROOT.parent
    source_manifest_path = data_root / "OPENFOOTBALL_SOURCE_SNAPSHOT.json"
    normalized_path = data_root / "normalized_matches.jsonl"
    if source_manifest_path.exists():
        prior_source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if prior_source.get("raw_data_sha256") != raw["raw_data_sha256"]:
            raise ValueError("FROZEN_RAW_SOURCE_HASH_MISMATCH")
        actual_retrieved_at = utc(datetime.fromisoformat(prior_source["download_timestamp"]))
        if normalized_path.exists():
            first_row = json.loads(normalized_path.open(encoding="utf-8").readline())
            actual_retrieved_at = utc(datetime.fromisoformat(first_row["retrieved_at"]))
    else:
        actual_retrieved_at = utc(retrieved_at or datetime.now(UTC))
    rows = parse_matches(retrieved_at=actual_retrieved_at)
    if not rows or any(row.kickoff_time.date() >= TARGET_CUTOFF for row in rows):
        raise ValueError("DATE_SAFE_CUTOFF_FAILED")
    full_dataset = dataset(rows)
    source_snapshot_file = {
        "provider": "OPENFOOTBALL",
        "repository": "https://github.com/openfootball/south-america",
        "source_commit": SOURCE_COMMIT,
        "source_commit_date": "2026-09-08",
        "license": "CC0-1.0",
        "raw_data_sha256": raw["raw_data_sha256"],
        "raw_file_sha256": raw["files"],
        "download_timestamp": actual_retrieved_at.isoformat(),
        "date_safe": True,
        "filter": "completed match rows only; match_date < 2026-10-07",
    }
    if source_manifest_path.exists():
        saved_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if any(saved_manifest.get(key) != source_snapshot_file.get(key)
               for key in ("provider", "repository", "source_commit", "source_commit_date", "license",
                           "raw_data_sha256", "raw_file_sha256", "date_safe", "filter")):
            raise ValueError("FROZEN_SOURCE_MANIFEST_MISMATCH")
    else:
        source_manifest_path.write_text(json.dumps(source_snapshot_file, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    normalized_content = "".join(row.model_dump_json() + "\n" for row in rows)
    if normalized_path.exists():
        if normalized_path.read_text(encoding="utf-8") != normalized_content:
            raise ValueError("FROZEN_NORMALIZED_DATA_MISMATCH")
    else:
        normalized_path.write_text(normalized_content, encoding="utf-8")
    by_season = Counter(row.season for row in rows)
    validation: dict[str, dict[str, float | int]] = {}
    holdout: dict[str, dict[str, float | int]] = {}
    for model_id, model_type in MODEL_CLASSES.items():
        validation[model_id] = evaluate_model(
            model_type, rows, date(2025, 1, 1), date(2026, 1, 1), VALIDATION_CUTOFF
        )
        holdout[model_id] = evaluate_model(
            model_type, rows, TRAINING_CUTOFF, TARGET_CUTOFF, TRAINING_CUTOFF
        )

    final_models: dict[str, Any] = {}
    cutoff_time = datetime.combine(TRAINING_CUTOFF, time.min, UTC)
    for model_id, model_type in MODEL_CLASSES.items():
        final_models[model_id] = _fit(model_type, rows, TRAINING_CUTOFF)
        if final_models[model_id].trained_until != cutoff_time:
            raise ValueError("MODEL_TRAINING_CUTOFF_MISMATCH")

    holdout_rows = [row for row in rows if row.kickoff_time.date() >= TRAINING_CUTOFF]
    goldens: list[dict[str, Any]] = []
    used_teams: set[str] = set()
    for row in holdout_rows:
        teams = {row.home_team_id, row.away_team_id}
        if teams & used_teams:
            continue
        if not all(model.supports_fixture(_fixture(row)) for model in final_models.values()):
            continue
        individual = {key: list(_predict(model, row)) for key, model in final_models.items()}
        ensemble = [sum(values[index] for values in individual.values()) / len(individual)
                    for index in range(3)]
        goldens.append({
            "golden_id": f"GOLDEN_BRAZIL_CLUB_{len(goldens) + 1:03d}",
            "fixture_id": row.match_id,
            "match_date": row.kickoff_time.date().isoformat(),
            "competition_id": BRAZIL_SERIE_A,
            "home_team_id": row.home_team_id,
            "away_team_id": row.away_team_id,
            "neutral_venue": row.neutral_venue,
            "model_probabilities": individual,
            "ensemble_hda": ensemble,
        })
        used_teams.update(teams)
        if len(goldens) == 2:
            break
    if len(goldens) != 2:
        raise ValueError("GOLDEN_BRAZIL_FIXTURES_INSUFFICIENT")

    models_root = PROJECT_ROOT / "cloud_release" / "models" / "club_brazil"
    models_root.mkdir(parents=True, exist_ok=True)
    code_hashes = {
        "model_wrappers": digest_file(Path(__file__).with_name("club_brazil.py")),
        "build_code": digest_file(Path(__file__)),
        "team_mapping": digest_file(TEAM_MAP_PATH),
    }
    model_records: dict[str, Any] = {}
    for model_id, model in final_models.items():
        folder_name = {
            "CLUB_ELO_BRAZIL_V1": "club_elo_brazil_v1",
            "CLUB_DIXON_COLES_BRAZIL_V1": "club_dixon_coles_brazil_v1",
            "CLUB_BIVARIATE_POISSON_BRAZIL_V1": "club_bivariate_poisson_brazil_v1",
        }[model_id]
        directory = models_root / folder_name
        if directory.exists():
            raise FileExistsError(f"FROZEN_ARTIFACT_EXISTS:{directory}")
        artifact = model.save(directory / "artifact")
        model_records[model_id] = {
            "model_name": model.model_name,
            "model_version": model.model_version,
            "artifact_path": f"{folder_name}/artifact",
            "artifact_sha256": artifact.payload_sha256,
            "training_cutoff": TRAINING_CUTOFF.isoformat(),
            "training_range": [min(row.kickoff_time.date() for row in rows
                                    if row.match_id in model.training_ids).isoformat(),
                               max(row.kickoff_time.date() for row in rows
                                   if row.match_id in model.training_ids).isoformat()],
            "training_match_count": len(model.training_ids),
            "training_team_count": len(model.teams),
            "training_data_hash": model.training_data_hash,
            "config": model.config.model_dump(mode="json"),
            "supported_teams": sorted(model.teams),
            "supported_competition": BRAZIL_SERIE_A,
            "validation_metrics": validation[model_id],
            "final_holdout_metrics": holdout[model_id],
        }
        per_model_manifest = {
            "model_id": model_id,
            **model_records[model_id],
            "data_source": "OPENFOOTBALL",
            "data_license": "CC0-1.0",
            "source_commit": SOURCE_COMMIT,
            "raw_data_sha256": raw["raw_data_sha256"],
            "normalized_dataset_sha256": full_dataset.data_hash,
            "code_sha256": code_hashes,
            "calibration": "Model-native only; no market calibration or separate learned calibrator",
            "date_safe": True,
            "runtime_fit": False,
        }
        (directory / "manifest.json").write_text(
            json.dumps(per_model_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    target_aliases = ["Vitória", "Chapecoense", "Botafogo", "Vasco da Gama", "Remo",
                      "Grêmio", "Red Bull Bragantino", "Mirassol", "Internacional", "Corinthians"]
    aliases = load_mapping()
    supported_ids = set.intersection(*(set(model.teams) for model in final_models.values()))
    target_support = {name: aliases[name] in supported_ids for name in target_aliases}
    if not all(target_support.values()):
        raise ValueError(f"TARGET_TEAM_HISTORY_UNAVAILABLE:{target_support}")

    source_snapshot = {
        "data_source": "OPENFOOTBALL",
        "data_license": "CC0-1.0",
        "repository_url": "https://github.com/openfootball/south-america",
        "source_commit": SOURCE_COMMIT,
        "source_commit_date": "2026-09-08",
        "source_as_of_date_sentinel": SOURCE_AS_OF.isoformat(),
        "source_as_of_semantics": "DATE_SAFE_BATCH boundary sentinel; exact upstream commit time is not asserted",
        "download_timestamp": utc(retrieved_at or datetime.now(UTC)).isoformat(),
        "raw_data_sha256": raw["raw_data_sha256"],
        "raw_file_sha256": raw["files"],
        "normalized_dataset_sha256": full_dataset.data_hash,
        "training_cutoff": TRAINING_CUTOFF.isoformat(),
        "date_safe": True,
        "temporal_mode": "DATE_SAFE_BATCH",
        "match_date_filter": "match_date < 2026-10-07",
        "source_matches_by_season": dict(sorted(by_season.items())),
        "source_match_count": len(rows),
        "source_team_count": len(full_dataset.known_team_ids),
        "source_date_range": [min(r.kickoff_time.date() for r in rows).isoformat(),
                              max(r.kickoff_time.date() for r in rows).isoformat()],
        "validation_window": ["2025-01-01", "2025-12-31"],
        "validation_training_cutoff": VALIDATION_CUTOFF.isoformat(),
        "final_holdout_window": [TRAINING_CUTOFF.isoformat(),
                                 max(r.kickoff_time.date() for r in rows).isoformat()],
        "neutral_venue_policy": "SOURCE_FIELD_ABSENT_DOMESTIC_HOME_ROLE_ENCODED_NON_NEUTRAL",
        "timezone_policy": "DATE_SAFE_BATCH; no exact kickoff timestamp is inferred from local match time",
        "recency_policy": {"training_window_days": 1095, "goal_model_time_decay_xi": 0.001,
                           "elo_chronological_updates": True},
        "target_team_history_coverage": target_support,
        "temporal_assumptions": list(TEMPORAL_ASSUMPTIONS),
    }
    snapshot_material = json.dumps({"models": model_records, "source": source_snapshot,
                                    "goldens": goldens}, sort_keys=True).encode()
    snapshot_id = "R3-BRAZIL-" + hashlib.sha256(snapshot_material).hexdigest()[:24]
    manifest = {
        "snapshot_id": snapshot_id,
        "release_status": "FROZEN_BLIND_TEST_ELIGIBLE",
        "data": source_snapshot,
        "code_sha256": code_hashes,
        "models": model_records,
        "ensemble": {"method": "EQUAL_WEIGHT_AVAILABLE_MODELS_V1",
                     "models": list(model_records), "calibration": "MODEL_NATIVE; NO_ADDITIONAL_CALIBRATOR"},
        "golden_references": goldens,
        "golden_test_pass": False,
        "cloud_portable": False,
        "supports_fixture": "Exact competition ID and explicit canonical team IDs only; unknown venue is rejected",
    }
    manifest_path = models_root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"FROZEN_SNAPSHOT_EXISTS:{manifest_path}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _fixture(row: TrainingMatch) -> Fixture:
    """Build a date-safe fixture sentinel for inference, not an asserted kickoff time."""
    return Fixture(
        match_id=row.match_id,
        competition_id=BRAZIL_SERIE_A,
        home_team_id=row.home_team_id,
        away_team_id=row.away_team_id,
        kickoff_time=row.kickoff_time + timedelta(days=1),
        source=row.source,
        retrieved_at=row.retrieved_at,
        as_of_time=row.as_of_time,
        data_version=row.data_version,
        season=row.season,
        neutral_venue=row.neutral_venue,
    )
