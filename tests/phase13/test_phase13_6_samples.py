"""SYNTHETIC_TEST sample lineage, separation and independent model thresholds."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.research.live_data.model_readiness import (
    ModelReadinessEvaluator,
)
from erguoyuan_football.research.live_data.model_requirements import (
    ModelDataRequirement,
    ModelSamplePolicy,
    RequirementLevel,
)
from erguoyuan_football.research.live_data.readiness import (
    DataAvailability,
    DataReadinessItem,
    _item,
)
from erguoyuan_football.research.samples.match_deduplicator import (
    HistoricalMatchSample,
    deduplicate_matches,
)
from erguoyuan_football.research.samples.sample_builder import SampleBuilder
from erguoyuan_football.research.samples.sample_repository import SampleRepository
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.time_utils import utc_iso

AT = datetime(2026, 10, 1, tzinfo=UTC)
COMP = CompetitionIdentity("UEFA_W_U17", "Women U17", "UEFA", "Europe", "NATIONAL")
HOME = TeamIdentity("UEFA_GER_W_U17", "Germany Women U17", "Germany", "UEFA",
                    ["Germany Women U17"], "NATIONAL", "WOMEN", "U17", "FIRST_TEAM")
AWAY = TeamIdentity("UEFA_GRE_W_U17", "Greece Women U17", "Greece", "UEFA",
                    ["Greece Women U17"], "NATIONAL", "WOMEN", "U17", "FIRST_TEAM")


def _match(index: int, home: str, away: str, *, comp: str = "UEFA_W_U17",
           gender: str = "WOMEN", age: str = "U17", federation: str = "UEFA",
           score: tuple[int, int] = (1, 0), provider: str = "SYNTHETIC_TEST"
           ) -> HistoricalMatchSample:
    return HistoricalMatchSample(
        str(index), home, away, comp, AT - timedelta(days=index + 1),
        score[0], score[1], (f"SYNTHETIC_TEST_E{index}_{provider}",),
        (provider,), 2, AT - timedelta(days=1), federation, gender,
        age, "FRIENDLY",
    )


def _build(rows: tuple[HistoricalMatchSample, ...]):
    return SampleBuilder(SampleRepository(rows)).build(
        HOME, AWAY, COMP, cutoff=AT, competition_type="FRIENDLY")


def test_age_gender_and_squad_separation_with_prior_lineage() -> None:
    rows = (
        _match(1, HOME.team_id, "W_U17_OPP"),
        _match(2, AWAY.team_id, "W_U17_OPP2"),
        _match(3, "UEFA_GER_W_U16", "W_U16_OPP", age="U16"),
        _match(4, "UEFA_GER_M_U17", "M_U17_OPP", gender="MEN"),
        _match(5, "BARCELONA_B", "OPP", gender="WOMEN"),
        _match(6, "OTHER_W_U17", "OPP", comp="OTHER_W_U17"),
    )
    hierarchy = _build(rows)
    assert hierarchy.home_team_direct_matches == 1
    assert hierarchy.away_team_direct_matches == 1
    assert {row.match_id for row in hierarchy.age_group_prior} == {"3"}
    assert "4" not in {row.match_id for row in (*hierarchy.home_direct,
        *hierarchy.away_direct, *hierarchy.comparable, *hierarchy.age_group_prior)}
    assert {row.match_id for row in hierarchy.comparable} == {"6"}
    assert all(row.evidence_ids for row in hierarchy.age_group_prior)


def test_deduplication_and_score_conflict_exclusion() -> None:
    first = _match(1, HOME.team_id, AWAY.team_id, provider="SYNTHETIC_A")
    duplicate = _match(1, HOME.team_id, AWAY.team_id, provider="SYNTHETIC_B")
    merged = deduplicate_matches((first, duplicate))
    assert len(merged.samples) == 1
    assert set(merged.samples[0].provider_ids) == {"SYNTHETIC_A", "SYNTHETIC_B"}
    conflict = _match(1, HOME.team_id, AWAY.team_id, provider="SYNTHETIC_C",
                      score=(0, 2))
    rejected = deduplicate_matches((first, duplicate, conflict))
    assert not rejected.samples and len(rejected.conflicts) == 1
    assert _build((first, duplicate, conflict)).quality.source_conflict_count == 1


def test_future_retrieval_and_event_time_excluded() -> None:
    current = _match(1, HOME.team_id, AWAY.team_id)
    future_fetch = HistoricalMatchSample(**{**current.__dict__,
        "match_id": "future", "fetched_at": AT + timedelta(seconds=1)})
    future_event = HistoricalMatchSample(**{**current.__dict__,
        "match_id": "future-event", "kickoff": AT + timedelta(days=1)})
    hierarchy = _build((current, future_fetch, future_event))
    assert hierarchy.home_team_direct_matches == 1


def test_model_specific_39_does_not_trigger_global_gate() -> None:
    rows = tuple(_match(index, HOME.team_id if index % 2 else AWAY.team_id,
                        f"OPP_{index}") for index in range(1, 40))
    hierarchy = _build(rows)
    items = {"HISTORICAL_RESULTS": DataReadinessItem(
        "HISTORICAL_RESULTS", DataAvailability.AVAILABLE)}
    evaluator = ModelReadinessEvaluator()
    small_policy = ModelDataRequirement("SYNTHETIC_ELO", {
        "HISTORICAL_RESULTS": RequirementLevel.REQUIRED},
        ModelSamplePolicy(12, 3, 0, False, min_training_team_matches=1))
    larger_policy = ModelDataRequirement("SYNTHETIC_GOAL", {
        "HISTORICAL_RESULTS": RequirementLevel.REQUIRED},
        ModelSamplePolicy(40, 3, 0, False, min_training_team_matches=1))
    assert evaluator.evaluate(small_policy, items, hierarchy,
                              fixture_verified=True).status in {"READY", "DEGRADED"}
    assert evaluator.evaluate(larger_policy, items, hierarchy,
                              fixture_verified=True).status == "BLOCKED"


def test_low_direct_but_competition_sufficient_for_hierarchical_policy() -> None:
    direct = tuple(_match(index, HOME.team_id if index % 2 == 0 else AWAY.team_id,
                          f"OPP_{index % 3}") for index in range(1, 12))
    competition = tuple(_match(index, f"A_{index % 10}", f"B_{index % 10}")
                        for index in range(12, 162))
    hierarchy = _build(direct + competition)
    requirement = ModelDataRequirement("SYNTHETIC_HIERARCHICAL", {
        "HISTORICAL_RESULTS": RequirementLevel.REQUIRED},
        ModelSamplePolicy(40, 3, 20, False))
    item = {"HISTORICAL_RESULTS": DataReadinessItem(
        "HISTORICAL_RESULTS", DataAvailability.AVAILABLE)}
    assert hierarchy.home_team_direct_matches == 5
    assert hierarchy.away_team_direct_matches == 6
    assert hierarchy.competition_matches == 150
    assert ModelReadinessEvaluator().evaluate(requirement, item, hierarchy,
                                              fixture_verified=True).ready


def test_prior_allowed_only_when_policy_explicit() -> None:
    direct = (_match(1, HOME.team_id, "OPP"), _match(2, AWAY.team_id, "OPP2"))
    priors = tuple(_match(index, "GER_W_U16", f"YOUTH_{index}", age="U16")
                   for index in range(3, 53))
    hierarchy = _build(direct + priors)
    item = {"HISTORICAL_RESULTS": DataReadinessItem(
        "HISTORICAL_RESULTS", DataAvailability.AVAILABLE)}
    allowed = ModelDataRequirement("SYNTHETIC_PRIOR_MODEL", {
        "HISTORICAL_RESULTS": RequirementLevel.REQUIRED},
        ModelSamplePolicy(40, 1, 0, True, min_training_team_matches=1))
    denied = ModelDataRequirement("SYNTHETIC_NO_PRIOR", {
        "HISTORICAL_RESULTS": RequirementLevel.REQUIRED},
        ModelSamplePolicy(40, 1, 0, False, min_training_team_matches=1))
    evaluator = ModelReadinessEvaluator()
    assert evaluator.evaluate(allowed, item, hierarchy, fixture_verified=True).ready
    assert not evaluator.evaluate(denied, item, hierarchy, fixture_verified=True).ready


def test_extreme_low_sample_not_ready() -> None:
    hierarchy = _build((_match(1, HOME.team_id, "OPP"),
                        _match(2, AWAY.team_id, "OPP")))
    requirement = ModelDataRequirement("SYNTHETIC_ELO", {
        "HISTORICAL_RESULTS": RequirementLevel.REQUIRED},
        ModelSamplePolicy(12, 3, 0, False))
    item = {"HISTORICAL_RESULTS": DataReadinessItem(
        "HISTORICAL_RESULTS", DataAvailability.AVAILABLE)}
    verdict = ModelReadinessEvaluator().evaluate(requirement, item, hierarchy,
                                                  fixture_verified=True)
    assert verdict.status == "BLOCKED"
    assert any(reason.startswith("INFERENCE_DIRECT_SAMPLE_LOW")
               for reason in verdict.reasons)


def test_ml_artifact_and_features_require_compatible_pit_evidence() -> None:
    stamp = utc_iso(AT - timedelta(days=1))
    record = EvidenceRecord(
        "SYNTHETIC_TEST_ML", "ML_ARTIFACT",
        {"model_id": "CORE_XGBOOST_V1", "artifact_exists": True,
         "schema_compatible": True, "trained_until": stamp},
        "SYNTHETIC_TEST", None, stamp, "HIGH", "https://synthetic.invalid/ml",
        stamp,
    )
    item = _item("ML_ARTIFACT", [record], AT)
    assert item.status == DataAvailability.AVAILABLE
    assert item.model_ids == ("CORE_XGBOOST_V1",)
    wrong = EvidenceRecord(**{**record.__dict__, "evidence_id": "SYNTHETIC_TEST_FUTURE",
                              "value": {**record.value,
                                        "trained_until": utc_iso(AT + timedelta(days=1))}})
    assert _item("ML_ARTIFACT", [wrong], AT).status == DataAvailability.INVALID
    feature = EvidenceRecord(**{**record.__dict__, "evidence_id": "SYNTHETIC_TEST_FEATURE",
                                "data_type": "ML_FEATURE_VECTOR", "value": {
                                    "coverage_complete": False,
                                    "schema_compatible": True}})
    assert _item("ML_FEATURE_VECTOR", [feature], AT).status == DataAvailability.INVALID
