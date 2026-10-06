"""Build ordered pre-match features from already frozen CORE evidence only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from erguoyuan_football.contracts.common import Availability, ExecutionStatus
from erguoyuan_football.data.snapshots import HistoricalResult, PredictionSnapshot
from erguoyuan_football.markets.schemas import MarketType
from erguoyuan_football.ml.feature_contract import (
    FORM_FEATURES,
    MARKET_FEATURES,
    MODEL_PREFIXES,
    XG_FEATURES,
    build_feature_schema,
    validate_feature_values,
)
from erguoyuan_football.ml.feature_lineage import OOSFeatureValidator
from erguoyuan_football.ml.schemas import (
    BasePredictionEvidence,
    FeatureLineage,
    FeatureMode,
    MLFeatureVector,
    ModelFamily,
    stable_hash,
)

FEATURE_VERSION = "ML_FEATURE_V1"


def _team_form(history: tuple[HistoricalResult, ...], team_id: str,
               prediction_time: datetime) -> tuple[dict[str, float | None], tuple[HistoricalResult, ...]]:
    relevant = [row for row in history if team_id in (row.fixture.home_team_id, row.fixture.away_team_id)]
    relevant.sort(key=lambda row: (row.fixture.kickoff_time, row.fixture.match_id))
    recent = tuple(relevant[-5:])
    if not recent:
        return {"goals_for_avg": None, "goals_against_avg": None,
                "points_avg": None, "rest_days": None}, ()
    goals_for, goals_against, points = [], [], []
    for row in recent:
        home = row.fixture.home_team_id == team_id
        gf = row.result.home_goals if home else row.result.away_goals
        ga = row.result.away_goals if home else row.result.home_goals
        goals_for.append(gf)
        goals_against.append(ga)
        points.append(3 if gf > ga else 1 if gf == ga else 0)
    rest = (prediction_time - recent[-1].fixture.kickoff_time).total_seconds() / 86_400
    return {"goals_for_avg": sum(goals_for) / len(recent),
            "goals_against_avg": sum(goals_against) / len(recent),
            "points_avg": sum(points) / len(recent), "rest_days": rest}, recent


class MLFeatureBuilder:
    """One pass over frozen snapshots and evidence; never queries an external service."""

    def __init__(self) -> None:
        self.oos_validator = OOSFeatureValidator()

    def build(self, snapshot: PredictionSnapshot, base_predictions: Sequence[BasePredictionEvidence], *,
              mode: FeatureMode, family: ModelFamily, for_training: bool) -> MLFeatureVector:
        schema = build_feature_schema(mode, family)
        fixture = snapshot.match_data_snapshot
        if snapshot.match_id != fixture.match_id or snapshot.prediction_time >= fixture.kickoff_time:
            raise ValueError("ML_SNAPSHOT_IDENTITY_OR_TIME_MISMATCH")
        evidence_by_model: dict[str, BasePredictionEvidence] = {}
        for evidence_item in base_predictions:
            self.oos_validator.validate(evidence_item, snapshot, for_training=for_training)
            model_id = evidence_item.prediction.model_id
            if model_id in evidence_by_model:
                raise ValueError("DUPLICATE_BASE_MODEL_FEATURE")
            evidence_by_model[model_id] = evidence_item
        values: dict[str, float | str | None] = {name: None for name in schema.feature_names}
        lineage: list[FeatureLineage] = [FeatureLineage(feature_names=("horizon_seconds", "neutral_venue"),
            source_type="FIXTURE", source_id=fixture.source, as_of_time=fixture.as_of_time,
            retrieved_at=fixture.retrieved_at, data_version=fixture.data_version,
            dependency_ids=(fixture.match_id,))]
        values["horizon_seconds"] = (fixture.kickoff_time - snapshot.prediction_time).total_seconds()
        values["neutral_venue"] = float(fixture.neutral_venue) if fixture.neutral_venue is not None else None
        used_base: list[BasePredictionEvidence] = []
        for model_id, prefix in MODEL_PREFIXES.items():
            if mode == FeatureMode.NO_MARKET and prefix == "market_bayes":
                continue
            base_evidence = evidence_by_model.get(model_id)
            if base_evidence is not None and self._uses_market(base_evidence) and mode == FeatureMode.NO_MARKET:
                continue
            prediction = base_evidence.prediction if base_evidence is not None else None
            available = prediction is not None and prediction.execution_status == ExecutionStatus.SUCCESS
            values[f"{prefix}_available"] = float(available)
            if not available or prediction is None:
                continue
            if base_evidence is None:
                raise AssertionError("available prediction has no evidence")
            values[f"{prefix}_p_home"] = prediction.p_home
            values[f"{prefix}_p_draw"] = prediction.p_draw
            values[f"{prefix}_p_away"] = prediction.p_away
            if f"{prefix}_lambda_home" in values:
                values[f"{prefix}_lambda_home"] = prediction.lambda_home
                values[f"{prefix}_lambda_away"] = prediction.lambda_away
            used_base.append(base_evidence)
            lineage.append(FeatureLineage(
                feature_names=tuple(name for name in schema.feature_names if name.startswith(prefix + "_")),
                source_type="BASE_MODEL", source_id=model_id,
                as_of_time=prediction.prediction_time, retrieved_at=prediction.prediction_time,
                data_version=f"{prediction.model_version}:{prediction.input_data_version}",
                dependency_ids=(prediction.prediction_id,)))
        form_used: list[HistoricalResult] = []
        for side, team_id in (("home", fixture.home_team_id), ("away", fixture.away_team_id)):
            form, history = _team_form(snapshot.historical_results, team_id, snapshot.prediction_time)
            form_used.extend(history)
            for key, value in form.items():
                values[f"{side}_{key}"] = value
        values["form_available"] = float(bool(form_used))
        if form_used:
            lineage.append(FeatureLineage(feature_names=FORM_FEATURES + ("form_available",),
                source_type="HISTORICAL_RESULTS", source_id="CORE_MATCH_RESULTS",
                as_of_time=max(row.result.as_of_time for row in form_used),
                retrieved_at=max(row.result.retrieved_at for row in form_used),
                data_version=stable_hash(sorted(row.result.data_version for row in form_used)),
                dependency_ids=tuple(sorted({row.result.result_id for row in form_used}))))
        xg_used = []
        for side, team_id in (("home", fixture.home_team_id), ("away", fixture.away_team_id)):
            for metric in ("xg", "xga"):
                matches = [item for item in snapshot.xg_snapshot if item.team_id == team_id
                           and item.availability == Availability.AVAILABLE and metric in item.metrics]
                if len({item.metrics[metric] for item in matches}) > 1:
                    raise ValueError("AMBIGUOUS_XG_SOURCE")
                if matches:
                    values[f"{side}_{metric}"] = float(matches[0].metrics[metric])
                    xg_used.extend(matches)
        values["xg_available"] = float(bool(xg_used))
        if xg_used:
            lineage.append(FeatureLineage(feature_names=XG_FEATURES + ("xg_available",),
                source_type="XG_SNAPSHOT", source_id="CORE_XG_SNAPSHOTS",
                as_of_time=max(item.as_of_time for item in xg_used),
                retrieved_at=max(item.retrieved_at for item in xg_used),
                data_version=stable_hash(sorted(item.data_version for item in xg_used)),
                dependency_ids=tuple(sorted({item.snapshot_id for item in xg_used}))))
        if mode == FeatureMode.WITH_MARKET:
            consensus = [item for item in snapshot.canonical_market_consensus
                         if item.market_type == MarketType.MATCH_1X2]
            if len(consensus) > 1:
                raise ValueError("AMBIGUOUS_MARKET_CONSENSUS")
            values["market_available"] = float(bool(consensus))
            if consensus:
                market_consensus = consensus[0]
                quote_by_id = {quote.quote_id: quote for quote in
                               (snapshot.canonical_market_snapshot.quotes
                                if snapshot.canonical_market_snapshot is not None else ())}
                quotes = [quote_by_id[quote_id] for quote_id in market_consensus.quote_ids
                          if quote_id in quote_by_id]
                if len(quotes) != len(market_consensus.quote_ids):
                    raise ValueError("MARKET_QUOTE_LINEAGE_MISSING")
                for outcome in ("HOME", "DRAW", "AWAY"):
                    values[f"market_consensus_{outcome.lower()}"] = market_consensus.probabilities[outcome]
                values["market_overround"] = market_consensus.overround_mean
                values["market_freshness_seconds"] = float(market_consensus.freshness_seconds)
                lineage.append(FeatureLineage(feature_names=MARKET_FEATURES,
                    source_type="MARKET", source_id=market_consensus.market_snapshot_id,
                    as_of_time=max(quote.as_of_time for quote in quotes if quote.as_of_time is not None),
                    retrieved_at=max(quote.retrieved_at for quote in quotes),
                    data_version=market_consensus.devig_policy_version,
                    dependency_ids=market_consensus.quote_ids))
        if family == ModelFamily.CATBOOST:
            values["competition_id"] = fixture.competition_id
        validate_feature_values(values, schema)
        draft: dict[str, Any] = {
            "match_id": fixture.match_id,
            "prediction_snapshot_id": snapshot.prediction_snapshot_id,
            "input_data_version": snapshot.input_data_version,
            "prediction_time": snapshot.prediction_time, "kickoff_time": fixture.kickoff_time,
            "feature_version": FEATURE_VERSION, "feature_schema_hash": schema.schema_hash,
            "feature_mode": mode, "model_family": family, "features": values,
            "feature_lineage": tuple(lineage), "base_prediction_evidence": tuple(used_base),
        }
        provisional = MLFeatureVector.model_construct(feature_data_hash="PENDING", **draft)
        return MLFeatureVector(feature_data_hash=stable_hash(
            provisional.model_dump(mode="json", exclude={"feature_data_hash"})), **draft)

    def build_many(self, snapshots: Sequence[PredictionSnapshot],
                   base_predictions: Mapping[str, Sequence[BasePredictionEvidence]], *,
                   mode: FeatureMode, family: ModelFamily,
                   for_training: bool) -> tuple[MLFeatureVector, ...]:
        """Batch over already loaded snapshots without one query per feature."""
        return tuple(self.build(snapshot, base_predictions.get(snapshot.match_id, ()), mode=mode,
                                family=family, for_training=for_training) for snapshot in snapshots)

    @staticmethod
    def _uses_market(item: BasePredictionEvidence) -> bool:
        prediction = item.prediction
        return bool(prediction.metadata.get("uses_market")) or any(
            tag.startswith("MARKET_") for tag in prediction.dependency_tags)
