"""Build model-specific, PIT-filtered inputs from the canonical sample hierarchy."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from erguoyuan_football.models.training import TrainingDataset, TrainingMatch
from erguoyuan_football.research.samples.bayesian_prior import BayesianPrior
from erguoyuan_football.research.samples.bayesian_prior_builder import (
    BayesianPriorBuilder,
)
from erguoyuan_football.research.samples.match_deduplicator import HistoricalMatchSample
from erguoyuan_football.research.samples.sample_hierarchy import SampleHierarchy
from erguoyuan_football.research.samples.sample_repository import SampleRepository


class PriorSupport(StrEnum):
    """Declared mathematical scope of a model input adapter."""

    DIRECT_ONLY = "DIRECT_ONLY"
    DIRECT_PLUS_COMPETITION = "DIRECT_PLUS_COMPETITION"
    HIERARCHICAL_PRIOR = "HIERARCHICAL_PRIOR"
    FEATURE_BASED = "FEATURE_BASED"
    MARKET_ONLY = "MARKET_ONLY"
    CONTEXT_ONLY = "CONTEXT_ONLY"


class ModelInputError(ValueError):
    """Invalid hierarchy, missing sample rows or point-in-time violation."""


@dataclass(frozen=True)
class ModelInputBundle:
    """Immutable-by-contract model input with isolated sample types and lineage."""

    model_name: str
    direct_samples: tuple[HistoricalMatchSample, ...]
    competition_samples: tuple[HistoricalMatchSample, ...]
    comparable_samples: tuple[HistoricalMatchSample, ...]
    prior_samples: tuple[HistoricalMatchSample, ...]
    feature_data: Mapping[str, object]
    evidence_ids: tuple[str, ...]
    cutoff: datetime
    lineage: Mapping[str, tuple[str, ...]]
    degraded: bool
    prior_support: PriorSupport
    training_dataset: TrainingDataset
    input_schema_version: str = "MODEL_INPUT_BUNDLE_V1"
    sample_hierarchy_version: str = "SAMPLE_HIERARCHY_V1"
    prior_type: str | None = None
    prior_parameter_source: str | None = None
    prior_strength: float | None = None
    prior_derived_from: tuple[str, ...] = ()
    bayesian_prior: BayesianPrior | None = None

    @property
    def sample_counts(self) -> dict[str, int]:
        """Return counts by semantic sample category, with prior never relabelled."""
        return {"direct": len(self.direct_samples),
                "competition": len(self.competition_samples),
                "comparable": len(self.comparable_samples),
                "prior": len(self.prior_samples),
                "prior_distribution": (self.bayesian_prior.sample_count
                                       if self.bayesian_prior else 0),
                "training": len(self.training_dataset.matches)}


class ModelInputAdapter(ABC):
    """One model's declared sample policy; no fitting or prediction occurs here."""

    model_name: str
    prior_support: PriorSupport
    sample_levels: tuple[str, ...]

    def __init__(self, *, prior_support: PriorSupport,
                 sample_levels: tuple[str, ...]) -> None:
        self.prior_support = prior_support
        self.sample_levels = sample_levels

    def build_input(self, repository: SampleRepository, hierarchy: SampleHierarchy,
                    cutoff: datetime, *, degraded: bool = False,
                    feature_data: Mapping[str, object] | None = None) -> ModelInputBundle:
        """Resolve every lineage row, enforce cutoff, preserve category boundaries."""
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ModelInputError("UTC_CUTOFF_REQUIRED")
        if hierarchy.generated_at > cutoff:
            raise ModelInputError("HIERARCHY_GENERATED_AFTER_CUTOFF")
        rows_by_id: dict[str, HistoricalMatchSample] = {}
        for row in repository.samples:
            if row.match_id in rows_by_id and rows_by_id[row.match_id] != row:
                raise ModelInputError(f"DUPLICATE_SAMPLE_ID_CONFLICT:{row.match_id}")
            rows_by_id[row.match_id] = row

        def materialize(name: str, lineages) -> tuple[HistoricalMatchSample, ...]:
            result = []
            for lineage in lineages:
                row = rows_by_id.get(lineage.match_id)
                if row is None:
                    raise ModelInputError(f"SAMPLE_LINEAGE_MISSING:{lineage.match_id}")
                if row.kickoff >= cutoff or row.fetched_at > cutoff:
                    raise ModelInputError(f"FUTURE_EVIDENCE_REJECTED:{row.match_id}")
                if (row.competition_id != lineage.competition_id
                        or row.evidence_ids != lineage.evidence_ids):
                    raise ModelInputError(f"SAMPLE_LINEAGE_MISMATCH:{row.match_id}")
                result.append(row)
            unique = {row.fingerprint: row for row in result}
            return tuple(sorted(unique.values(), key=lambda row: (row.kickoff, row.match_id)))

        direct_lines = tuple({row.match_id: row for row in
                              (*hierarchy.home_direct, *hierarchy.away_direct)}.values())
        direct = materialize("DIRECT", direct_lines)
        competition = materialize("COMPETITION", hierarchy.competition)
        comparable = materialize("COMPARABLE", hierarchy.comparable)
        federation_prior = materialize("FEDERATION_PRIOR", hierarchy.federation_prior)
        age_prior = materialize("AGE_GROUP_PRIOR", hierarchy.age_group_prior)
        gender_prior = materialize("GENDER_PRIOR", hierarchy.gender_prior)
        prior_by_id = {row.match_id: row for row in
                       (*federation_prior, *age_prior, *gender_prior)}
        prior = tuple(sorted(prior_by_id.values(), key=lambda row: (row.kickoff, row.match_id)))
        selected: list[HistoricalMatchSample] = []
        for category, rows in (("DIRECT", direct), ("COMPETITION", competition),
                               ("COMPARABLE", comparable), ("PRIOR", prior)):
            if category in self.sample_levels:
                selected.extend(rows)
        selected_ids = [row.match_id for row in selected]
        if len(selected_ids) != len(set(selected_ids)):
            raise ModelInputError("SAMPLE_CATEGORY_OVERLAP")
        selected_rows = tuple(sorted(selected, key=lambda row: (row.kickoff, row.match_id)))
        training = self._training_dataset(selected_rows)
        bayesian_prior = (BayesianPriorBuilder().build(repository, hierarchy, cutoff)
                          if self.model_name == "BAYESIAN_HIERARCHICAL_V1" else None)
        prior_evidence = tuple(sorted({evidence for row in prior
                                      for evidence in row.evidence_ids}))
        if bayesian_prior is not None:
            prior_evidence = bayesian_prior.evidence_ids
        evidence_ids = tuple(sorted({evidence for row in selected_rows
                                     for evidence in row.evidence_ids} | set(prior_evidence)))
        return ModelInputBundle(
            self.model_name, direct, competition, comparable, prior,
            dict(feature_data or {}), evidence_ids, cutoff,
            {"direct": tuple(e for row in direct for e in row.evidence_ids),
             "competition": tuple(e for row in competition for e in row.evidence_ids),
             "comparable": tuple(e for row in comparable for e in row.evidence_ids),
             "federation_prior": tuple(e for row in federation_prior for e in row.evidence_ids),
             "age_group_prior": tuple(e for row in age_prior for e in row.evidence_ids),
             "gender_prior": tuple(e for row in gender_prior for e in row.evidence_ids),
             "prior": prior_evidence}, degraded, self.prior_support, training,
            prior_type=(bayesian_prior.prior_type if bayesian_prior else
                        "AGE_GROUP_OR_FEDERATION" if prior else None),
            prior_parameter_source=("EMPIRICAL_PIT_HIERARCHICAL_RATE_PRIOR"
                                    if bayesian_prior else
                                    "NOT_USED_EXISTING_MODEL_HAS_NO_EXTERNAL_PRIOR_HOOK"
                                    if prior else None),
            prior_strength=(bayesian_prior.effective_strength if bayesian_prior else None),
            prior_derived_from=prior_evidence, bayesian_prior=bayesian_prior,
        )

    @staticmethod
    def _training_dataset(rows: tuple[HistoricalMatchSample, ...]) -> TrainingDataset:
        """Convert observed completed-score evidence with retrieval-time PIT bounds."""
        if not rows:
            return TrainingDataset(matches=(), known_team_ids=frozenset(),
                                   dataset_kind="REAL")
        teams = frozenset(team for row in rows for team in
                          (row.home_team_id, row.away_team_id))
        synthetic = any("SYNTHETIC_TEST" in provider.upper()
                        for row in rows for provider in row.provider_ids)
        training_rows = []
        for row in rows:
            if row.fetched_at <= row.kickoff:
                raise ModelInputError(f"RESULT_OBSERVATION_BEFORE_KICKOFF:{row.match_id}")
            version = hashlib.sha256("|".join(sorted(row.evidence_ids)).encode()).hexdigest()
            # The provider retrieval timestamp is the first defensible time this system
            # can confirm the completed result; it is used conservatively for PIT.
            training_rows.append(TrainingMatch(
                match_id=row.match_id, competition_id=row.competition_id,
                season=str(row.kickoff.year), kickoff_time=row.kickoff,
                home_team_id=row.home_team_id, away_team_id=row.away_team_id,
                home_goals=row.home_goals, away_goals=row.away_goals,
                neutral_venue=row.neutral_venue,
                source="+".join(row.provider_ids) if not synthetic else "SYNTHETIC_TEST",
                completed_at=row.fetched_at, as_of_time=row.fetched_at,
                retrieved_at=row.fetched_at, data_version=version,
            ))
        return TrainingDataset(matches=tuple(training_rows), known_team_ids=teams,
                               dataset_kind="SYNTHETIC_TEST" if synthetic else "REAL")

    @abstractmethod
    def supports_prior_execution(self) -> bool:
        """Whether the existing model math consumes external prior samples."""


class DixonColesInputAdapter(ModelInputAdapter):
    """Dixon-Coles fits target-competition direct data and league baseline rows."""

    model_name = "DIXON_COLES_V1"

    def __init__(self) -> None:
        super().__init__(prior_support=PriorSupport.DIRECT_PLUS_COMPETITION,
                         sample_levels=("DIRECT", "COMPETITION"))

    def supports_prior_execution(self) -> bool:
        return False


class BivariatePoissonInputAdapter(ModelInputAdapter):
    """Bivariate Poisson uses direct and target-competition scored matches."""

    model_name = "BIVARIATE_POISSON_V1"

    def __init__(self) -> None:
        super().__init__(prior_support=PriorSupport.DIRECT_PLUS_COMPETITION,
                         sample_levels=("DIRECT", "COMPETITION"))

    def supports_prior_execution(self) -> bool:
        return False


class HierarchicalBayesianInputAdapter(ModelInputAdapter):
    """Fit target-team rows and supply disjoint competition evidence as a prior."""

    model_name = "BAYESIAN_HIERARCHICAL_V1"

    def __init__(self) -> None:
        super().__init__(prior_support=PriorSupport.DIRECT_PLUS_COMPETITION,
                         sample_levels=("DIRECT",))

    def supports_prior_execution(self) -> bool:
        return True


class EloInputAdapter(ModelInputAdapter):
    """Elo learns chronological local ratings from competition result history."""

    model_name = "ELO_V1"

    def __init__(self) -> None:
        super().__init__(prior_support=PriorSupport.DIRECT_PLUS_COMPETITION,
                         sample_levels=("DIRECT", "COMPETITION"))

    def supports_prior_execution(self) -> bool:
        return False


class PiRatingInputAdapter(ModelInputAdapter):
    """Pi rating receives only chronological source-backed competition results."""

    model_name = "PI_RATING_V1"

    def __init__(self) -> None:
        super().__init__(prior_support=PriorSupport.DIRECT_PLUS_COMPETITION,
                         sample_levels=("DIRECT", "COMPETITION"))

    def supports_prior_execution(self) -> bool:
        return False


class DynamicBayesianInputAdapter(ModelInputAdapter):
    """Dynamic Bayes receives time-stamped direct and competition histories."""

    model_name = "DYNAMIC_BAYESIAN_POISSON_V1"

    def __init__(self) -> None:
        super().__init__(prior_support=PriorSupport.DIRECT_PLUS_COMPETITION,
                         sample_levels=("DIRECT", "COMPETITION"))

    def supports_prior_execution(self) -> bool:
        return False


class SPILikeInputAdapter(ModelInputAdapter):
    """SPI-like model uses results; missing xG is not synthesized from other stats."""

    model_name = "CORE_SPI_LIKE_V1"

    def __init__(self) -> None:
        super().__init__(prior_support=PriorSupport.DIRECT_PLUS_COMPETITION,
                         sample_levels=("DIRECT", "COMPETITION"))

    def supports_prior_execution(self) -> bool:
        return False


class XGEloInputAdapter(ModelInputAdapter):
    """xG-Elo-like input requires source-backed xG and xGA feature evidence."""

    model_name = "CORE_OPTA_XG_ELO_LIKE_V1"

    def __init__(self) -> None:
        super().__init__(prior_support=PriorSupport.FEATURE_BASED,
                         sample_levels=("DIRECT", "COMPETITION"))

    def supports_prior_execution(self) -> bool:
        return False


class HistoricalMarketInputAdapter(ModelInputAdapter):
    """Historical-market model requires independent PIT current odds inputs."""

    model_name = "HISTORICAL_MARKET_BAYESIAN_POISSON_V1"

    def __init__(self) -> None:
        super().__init__(prior_support=PriorSupport.MARKET_ONLY,
                         sample_levels=("DIRECT", "COMPETITION"))

    def supports_prior_execution(self) -> bool:
        return False


class XGBoostInputAdapter(ModelInputAdapter):
    """XGBoost requires a matching frozen feature vector and trained artifact."""

    model_name = "CORE_XGBOOST_V1"

    def __init__(self) -> None:
        super().__init__(prior_support=PriorSupport.FEATURE_BASED, sample_levels=())

    def supports_prior_execution(self) -> bool:
        return False


class CatBoostInputAdapter(ModelInputAdapter):
    """CatBoost requires a matching frozen feature vector and trained artifact."""

    model_name = "CORE_CATBOOST_V1"

    def __init__(self) -> None:
        super().__init__(prior_support=PriorSupport.FEATURE_BASED, sample_levels=())

    def supports_prior_execution(self) -> bool:
        return False


def default_model_input_adapters() -> dict[str, ModelInputAdapter]:
    """Construct the 11 explicit input adapters without constructing any model."""
    adapters: tuple[ModelInputAdapter, ...] = (
        DixonColesInputAdapter(), BivariatePoissonInputAdapter(),
        HierarchicalBayesianInputAdapter(), EloInputAdapter(), PiRatingInputAdapter(),
        DynamicBayesianInputAdapter(), SPILikeInputAdapter(), XGEloInputAdapter(),
        HistoricalMarketInputAdapter(), XGBoostInputAdapter(), CatBoostInputAdapter(),
    )
    return {adapter.model_name: adapter for adapter in adapters}
