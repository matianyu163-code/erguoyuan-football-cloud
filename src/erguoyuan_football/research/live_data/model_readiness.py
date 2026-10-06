"""Independent model data qualification; no fit or prediction imports."""

from __future__ import annotations

from collections.abc import Mapping

from erguoyuan_football.research.live_data.model_requirements import (
    ModelDataRequirement,
    RequirementLevel,
)
from erguoyuan_football.research.live_data.readiness import (
    DataAvailability,
    DataReadinessItem,
    ModelReadiness,
)
from erguoyuan_football.research.samples.sample_hierarchy import SampleHierarchy


class ModelReadinessEvaluator:
    """Check each real model's declared data/sample policy independently."""

    def evaluate(self, requirement: ModelDataRequirement,
                 items: Mapping[str, DataReadinessItem],
                 hierarchy: SampleHierarchy | None, *, fixture_verified: bool
                 ) -> ModelReadiness:
        """A successful verdict licenses only a future fit attempt, never predicts."""
        policy = requirement.sample_policy
        reasons: list[str] = []
        missing_required = [name for name, level in requirement.inputs.items()
                            if level == RequirementLevel.REQUIRED and
                            items[name].status != DataAvailability.AVAILABLE]
        missing_optional = [name for name, level in requirement.inputs.items()
                            if level != RequirementLevel.REQUIRED and
                            items[name].status != DataAvailability.AVAILABLE]
        if not fixture_verified:
            reasons.append("FIXTURE_NOT_VERIFIED")
        if policy is None:
            reasons.append("MODEL_SAMPLE_POLICY_MISSING")
        if hierarchy is None:
            reasons.append("SAMPLE_HIERARCHY_MISSING")
        training_sample_count = 0
        team_coverage = 0
        training_team_ids: tuple[str, ...] = ()
        if policy is not None and hierarchy is not None:
            home_count = hierarchy.home_team_direct_matches
            away_count = hierarchy.away_team_direct_matches
            if min(home_count, away_count) < policy.min_direct_each:
                reasons.append(f"INFERENCE_DIRECT_SAMPLE_LOW:{home_count}/{away_count}"
                               f"<{policy.min_direct_each}")
            direct_lines = {row.match_id: row for row in
                            (*hierarchy.home_direct, *hierarchy.away_direct)}
            direct_ids = set(direct_lines)
            competition_ids = {row.match_id for row in hierarchy.competition}
            same_competition_ids = competition_ids | {
                row.match_id for row in (*hierarchy.home_direct, *hierarchy.away_direct)
                if row.competition_id == hierarchy.competition_id}
            sample_lines = {
                "DIRECT": direct_lines,
                "COMPETITION": {row.match_id: row for row in hierarchy.competition},
                "COMPARABLE": {row.match_id: row for row in hierarchy.comparable},
                "PRIOR": {row.match_id: row for row in (
                    *hierarchy.federation_prior, *hierarchy.age_group_prior,
                    *hierarchy.gender_prior)},
            }
            training_levels = (requirement.training.sample_levels
                               if requirement.training else ())
            if training_levels:
                total_ids = set().union(*(set(sample_lines[level]) for level in training_levels
                                          if level in sample_lines))
                if policy.allow_multi_competition:
                    total_ids &= {row.match_id for row in (
                        *hierarchy.home_direct, *hierarchy.away_direct,
                        *hierarchy.competition, *hierarchy.comparable,
                        *hierarchy.federation_prior, *hierarchy.age_group_prior,
                        *hierarchy.gender_prior)}
            else:
                total_ids = ((direct_ids | competition_ids) if policy.allow_multi_competition
                             else set(same_competition_ids))
            if policy.prior_allowed and not training_levels:
                total_ids |= {row.match_id for row in (*hierarchy.federation_prior,
                    *hierarchy.age_group_prior, *hierarchy.gender_prior)}
            training_minimum = (
                requirement.training.min_training_rows
                if requirement.training and requirement.training.min_training_rows is not None
                else policy.training_min_matches if policy.training_min_matches is not None
                else policy.min_total)
            if len(total_ids) < training_minimum:
                reasons.append(f"TRAINING_SAMPLE_LOW:{len(total_ids)}<{training_minimum}")
            if len(same_competition_ids) < policy.min_competition:
                reasons.append(f"INFERENCE_COMPETITION_SAMPLE_LOW:{len(same_competition_ids)}"
                               f"<{policy.min_competition}")
            training_matches = {row.match_id: row for group in sample_lines.values()
                                for row in group.values() if row.match_id in total_ids}
            training_team_ids = tuple(sorted({
                team for row in training_matches.values()
                for team in (row.home_team_id, row.away_team_id)
                if team and team != "UNKNOWN"
            }))
            team_coverage = len(training_team_ids)
            training_sample_count = len(training_matches)
            if (policy.min_total > 0 and team_coverage
                    < policy.min_training_team_matches):
                reasons.append("TRAINING_TEAM_COVERAGE_LOW:"
                               f"{team_coverage}"
                               f"<{policy.min_training_team_matches}")
            if requirement.inference and requirement.inference.supports_prior:
                prior_count = sum(len(sample_lines[level]) for level in
                                  ("COMPETITION", "COMPARABLE", "PRIOR"))
                if prior_count < requirement.inference.min_prior_samples:
                    reasons.append(f"INFERENCE_PRIOR_SAMPLE_LOW:{prior_count}<"
                                   f"{requirement.inference.min_prior_samples}")
            if policy.require_three_outcomes and (
                items["THREE_WAY_MAPPER_TRAINING"].status != DataAvailability.AVAILABLE):
                missing_required.append("THREE_WAY_MAPPER_TRAINING")
            if policy.require_xg_for_named_mode and (
                items["XG"].status != DataAvailability.AVAILABLE):
                missing_required.append("XG")
            if policy.require_current_odds and (
                items["ODDS"].status != DataAvailability.AVAILABLE):
                missing_required.append("ODDS")
            if policy.require_artifact and (
                items["ML_ARTIFACT"].status != DataAvailability.AVAILABLE
                    or requirement.model_id not in items["ML_ARTIFACT"].model_ids):
                missing_required.append("ML_ARTIFACT")
            if hierarchy.quality.source_conflict_count:
                reasons.append("SOURCE_CONFLICT")
        missing_required = list(dict.fromkeys(missing_required))
        reasons.extend(f"{name}:{items[name].status.value}" for name in missing_required)
        ready = not reasons
        preferred_direct = policy.preferred_direct_each if policy is not None else 0
        degraded = ready and (bool(missing_optional) or bool(hierarchy and
            (hierarchy.quality.roster_continuity_unknown or
             hierarchy.quality.opponent_diversity_score < 0.3 or
             min(hierarchy.home_team_direct_matches,
                 hierarchy.away_team_direct_matches) < preferred_direct)))
        uncertainty = ("VERY_HIGH" if not ready else "HIGH" if degraded else
                       "MEDIUM" if hierarchy and min(
                           hierarchy.home_team_direct_matches,
                           hierarchy.away_team_direct_matches) < 10 else "LOW")
        training_blocked = any(reason.startswith(("TRAINING_", "SOURCE_CONFLICT"))
                               for reason in reasons)
        inference_blocked = any(reason.startswith(("INFERENCE_", "DIRECT_SAMPLE_LOW",
                                                   "COMPETITION_SAMPLE_LOW"))
                                for reason in reasons)
        non_artifact_missing = [name for name in missing_required if name != "ML_ARTIFACT"]
        input_status = "BLOCKED" if inference_blocked or non_artifact_missing else "READY"
        training_status = "BLOCKED" if training_blocked else "READY"
        artifact_status = ("NOT_REQUIRED" if not policy or not policy.require_artifact else
                           "READY" if "ML_ARTIFACT" not in missing_required else "BLOCKED")
        execution_status = ("READY" if training_status == "READY"
                            and input_status == "READY"
                            and artifact_status != "BLOCKED" else "BLOCKED")
        stage_statuses = {"TRAINING_READY": training_status,
                          "ARTIFACT_READY": artifact_status,
                          "INPUT_READY": input_status,
                          "EXECUTION_READY": execution_status,
                          "OUTPUT_VALID": "NOT_RUN"}
        return ModelReadiness(
            requirement.model_id, ready, degraded, tuple(missing_required),
            tuple(missing_optional),
            tuple(name for name in requirement.inputs if
                  items[name].status == DataAvailability.STALE),
            tuple(name for name in requirement.inputs if
                  items[name].status == DataAvailability.CONFLICT),
            tuple(reasons), "DEGRADED" if degraded else "READY" if ready else "BLOCKED",
            uncertainty, stage_statuses,
            training_sample_count, team_coverage, training_team_ids,
        )
