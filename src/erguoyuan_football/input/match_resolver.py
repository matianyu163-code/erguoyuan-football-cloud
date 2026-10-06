"""Resolve only exact, trusted aliases and unique pre-match fixtures."""

from datetime import date

from erguoyuan_football.contracts.common import utc
from erguoyuan_football.data.store import Store
from erguoyuan_football.input.schemas import MatchRequest, ResolutionStatus


class MatchResolver:
    def __init__(self, store: Store, *, min_input_confidence: float = 1.0):
        if not 0 <= min_input_confidence <= 1:
            raise ValueError("confidence threshold must be within [0,1]")
        self.store = store
        self.min_input_confidence = min_input_confidence

    def resolve(self, request: MatchRequest, *, prediction_time, match_date: date | None = None) -> MatchRequest:
        if request.resolution_status == ResolutionStatus.INVALID:
            return request
        if request.input_type == "SCREENSHOT" and request.reason != "VISION_VERIFIED":
            return request
        if request.input_type == "SCREENSHOT" and (request.input_confidence is None or request.input_confidence < self.min_input_confidence):
            return MatchRequest.model_validate({**request.model_dump(), "match_id": None,
                "resolution_status": ResolutionStatus.AMBIGUOUS, "reason": "LOW_VISION_CONFIDENCE"})
        at = utc(prediction_time)
        fixtures = [f for f in self.store.fixtures_at(at) if f.kickoff_time > at
                    and (match_date is None or f.kickoff_time.date() == match_date)]
        ambiguity = False
        pairs = []
        single_ids = ()
        if request.home_team_name and request.away_team_name:
            home = self.store.alias_ids(request.home_team_name, at)
            away = self.store.alias_ids(request.away_team_name, at)
            ambiguity = len(home) > 1 or len(away) > 1
            pairs = [(h, a) for h in home for a in away]
        elif request.team_query:
            single_ids = self.store.alias_ids(request.team_query, at)
            # Enumerate all whitespace partitions using the alias dictionary, never a best guess.
            words = request.team_query.split()
            for split in range(1, len(words)):
                home = self.store.alias_ids(" ".join(words[:split]), at)
                away = self.store.alias_ids(" ".join(words[split:]), at)
                pairs.extend((h, a) for h in home for a in away)
                ambiguity |= (len(home) > 1 and bool(away)) or (len(away) > 1 and bool(home))
            ambiguity |= len(single_ids) > 1 or len(set(pairs)) > 1 or bool(single_ids and pairs)
        competition_ids = None
        if request.competition_name:
            competition_ids = self.store.alias_ids(request.competition_name, at, competition=True)
            ambiguity |= len(competition_ids) > 1
        filtered = []
        for f in fixtures:
            if request.match_id and request.match_id != f.match_id:
                continue
            if request.lottery_match_no and request.lottery_match_no != f.lottery_match_no:
                continue
            if request.kickoff_time and request.kickoff_time != f.kickoff_time:
                continue
            if competition_ids is not None and f.competition_id not in competition_ids:
                continue
            if (request.home_team_name or request.away_team_name or request.team_query) and \
                    (f.home_team_id, f.away_team_id) not in pairs and not (
                        f.home_team_id in single_ids or f.away_team_id in single_ids
                    ):
                continue
            if not any((request.match_id, request.lottery_match_no, request.team_query,
                        request.home_team_name and request.away_team_name)):
                continue
            filtered.append(f)
        updates = {"match_id": None, "home_team_id": None, "away_team_id": None, "competition_id": None,
                   "candidates": tuple(f.match_id for f in filtered)}
        if ambiguity or len(filtered) > 1:
            updates.update(resolution_status=ResolutionStatus.AMBIGUOUS, reason="MULTIPLE_IDENTITIES_OR_FIXTURES")
        elif not filtered:
            updates.update(resolution_status=ResolutionStatus.NOT_FOUND, reason="NO_EXACT_FIXTURE")
        else:
            f = filtered[0]
            updates.update(match_id=f.match_id, lottery_match_no=f.lottery_match_no,
                           home_team_id=f.home_team_id, home_team_name=self.store.team(f.home_team_id).team_name,
                           away_team_id=f.away_team_id, away_team_name=self.store.team(f.away_team_id).team_name,
                           competition_id=f.competition_id,
                           competition_name=self.store.competition(f.competition_id).competition_name,
                           kickoff_time=f.kickoff_time, resolution_status=ResolutionStatus.RESOLVED, reason=None)
        return MatchRequest.model_validate({**request.model_dump(), **updates})
