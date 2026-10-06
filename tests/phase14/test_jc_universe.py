"""JC production tasks have deterministic higher fetch priority."""

from erguoyuan_football.research.data_priority_scheduler import (
    DataFetchTask,
    DataKind,
    DataPriorityScheduler,
)
from erguoyuan_football.research.match_universe import MatchUniverse


def test_jc_priority_is_fixture_then_odds_then_form_context_then_history() -> None:
    tasks = tuple(DataFetchTask(f"JC_{kind.value}", MatchUniverse.JC_PRODUCTION,
                                kind, index)
                  for index, kind in enumerate((DataKind.HISTORICAL, DataKind.INJURY,
                      DataKind.LINEUP, DataKind.RECENT_FORM, DataKind.ODDS, DataKind.FIXTURE)))
    tasks += (DataFetchTask("RESEARCH", MatchUniverse.GLOBAL_RESEARCH,
                            DataKind.FIXTURE, 0),)
    ordered = DataPriorityScheduler().order(tasks)
    assert [task.kind for task in ordered[:6]] == [
        DataKind.FIXTURE, DataKind.ODDS, DataKind.RECENT_FORM,
        DataKind.LINEUP, DataKind.INJURY, DataKind.HISTORICAL,
    ]
    assert ordered[-1].universe == MatchUniverse.GLOBAL_RESEARCH
