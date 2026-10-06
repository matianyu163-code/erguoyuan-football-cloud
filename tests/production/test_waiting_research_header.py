"""The active research request must not inherit an older JC blocker banner."""

from __future__ import annotations

from erguoyuan_football.app.session import ApplicationResult
from erguoyuan_football.app.ui.window import (
    WAITING_FOR_RESEARCH_MESSAGE,
    DesktopWindow,
    prediction_header,
)
from production.status import ProductionStatusView


def _result(status: str, text: str, request_id: str | None = None
            ) -> ApplicationResult:
    return ApplicationResult(status, (), None, text, None, {}, (), request_id)


def _old_jc_state() -> ProductionStatusView:
    return ProductionStatusView(
        "PRODUCTION_BLOCKED", "生产预测当前受阻。",
        "本次预测未完成：JC_FIXTURE_METADATA_INCOMPLETE。", "old-prediction",
    )


def test_waiting_v7_status_replaces_stale_jc_header() -> None:
    result = _result("PRODUCTION_TRIAL",
                     "FINAL_V7_STATUS: WAITING_FOR_RESEARCH")
    assert prediction_header(result, _old_jc_state()) == WAITING_FOR_RESEARCH_MESSAGE


def test_bridge_primary_status_replaces_stale_jc_header() -> None:
    result = _result("PRODUCTION_TRIAL",
                     "PRIMARY STATUS: BRIDGE_RESEARCH_REQUIRED")
    assert prediction_header(result, _old_jc_state()) == WAITING_FOR_RESEARCH_MESSAGE


def test_waiting_poll_replaces_stale_jc_header() -> None:
    result = _result("WAITING_FOR_RESEARCH", "等待 Research Agent 数据", "request-1")
    assert prediction_header(result, _old_jc_state()) == WAITING_FOR_RESEARCH_MESSAGE


def test_real_jc_blocker_remains_visible() -> None:
    result = _result("PRODUCTION_TRIAL",
                     "PRIMARY BLOCKER: JC_FIXTURE_METADATA_INCOMPLETE")
    assert prediction_header(result, _old_jc_state()) == (
        "本次预测未完成：JC_FIXTURE_METADATA_INCOMPLETE。")


def test_desktop_finish_displays_active_research_header() -> None:
    class _Control:
        def __init__(self) -> None:
            self.value = ""

        def set(self, value: str) -> None:
            self.value = value

        def get(self, *_args: object) -> str:
            return "SYNTHETIC_TEST HOME VS SYNTHETIC_TEST AWAY"

        def configure(self, **_kwargs: object) -> None:
            return None

    window = object.__new__(DesktopWindow)
    window.status = _Control()
    window.input = _Control()
    window.check_research_button = _Control()
    window.run_button = _Control()
    window.bridge_request_id = None
    window.bridge_raw_input = None
    window._production_status = _old_jc_state  # type: ignore[method-assign]
    shown: list[str] = []
    window._show = shown.append  # type: ignore[method-assign]

    result = _result("WAITING_FOR_BRIDGE_RESEARCH",
                     "RUN MODE: SMART_BRIDGE\n"
                     "PRIMARY STATUS: BRIDGE_RESEARCH_REQUIRED\n"
                     "FINAL_V7_STATUS: WAITING_FOR_RESEARCH", "request-1")
    window._finish(result)

    assert shown[0].startswith(WAITING_FOR_RESEARCH_MESSAGE + "\n\n")
    assert "本次预测未完成：JC_FIXTURE_METADATA_INCOMPLETE" not in shown[0]
    assert window.status.value == WAITING_FOR_RESEARCH_MESSAGE
