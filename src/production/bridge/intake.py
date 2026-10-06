"""Smart desktop intake for externally researched, CORE-validated fixtures."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from erguoyuan_football.app.input.match_input import MatchInputParserV2
from production.bridge.contracts import CoreDataPacketV1, CoreResultPacketV1
from production.bridge.runner import BridgeRunner
from production.config import ProductionConfig
from production.jc_metadata import JCMetadataParser


class CoreResearchRequestV1(BaseModel):
    """External-research request; it carries no predicted or inferred facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["CORE_RESEARCH_REQUEST_V1"] = "CORE_RESEARCH_REQUEST_V1"
    request_id: str
    created_at: datetime
    raw_input: str
    home_raw: str
    away_raw: str
    competition_hint: str | None = None
    date_hint: date | None = None
    jc_match_code: str | None = None
    requested_output: Literal["CORE_DATA_PACKET_V1"] = "CORE_DATA_PACKET_V1"


class AmbiguousFixtureCandidate(BaseModel):
    """One externally discovered option, displayed without automatic selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    competition: str
    kickoff_original: str
    kickoff_timezone: str
    kickoff_utc: datetime
    source_url: str


class AmbiguousFixtureResponse(BaseModel):
    """Research response requiring an explicit user's fixture choice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["BRIDGE_FIXTURE_AMBIGUOUS_V1"]
    request_id: str
    candidates: tuple[AmbiguousFixtureCandidate, ...] = Field(min_length=2)


@dataclass(frozen=True)
class SmartInputDecision:
    """Route a desktop input without trying to resolve the teams or fixture."""

    route: Literal["JC_PRODUCTION", "BRIDGE_RESEARCH", "INVALID"]
    home: str | None
    away: str | None
    competition: str | None
    kickoff_local: str | None
    timezone: str
    jc_match_code: str | None
    date_hint: date | None
    error: str | None = None


@dataclass(frozen=True)
class BridgePollResult:
    """A research request's current state and optional final CORE result."""

    status: str
    request_id: str
    message: str
    result: CoreResultPacketV1 | None = None
    ambiguous: AmbiguousFixtureResponse | None = None


def _date_hint(raw: str) -> date | None:
    match = re.search(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", raw)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def classify_smart_input(raw: str, *, jc_fields: dict[str, str] | None = None
                         ) -> SmartInputDecision:
    """Use explicit syntax and metadata only; ambiguous teams are never guessed."""
    parser = MatchInputParserV2()
    metadata_fields = jc_fields or {}
    metadata = JCMetadataParser().parse(
        raw,
        jc_match_code=metadata_fields.get("jc_match_code"),
        competition=metadata_fields.get("competition"),
        home_team=metadata_fields.get("home_team"),
        away_team=metadata_fields.get("away_team"),
        kickoff_local=metadata_fields.get("kickoff_local"),
        timezone=metadata_fields.get("timezone") or "Asia/Shanghai",
    )
    if (metadata.competition_original
            and metadata.home_team and metadata.away_team and metadata.kickoff_utc
            and metadata.kickoff_original
            and not metadata.missing_fields):
        return SmartInputDecision(
            "JC_PRODUCTION", metadata.home_team, metadata.away_team,
            metadata.competition_original, metadata.kickoff_original,
            metadata.source_timezone, metadata.jc_match_code,
            _date_hint(raw) or date.fromisoformat(metadata.kickoff_original[:10]),
        )
    structure = parser.parse_structure(raw)
    if not structure.home_raw or not structure.away_raw:
        return SmartInputDecision("INVALID", None, None, None, None,
                                  metadata_fields.get("timezone") or "Asia/Shanghai",
                                  None, _date_hint(raw), "MATCH_SYNTAX_INVALID")
    competition = (metadata_fields.get("competition", "").strip()
                   or structure.competition_raw)
    return SmartInputDecision(
        "BRIDGE_RESEARCH", structure.home_raw, structure.away_raw,
        competition, metadata.kickoff_original, metadata.source_timezone,
        metadata.jc_match_code, _date_hint(raw),
    )


class SmartBridgeIntake:
    """Write requests and accept response packets through the explicit file bridge."""

    def __init__(self, config: ProductionConfig, project_root: Path) -> None:
        self.config = config
        self.root = project_root.resolve()
        self.bridge_root = self.root / "data" / "bridge"
        self.inbox = self.bridge_root / "inbox"
        self.ready = self.bridge_root / "ready"
        self.results = self.bridge_root / "results"
        self.database = config.record_database

    def create_request(self, raw_input: str, decision: SmartInputDecision
                       ) -> CoreResearchRequestV1:
        """Archive a valid team-pair request and append its requested-state event."""
        if decision.route != "BRIDGE_RESEARCH" or not decision.home or not decision.away:
            raise ValueError("SMART_BRIDGE_TEAMS_REQUIRED")
        request = CoreResearchRequestV1(
            request_id=str(uuid4()), created_at=datetime.now(UTC),
            raw_input=raw_input, home_raw=decision.home, away_raw=decision.away,
            competition_hint=decision.competition, date_hint=decision.date_hint,
            jc_match_code=decision.jc_match_code,
        )
        self.inbox.mkdir(parents=True, exist_ok=True)
        path = self.inbox / f"{request.request_id}.json"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(request.model_dump_json(indent=2) + "\n")
        self._append_event(request.request_id, "BRIDGE_RESEARCH_REQUESTED",
                           "BRIDGE_AGENT_NOT_CONNECTED")
        return request

    def poll(self, request_id: str) -> BridgePollResult:
        """Read a request's result, ambiguity response, or ready packet exactly by ID."""
        request_path = self.inbox / f"{request_id}.json"
        if not request_path.is_file():
            return BridgePollResult("REQUEST_NOT_FOUND", request_id,
                                    "研究请求不存在或已移动。")
        result_path = self.results / f"{request_id}.result.json"
        if result_path.is_file():
            try:
                result = CoreResultPacketV1.model_validate_json(
                    result_path.read_text(encoding="utf-8"))
            except ValidationError as error:
                return BridgePollResult("RESULT_INVALID", request_id,
                                        f"结果文件格式无效：{error}")
            return BridgePollResult(result.status, request_id, result.v7_output, result)

        ambiguous_path = self.ready / f"{request_id}.ambiguous.json"
        if ambiguous_path.is_file():
            try:
                ambiguous = AmbiguousFixtureResponse.model_validate_json(
                    ambiguous_path.read_text(encoding="utf-8"))
            except ValidationError as error:
                return BridgePollResult("AMBIGUOUS_RESPONSE_INVALID", request_id,
                                        f"歧义候选文件格式无效：{error}")
            if ambiguous.request_id != request_id:
                return BridgePollResult("AMBIGUOUS_RESPONSE_INVALID", request_id,
                                        "研究请求编号不匹配。")
            selection_path = self.inbox / f"{request_id}.selection.json"
            if not selection_path.is_file():
                choices = "\n".join(
                    f"{row.candidate_id}｜{row.competition}｜{row.kickoff_original} "
                    f"({row.kickoff_timezone})"
                    for row in ambiguous.candidates)
                return BridgePollResult("BRIDGE_FIXTURE_AMBIGUOUS", request_id,
                                        "发现多个可能比赛，未自动选择：\n" + choices,
                                        ambiguous=ambiguous)
            try:
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                selected_id = str(selection["candidate_id"])
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
                return BridgePollResult("BRIDGE_SELECTION_INVALID", request_id,
                                        f"用户选择文件无效：{error}")
            if selected_id not in {row.candidate_id for row in ambiguous.candidates}:
                return BridgePollResult("BRIDGE_SELECTION_INVALID", request_id,
                                        "用户选择的比赛不在候选列表中。")
            if not (self.ready / f"{request_id}.packet.json").is_file():
                return BridgePollResult(
                    "WAITING_FOR_RESEARCH", request_id,
                    f"已记录用户选择 {selected_id}，等待 Research Agent 生成对应数据包。",
                )

        packet_path = self.ready / f"{request_id}.packet.json"
        if not packet_path.is_file():
            return BridgePollResult(
                "WAITING_FOR_RESEARCH", request_id,
                "等待 Research Agent 完成数据准备。BRIDGE_AGENT_NOT_CONNECTED；"
                "请由外部 Research Agent 将数据包放入 data/bridge/ready/。",
            )
        try:
            packet = CoreDataPacketV1.model_validate_json(
                packet_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as error:
            self._append_event(request_id, "BRIDGE_PACKET_REJECTED", str(error))
            return BridgePollResult("BRIDGE_PACKET_INVALID", request_id,
                                    f"研究数据包无效：{error}")
        if packet.request_id != request_id:
            self._append_event(request_id, "BRIDGE_PACKET_REJECTED",
                               "BRIDGE_PACKET_REQUEST_ID_MISMATCH")
            return BridgePollResult("BRIDGE_PACKET_REQUEST_ID_MISMATCH", request_id,
                                    "研究数据包的请求编号不匹配，CORE 未执行。")
        self._append_event(request_id, "BRIDGE_PACKET_RECEIVED")
        runner = BridgeRunner(self.config, project_root=self.root)
        try:
            result = runner.run(packet)
        finally:
            runner.close()
        self.results.mkdir(parents=True, exist_ok=True)
        with result_path.open("x", encoding="utf-8") as handle:
            json.dump(result.model_dump(mode="json"), handle,
                      ensure_ascii=False, indent=2)
            handle.write("\n")
        self._append_event(request_id, "BRIDGE_RESULT_SAVED", result.status)
        return BridgePollResult(result.status, request_id, result.v7_output, result)

    def record_user_selection(self, request_id: str, candidate_id: str) -> Path:
        """Persist an explicit choice for the research agent; never selects implicitly."""
        response_path = self.ready / f"{request_id}.ambiguous.json"
        response = AmbiguousFixtureResponse.model_validate_json(
            response_path.read_text(encoding="utf-8"))
        if response.request_id != request_id:
            raise ValueError("BRIDGE_REQUEST_ID_MISMATCH")
        if candidate_id not in {row.candidate_id for row in response.candidates}:
            raise ValueError("BRIDGE_CANDIDATE_NOT_FOUND")
        selection_path = self.inbox / f"{request_id}.selection.json"
        payload = {"schema_version": "CORE_RESEARCH_SELECTION_V1",
                   "request_id": request_id, "candidate_id": candidate_id,
                   "selected_at": datetime.now(UTC).isoformat()}
        with selection_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        self._append_event(request_id, "USER_FIXTURE_SELECTED", candidate_id)
        return selection_path

    def _append_event(self, request_id: str, status: str, detail: str = "") -> None:
        """Create an append-only intake audit event in the existing trial database."""
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS bridge_research_events (
                event_id TEXT PRIMARY KEY, request_id TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, detail TEXT NOT NULL)""")
            connection.execute("""INSERT INTO bridge_research_events VALUES (?,?,?,?,?)""",
                (str(uuid4()), request_id, status, datetime.now(UTC).isoformat(), detail))
