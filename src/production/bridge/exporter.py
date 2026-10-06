"""Append-only packet archives and machine-readable result export."""

from __future__ import annotations

import json
import re
from pathlib import Path

from production.bridge.contracts import CoreDataPacketV1, CoreResultPacketV1
from production.bridge.validator import packet_hash


def archive_packet(packet: CoreDataPacketV1, root: Path) -> Path:
    """Preserve an exact normalized request under its hash without overwriting."""
    target = root / "data" / "bridge" / "requests" / (packet_hash(packet) + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = packet.model_dump_json(indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != payload:
            raise ValueError("BRIDGE_ARCHIVE_HASH_COLLISION")
    else:
        target.write_text(payload, encoding="utf-8")
    return target


def export_result(result: CoreResultPacketV1, root: Path) -> Path:
    """Write a unique result per saved prediction record."""
    if result.prediction_id is None or not re.fullmatch(
            r"[0-9a-f-]{36}", result.prediction_id):
        raise ValueError("BRIDGE_PREDICTION_ID_REQUIRED")
    target = root / "data" / "bridge" / "results" / (result.prediction_id + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target
