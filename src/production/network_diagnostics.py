"""Read-only DNS/TCP/TLS/HTTPS diagnostics shared by source and packaged entrypoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import ssl
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry

OFFICIAL_PROBE_DOMAINS = ("www.uefa.com", "www.onsoranje.nl", "www.dfb.de")


def network_policy_blocked(error: BaseException) -> bool:
    """Recognize Windows permission denial even inside a transport exception chain."""
    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if getattr(current, "winerror", None) == 10013 or "WinError 10013" in str(current):
            return True
        for child in (current.__cause__, current.__context__, getattr(current, "cause", None)):
            if isinstance(child, BaseException):
                pending.append(child)
    return False


def probe_domain(domain: str, *, process: str) -> dict[str, Any]:
    """Probe approved official hosts without altering proxy/firewall/TLS settings."""
    if domain not in OFFICIAL_PROBE_DOMAINS:
        raise ValueError("OFFICIAL_PROBE_DOMAIN_NOT_ALLOWLISTED")
    row: dict[str, Any] = {
        "process": process, "pid": os.getpid(), "executable_path": sys.executable,
        "frozen": bool(getattr(sys, "frozen", False)), "domain": domain,
        "created_at": datetime.now(UTC).isoformat(), "dns_result": "NOT_RUN",
        "socket_result": "NOT_RUN", "tls_result": "NOT_RUN", "http_result": "NOT_RUN",
        "status": "NOT_RUN", "failure_layer": None, "verify_tls": True,
        "proxy_environment_present": any(os.environ.get(key) for key in
                                          ("HTTPS_PROXY", "https_proxy", "ALL_PROXY")),
    }
    stage = "dns_result"
    try:
        addresses = socket.getaddrinfo(domain, 443, type=socket.SOCK_STREAM)
        row[stage] = "PASS"
        row["dns_addresses"] = sorted({str(item[4][0]) for item in addresses})
        stage = "socket_result"
        with socket.create_connection((domain, 443), timeout=8) as connection:
            row[stage] = "PASS"
            stage = "tls_result"
            with ssl.create_default_context().wrap_socket(connection, server_hostname=domain) as secured:
                row[stage] = "PASS"
                row["tls_version"] = secured.version()
        row["status"] = "TRANSPORT_READY"
    except (OSError, ValueError) as error:
        row[stage] = "FAILED"
        row["failure_layer"] = "Windows permission" if network_policy_blocked(error) else stage
        row["status"] = "NETWORK_POLICY_BLOCKED" if network_policy_blocked(error) else "NETWORK_FAILED"
        row["transport_error"] = f"{type(error).__name__}: {error}"
    # HTTP uses the application's normal AUTO route (which may use an existing
    # proxy). Keep direct transport results distinct from HTTP route results.
    definition = SourceDefinition(source_id="OFFICIAL_PROBE", display_name=domain,
        category="OFFICIAL_DIAGNOSTIC", base_url=f"https://{domain}",
        endpoints={"head": f"https://{domain}/"}, schema_version="OFFICIAL_PROBE_V1")
    client = CoreNetworkClient(ExternalSourceRegistry((definition,)))
    try:
        response = client.fetch_text("OFFICIAL_PROBE", "head", max_bytes=3_000_000)
        row["http_result"] = response.status_code
        row["http_final_url"] = response.final_url
        row["http_hash"] = response.content_hash
        row["status"] = "HTTPS_READY"
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        # A diagnostic must report transport-library failures, not crash before
        # recording the failing executable identity. No prediction is generated.
        row["http_result"] = "FAILED"
        row["http_error"] = f"{type(error).__name__}: {error}"
        if network_policy_blocked(error):
            row["status"] = "NETWORK_POLICY_BLOCKED"
            row["failure_layer"] = "Windows permission"
        elif row["status"] == "TRANSPORT_READY":
            row["status"] = "HTTP_FAILED"
            row["failure_layer"] = "HTTP"
    finally:
        row["http_audit"] = [item.model_dump(mode="json") for item in client.audit.records()]
        client.close()
    return row


def write_diagnostics(path: Path, *, process: str) -> int:
    """Persist real per-process diagnostics even in a windowed executable."""
    executable = Path(sys.executable)
    rows = [probe_domain(domain, process=process) for domain in OFFICIAL_PROBE_DOMAINS]
    report = {"process": process, "executable_path": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(row["status"] == "HTTPS_READY" for row in rows) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(write_diagnostics(arguments.output, process="PYTHON_INTERPRETER"))
