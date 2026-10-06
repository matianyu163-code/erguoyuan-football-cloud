"""Normal Chromium rendering and append-only evidence for public JC pages."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.store import canonical_bytes, sha256, write_once

PAGES = {
    "schedule": "https://www.sporttery.cn/jc/zqszsc/index.html",
    "spf": "https://www.sporttery.cn/jc/jsq/zqspf/index.html",
}
MATCH_NUMBER = re.compile(r"周[一二三四五六日天]\d{3}")
PARSER_VERSION = "JC_RENDERED_PARSER_V1"
CHALLENGE_TERMS = ("验证码", "安全验证", "人机验证", "captcha")
RESTRICTED_TERMS = ("访问异常", "访问受限", "access denied")


@dataclass(frozen=True)
class RenderedPage:
    """A single visible rendered page and its durable evidence metadata."""

    endpoint: str
    url: str
    fetched_at: datetime
    http_status: int | None
    html: str
    visible_text: str
    screenshot: bytes
    visible_rows: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


class JCRenderedPageFetcher:
    """Render only allowlisted public pages with unmodified browser settings."""

    def __init__(self, evidence_root: Path, *, executable_path: Path | None = None,
                 cache_ttl_seconds: int = 300) -> None:
        self.evidence_root = evidence_root.resolve()
        self.executable_path = executable_path
        self.cache_ttl_seconds = cache_ttl_seconds
        self.evidence_root.mkdir(parents=True, exist_ok=True)

    def _cached(self, endpoint: str, at: datetime) -> RenderedPage | None:
        for path in sorted(self.evidence_root.glob(f"*_{endpoint}.metadata.json"), reverse=True):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(metadata["fetched_at"])
            if fetched <= at and at - fetched < timedelta(seconds=self.cache_ttl_seconds):
                prefix = path.name.removesuffix(".metadata.json")
                html = (self.evidence_root / f"{prefix}.html").read_text(encoding="utf-8")
                visible = (self.evidence_root / f"{prefix}.txt").read_text(encoding="utf-8")
                screenshot = (self.evidence_root / f"{prefix}.png").read_bytes()
                rows_bytes = (self.evidence_root / f"{prefix}.rows.json").read_bytes()
                rows = json.loads(rows_bytes)["rows"]
                if (sha256(html.encode()) != metadata["dom_sha256"] or
                    sha256(visible.encode()) != metadata["visible_text_sha256"] or
                    sha256(screenshot) != metadata["screenshot_sha256"] or
                    sha256(rows_bytes) != metadata["visible_rows_sha256"]):
                    raise ValueError("JC_RENDERED_CACHE_HASH_INVALID")
                return RenderedPage(endpoint, metadata["url"], fetched,
                    metadata["http_status"], html, visible, screenshot, tuple(rows), metadata)
        return None

    def capture(self, endpoint: str, *, use_cache: bool = True) -> RenderedPage:
        if endpoint not in PAGES:
            raise ValueError("JC_RENDERED_ENDPOINT_NOT_ALLOWED")
        now = datetime.now(UTC)
        cached = self._cached(endpoint, now) if use_cache else None
        if cached is not None:
            return cached
        try:
            from playwright.sync_api import Error, TimeoutError as PlaywrightTimeout, sync_playwright
        except ImportError as error:
            raise RuntimeError("BROWSER_UNAVAILABLE:PLAYWRIGHT_NOT_INSTALLED") from error
        try:
            with sync_playwright() as playwright:
                options: dict[str, Any] = {"headless": True}
                if self.executable_path is not None:
                    options["executable_path"] = str(self.executable_path)
                browser = playwright.chromium.launch(**options)
                try:
                    page = browser.new_page()
                    response = page.goto(PAGES[endpoint], wait_until="domcontentloaded",
                                         timeout=30000)
                    if response is None:
                        raise RuntimeError("PAGE_TIMEOUT:NO_RESPONSE")
                    deadline = datetime.now(UTC) + timedelta(seconds=10)
                    visible = page.locator("body").inner_text(timeout=10000)
                    while not MATCH_NUMBER.search(visible) and datetime.now(UTC) < deadline:
                        page.wait_for_timeout(500)
                        visible = page.locator("body").inner_text(timeout=10000)
                    html = page.content()
                    screenshot = page.screenshot(full_page=True)
                    rows: list[dict[str, Any]] = []
                    for row in page.locator("#mainTbl tr.listTr").all():
                        if not row.is_visible():
                            continue
                        fields: dict[str, Any] = {"provider_row_id": row.get_attribute("id"),
                            "matchnumdate": row.get_attribute("matchnumdate")}
                        cells = []
                        for cell in row.locator("td").all():
                            if cell.is_visible():
                                cells.append(cell.inner_text())
                        fields["visible_cells"] = cells
                        for key, selector in (("competition", "td.lname"),
                                              ("home_team", ".team-left .vs-left-padding"),
                                              ("away_team", ".team-right .vs-right-padding"),
                                              ("handicap", ".hhadGL")):
                            locator = row.locator(selector)
                            fields[key] = locator.inner_text() if locator.count() == 1 and (
                                locator.is_visible()) else None
                        for key, selector in (("spf", ".hadOdds .oddsItem"),
                                              ("rqspf", ".hhadOdds .oddsItem")):
                            fields[key] = [item.inner_text() for item in row.locator(selector).all()
                                           if item.is_visible()]
                        rows.append(fields)
                    fetched_at = datetime.now(UTC)
                    final_url = page.url
                    status = response.status
                finally:
                    browser.close()
        except PlaywrightTimeout as error:
            raise RuntimeError("PAGE_TIMEOUT") from error
        except Error as error:
            raise RuntimeError(f"BROWSER_UNAVAILABLE:{type(error).__name__}") from error
        if final_url != PAGES[endpoint]:
            raise ValueError("JC_RENDERED_UNEXPECTED_URL")
        prefix = f"{fetched_at.strftime('%Y%m%dT%H%M%S%fZ')}_{endpoint}"
        rows_bytes = canonical_bytes({"rows": rows})
        metadata = {"url": final_url, "fetched_at": fetched_at.isoformat(),
            "http_status": status, "dom_sha256": sha256(html.encode()),
            "visible_text_sha256": sha256(visible.encode()),
            "screenshot_sha256": sha256(screenshot), "parser_version": PARSER_VERSION,
            "browser": "chromium", "provider": "JC_RENDERED_PUBLIC",
            "visible_match_numbers": sorted(set(MATCH_NUMBER.findall(visible))),
            "visible_rows_sha256": sha256(rows_bytes), "visible_row_count": len(rows)}
        write_once(self.evidence_root / f"{prefix}.html", html.encode())
        write_once(self.evidence_root / f"{prefix}.txt", visible.encode())
        write_once(self.evidence_root / f"{prefix}.png", screenshot)
        write_once(self.evidence_root / f"{prefix}.rows.json", rows_bytes)
        write_once(self.evidence_root / f"{prefix}.metadata.json", canonical_bytes(metadata))
        return RenderedPage(endpoint, final_url, fetched_at, status, html,
                            visible, screenshot, tuple(rows), metadata)


def page_status(page: RenderedPage) -> str:
    """Classify normal user-visible text without reading hidden scripts."""
    text = page.visible_text.casefold()
    if any(term in text for term in CHALLENGE_TERMS):
        return "ACCESS_CHALLENGE"
    if any(term in text for term in RESTRICTED_TERMS) or page.http_status in {401, 403, 429}:
        return "ACCESS_RESTRICTED"
    if page.http_status != 200:
        return "UNAVAILABLE"
    return "HAS_MATCHES" if page.visible_rows or MATCH_NUMBER.search(page.visible_text) else "NO_ACTIVE_MATCHES"
