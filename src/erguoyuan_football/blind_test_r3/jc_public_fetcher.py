"""Low-frequency retrieval of the two explicitly public Sporttery pages."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from erguoyuan_football.blind_test_r3.jc_public_store import JCPublicStore
from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.config import RateLimitPolicy, RetryPolicy
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry

SOURCE_ID = "JC_OFFICIAL_PUBLIC"
PAGES = {
    "schedule": "https://www.sporttery.cn/jc/zqszsc/index.html",
    "spf": "https://www.sporttery.cn/jc/jsq/zqspf/index.html",
}


class JCPublicPageFetcher:
    """Use the existing audited HTML transport; never discover internal URLs."""

    def __init__(self, store: JCPublicStore, *, http_client: Any | None = None) -> None:
        source = SourceDefinition(source_id=SOURCE_ID, display_name="Sporttery public pages",
            category="MARKET", base_url="https://www.sporttery.cn", endpoints=PAGES,
            supports_live=True, schema_version="JC_PUBLIC_HTML_V1",
            retry_policy=RetryPolicy(max_attempts=1),
            rate_limit_policy=RateLimitPolicy(requests_per_second=1 / 60,
                requests_per_minute=1, burst=1))
        self.store = store
        self.client = CoreNetworkClient(ExternalSourceRegistry((source,)),
                                        http_client=http_client)

    def close(self) -> None:
        self.client.close()

    def fetch(self, endpoint: str, *, at: datetime | None = None) -> dict[str, Any]:
        if endpoint not in PAGES:
            raise ValueError("JC_PUBLIC_ENDPOINT_NOT_ALLOWED")
        current = at or datetime.now(UTC)
        if current.tzinfo is None:
            raise ValueError("JC_PUBLIC_FETCH_TIMEZONE_REQUIRED")
        cached = self.store.fresh_page(endpoint, at=current, ttl_seconds=60)
        if cached is not None:
            return cached
        response = self.client.fetch_text(SOURCE_ID, endpoint, max_bytes=2_000_000)
        if response.final_url != PAGES[endpoint]:
            raise ValueError("JC_PUBLIC_UNEXPECTED_REDIRECT")
        return self.store.append_page(endpoint=endpoint, url=response.final_url,
            fetched_at=response.retrieved_at, http_status=response.status_code,
            body=response.body, transport_sha256=response.content_hash,
            parser_version="JC_PUBLIC_PARSER_V1")
