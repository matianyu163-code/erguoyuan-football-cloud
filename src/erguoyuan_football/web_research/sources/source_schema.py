"""Source metadata for research; no real endpoint is inferred from a label."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from erguoyuan_football.web_research.policies.source_policy import SOURCE_TIER


@dataclass(frozen=True)
class SourceRecord:
    """Explicit HTTPS source allowed for evidence collection."""

    source_id: str
    name: str
    source_type: str
    tier: int
    url: str
    enabled: bool
    allowed_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id or not self.name:
            raise ValueError("SOURCE_ID_AND_NAME_REQUIRED")
        if self.source_type not in SOURCE_TIER or self.tier != SOURCE_TIER[self.source_type]:
            raise ValueError("SOURCE_TIER_INVALID")
        parsed = urlparse(self.url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("SOURCE_HTTPS_URL_REQUIRED")
        if any(not domain or "/" in domain or "@" in domain
               for domain in self.allowed_domains):
            raise ValueError("INVALID_ALLOWED_DOMAIN")
