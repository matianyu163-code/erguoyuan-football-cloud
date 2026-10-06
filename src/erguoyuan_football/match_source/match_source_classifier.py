"""Deterministically classify match input provenance without guessing fixtures."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime

from erguoyuan_football.match_source.match_source import MatchSourceType
from erguoyuan_football.match_source.match_source_audit import (
    MatchSourceAudit,
    new_audit,
)

_JC_MARKERS = re.compile(r"中国体育彩票竞彩足球|中国竞彩|竞彩足球")
_LOTTERY_LABEL = re.compile(r"竞彩(?:编号(?:为)?|比赛编号|比赛)\s*[：:]?", re.IGNORECASE)
_LOTTERY_NUMBER = re.compile(r"周\s*[一二三四五六日天末1-7]?\s*\d{3}(?!\d)",
                             re.IGNORECASE)
_YOUTH_MARKER = re.compile(r"(?<![A-Z0-9_])U(?:15|16|17|18|19|20|21|22|23)(?![A-Z0-9_])",
                           re.IGNORECASE)


class MatchSourceClassifier:
    """Classify explicit ticket inputs, age-group research, and normal discovery."""

    def classify(self, raw_text: str | None, *, jc_confirmed: bool = False,
                 declared_type: MatchSourceType | str | None = None,
                 created_time: datetime | None = None) -> MatchSourceAudit:
        """Return source class and auditable rule evidence; no network is accessed."""
        text = unicodedata.normalize("NFKC", raw_text or "").strip()
        if jc_confirmed or _declared(declared_type) == MatchSourceType.USER_JC_CONFIRMED:
            return new_audit(MatchSourceType.USER_JC_CONFIRMED, confidence=1.0,
                evidence=_evidence(text, "DECLARED_USER_JC_CONFIRMED"),
                created_time=created_time)
        if _declared(declared_type) == MatchSourceType.AUTO_DISCOVERY:
            return new_audit(MatchSourceType.AUTO_DISCOVERY, confidence=1.0,
                evidence=_evidence(text, "DECLARED_AUTO_RESEARCH"),
                created_time=created_time)
        if _declared(declared_type) == MatchSourceType.RESEARCH_TEST:
            return new_audit(MatchSourceType.RESEARCH_TEST, confidence=1.0,
                evidence=_evidence(text, "DECLARED_RESEARCH_TEST"),
                created_time=created_time)
        if _JC_MARKERS.search(text) or _LOTTERY_NUMBER.search(text):
            evidence = tuple(label for pattern, label in (
                (_JC_MARKERS, "EXPLICIT_CHINA_SPORTTERY_MARKER"),
                (_LOTTERY_NUMBER, "SPORTTERY_WEEK_MATCH_NUMBER")) if pattern.search(text))
            return new_audit(MatchSourceType.USER_JC_CONFIRMED, confidence=1.0,
                             evidence=_evidence(text, *evidence), created_time=created_time)
        youth = _YOUTH_MARKER.findall(text)
        if youth:
            levels = tuple(sorted({item.upper() for item in youth}))
            return new_audit(MatchSourceType.RESEARCH_TEST, confidence=1.0,
                evidence=_evidence(text, *(f"EXPLICIT_YOUTH_MARKER:{item}"
                                           for item in levels)),
                created_time=created_time)
        return new_audit(MatchSourceType.AUTO_DISCOVERY, confidence=1.0,
            evidence=_evidence(text, "NO_USER_JC_OR_RESEARCH_MARKER"),
            created_time=created_time)

    def clean_match_text(self, raw_text: str) -> str:
        """Remove only recognized request metadata before syntax/team parsing."""
        text = unicodedata.normalize("NFKC", raw_text).strip()
        text = _JC_MARKERS.sub(" ", text)
        text = _LOTTERY_LABEL.sub(" ", text)
        text = _LOTTERY_NUMBER.sub(" ", text)
        return " ".join(text.split())


def _declared(value: MatchSourceType | str | None) -> MatchSourceType | None:
    if value is None:
        return None
    try:
        return value if isinstance(value, MatchSourceType) else MatchSourceType(value)
    except ValueError:
        return None


def _evidence(text: str, *rules: str) -> tuple[str, ...]:
    """Keep classification rule evidence and a privacy-preserving input digest."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (*rules, f"INPUT_SHA256:{digest}")
