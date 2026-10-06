"""Country-to-association official source candidates for global fixture research."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssociationSource:
    """A vetted association identity and its official public web domains."""

    country_id: str
    association_id: str
    official_domains: tuple[str, ...]
    supported_languages: tuple[str, ...]
    country_aliases: tuple[tuple[str, str], ...] = ()
    source_tier: str = "A"

    def __post_init__(self) -> None:
        if not self.country_id or not self.association_id or not self.official_domains:
            raise ValueError("ASSOCIATION_SOURCE_IDENTITY_REQUIRED")
        if self.source_tier != "A":
            raise ValueError("ASSOCIATION_SOURCE_MUST_BE_TIER_A")
        for domain in self.official_domains:
            if (
                not domain
                or "/" in domain
                or ":" in domain
                or domain.startswith(".")
                or domain.endswith(".")
            ):
                raise ValueError("INVALID_ASSOCIATION_DOMAIN")


_SOURCES = (
    AssociationSource(
        "IRL",
        "FAI",
        ("fai.ie",),
        ("en", "ga"),
        (("en", "Republic of Ireland"), ("ga", "Éire"), ("nl", "Ierland")),
    ),
    AssociationSource(
        "NLD", "KNVB", ("onsoranje.nl", "knvb.nl"), ("nl", "en"), (("nl", "Nederland"),)
    ),
    AssociationSource(
        "DEU", "DFB", ("dfb.de", "datencenter.dfb.de"), ("de", "en"), (("de", "Deutschland"),)
    ),
    AssociationSource("CZE", "FACR", ("fotbal.cz",), ("cs", "en"), (("cs", "Česko"),)),
    AssociationSource("FRA", "FFF", ("fff.fr",), ("fr", "en")),
    AssociationSource("USA", "USSF", ("ussoccer.com",), ("en",)),
    AssociationSource(
        "KAZ",
        "KFF",
        ("kff.kz",),
        ("kk", "ru", "en"),
        (("kk", "Қазақстан"), ("ru", "Казахстан")),
    ),
    AssociationSource(
        "MDA",
        "FMF",
        ("fmf.md",),
        ("ro", "ru", "en"),
        (("ro", "Moldova"), ("ru", "Молдова")),
    ),
    AssociationSource("RUS", "RFS", ("rfs.ru",), ("ru", "en"), (("ru", "Россия"),)),
    AssociationSource(
        "SAU", "SAFF", ("saff.com.sa",), ("ar", "en"), (("ar", "السعودية"),)
    ),
    AssociationSource("JPN", "JFA", ("jfa.jp",), ("ja", "en"), (("ja", "日本"),)),
    AssociationSource(
        "KOR", "KFA", ("kfa.or.kr",), ("ko", "en"), (("ko", "대한민국"),)
    ),
    AssociationSource("ESP", "RFEF", ("rfef.es",), ("es", "en"), (("es", "España"),)),
    AssociationSource("GBR", "THE_FA", ("thefa.com",), ("en",)),
    AssociationSource("ITA", "FIGC", ("figc.it",), ("it", "en")),
    AssociationSource("PRT", "FPF", ("fpf.pt",), ("pt", "en")),
    AssociationSource("CHN", "CFA", ("thecfa.cn",), ("zh", "en")),
    AssociationSource(
        "AUS", "FOOTBALL_AUSTRALIA", ("footballaustralia.com.au",), ("en",)
    ),
    AssociationSource("BRA", "CBF", ("cbf.com.br",), ("pt",)),
    AssociationSource("ARG", "AFA", ("afa.com.ar",), ("es",)),
    AssociationSource("TUR", "TFF", ("tff.org",), ("tr", "en")),
    AssociationSource("POL", "PZPN", ("pzpn.pl",), ("pl", "en")),
    AssociationSource("NOR", "NFF", ("fotball.no",), ("no", "en")),
    AssociationSource("SWE", "SVFF", ("svenskfotboll.se",), ("sv", "en")),
    AssociationSource("DEN", "DBU", ("dbu.dk",), ("da", "en")),
    AssociationSource("BEL", "RBFA", ("rbfa.be",), ("nl", "fr", "en")),
    AssociationSource("AUT", "OFB", ("oefb.at",), ("de", "en"), (("de", "Österreich"),)),
    AssociationSource("CHE", "SFV", ("football.ch",), ("de", "fr", "it", "en")),
    AssociationSource("UKR", "UAF", ("uaf.ua",), ("uk", "en")),
)


class AssociationSourceRegistry:
    """Resolve official association candidates from canonical country IDs."""

    version = "ASSOCIATION_SOURCE_REGISTRY_V1"

    def __init__(self, sources: tuple[AssociationSource, ...] = _SOURCES) -> None:
        self._sources = {source.country_id: source for source in sources}
        if len(self._sources) != len(sources):
            raise ValueError("DUPLICATE_ASSOCIATION_COUNTRY")

    def get(self, country_id: str) -> AssociationSource | None:
        """Return the vetted source candidate for an ISO-3 country ID."""
        return self._sources.get(country_id.upper())

    def for_countries(self, *country_ids: str) -> tuple[AssociationSource, ...]:
        """Return unique sources in stable country order."""
        rows = {
            source.association_id: source
            for country_id in country_ids
            if (source := self.get(country_id)) is not None
        }
        return tuple(sorted(rows.values(), key=lambda row: row.association_id))

    def all(self) -> tuple[AssociationSource, ...]:
        """Expose the immutable registry for audits and diagnostics."""
        return tuple(self._sources[key] for key in sorted(self._sources))
