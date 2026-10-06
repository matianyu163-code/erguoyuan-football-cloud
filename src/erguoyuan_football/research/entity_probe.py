"""Research-only universal entity diagnostic; fixture and prediction stay separate."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from erguoyuan_football.knowledge.entities.competition_hint_parser import (
    CompetitionHintParser,
)
from erguoyuan_football.knowledge.entities.openligadb_discovery import (
    OpenLigaDBTeamDiscovery,
)
from erguoyuan_football.knowledge.entities.universal_team_resolver import (
    UniversalTeamResolver,
)
from erguoyuan_football.knowledge.entities.verified_entity_store import (
    VerifiedEntityStore,
)
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBConfig,
    OpenLigaDBProvider,
)
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore

ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    """Show identity evidence and scope without claiming a verified match."""
    parser = argparse.ArgumentParser(description="Phase 13.6 entity diagnostic")
    parser.add_argument("name")
    parser.add_argument("--competition")
    parser.add_argument("--live", action="store_true", help="Allow scoped real discovery")
    parser.add_argument("--store", type=Path,
                        default=ROOT / "data/dynamic_entity_store.sqlite")
    args = parser.parse_args(argv)
    competition = (CompetitionHintParser().parse(args.competition)
                   if args.competition else None)
    store = VerifiedEntityStore(args.store)
    provider = None
    evidence = None
    try:
        discovery: tuple[OpenLigaDBTeamDiscovery, ...] = ()
        if args.live:
            provider = OpenLigaDBProvider(OpenLigaDBConfig.from_yaml(
                ROOT / "config/phase13_5_provider.yaml"))
            evidence_path = args.store.with_name(args.store.stem + "_evidence.sqlite")
            evidence = EvidenceStore(evidence_path, provider.sources)
            discovery = (OpenLigaDBTeamDiscovery(provider, evidence),)
        resolver = UniversalTeamResolver(store=store, discovery=discovery)
        parsed = resolver.parser.parse(args.name)
        local = TeamResolver().resolve(args.name)
        result = resolver.resolve(args.name, competition=competition,
                                  allow_discovery=args.live)
        print(f"RAW INPUT: {args.name}")
        print(f"PARSED HINTS: {parsed}")
        print(f"COMPETITION HINT: {competition}")
        print(f"LOCAL MATCH: {local.team_id if local else 'NONE'}")
        stored = store.find(args.name, as_of=datetime.now(UTC))
        print(f"DYNAMIC STORE: {len(stored)} candidate(s)")
        print(f"LIVE CANDIDATES: {len(result.candidates)}")
        print(f"SELECTED IDENTITY: {result.identity.team_id if result.identity else 'NONE'}")
        print(f"VERIFICATION STATUS: {result.status}")
        print("FIXTURE: NOT VERIFIED")
        print("PREDICTION: NOT EXECUTED")
        return 0 if result.identity is not None else 2
    finally:
        if evidence is not None:
            evidence.close()
        if provider is not None:
            provider.close()
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
