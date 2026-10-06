# CORE Research Agent Protocol — V1

External research supplies **facts and citations only**. CORE computes model probabilities. The
agent must not estimate outcomes, tune model parameters, declare samples eligible, or bypass PIT.
The wire versions are `CORE_DATA_PACKET_V1`, `CORE_RESULT_PACKET_V1`, and `CORE_BRIDGE_V1`.

## Smart desktop request/response handoff

The ordinary desktop mode is `SMART_BRIDGE`. A team-pair input without complete, explicit
Sporttery metadata creates a `CORE_RESEARCH_REQUEST_V1` JSON file at
`data/bridge/inbox/<request_id>.json` and an append-only `BRIDGE_RESEARCH_REQUESTED` event in
the trial database. The request contains the raw input, parsed home/away names, optional
competition/date hints, and the requested `CORE_DATA_PACKET_V1` output. No team identity,
fixture, kickoff, probability, or source evidence is guessed by this step.

An external agent may read that request and research real, cited fixture/history facts. If one
fixture is confirmed, it writes `data/bridge/ready/<request_id>.packet.json`. If more than one
fixture is plausible, it writes `data/bridge/ready/<request_id>.ambiguous.json` using
`BRIDGE_FIXTURE_AMBIGUOUS_V1`; CORE shows all candidates and requires a user-selected
`candidate_id`, persisted as `data/bridge/inbox/<request_id>.selection.json`. It must not
choose a candidate automatically. The desktop's **检查研究结果** action polls request-specific
files. It reports `BRIDGE_AGENT_NOT_CONNECTED` while no result is present; this file-drop
protocol by itself does not establish a live ChatGPT Work/Codex agent connection.

After a packet arrives, CORE verifies the request ID and runs the regular bridge validator,
PIT checks, history merge/conflict checks, snapshot creation, planner, eligible models, V7
diagnostic, and append-only prediction record. The final packet is also exposed as
`data/bridge/results/<request_id>.result.json` (the prediction-ID result archive remains intact).
Complete user-confirmed JC text with code, competition, teams, and local kickoff bypasses the
research inbox and continues through the existing `JC_PRODUCTION` path. The explicit
`JC_PRODUCTION` and `AUTO_RESEARCH` modes remain available.

Start with the named fixture: competition, canonical home/away team names, exact local
kickoff, IANA timezone, UTC kickoff, and whether the user explicitly confirmed it as a
China Sporttery fixture. Research at least one fixture/schedule citation. For each
source, record its HTTPS URL, publisher, A/B/C tier, fetched time and publication time
if known. Tier A is an official association or competition, Tier B a registered public
football data source, and Tier C a registered secondary results source. A domain being
allowed does **not** prove that its cited page contains the claimed match; review the
page and retain the exact URL. Forums, blogs, odds tips and prediction sites are not
production history sources.

### Historical-data acquisition policy

The Research Agent collects source-backed candidates; it does not decide how many
samples a model needs, whether a row is eligible, or whether a model is ready. CORE
owns PIT filtering, deduplication, conflict checks, sample windows, and model-specific
readiness. Never stop after finding only three or four rows while usable sources remain.

For each target fixture, attempt to collect up to the 30 most recent completed matches
for the home team and up to 30 for the away team. Search in descending order of
coverage targets: 30, then 20, then 10, then 5. If fewer than 30 are available from
the first source, continue with other permitted sources and competitions before
stopping. Prefer the latest 24 months; if that period does not supply the target,
expand to 36 months and then farther back as needed. Preserve each actual match date;
CORE decides sample weighting and eligibility. Collect direct home-versus-away history
separately, up to 10 matches when available. Lack of direct history must not stop team
history collection.

Use multiple lawful sources where useful. Prefer Tier A official competition,
association, FIFA, UEFA, league, or club sources; then use registered Tier B
structured football-data providers. A single official schedule/results page is not
the stopping condition. Keep the source and exact URL on every row; do not count one
match twice to reach a coverage target. For national teams, combine senior-team
Nations League, continental/world championships, qualifiers, friendlies, and other
senior national-team competitions where available. The entity must remain the same
senior national team: never mix U21/U19/U17 or women's/futsal teams into the senior
men's history (or vice versa).

Before writing a packet, report `HOME_RAW_HISTORY_COUNT`, `AWAY_RAW_HISTORY_COUNT`,
and `DIRECT_RAW_COUNT`. These are candidate-row counts before CORE deduplication;
overlap between categories is allowed only when useful and must carry the same real
match facts. After Bridge ingestion, report `DEDUPED_COUNT` and `PIT_ELIGIBLE_COUNT`
from CORE's actual ingestion/audit output when exposed. Do not infer a model's sample
requirement from these counts. If CORE still returns `HISTORY_INSUFFICIENT`, report
each model's required sample, available sample, and block reason only when CORE
exposes those diagnostics; otherwise state that the per-model diagnostic is not
exposed.

Every history row needs an exact timezone-aware kickoff, teams, score, competition,
neutral-venue status, source URL/tier, and a real `fetched_at`. Use the actual
retrieval time; never backdate retrieval to make a replay pass. A result must have
been fetched after its kickoff and no later than the packet's `research_as_of`, and
`research_as_of` must precede the target kickoff. For a replay with a frozen
`research_as_of`, newly retrieved material with `fetched_at` later than that cutoff
cannot be added to that packet. Do not rewrite a previously submitted/archived packet
or reuse its request ID with changed content; request a new prediction timestamp and
request ID if the user authorizes a fresh PIT run. If only a calendar date is
available, leave `kickoff` null. CORE will reject that row for exact-time production
rather than invent a time. Do not include the target result, later fixtures, later
advancement, or post-cutoff page revisions.

Use a JSON object shaped as follows (field names and versions are fixed; values below
are **format placeholders**, not a match or prediction):

```json
{
  "schema_version": "CORE_DATA_PACKET_V1",
  "request_id": "your-unique-request-id",
  "created_at": "UTC timestamp",
  "research_as_of": "UTC timestamp before kickoff",
  "match": {
    "jc_match_code": null,
    "competition": "verified competition name",
    "home": "verified home name",
    "away": "verified away name",
    "kickoff_original": "local ISO timestamp",
    "kickoff_timezone": "IANA timezone",
    "kickoff_utc": "UTC ISO timestamp",
    "source_type": "AUTO_DISCOVERY",
    "neutral_venue": false
  },
  "entities": {
    "home_entity": "CORE-resolvable canonical ID",
    "away_entity": "CORE-resolvable canonical ID"
  },
  "fixture_evidence": {"sources": [{
    "source": "publisher", "source_url": "https://publisher.example/fixture",
    "source_tier": "A", "fetched_at": "UTC timestamp",
    "published_at": null, "evidence_type": "FIXTURE"
  }]},
  "history": {
    "home_matches": [], "away_matches": [], "direct_matches": [],
    "competition_matches": []
  },
  "optional": {"odds": [], "xg": [], "lineup": [], "injuries": [],
               "rankings": [], "news": []}
}
```

Each history row requires `date`, `kickoff`, `competition`, `home`, `away`,
`home_goals`, `away_goals`, `neutral_venue`, `source`, `source_url`,
`source_tier`, and `fetched_at`; `published_at`, `match_id`, `home_entity`, and
`away_entity` are optional. Put the same real match in multiple categories only when
necessary; CORE deduplicates it and rejects conflicting scores.

Run `python -m production.bridge.runner --input packet.json`. The desktop EXE and
production launcher also accept `--bridge-input packet.json`. The result is a single
`CORE_RESULT_PACKET_V1` JSON object with the primary blocker, snapshot ID, per-model
raw probabilities if any, V7 diagnostic, and prediction record ID. Requests and
results are archived under `data/bridge/requests/` and `data/bridge/results/`.
Bridge citations are checked for allowed domains and PIT, but content is not
cryptographically authenticated by the bridge. A source citation alone is **not**
provider-verified fixture content. Optional odds, xG, lineup, injuries, rankings, and
news are archived only; they cannot enter models until separately validated by typed
CORE data interfaces. No automatic betting or publishing is enabled.
