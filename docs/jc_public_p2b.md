# JC official public market, R3 P2B

This supplement reads only the two configured public pages on `www.sporttery.cn`.
It records raw HTML and hashes under `blind_test/r3/market/jc_public/raw/` and
separate append-only JC rows in `blind_test/r3/market/jc_public/jc_public.sqlite`.
It never changes the frozen YY-CORE prediction or its original lock.

Run from the project root with the Python 3.11 environment:

```powershell
.venv\Scripts\python.exe -m erguoyuan_football.blind_test_r3.jc_public_lock <prediction_id>
```

The public HTML currently exposes no reliable semantic fixture/bonus rows to
this parser. The resulting supplement therefore records
`PUBLIC_PAGE_AUTOMATION_UNAVAILABLE` and retains `MODEL_ONLY` Fusion. Do not
promote the user's screenshot prices to `OFFICIAL_PUBLIC_PAGE` evidence.

For a person-confirmed pre-kickoff market snapshot, create a new JSON or CSV
file. JSON requires the following explicit evidence fields in addition to
fixture identity and prices:

```json
{
  "source": "USER_CONFIRMED_MARKET",
  "confirmation_status": "CONFIRMED",
  "confirmed_by": "name-or-identifier",
  "evidence_description": "How and where the displayed fixed bonuses were checked",
  "jc_match_number": "周一001",
  "competition": "competition name",
  "home_team": "home name",
  "away_team": "away name",
  "kickoff_time": "2026-10-05T16:00:00+00:00",
  "captured_at": "2026-10-05T10:00:00+00:00",
  "confirmed_at": "2026-10-05T10:01:00+00:00",
  "spf": {"home": 1.80, "draw": 3.30, "away": 4.50},
  "rqspf": {"handicap": -1, "home": 3.10, "draw": 3.45, "away": 1.95}
}
```

Times must include a UTC offset and precede kickoff. The person-confirmed
path is tagged `USER_CONFIRMED`, never `OFFICIAL_PUBLIC_PAGE`:

```powershell
.venv\Scripts\python.exe -m erguoyuan_football.blind_test_r3.jc_public_lock <prediction_id> --user-confirmed <new-input.json>
```

The illustrative prices above are test values only. They are not real JC
fixed bonuses and must not be used for a live prediction.
