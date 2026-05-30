# Network Investigation — `stats.nba.com` throttle in the dev sandbox

**Date:** 2026-05-30
**Context:** Prior sessions reported NBA-API fetches timing out here. This pins
down *exactly* what is and isn't reachable so we stop mislabeling the sandbox as
"offline" and know which fetches must move to the user's normal environment.

## Diagnostic (run from the sandbox)

| Target | Test | Result |
|---|---|---|
| `github.com` | `curl https://github.com` | **HTTP 200 in 0.46 s** |
| `stats.nba.com` | DNS (`getent hosts`) | resolves → Akamai edge (`e8017.dsci.akamaiedge.net`, IPv6 `2600:1404:…`) |
| `stats.nba.com:443` | raw TCP connect | **connects OK** |
| `stats.nba.com` | `curl https://stats.nba.com/stats/scoreboardv2?...` | **times out (>20 s)** |
| `cdn.nba.com` | `curl …/liveData/scoreboard/...` | **HTTP 403 in 0.12 s** (reachable, just needs auth headers) |

## Conclusion

The sandbox is **not offline**. General internet works (GitHub/git push fine).
The failure is **`stats.nba.com`-specific and at the application layer**:

- DNS resolves and the **TCP/443 handshake completes**, so it is *not* a DNS or
  firewall port block.
- The HTTPS request then **hangs and times out** — the response is silently
  black-holed *after* connect. This is the signature of **Akamai datacenter-IP
  throttling**: the stats host (behind Akamai) drops/stalls requests from
  cloud/datacenter IP ranges, which this sandbox is on.
- `cdn.nba.com` (also NBA, also Akamai, but a different/cacheable property)
  answers **instantly with 403** — reachable, just rejecting our headers. So the
  block is not "all of NBA," only the `stats` origin's anti-bot path.

This matches the existing memory note ([[feedback_not_offline_nba_throttle]]):
only `stats.nba.com` is throttled; never call the environment offline.

## Implications for the pipeline

- **Anything hitting `stats.nba.com`** (NBA-API: `leaguedashptstats`,
  `commonteamroster`, `boxscoresummaryv2`, `scoreboardv2`, clutch stats, the
  Phase-3 game-log/inactives backfills, and the new
  `src/data_fetch/fetch_pregame_lineups.py`) **cannot run here.** Run them from
  the user's normal (residential-IP) environment.
- **Everything offline runs here fine** — Step 0a lineup validation/tuning uses
  only local `pbp_with_lineups_*` + profiles and completes in the sandbox.

## Workarounds (for the networked env or future automation)

1. **Run from a residential IP** (the user's normal machine) — the established
   fix; the prior NBA fetches in this repo were all collected that way.
2. **`curl_cffi` + `impersonate="chrome124"` + full ~30-param payloads** — needed
   for `leaguedashptstats` regardless of IP ([[feedback_nba_api_full_params]]);
   necessary but **not sufficient** here because the throttle is IP-based, not
   just fingerprint-based.
3. **`cdn.nba.com` liveData endpoints** — reachable from the sandbox; usable for
   *live* scoreboard/boxscore JSON with correct headers, but does not cover the
   historical `leaguedash*` aggregates.
4. A residential/mobile **proxy** in front of `stats.nba.com` if sandbox-side
   automation is ever required (not currently needed).

## Verdict

No *plain-curl* request to `stats.nba.com` succeeds in-sandbox — but see the
correction below: a `curl_cffi` TLS-impersonated request **does**.

---

## CORRECTION (2026-05-30, later same day) — curl_cffi gets through

Running `src/data_fetch/fetch_matchup_data.py` (which uses
`curl_cffi … impersonate="chrome110"`) from the sandbox **successfully fetched all
9 seasons** of `leagueseasonmatchups` (2017-18…2025-26, 1.22M rows) — no timeouts.

So the throttle is **TLS-fingerprint / request-shape dependent, not a blanket
datacenter-IP block**:

- **Plain `curl` / default Python `requests`** → timeout (the diagnostic above).
- **`curl_cffi` with `impersonate="chrome110/124"`** → succeeds for at least
  `leagueseasonmatchups`.

Akamai is filtering on TLS/JA3 fingerprint + headers; a real-browser impersonation
passes. **Revised guidance:** prefer `curl_cffi` + `impersonate` for any
`stats.nba.com` fetch — it may well work in-sandbox. Still validate per-endpoint
(some endpoints/anti-bot paths may differ, e.g. `scoreboardv2` was not retried
here), and keep the networked-env fallback for anything that does time out. This
updates [[feedback_not_offline_nba_throttle]]: the sandbox is not just "not
offline" — `stats.nba.com` itself is reachable with the right client.
