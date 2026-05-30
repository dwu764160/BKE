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

No code change fixes this in-sandbox — it is an IP-reputation throttle on one
host. The correct handling is the split already adopted: **offline work
(Step 0a) here; all `stats.nba.com` fetches (Step 0b live lineups, Phase-3
backfills) in the networked environment.**
