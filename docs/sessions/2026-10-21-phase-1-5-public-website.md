# Session 2: Phase 1.5, public website (2026-10-20 → 2026-10-21)

**Goal:** put SignalDesk on a public URL for recruiters (Streamlit Community Cloud), while keeping it
useful for my own analysis. I approved a plan with these defaults:
- a demo portfolio that works without Yahoo Finance: AAPL 20%, MSFT 20%, NVDA 15%, JPM 15%, XOM 15%,
  JNJ 15%, compared with SPY over 5 years
- request limits for the public site
- a custom live mode
- no React or Next.js

**Who did the work:** the same agent team as Phase 1, under the supervisor (Claude Code):
- **Builders:** `data-engineer`, `quant-dev` and `dashboard-dev`.
- **Reviewer:** the independent, read-only `risk-reviewer`. Nothing was committed without its PASS.

**Branch:** all work happened on `phase-1.5` and reaches `main` (the live site) only through a reviewed
pull request.

---

## 1. What got built

| Work package | Builder | What it added |
|---|---|---|
| **WP1: data for the public** | data-engineer | **Demo snapshot:** a committed 5-year price file (`data/demo/`) that loads offline. **Request limits** (`signaldesk/limits.py`): at most 10 tickers, 10 calendar years, 10 live loads per visitor per hour, 30 per hour for the whole site, and one download at a time. **Cache:** safe for many visitors, and it never deletes files it didn't create. **Errors:** messages that don't blame the visitor for Yahoo's problems. **Clock:** "today" means New York's date |
| **WP2 / WP2b: portfolio math** | quant-dev | **Buy-and-hold:** `portfolio_value`. **Drawdowns:** `drawdown_series`, the underwater curve. **Extreme stocks:** `rolling_volatility` and `return_correlation`, which accept extreme short samples (fixes A4). **Late starts:** a warning when a late-starting holding shortens the history. **Typed weights:** `normalize_weights` |
| **WP3: the dashboard** | dashboard-dev | **Modes:** demo (default) and custom live data. **Page:** key-number cards (portfolio vs benchmark over the **same** dates) and six tabs (Overview, Risk, Correlation, Holdings, Data, About). **Downloads:** nothing until "Run analysis" is pressed. **Code:** a `ui/` package that has no math, enforced by tests. **Config:** `.streamlit/config.toml` hides error details from visitors |
| **WP4: CI** | supervisor | **CI:** a GitHub Actions workflow running the whole offline test suite on every push and pull request, installing the exact tested versions from `constraints.txt`. **Requirements:** `requirements.txt` holds only what the app needs; `requirements-dev.txt` adds pytest |

**Tests:** 371 → **1,319**, all offline. In the strict check, every test also passes with the network
blocked at the socket and libcurl level.

**The demo's numbers** (5 years to 2026-10-19), checked independently by the reviewer in plain numpy:

| | Annual return | Volatility | Sharpe | Max drawdown | 1-day VaR 95% | Growth of $1 |
|---|---|---|---|---|---|---|
| Demo portfolio | 28.0% | 22.4% | 1.21 | 23.8% | 2.1% | 3.41 |
| SPY | 12.7% | 17.1% | 0.78 | 24.5% | 1.6% | 1.81 |

## 2. Delegation log

| # | Agent | Task | Result | Verdict |
|---|---|---|---|---|
| 1 | data-engineer | WP1 | Interrupted by a usage pause after building most of it | – |
| 2 | data-engineer | Check the interrupted work against the spec and finish it | Everything met; demo tests added; 711 tests | – |
| 3 | risk-reviewer | Review WP1 | Snapshot verified against a fresh Yahoo download (same 1,254 dates) | **NEEDS FIXES** (W1–W4) |
| 4 | data-engineer | Fix round 1 | 775 tests | – |
| 5 | risk-reviewer | Re-review | 0 of 2,922 days wrongly refused; 46/46 planted bugs caught | **PASS** |
| 6–7 | data-engineer → reviewer | Polish: an offline guard that can see libcurl | | **PASS** |
| 8 | quant-dev | WP2 | 864 tests; the demo numbers above | – |
| 9 | risk-reviewer | Review WP2 (resumed after a usage limit) | Every demo number recomputed in plain numpy and matched | **PASS** (Q1 flagged) |
| 10–13 | quant-dev → reviewer | Late-start warning; `normalize_weights` | 893 tests | **PASS** ×2 |
| 14 | dashboard-dev | WP3 | 1,050 tests; live demo and custom runs work | – |
| 15 | risk-reviewer | Review WP3 (resumed after a usage limit) | All numbers correct for 1Y/3Y/5Y; all 6 tabs checked in a real browser | **NEEDS FIXES** (U1) |
| 16 | data-engineer | Site-wide cap in `limits.py`; guard hardening | 1,067 tests | – |
| 17 | dashboard-dev | Fix round 1 (U1–U7) | 1,212 tests | – |
| 18 | risk-reviewer | Re-review | Reload attack now stops at 30 downloads | **PASS** |
| 19–20 | dashboard-dev → reviewer | Polish (U8–U13), resumed after a usage limit | 1,295 tests | **PASS** (U14 flagged) |
| 21–22 | dashboard-dev → reviewer | U14: tables sort by value again; U15 guard additions (paused overnight and resumed) | 1,319 tests; sorting checked in a real browser | **PASS** |

## 3. What the reviewer caught

The full write-ups are in the [catch log](../catch-log.md#session-2-phase-15-public-website-2026-10-20--21), C14–C22. The highlights:

- **A reload reset the rate limit** ([C19](../catch-log.md#c19-reloading-the-page-reset-the-rate-limit)).
  Each visitor's budget was tied to the browser session, so a script opening new sessions could have
  sent about 1,000 Yahoo downloads an hour and had the server's IP blocked for everyone. There is now
  a site-wide cap of 30 an hour. The reviewer re-ran its attack: 35 new sessions produced exactly 30
  downloads.
- **The 10-year limit would have refused a "10Y" button every single day**
  ([C14](../catch-log.md#c14-the-10-year-limit-refused-10y-every-day)). It counted days ÷ 365.25, so
  three leap days tipped a real 10-year range over the limit. The reviewer found it by checking all
  2,922 days from 2026 to 2033.
- **A late-starting holding silently shortened the portfolio**
  ([C18](../catch-log.md#c18-a-late-starting-holding-silently-shortened-the-portfolio)). The page
  would have compared one year of portfolio with five years of SPY. This is exactly the kind of
  sample-period trap the Phase-2 risk agent is meant to catch, so the code now says so out loud.

## 4. Decisions made in this session

Recorded in [decisions.md](../decisions.md#phase-15-public-website), PW-1 to PW-14. The most important:
- **Hosting and data:** Streamlit Community Cloud, with a committed demo snapshot so Yahoo outages
  can't break the first impression.
- **Safe default:** the public limits apply unless the server is explicitly started with
  `SIGNALDESK_MODE=local`.
- **One concurrency issue stays open (R7, [C21](../catch-log.md#c21-a-visitors-warnings-can-get-lost-between-threads)).**
  The numbers are never affected, but a warning text can occasionally be lost when two visitors use
  the site at once. The real fix is for `risk.py` to return its diagnostics as data, which the Phase-2
  risk agent needs anyway. It is the first Phase-2 preparation task.

## 5. Process notes

- **Two usage-limit interruptions:**
  - The first WP1 run was cut off midway. A fresh data-engineer checked the half-finished work
    against the full spec instead of starting over.
  - Two reviews were also cut off. Each was resumed with its earlier context, so no checks were
    repeated or lost.
- **Git attribution was cleaned up** before this phase. History was restarted with a single commit
  under my own identity, and commits no longer carry an AI co-author line. The README still explains
  openly that AI agents wrote the code under my design and review process.
- **Custom agent types now load natively,** after a restart of Claude Code. In Phase 1 they had to be
  emulated.

## 6. Still open

- **Since Phase 1:**
  - R7: warnings across threads (C21), the first Phase-2 prep task.
  - D6: a version tag in cache file names.
  - A7: thin-tail VaR flag in the table.
- **Small suggestions the reviewer left for later:**
  - `DemoData` could expose when the snapshot was built.
  - The two kinds of gap note could get separate warning classes.
  - `normalize_weights` could report the typed total.
- **Before go-live (my part):** deploy on Streamlit Community Cloud (Python 3.13), then add the live
  link and a fresh screenshot to the README.
