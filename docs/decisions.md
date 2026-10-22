# Design decisions

Each decision records *what* was chosen, *why*, and what was rejected. Engineering teams
call these "Architecture Decision Records". New decisions go at the bottom of the right section.

Legend: **Who** = who made the call. "Supervisor" means the Claude Code session acting as supervisor;
"me" means a decision I made myself.

---

## Process: how the agents work together

### P1. A supervisor delegates, builders build, a reviewer checks
- **Decision:** one supervisor agent plans and delegates. Three builder agents each own one layer
  (data → math → dashboard, in that order). One reviewer agent checks every result before it is committed.
- **Why:** each agent stays focused on one job, and no work goes into git without a second pair of eyes.
- **Who:** me (`CLAUDE.md`, `.claude/agents/`).

### P2. The reviewer is read-only and runs on a different model
- **Decision:** `risk-reviewer` can read files and run tests but cannot edit anything. It runs on Opus,
  while the builders run on Sonnet.
- **Why:** a reviewer that can "just fix it" stops being independent. A different model is less likely
  to share the builder's blind spots.
- **Who:** me (`.claude/agents/risk-reviewer.md`).

### P3. At most 3 fix rounds per builder, then commit
- **Decision:** on NEEDS FIXES, critical and warning issues go back to the same builder; on PASS, the
  supervisor commits. After 3 rounds it commits anyway and logs what's left.
- **Why:** it guarantees progress and stops endless review loops.
- **Who:** me (overnight prompt).

### P4. Same agent for fixes, fresh reviewer for each new layer
- **Decision:** fix requests are sent back to the builder that wrote the code, which still remembers it.
  Each layer gets a new reviewer instance.
- **Why:** builders fix faster with context, and a fresh reviewer per layer keeps reviews independent.
- **Who:** supervisor.

### P5. Every catch gets documented
- **Decision:** every caught mistake goes into [`catch-log.md`](catch-log.md), using a fixed template.
- **Why:** catches are the evidence that the review process works. That's the point of the project.
- **Who:** me (`CLAUDE.md`).

---

## Data layer (`signaldesk/data.py`)

### D-1. Use adjusted close prices
- **Decision:** `yfinance` with `auto_adjust=True`, so prices include splits and dividends.
- **Why:** returns then include dividends, and splits don't look like crashes. The reviewer verified this
  against a real NVDA split and an AAPL dividend.
- **Caveat:** Yahoo rescales old prices after each new dividend. Returns are fine, but don't compare price
  *levels* across downloads made on different days.

### D-2. Never fill missing prices; report them instead
- **Decision:** gaps stay blank (NaN). `find_gaps` reports blank cells, and `find_missing_days` reports
  weekdays with no row at all.
- **Rejected:** forward-filling, which invents prices and hides problems.

### D-3. End dates are exclusive and capped at New York's "today"
- **Decision:** `load_prices(..., end)` excludes `end`, and a future `end` is capped at today's date in New York.
- **Why:** without the cap, partial data was cached forever ([C3](catch-log.md#c3-a-future-end-date-froze-the-cache)).
  Using the computer's own date let an unfinished midday price in ([C4](catch-log.md#c4-the-fix-for-c3-used-the-wrong-clock)).
- **Who:** supervisor, after reviewer findings.

### D-4. An unknown ticker fails the whole request
- **Decision:** raise `PriceDataError` and cache nothing.
- **Rejected:** quietly dropping the ticker, because then the user doesn't get what they asked for, silently.

### D-5. A CSV cache in the project folder
- **Decision:** one CSV file per request in `<project>/data/cache/` (ignored by git), read back
  with exact float precision.
- **Why:** it is simple, easy to inspect, and gives the same numbers every time ([C13](catch-log.md#c13-self-catch-the-csv-cache-changed-the-last-digit-of-prices)).

---

## Math layer (`signaldesk/risk.py`)

### M-1. Returns across a gap are measured from the last valid price
- **Decision:** each ticker's return uses *its own* previous valid price. When that skips a gap, a
  `GapWarning` says so, including how long the longest gap is.
- **Rejected:** the supervisor's first rule ("blank the returns next to a missing price"). With real data
  it changed AAPL's annual return from 21.7% to 28.8% just by adding foreign stocks to the table
  ([C1](catch-log.md#c1-the-supervisors-gap-rule-distorted-returns)).
- **Who:** supervisor, after the reviewer proved the first rule wrong.

### M-2. A year has 252 trading days
- **Decision:** annualize with 252 days.
- **Caveat:** this is only valid for stocks and ETFs. Crypto trades 365 days a year and comes out wrong;
  the documentation says so.

### M-3. Formula conventions
- **Volatility:** sample standard deviation (`ddof=1`) × √252.
- **Annualized return:** geometric (CAGR), `(∏(1+r))^(252/n) − 1`. This is what an investor actually earned per year.
- **Sharpe:** arithmetic mean of excess returns ÷ standard deviation × √252. The risk-free rate is an annual fraction.
- **VaR:** historical, `−quantile(returns, 1 − confidence)`, with linear interpolation.

### M-4. Losses are positive numbers
- **Decision:** max drawdown and VaR are both returned as positive numbers (0.25 = a 25% loss).
- **Why:** one convention for both loss measures, so nobody has to guess the sign.

### M-5. Warn on short samples instead of refusing them
- **Decision:** below 60 daily returns (about 3 months), functions still return a number but emit a
  `ShortSampleWarning`. VaR also warns when fewer than 5 observations are in the loss tail (fewer than
  100 returns at 95%).
- **Why:** the number can still be useful, but the reader, and later the Phase-2 risk agent, must know it
  is weak. Three days of a meme stock annualize to a CAGR of about 10^61.

### M-6. One bad ticker never hides the others
- **Decision:** `risk_summary` gives an uncomputable ticker an empty row plus a `SkippedTickerWarning`.
  The individual functions still raise errors.
- **Why:** a recent IPO or a bad price shouldn't delete the whole table ([C9](catch-log.md#c9-one-bad-ticker-deleted-the-whole-metrics-table)).

### M-7. Reject prices where returns are expected
- **Decision:** return-taking functions raise an error when a column's median |value| > 0.5.
- **Why:** it catches an easy mistake that otherwise gave 46,100% volatility silently ([C7](catch-log.md#c7-prices-passed-as-returns-gave-46100-volatility-silently)).
- **Caveat:** it's a heuristic, and prices below $0.50 aren't caught.

---

## Dashboard (`app.py`)

### UI-1. No math in the dashboard, enforced by tests
- **Decision:** every number comes from `risk.py` or `data.py`. Tests scan `app.py` and each chart's
  specification for calculations.
- **Why:** the same functions will be the AI agents' tools in Phase 2, so there must be a single source of truth.

### UI-2. The end date in the sidebar is inclusive
- **Decision:** the date picker means "up to and including". The app adds 1 day before calling
  `load_prices`, which is end-exclusive.
- **Why:** that's how people read a date range.

### UI-3. The risk-free rate is typed as a fraction, within −5% … 20%
- **Decision:** the user types 0.04 for 4%, so the app does no unit conversion. The bounds catch "4.0"
  typed by mistake, and a caption shows the rate used.

### UI-4. No extra Streamlit cache *(superseded in Phase 1.5 by PW-6)*
- **Decision:** no `st.cache_data`. The CSV cache is enough.
- **Why:** a second cache could freeze data that the data layer deliberately refreshes every day.

### UI-5. No new dependencies
- **Decision:** charts use Altair, which ships with Streamlit.

---

## Phase 1.5: public website

### PW-1. Host on Streamlit Community Cloud
- **Decision:** deploy the existing Streamlit app from the GitHub repo's `main` branch, with Python 3.13.
- **Why:** it's free, redeploys on every push, and needs no servers or Docker.
- **Rejected:**
  - Hugging Face Spaces: a fine backup if the limits bite.
  - Render, Railway or Fly: more setup and usually paid.
  - Rewriting in React or Next.js: not yet.
- **Who:** me.

### PW-2. The default view is a committed demo snapshot
- **Decision:**
  - `data/demo/` holds 5 years of adjusted closes for the demo portfolio and SPY, and the page opens
    on it.
  - `scripts/refresh_demo_data.py` refreshes it; that is a manual step followed by a commit.
- **Why:** Yahoo Finance is known to rate-limit or block shared cloud servers. A recruiter's first
  click must never depend on that. It also makes the demo numbers reproducible.
- **Who:** me (approved default).

### PW-3. Public limits are the default; local use is unlimited
- **Decision:** `SIGNALDESK_MODE=local` removes the limits for my own analysis. Anything else, or
  nothing, means the public limits.
- **Why:** forgetting to set a variable on the server must never remove the protection.
- **Who:** supervisor.

### PW-4. Public limits
- **Decision:**
  - At most 10 tickers per request.
  - At most 10 **calendar** years.
  - At most 10 live analyses per visitor per hour.
  - At most **30 live downloads per hour for the whole site**.
  - One download at a time.
  - Answers from the cache are free.
- **Why:**
  - The per-visitor limit resets when a session is reloaded, so the site-wide limit is what really
    protects the shared server IP from being blocked
    ([C19](catch-log.md#c19-reloading-the-page-reset-the-rate-limit)).
  - Calendar years, because counting days ÷ 365.25 refused real 10-year ranges
    ([C14](catch-log.md#c14-the-10-year-limit-refused-10y-every-day)).
- **Who:** me (approved defaults); supervisor (site cap and calendar years, after reviewer findings).

### PW-5. Honest error messages instead of guessing the cause
- **Decision:** when Yahoo returns nothing, the message names every possible cause: provider
  rate-limiting or down, a symbol that doesn't exist, or no trading days in the range.
- **Rejected:** parsing yfinance's log output to tell the causes apart. That is too brittle, because
  yfinance swallows the real errors ([C15](catch-log.md#c15-yahoos-rate-limits-never-reached-our-code)).
- **Note:** this reworded one Phase-1 message on purpose. The supervisor signed off on it.

### PW-6. Caching for many visitors
- **Decision:**
  - Demo files are cached in memory.
  - Live prices go into a small in-memory cache, kept 6 hours and keyed by New York's date.
  - The CSV cache uses a unique temp file per write.
  - Old cache files are pruned, but only names the cache itself created, and only in the default
    folder.
- **Why:** many visitors share one process (threads). A clean-up must never delete a user's file
  ([C16](catch-log.md#c16-the-cache-clean-up-deleted-a-users-own-file)).

### PW-7. The demo portfolio is buy-and-hold
- **Decision:** `portfolio_value` splits $1 by the starting weights and never rebalances.
- **Why:** it's the easiest to explain, and it matches "growth of $1". Rebalancing daily would have
  given a 5-year growth of 3.22 instead of 3.41 (NVDA grew into 43% of the portfolio).
- **Who:** supervisor.

### PW-8. Compare the portfolio and the benchmark over the same dates
- **Decision:**
  - The benchmark and every holding are measured over the portfolio's own dates, and the page shows
    the start date.
  - A late-starting holding triggers a warning.
- **Why:** otherwise one year of portfolio could be compared with five years of SPY
  ([C18](catch-log.md#c18-a-late-starting-holding-silently-shortened-the-portfolio)).

### PW-9. Typed weights are rescaled in `risk.py`
- **Decision:** visitors type weights in percent or any relative numbers (1/1/1 means equal weight),
  and `risk.normalize_weights` turns them into fractions.
- **Why:** even dividing by 100 is math, and math belongs in `risk.py`.

### PW-10. The dashboard is a `ui/` package with an enforced "no math" rule
- **Decision:** `app.py` is a thin entry point. Tests scan every UI file and every chart spec for
  arithmetic, with four pinned exceptions for layout and dates.
- **Why:** the first guard missed 24 of 26 tricks
  ([C20](catch-log.md#c20-the-no-math-guard-missed-24-of-26-tricks)).

### PW-11. Visitors never see internals
- **Decision:**
  - `showErrorDetails = "none"` and `toolbarMode = "viewer"` in `.streamlit/config.toml`.
  - Logged catch-alls keep the title, footer and a way back to the demo when something unexpected
    fails.
- **Why:** a traceback with server paths is unprofessional and leaks details.

### PW-12. Warnings across threads: reduced, not fixed (R7 stays open)
- **Decision:** keep a lock around the warning capture, and document honestly that other code can
  still interfere.
- **Why:** the full fix is `risk.py` returning diagnostics as data. The Phase-2 risk agent needs that
  anyway, so it's the first Phase-2 preparation task. Numbers are never affected
  ([C21](catch-log.md#c21-a-visitors-warnings-can-get-lost-between-threads)).

### PW-13. CI runs every test on every push
- **Decision:**
  - GitHub Actions runs `pytest` on Python 3.13 for every push and pull request.
  - `requirements.txt` holds only what the app needs (Streamlit Cloud installs it), and
    `requirements-dev.txt` adds pytest.
  - CI installs with `constraints.txt`, a `pip freeze` of the tested environment.
  - `requirements.txt` also pins `curl_cffi`, `altair` and `pyarrow`, because the tests patch or
    inspect them.
- **Why:** proof that the tests pass on a clean machine, visible to anyone, with the same package
  versions that were tested.

### PW-14. Work on a branch; `main` is the live site
- **Decision:** Phase 1.5 was built on `phase-1.5` and merged through a pull request, after my
  approval.
- **Why:** half-finished work can never reach the public site by accident.
