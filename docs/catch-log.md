# Catch log

Every time an agent catches a mistake, whether it belongs to another agent, to the supervisor, or to
the agent itself, it gets an entry here. This is the evidence for the project's main idea: **a
skeptical, read-only reviewer makes a team of AI agents more trustworthy.**

The IDs in brackets (for example `[R1]`) are the reviewer's original issue numbers, so each entry
can be traced back to the [session journals](sessions/).

## How to add an entry

Copy this template and give the entry the next number:

```markdown
### C<n>: <short title>
- **Session:** <date / phase> · **Layer:** <file> · **Caught by:** <agent> · **Mistake by:** <agent>
- **What was wrong:** what the code or spec did, in plain words.
- **How it was found:** the method used (real data, hand calculation, planted bug, ...).
- **Evidence:** the concrete numbers that proved it.
- **Fix:** what changed.
- **Guard:** the test that now fails if the mistake comes back.
- **Lesson:** what to remember next time.
```

## Index

| # | Title | Caught by | Mistake by | Severity |
|---|---|---|---|---|
| [C1](#c1-the-supervisors-gap-rule-distorted-returns) | The supervisor's gap rule distorted returns ★ | risk-reviewer | **supervisor** | blocking |
| [C2](#c2-tests-checked-that-the-page-ran-not-what-it-showed) | Tests checked that the page ran, not what it showed ★ | risk-reviewer | dashboard-dev | blocking |
| [C3](#c3-a-future-end-date-froze-the-cache) | A future end date froze the cache | risk-reviewer | data-engineer | blocking |
| [C4](#c4-the-fix-for-c3-used-the-wrong-clock) | The fix for C3 used the wrong clock | risk-reviewer | data-engineer | blocking |
| [C5](#c5-two-fixes-together-reported-52-fake-missing-days) | Two fixes together reported 52 fake missing days | risk-reviewer | data-engineer | blocking |
| [C6](#c6-a-timezone-test-that-could-not-fail) | A timezone test that could not fail | risk-reviewer | data-engineer | blocking |
| [C7](#c7-prices-passed-as-returns-gave-46100-volatility-silently) | Prices passed as returns gave 46,100% volatility, silently | risk-reviewer | quant-dev | suggestion |
| [C8](#c8-missing-trading-days-were-invisible) | Missing trading days were invisible | risk-reviewer | data-engineer | blocking |
| [C9](#c9-one-bad-ticker-deleted-the-whole-metrics-table) | One bad ticker deleted the whole metrics table | risk-reviewer | quant-dev | suggestion |
| [C10](#c10-the-no-math-in-the-dashboard-test-was-blind-to-chart-code) | The "no math in the dashboard" test was blind to chart code | risk-reviewer | dashboard-dev | blocking |
| [C11](#c11-a-40-day-data-outage-was-called-a-holiday) | A 40-day data outage was called a holiday | risk-reviewer | quant-dev | suggestion |
| [C12](#c12-gmes-real-squeeze-was-rejected-as-prices) | GME's real squeeze was rejected as "prices" | risk-reviewer | quant-dev | suggestion |
| [C13](#c13-self-catch-the-csv-cache-changed-the-last-digit-of-prices) | Self-catch: the CSV cache changed the last digit of prices | data-engineer | data-engineer | – |
| [More](#smaller-catches) | 18 smaller catches | | | |

★ = highlighted in the README.

---

## Session 1: overnight Phase 1 build (2026-10-20)

### C1: The supervisor's gap rule distorted returns
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `signaldesk/risk.py` · **Caught by:** risk-reviewer · **Mistake by:** supervisor (spec) `[R1]`
- **What was wrong:** the supervisor told quant-dev: *"if a price is missing, the returns next to it
  are blank, and statistics skip blank returns."* It sounds careful, but it throws away real price moves.
  In a table that mixes US and foreign stocks, AAPL has blank cells on US holidays, because the foreign
  markets were open. AAPL's numbers then depended on **which other stocks were in the table**.
- **How it was found:** the reviewer downloaded real 2024–2025 prices and compared AAPL on its own with
  AAPL next to SAP.DE (Germany) and 7203.T (Japan).
- **Evidence:**

  | | AAPL alone | AAPL + foreign stocks |
  |---|---|---|
  | Annualized return | 21.7% | **28.8%** |
  | Sharpe ratio | 0.84 | **1.05** |
  | Rolling volatility points missing | – | **99%** |

  The growth chart still showed +47.8% (about 21.7% a year), so two functions contradicted each other.
  A single missing price on AAPL's biggest day (2025-04-09, +15.3%) cut its annual return to 15.9%.
- **Fix:** each stock's return is now measured from **its own last valid price**. When that skips over a
  gap, a `GapWarning` says so out loud. Re-checked on the same real data: **21.73% both alone and mixed**,
  with missing rolling points down from 99% to 3.9%.
- **Guard:** 10 tests fail if the old rule comes back, including
  `test_risk_summary_row_is_identical_alone_or_beside_a_ticker_with_another_calendar`.
- **Lesson:** the reviewer questioned the **spec**, not just the code. The builder had implemented the
  wrong rule perfectly. A checker has to sit above everyone, including the one giving the orders.

### C2: Tests checked that the page ran, not what it showed
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `app.py`, `tests/test_app.py` · **Caught by:** risk-reviewer · **Mistake by:** dashboard-dev `[A1]`
- **What was wrong:** the dashboard's numbers were all correct, but its tests only checked that the page
  loaded and that raw data existed. They never read the text a user actually sees.
- **How it was found:** **mutation testing.** The reviewer made 16 small, realistic display bugs in copies
  of `app.py` and ran the tests against each one.
- **Evidence:** **13 of 16 bugs passed every test**, including:
  - max drawdown shown as **"−28.4%"** (sign flipped)
  - max drawdown shown as **"2843.9%"** (multiplied by 100 twice)
  - Sharpe shown as a percentage
  - heatmap colours reversed
  - a rolling chart titled "20-day" while computing over 60 days
- **Fix:** new tests read the exact strings on screen and the exact data inside each chart, and compare
  them with `risk.py`. One test uses prices `[100, 110, 99, 108.9]` and checks hand-calculated strings
  such as "183.3%" and "4.58".
- **Guard:** the reviewer re-ran its own 16 bugs, and **16 of 16 are now caught**.
- **Lesson:** "the tests pass" is not the same as "the tests test something". Planting bugs on purpose
  is the quickest way to find out which one you have.

### C3: A future end date froze the cache
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `signaldesk/data.py` · **Caught by:** risk-reviewer · **Mistake by:** data-engineer `[D1]`
- **What was wrong:** asking for prices through a future date saved whatever existed so far under that
  future date's file name, and returned the same frozen data forever.
- **How it was found:** a live download, then inspecting the cache.
- **Evidence:** `load_prices(['AAPL'], '2026-10-01', '2026-12-31')` returned 13 rows and saved them as
  `..._2026-12-31.csv`. Every later call returned those same 13 rows.
- **Fix:** the end date is capped at "today" before the cache name is built.
- **Guard:** tests with a pinned clock check the capped file name and that the cache refreshes the next day.
- **Lesson:** a cache key must describe the data you *got*, not the data you *asked for*.

### C4: The fix for C3 used the wrong clock
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `signaldesk/data.py` · **Caught by:** risk-reviewer · **Mistake by:** data-engineer (in a fix) `[D9]`
- **What was wrong:** "today" came from the computer's clock. On a computer in Tokyo or Sydney the date
  changes while the US market is still open, so an **unfinished midday price** would be saved as that
  day's close, permanently.
- **How it was found:** the reviewer read yfinance's source code to see how it interprets dates, then
  simulated a Tokyo clock.
- **Evidence:** a noon price of **247.13** was still being served weeks later, instead of the real close
  of **249.50**.
- **Fix:** "today" is now New York's date. Every exchange's session for any earlier date has finished.
- **Guard:** exact tests around New York midnight (summer and winter time), plus an end-to-end
  "Tokyo clock" test.
- **Lesson:** fixes need review too. This bug did not exist until C3 was fixed.

### C5: Two fixes together reported 52 fake missing days
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `signaldesk/data.py` · **Caught by:** risk-reviewer · **Mistake by:** data-engineer (interaction of two fixes) `[D13]`
- **What was wrong:** the gap report (added for C8) didn't apply the same "cap at today" rule as the
  loader (C3). Each fix was correct on its own; together they disagreed.
- **How it was found:** the reviewer used the two functions together, exactly as the docstring told users to.
- **Evidence:** complete data through Oct 19, with the end date Dec 31, produced **52 "missing" days**,
  all of them in the future.
- **Fix:** both functions now share one helper, `_limit_end_to_today`.
- **Guard:** an end-to-end test asserts 0 false missing days. Removing the cap fails with `assert 52 == 0`.
- **Lesson:** this was the third bug in a chain (C3 → C4 → C5). Re-review after **every** fix, and test
  functions in the combinations people will actually use.

### C6: A timezone test that could not fail
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `tests/test_data.py` · **Caught by:** risk-reviewer · **Mistake by:** data-engineer `[D3]`
- **What was wrong:** the test checked that the timezone was removed from the dates, but not that the
  dates were still *correct*.
- **How it was found:** a planted bug. Swapping `tz_localize(None)` for `tz_convert(None)` moves every
  Tokyo closing price onto the **previous day**, which is a look-ahead-style error, yet all 60 tests still passed.
- **Fix:** tests now assert the exact dates for New York, Tokyo, Sydney and UTC. While fixing it, the
  builder found **the same bug in a second function** and fixed that too.
- **Guard:** `test_clean_keeps_the_local_exchange_date_for_timezone_aware_indexes[Asia/Tokyo]`.
- **Lesson:** testing only New York and UTC would not have caught it. Pick test cases where the bug
  would actually show.

### C7: Prices passed as returns gave 46,100% volatility, silently
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `signaldesk/risk.py` · **Caught by:** risk-reviewer · **Mistake by:** quant-dev `[R4]`
- **What was wrong:** the functions that expect daily returns happily accepted prices.
- **How it was found:** the reviewer fed real prices into them, an easy mistake for a dashboard or a
  future AI agent.
- **Evidence:** `annualized_volatility(prices)` returned **461** (46,100%), `annualized_return(prices)`
  returned **infinity**, and nothing raised an error.
- **Fix:** they now refuse with *"looks like prices, not returns; call daily_returns first"*. The
  reviewer checked that this never trips on real returns over normal samples, including GME in January
  2021 and 3× leveraged ETFs.
- **Guard:** `test_return_functions_refuse_a_price_series`.
- **Lesson:** in Phase 2 an AI agent will call these functions. Make wrong calls loud.

### C8: Missing trading days were invisible
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `signaldesk/data.py` · **Caught by:** risk-reviewer · **Mistake by:** data-engineer `[D2]`
- **What was wrong:** the gap report only counted blank cells. A day missing from the table entirely was
  never reported, and the next return then quietly covered two days.
- **Evidence:** a hand-made table missing Wednesday 2026-03-04 reported **0 gaps**.
- **Fix:** a new `find_missing_days`. On real 2026 data it lists exactly the 7 US market holidays.
- **Lesson:** the Phase-2 risk agent's "is the sample long enough?" check needs honest day counts.

### C9: One bad ticker deleted the whole metrics table
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `signaldesk/risk.py` · **Caught by:** risk-reviewer · **Mistake by:** quant-dev `[R2, R12]`
- **What was wrong:** one ticker with too little data (a recent IPO), a flat price, or a zero price made
  `risk_summary` raise an error, so the dashboard showed no table at all.
- **Evidence:** real data with NVDA cut to 2 rows gave `ValueError ... 'NVDA' has 1`. Yahoo really returns
  a price of **0.0** for SHIB-USD in August 2020.
- **Fix:** only that ticker gets an empty row plus a `SkippedTickerWarning` that names the reason and
  date. The other tickers still show.
- **Lesson:** fail loudly for the one bad input, not for everything around it.

### C10: The "no math in the dashboard" test was blind to chart code
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `tests/test_app.py` · **Caught by:** risk-reviewer · **Mistake by:** dashboard-dev `[A2]`
- **What was wrong:** a test scanned `app.py` for math (a project rule), but math hidden inside the
  chart library's settings (Altair `transform_calculate`, `"mean(x)"`) slipped past it.
- **Evidence:** 3 planted bugs of that kind passed every test.
- **Fix:** each chart's specification is now checked for transforms and aggregations, and the code scan
  covers more operations.
- **Lesson:** a rule is only as strong as the check that enforces it.

### C11: A 40-day data outage was called a holiday
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `signaldesk/risk.py` · **Caught by:** risk-reviewer · **Mistake by:** quant-dev `[R11]`
- **What was wrong:** the gap warning counted gaps but not their length. A 40-row outage looked like
  "1 gap, normal for holidays", while its multi-week move was being counted as one day.
- **Fix:** the warning shows each ticker's longest gap. Anything over 5 rows says *"likely missing data"*,
  and the dashboard shows it as a warning instead of an info note.
- **Lesson:** a warning has to carry enough detail for the reader, human or agent, to judge it.

### C12: GME's real squeeze was rejected as "prices"
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `signaldesk/risk.py` · **Caught by:** risk-reviewer · **Mistake by:** quant-dev (side effect of C7) `[R8]`
- **What was wrong:** over a 1-week window, GME's real returns (+18%, +93%, +135%) tripped the C7 guard,
  and `risk_summary` blamed "prices" even though it had computed those returns itself.
- **Fix:** `risk_summary` skips that check internally. The public functions keep it.
- **Lesson:** every new guard needs a test with extreme but real data.

### C13: Self-catch: the CSV cache changed the last digit of prices
- **Session:** 2026-10-20 / Phase 1 · **Layer:** `signaldesk/data.py` · **Caught by:** data-engineer · **Mistake by:** data-engineer
- **What was wrong:** reading a price back from the CSV cache changed its last binary digit.
- **Evidence:** 467.70281982421875 came back as 467.7028198242187, so "cached" and "fresh" data were not
  identical, which breaks "same input, same output".
- **Fix:** `read_csv(float_precision="round_trip")`.
- **Guard:** a regression test with Yahoo-like prices. It fails without the fix.
- **Lesson:** the builder's own live smoke test found what its hand-made test data was too simple to show.

### Smaller catches

| ID | What was wrong | Fix |
|---|---|---|
| D4 | Two times on the same day passed the "start before end" check | Test added |
| D5 | Two timestamps on the same date both survived cleaning | Dates normalized to midnight |
| D8 | The cache folder depended on where the program was launched | Always `<project>/data/cache` |
| D10 | Days missing at the edges of the requested window weren't reported | Optional start/end for the gap report |
| D11 | Odd input crashed with a confusing pandas error | Clear error messages |
| D14 | A window that doesn't overlap the data said "nothing missing" | Raises a clear error |
| R3 | A VaR warning that used the wrong confidence passed all tests | Test added |
| R5 | VaR results had a garbage label: `np.float64(0.0500…44)` | Label removed |
| R6 | Two docstrings described behaviour slightly wrong | Corrected |
| R9 | The prices guard can't catch prices below $0.50 | Documented as a heuristic |
| R10 | Crypto (7 days a week) is annualized with a 252-day stock year | Documented: stocks/ETFs only |
| R13 | The guard test couldn't tell a median from a mean | Stronger test |
| A3 | A late-starting ticker silently had no line on the rolling chart | Caption explains why |
| A4 | One extreme ticker made two charts vanish with developer-only advice | Plain-English message (deeper fix open) |
| A5 | Typing 4.0 for a 4% rate gave Sharpe −27 with no hint | Input limited, caption shows the rate used |
| A6 | Long data gaps were shown as a soft info note | Shown as a warning |
| A8 | Unrelated pandas/numpy warnings were swallowed | Passed on to the terminal |
| A9 | "2708.1%" had no thousands separator | "2,708.1%" |

**Logged for later (not fixed yet):** D6, D7, D12, R7, A7, A10–A14, and the deeper A4 fix.
Details are in the [session journal](sessions/2026-10-20-overnight-build.md#5-unfinished-items-and-things-im-unsure-about).
