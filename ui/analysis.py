"""Run every risk.py call the page needs, once, and collect the warnings they raise.

No Streamlit and no math here: only calls into ``signaldesk/risk.py``, plus choosing which
columns and rows go where.

About the lock, and what it does NOT guarantee (a known limitation, not a fix):
    Python's ``warnings.catch_warnings`` changes settings of the WHOLE process, and Streamlit serves
    every visitor from its own thread in one process. ONE module-level ``threading.Lock`` is held
    around the block that captures warnings together with the risk.py calls that raise them. That
    guarantees this much: two analyses never run their capture blocks at the same time, so one
    analysis cannot take another's warnings or put the process-wide settings back in the wrong order.
    The calculations take milliseconds, so visitors hardly ever wait for each other.

    It is NOT a guarantee that a visitor's warnings are always complete. Other code enters
    ``catch_warnings`` WITHOUT our lock, and can overlap with our block: pandas itself (about 370
    times per page run, from internals such as ``pandas_dtype``) and Streamlit's own widgets (the
    holdings table editor). When that happens, a visitor's page can lose the risk warnings that
    were raised in that moment. The NUMBERS are never affected. The things the page also derives
    from the data itself (the short-sample flag in the table, the skipped-row note, the portfolio's
    start-date line) still show, because they do not depend on warnings. The real fix is for
    risk.py to return its diagnostics as data instead of as warnings; that is planned for Phase 2.
    Until then the logged race R7 is reduced, not fixed.
"""

from __future__ import annotations

import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import pandas as pd

from signaldesk import risk

CONFIDENCE = 0.95  # VaR confidence level (95%)
PORTFOLIO = risk.PORTFOLIO_NAME  # the name risk.portfolio_value gives its series: "Portfolio"

# Part of the sentence risk.py's GapWarning adds when a bridged gap is longer than any holiday.
# The tests check that risk.py still says it, so a reworded message cannot go unnoticed.
LONG_GAP_PHRASE = "longer than a holiday"
# risk.py starts every message with the name of the function that raised it. The portfolio's
# "valued at a last known price" gap note is a different thing from the "bridged returns" note.
PORTFOLIO_MESSAGE_PREFIX = "portfolio_value:"

# The kinds of warning that risk.py raises. Only these are shown on the page.
RISK_WARNINGS = (risk.SkippedTickerWarning, risk.ShortSampleWarning, risk.GapWarning)

# One lock for the whole process: it keeps our own capture blocks from overlapping (see the module
# docstring for what that does and does not guarantee). It is not re-entrant, so never take it twice
# in the same thread: ``analyze`` takes it exactly once.
_RISK_LOCK = threading.Lock()


@dataclass(frozen=True)
class Analysis:
    """The results of every risk.py call the page needs, computed once per run."""

    value: pd.Series  # the portfolio's buy-and-hold value, starting at 1.0
    benchmark: str
    holdings: list[str]
    frame: pd.DataFrame  # holdings, Portfolio and benchmark on the portfolio's dates
    kpi: pd.DataFrame  # risk_summary of Portfolio and benchmark (the key-number row)
    summary: pd.DataFrame  # risk_summary of every column of ``frame`` (NaN row = skipped)
    skipped: list[str]  # columns that risk_summary could not calculate
    usable: list[str]  # the other columns (these feed the charts)
    growth: pd.DataFrame | None  # growth of 1 dollar, usable columns only
    drawdown: pd.DataFrame | None  # loss from the previous peak, as positive fractions
    rolling_vol: pd.DataFrame | None  # None when the window is longer than the data
    n_returns: int  # number of daily-return rows (used in the "window too long" message)
    correlation: pd.DataFrame | None  # None when fewer than 2 usable holdings/benchmark
    problems: dict[str, str]  # chart name -> why risk.py refused to calculate it
    warnings: list[tuple[type, str]]  # (category, message), without repeats


# --------------------------------------------------------------------------- #
# Collecting warnings
# --------------------------------------------------------------------------- #

@contextmanager
def capture_risk_warnings(bucket: list):
    """Collect the warnings risk.py raises, so we can show them on the page. Holds the lock.

    The lock stops two of these blocks from overlapping. It does not stop OTHER code (pandas,
    Streamlit widgets) from entering ``catch_warnings`` at the same time, so the list can occasionally
    miss a warning; the module docstring explains why that is a known limitation.

    Python prints warnings to the server's terminal, which the person using the dashboard
    never sees. Recording them lets us put them on the page. Warnings of other kinds
    (for example from pandas or numpy) are not for the page, but recording would hide them
    from the terminal too, so those are passed on to ``warnings.showwarning`` afterwards.
    """
    caught: list = []
    with _RISK_LOCK:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")  # record every one, even repeats
                yield
        finally:
            # Runs after catch_warnings has finished, also when the calculation raised.
            for item in caught:
                if issubclass(item.category, RISK_WARNINGS):
                    bucket.append(item)
                else:
                    warnings.showwarning(
                        item.message, item.category, item.filename, item.lineno, item.file, item.line
                    )


def dedupe_warnings(caught: list) -> list[tuple[type, str]]:
    """Drop repeated warnings.

    risk_summary and daily_returns both report gaps, in nearly the same words. We keep the
    first of those (the summary's) and skip later ones. The portfolio's own gap note ("valued
    at a last known price") says something different, so it is kept once, separately. For the
    other kinds we remove exact repeats of (category, message).
    """
    result: list[tuple[type, str]] = []
    seen: dict[tuple[type, str], bool] = {}
    gap_families_shown: dict[str, bool] = {}
    for item in caught:
        category, message = item.category, str(item.message)
        if issubclass(category, risk.GapWarning):
            family = "portfolio" if message.startswith(PORTFOLIO_MESSAGE_PREFIX) else "returns"
            if family in gap_families_shown:
                continue
            gap_families_shown[family] = True
        if (category, message) in seen:
            continue
        seen[(category, message)] = True
        result.append((category, message))
    return result


def skipped_rows(summary: pd.DataFrame) -> pd.Series:
    """True for each ticker whose summary row is all NaN (risk_summary skipped it).

    We look at the NaN metrics and not at n_obs, because n_obs can be 0 or more than 0 for
    a skipped ticker (it is the real count of returns when the problem was too few of them
    or a flat price, and 0 when a price was bad).
    """
    return summary.drop(columns="n_obs").isna().all(axis=1)


# --------------------------------------------------------------------------- #
# The analysis
# --------------------------------------------------------------------------- #

def analyze(
    prices: pd.DataFrame,
    weights: dict[str, float],
    benchmark: str,
    window: int,
    risk_free_rate: float,
) -> Analysis:
    """Run every risk.py function the page needs, once, under the lock (see the module docstring).

    ``prices`` holds one column per holding plus the benchmark. ``weights`` are fractions that
    add up to 1 (the demo file's, or ``risk.normalize_weights`` output).

    The portfolio can start LATER than the first row of ``prices`` (a holding that began trading
    late). Everything is then measured over the portfolio's dates only: the benchmark and the
    holdings are cut to them first (a selection, not math), otherwise two numbers on the page
    would describe different periods. risk.py's "shorter history" warning is passed on to the page.

    Raises ValueError if the portfolio or the whole table cannot be calculated (the caller shows it).
    """
    holdings = list(weights)
    if benchmark in holdings:
        raise ValueError(f"The benchmark {benchmark} is also one of the holdings; pick a different benchmark.")

    caught: list = []
    problems: dict[str, str] = {}

    def attempt(chart: str, function, *args):
        """Call a risk.py function for a chart. If it refuses (ValueError), remember why.

        One chart that cannot be drawn must not take the metrics table down with it.
        """
        try:
            return function(*args)
        except ValueError as error:
            problems[chart] = str(error)
            return None

    with capture_risk_warnings(caught):
        value = risk.portfolio_value(prices[holdings], weights)
        # Same dates as the portfolio: a selection by date, not math.
        bench = prices[benchmark].loc[value.index]
        compared = pd.DataFrame({PORTFOLIO: value, benchmark: bench})

        # The key numbers. The table call below covers the same two columns and raises the same
        # warnings, so this call's warnings would only appear twice: they are muted here.
        with warnings.catch_warnings():
            for category in RISK_WARNINGS:
                warnings.simplefilter("ignore", category)
            kpi = risk.risk_summary(compared, confidence=CONFIDENCE, risk_free_rate=risk_free_rate)

        # The metrics table: every holding, the portfolio and the benchmark, all on the same dates.
        frame = pd.concat([prices.loc[value.index, holdings], compared], axis=1)
        summary = risk.risk_summary(frame, confidence=CONFIDENCE, risk_free_rate=risk_free_rate)

        # The charts only get columns that were calculated: the single-column functions
        # (growth, drawdown, correlation) refuse columns with too little data.
        is_skipped = skipped_rows(summary)
        skipped = list(summary.index[is_skipped.to_numpy()])
        usable = list(summary.index[~is_skipped.to_numpy()])
        usable_frame = frame[usable]

        growth = attempt("growth", risk.cumulative_returns, usable_frame)
        drawdown = attempt("drawdown", risk.drawdown_series, usable_frame)
        n_returns = len(risk.daily_returns(usable_frame))

        # A rolling window longer than the data raises ValueError, so check first.
        rolling_vol = None
        if window <= n_returns:
            rolling_vol = attempt("rolling", risk.rolling_volatility, usable_frame, window)

        # A correlation needs at least two columns; the portfolio itself is not one of them.
        correlation = None
        correlation_columns = [name for name in usable if name != PORTFOLIO]
        if len(correlation_columns) >= 2:
            correlation = attempt("correlation", risk.return_correlation, frame[correlation_columns])

    return Analysis(
        value=value,
        benchmark=benchmark,
        holdings=holdings,
        frame=frame,
        kpi=kpi,
        summary=summary,
        skipped=skipped,
        usable=usable,
        growth=growth,
        drawdown=drawdown,
        rolling_vol=rolling_vol,
        n_returns=n_returns,
        correlation=correlation,
        problems=problems,
        warnings=dedupe_warnings(caught),
    )
