"""SignalDesk math layer: returns, volatility, correlation, drawdown, VaR and Sharpe.

Every function here is a plain, pure function. It takes a pandas table, returns a number
or a table, and does nothing else: no downloading, no plotting, no printing. The same
input always gives the same output, and the input is never modified. The dashboard
(Phase 1) and the agents (Phase 2) both call these functions, so none of the math lives
anywhere else.

Shapes: every function accepts either a pandas Series (one ticker) or a DataFrame (one
column per ticker, like the table from ``data.load_prices``).
    - scalar metrics (volatility, VaR, Sharpe, ...): Series in -> float out,
      DataFrame in -> a Series out, indexed by ticker
    - time series (returns, cumulative growth, rolling volatility): Series in -> Series
      out, DataFrame in -> DataFrame out

Units: everything is a FRACTION, not a percent. 0.25 means 25%.

Conventions (each one is repeated in the docstring of the functions it affects)
    - Returns are SIMPLE daily returns: ``P_t / P_(t-1) - 1``.
    - A year has 252 trading days (``TRADING_DAYS_PER_YEAR``). Volatility, Sharpe and the
      annualized return all use that same number. Designed for exchange-traded
      stocks/ETFs (252 trading days a year). Assets that trade 7 days a week (e.g.
      crypto) will be annualized incorrectly: for example BTC in 2024 returned +111.5%,
      but the annualized return shown here would be 67.7%. There is no weekend detection.
    - Losses are POSITIVE numbers. ``historical_var`` and ``max_drawdown`` both return
      0.25 for "a 25% loss". Do not expect a minus sign.
    - Prices are expected to be ADJUSTED closes (already corrected for splits and
      dividends, which is what ``data.load_prices`` gives). We never adjust them again.

Missing values (gaps)
    The data layer never fills gaps: a missing price stays NaN. We keep that policy and
    measure every ticker against ITSELF, so a ticker's numbers never depend on which
    other tickers share the table:
    - ``daily_returns`` compares each price with that ticker's own previous VALID price.
      A row where the price is NaN gets a NaN return, and the next valid price is
      measured against the last valid one. [100, NaN, 110] gives returns [NaN, 0.10].
      A return that skips over one or more NaN rows is called a BRIDGED return. It
      covers more than one trading day but counts as one observation. This is normal
      for exchange holidays (a US holiday is a NaN row for AAPL when the table also holds
      Tokyo and Frankfurt stocks), and otherwise it means missing data. One
      ``GapWarning`` per call names the tickers, the number of bridged returns and the
      LONGEST bridge in rows (a holiday is 1 row, an outage can be 40). A bridge longer than
      ``MAX_HOLIDAY_GAP_ROWS`` (5) rows adds a stronger sentence: that is longer than any
      holiday, so it is probably missing data.
      Leading NaNs (a late start) and trailing NaNs are not bridges: there is no earlier
      or later price to span.
    - The statistics (volatility, VaR, Sharpe, annualized return) drop the NaN returns of
      each column separately and use that column's own count of valid returns.
    - Rolling volatility works on each ticker's own valid returns too, so a holiday row
      does not blank out the next ``window`` days.
    - ``cumulative_returns`` and ``max_drawdown`` look at price LEVELS and simply skip
      NaN prices. Everything therefore agrees: the annualized return of a ticker is
      exactly ``P_last_valid / P_first_valid`` annualized over that ticker's own number
      of returns, which is also the last valid value of ``cumulative_returns``.

Sample size
    Statistics from a few days of data are nearly meaningless, but the numbers still
    come out looking precise. So:
    - fewer than 2 valid data points in a column raises ``ValueError`` (a standard
      deviation needs at least 2 points);
    - fewer than ``MIN_RELIABLE_OBS`` (60, about 3 months of trading days) valid
      observations emits a ``ShortSampleWarning`` naming the column and the count. It is
      a warning, not an error, because the number is still returned.
    The Phase 2 risk agent reads these warnings (and ``GapWarning``) to decide whether a
    claim rests on a long enough, gap-free sample.

Prices versus returns
    The functions that take returns refuse input that looks like prices (a column whose
    median absolute value is above 0.5) with a ValueError: volatility of a price series
    is a number, but it means nothing. Call ``daily_returns`` first.
    This is only a HEURISTIC for typical stock prices (tens or hundreds of dollars). It
    cannot catch a price series that stays below about 0.5 (penny stocks, some crypto
    tokens), and the same price series would be mistaken for returns there.
    The guard can also refuse real returns: a squeeze like GME in January 2021 (+18%, +93%,
    +135%) has a median absolute return above 0.5. ``rolling_volatility`` and
    ``return_correlation`` take PRICES, compute the returns themselves (so they know they
    are returns) and skip the guard, like ``risk_summary`` does. Use them for charts.

Portfolios (``portfolio_value``)
    ``portfolio_value`` turns holdings and weights into ONE price-like series named
    "Portfolio" (it starts at 1.0), so every function above works on it unchanged. It is
    BUY-AND-HOLD: the weights apply on day one only and are never rebalanced. See its
    docstring for how the start date and missing prices are handled. If a late-starting
    holding makes the portfolio start after the first row of the table, a ``ShortSampleWarning``
    names it and the dropped dates: compare the portfolio with a benchmark on the same dates.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "MIN_RELIABLE_OBS",
    "MAX_HOLIDAY_GAP_ROWS",
    "MIN_RELIABLE_WINDOW",
    "MIN_TAIL_OBS",
    "PORTFOLIO_NAME",
    "TRADING_DAYS_PER_YEAR",
    "GapWarning",
    "ShortSampleWarning",
    "SkippedTickerWarning",
    "annualized_return",
    "annualized_volatility",
    "correlation_matrix",
    "cumulative_returns",
    "daily_returns",
    "drawdown_series",
    "historical_var",
    "max_drawdown",
    "portfolio_value",
    "return_correlation",
    "risk_summary",
    "rolling_volatility",
    "sharpe_ratio",
]

# --------------------------------------------------------------------------- #
# Constants and the warning class
# --------------------------------------------------------------------------- #

#: Number of trading days in a year, used to annualize (volatility, Sharpe, CAGR).
TRADING_DAYS_PER_YEAR = 252

#: Fewer valid observations than this and a statistic gets a ShortSampleWarning.
#: 60 trading days is roughly 3 months.
MIN_RELIABLE_OBS = 60

#: A rolling window shorter than this gets a ShortSampleWarning (too noisy).
MIN_RELIABLE_WINDOW = 20

#: Historical VaR needs at least this many observations in the loss tail.
MIN_TAIL_OBS = 5

#: A bridged return that skips more NaN rows than this is longer than any exchange holiday,
#: so ``GapWarning`` calls it probably missing data.
MAX_HOLIDAY_GAP_ROWS = 5

# A standard deviation below this counts as "zero". The standard deviation of a column
# of identical numbers is not always exactly 0.0 in floating point (it can be ~1e-17),
# and dividing by that would give a huge nonsense Sharpe ratio.
_ZERO_STD = 1e-12

_SQRT_TRADING_DAYS = math.sqrt(TRADING_DAYS_PER_YEAR)


class ShortSampleWarning(UserWarning):
    """Emitted when a number is based on too few observations to be trusted.

    It is only a warning: the number is still returned. Catch it (``pytest.warns`` or
    ``warnings.catch_warnings``) to find out which result is shaky. ``portfolio_value`` also uses
    it when a late-starting holding makes the portfolio cover a shorter period than the table.
    """


class GapWarning(UserWarning):
    """Emitted by ``daily_returns`` when some returns span missing prices ("bridged" returns).

    Normal for exchange holidays when tickers from different exchanges share a table;
    otherwise it means missing data (see ``data.find_gaps``). The message gives each
    ticker's number of bridged returns and its longest bridge in rows, and warns more
    strongly when a bridge is longer than ``MAX_HOLIDAY_GAP_ROWS``. The numbers are still
    returned.
    """


class SkippedTickerWarning(UserWarning):
    """Emitted by ``risk_summary`` when a ticker could not be calculated and got a NaN row.

    The message names the ticker and the reason (for example "only 1 valid return").
    """


# --------------------------------------------------------------------------- #
# Small private helpers
# --------------------------------------------------------------------------- #

def _prepare(
    data,
    what: str,
    *,
    ordered: bool = False,
    positive: bool = False,
    expect_returns: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Check the input and return ``(frame, is_series)``.

    ``frame`` is always a NEW float64 DataFrame (so the caller's data is never touched),
    even if a Series was given (it becomes a one-column frame). ``is_series`` tells the
    caller to turn the result back into a Series / float at the end.

    Raises ValueError for: not a pandas object, no rows or columns, duplicate column
    names, non-numeric columns, infinite values, and (when asked) an index that is not
    in increasing order, prices that are zero or negative, or a column that looks like
    prices when returns are expected. NaN is allowed: it means "gap".
    """
    if isinstance(data, pd.Series):
        is_series = True
        frame = data.to_frame(name="series" if data.name is None else data.name)
    elif isinstance(data, pd.DataFrame):
        is_series = False
        frame = data
    else:
        raise ValueError(f"{what} must be a pandas Series or DataFrame, got {type(data).__name__}.")

    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"{what} is empty (it has no rows or no columns), so there is nothing to calculate.")
    if frame.columns.has_duplicates:
        raise ValueError(f"{what} has duplicate column names; every ticker must appear once.")
    for label, dtype in frame.dtypes.items():
        # Floats and integers only: this also rejects bool, text, dates and complex numbers.
        if not (pd.api.types.is_float_dtype(dtype) or pd.api.types.is_integer_dtype(dtype)):
            raise ValueError(f"{what}: column {label!r} is not numeric (its dtype is {dtype}).")

    frame = frame.astype("float64")  # always a copy
    values = frame.to_numpy()
    infinite = np.isinf(values).any(axis=0)
    if infinite.any():
        bad = ", ".join(repr(c) for c in frame.columns[infinite])
        raise ValueError(f"{what}: infinite values found in {bad}. Check the data for a zero price.")
    if expect_returns:
        # Real daily returns have a median absolute value around 0.01 (and rarely exceed
        # 0.2). A median above 0.5 means the column is almost certainly a price series,
        # whose "volatility" or "VaR" would be a meaningless number.
        # This is only a HEURISTIC for typical stock prices (tens or hundreds of dollars):
        # a price series that stays below about 0.5 (penny stocks, some crypto tokens)
        # passes unnoticed. The MEDIAN is used so that one huge day (a +400% squeeze) in
        # an otherwise normal series does not trigger it.
        typical = frame.abs().median()  # per column, ignoring NaN
        prices_like = typical > 0.5
        if prices_like.any():
            bad = ", ".join(repr(c) for c in typical.index[prices_like])
            raise ValueError(
                f"{what}: {bad} looks like prices, not returns; call daily_returns first "
                f"(its median absolute value is above 0.5, but daily returns are usually "
                "around 0.01)."
            )
    if positive and (values <= 0).any():
        bad = ", ".join(repr(c) for c in frame.columns[(values <= 0).any(axis=0)])
        raise ValueError(f"{what}: prices must be greater than 0, but {bad} has zero or negative prices.")
    if ordered and not (frame.index.is_monotonic_increasing and frame.index.is_unique):
        raise ValueError(
            f"{what}: the index must be in increasing order with no repeated dates "
            "(oldest first), because the calculation depends on the order."
        )
    return frame, is_series


def _require_obs(frame: pd.DataFrame, caller: str, unit: str) -> pd.Series:
    """Raise ValueError if any column has fewer than 2 valid (non-NaN) values.

    Returns the count of valid values per column (used for the short-sample warnings).
    ``unit`` is "prices" or "returns", only used in the message.
    """
    counts = frame.count()
    too_few = counts[counts < 2]
    if not too_few.empty:
        detail = ", ".join(f"{label!r} has {n}" for label, n in too_few.items())
        raise ValueError(f"{caller}: need at least 2 valid {unit} per column, but {detail}.")
    return counts


def _bridge_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Per column: how many returns are bridged and how long the longest bridge is.

    A bridged return is one on a valid row whose previous row(s) are NaN while the column
    has an earlier valid price to measure against. Its length is the number of NaN rows it
    skips. Leading NaNs (no earlier price) and trailing NaNs (no return at all) are not
    bridges. Returns a table indexed like the columns, with the columns ``bridged`` (count)
    and ``longest`` (rows; 0 when there are no bridges).
    """
    bridged, longest = [], []
    for i in range(frame.shape[1]):
        valid_rows = np.flatnonzero(frame.iloc[:, i].notna().to_numpy())
        skipped = np.diff(valid_rows) - 1  # NaN rows between each pair of consecutive valid prices
        skipped = skipped[skipped > 0]
        bridged.append(len(skipped))
        longest.append(int(skipped.max()) if len(skipped) else 0)
    return pd.DataFrame({"bridged": bridged, "longest": longest}, index=frame.columns, dtype="int64")


def _rows(n: int) -> str:
    return f"{n} row" if n == 1 else f"{n} rows"


def _warn_gaps(caller: str, stats: pd.DataFrame) -> None:
    """Emit ONE GapWarning listing every column that has bridged returns. Same ``stacklevel`` rule.

    Each ticker is shown as ``'AAPL': 19 bridged (longest 1 row)``. If any longest bridge
    is above ``MAX_HOLIDAY_GAP_ROWS``, a stronger sentence names those tickers and says it is
    longer than a holiday.
    """
    with_bridges = stats[stats["bridged"] > 0]
    if with_bridges.empty:
        return
    detail = ", ".join(
        f"{label!r}: {row.bridged} bridged (longest {_rows(row.longest)})" for label, row in with_bridges.iterrows()
    )
    message = (
        f"{caller}: some returns span missing prices ({detail}). Each one is measured against "
        "that ticker's own previous valid price, so it covers more than one trading day but "
        "counts as one observation. Short bridges (a few rows) are normal for exchange "
        "holidays when tickers from different exchanges share a table; otherwise it means "
        "missing data (see data.find_gaps)."
    )
    long_ones = with_bridges[with_bridges["longest"] > MAX_HOLIDAY_GAP_ROWS]
    if not long_ones.empty:
        long_detail = ", ".join(f"{label!r} ({_rows(row.longest)})" for label, row in long_ones.iterrows())
        message += (
            f" A gap of more than {MAX_HOLIDAY_GAP_ROWS} rows is longer than a holiday: {long_detail}. "
            "This is likely missing data, and the returns across it are multi-week moves "
            "counted as one day."
        )
    warnings.warn(message, GapWarning, stacklevel=3)


def _warn_short_sample(caller: str, counts: pd.Series, unit: str = "observations") -> None:
    """Emit ONE ShortSampleWarning listing every column with fewer than MIN_RELIABLE_OBS.

    Call it directly from a public function: ``stacklevel=3`` makes the warning point at
    the line that called that public function.
    """
    short = counts[counts < MIN_RELIABLE_OBS]
    if short.empty:
        return
    detail = ", ".join(f"{label!r}: {n}" for label, n in short.items())
    warnings.warn(
        f"{caller}: fewer than {MIN_RELIABLE_OBS} valid {unit} (about 3 months of trading "
        f"days) for {detail}. The result is based on a short sample and may not be reliable.",
        ShortSampleWarning,
        stacklevel=3,
    )


def _warn_thin_tail(caller: str, counts: pd.Series, confidence: float) -> None:
    """Emit ONE ShortSampleWarning for columns with fewer than MIN_TAIL_OBS tail observations.

    The tail is the worst ``(1 - confidence)`` share of days. Same ``stacklevel`` rule as
    ``_warn_short_sample``.
    """
    tails = counts * (1.0 - confidence)
    # The small tolerance stops float noise from warning at the exact boundary
    # (for example 50 * (1 - 0.9) is 4.999999999999999, but it means exactly 5).
    thin = tails[tails < MIN_TAIL_OBS - 1e-9]
    if thin.empty:
        return
    detail = ", ".join(f"{label!r}: n={counts[label]}, about {tails[label]:.1f} in the tail" for label in thin.index)
    warnings.warn(
        f"{caller}: too few observations in the loss tail at confidence={confidence:g} "
        f"({detail}). Historical VaR needs at least {MIN_TAIL_OBS} observations in the tail "
        f"(n * (1 - confidence) >= {MIN_TAIL_OBS}), so this estimate rests on very few days.",
        ShortSampleWarning,
        stacklevel=3,
    )


def _real_number(value, name: str) -> float:
    """Return ``value`` as a float, or raise ValueError if it is not a finite real number.

    ``True``/``False`` are rejected on purpose: they are ints in Python, but passing one
    here is almost certainly a mistake.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    return float(value)


def _check_confidence(confidence) -> float:
    """Validate a VaR confidence level: a number strictly between 0 and 1."""
    level = _real_number(confidence, "confidence")
    if not 0.0 < level < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1 (for example 0.95), got {confidence!r}.")
    return level


def _finish_scalar(result: pd.Series, is_series: bool):
    """Per-column results -> a float for a Series input, or the Series (by ticker) for a DataFrame.

    The returned Series has no name: pandas would otherwise leave an accidental one behind
    (``DataFrame.quantile`` names its result after the quantile, for example).
    """
    if is_series:
        return float(result.iloc[0])
    result.name = None
    return result


def _finish_series(result: pd.DataFrame, original, is_series: bool):
    """Time-series results -> a Series (same name as the input) for a Series input, else the DataFrame."""
    return result.iloc[:, 0].rename(original.name) if is_series else result


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #

def daily_returns(prices):
    """Simple daily returns: how much the price changed since the ticker's previous price.

    Formula: ``r_t = P_t / P_prev - 1``, where ``P_prev`` is that ticker's own previous
    VALID (non-NaN) price. Units: a fraction (0.10 = +10%).
    Example: prices [100, 110, 99] give returns [0.10, -0.10].

    The first row has no previous price, so it is dropped: the result has one row fewer
    than the input and keeps the dates of the remaining rows.

    Gaps: each column is measured against itself, never against other tickers.
    - A row where the price is NaN gets a NaN return.
    - The next valid price is measured against the last valid one: [100, NaN, 110] gives
      [NaN, 0.10]. That return skips a missing row, so it is called BRIDGED: it covers
      more than one trading day but counts as one observation.
    - The first valid price of a late starter has no return (nothing before it), and
      trailing NaNs just give NaN. Neither is a bridge.
    Nothing is ever filled in or invented. If any column has bridged returns, ONE
    ``GapWarning`` is emitted naming each ticker and how many returns were bridged. That is
    normal for exchange holidays when tickers from different exchanges share a table, and
    otherwise means missing data (see ``data.find_gaps``). The numbers are still returned.

    Why this policy: a ticker's returns, and every statistic built on them, then do not
    depend on which other tickers happen to be in the table, and they agree with
    ``cumulative_returns`` (the compounded returns equal the price growth).

    Input: adjusted close prices, oldest first, all greater than 0. Needs at least 2
    valid prices per column, otherwise ValueError (naming the column). It does not warn
    about sample size: it only reshapes the data, it does not estimate anything.
    """
    frame, is_series = _prepare(prices, "prices", ordered=True, positive=True)
    _require_obs(frame, "daily_returns", "prices")
    # frame.ffill().shift(1) is, on every row, the last valid price BEFORE that row (per
    # column). The ffill is used only to look that price up; it is never put in the result.
    # On a row whose own price is NaN the division gives NaN, so no value is invented.
    previous_valid = frame.ffill().shift(1)
    returns = (frame / previous_valid - 1.0).iloc[1:]
    _warn_gaps("daily_returns", _bridge_stats(frame))
    return _finish_series(returns, prices, is_series)


def cumulative_returns(prices):
    """Growth of 1 unit invested at the start: how many times the first price the price is now.

    Formula: ``G_t = P_t / P_first``, where ``P_first`` is the first valid (non-NaN)
    price of that column. So every column starts at 1.0 (1.10 means +10% since the
    start, 0.80 means -20%). Units: a multiple, not a percent.

    The result has the same dates as the input. It is NaN where the price is NaN and
    before a column's first valid price (a ticker that starts later starts at 1.0 on its
    own first day).

    Gaps: NaN prices are skipped (this works on price levels). That is consistent with
    ``daily_returns``: the compounded returns of a ticker equal its last valid value here,
    ``P_last_valid / P_first_valid``.

    Needs at least 2 valid prices per column, otherwise ValueError. No sample-size warning
    (it is a path, not an estimate).
    """
    frame, is_series = _prepare(prices, "prices", ordered=True, positive=True)
    _require_obs(frame, "cumulative_returns", "prices")
    first_valid = pd.Series(
        [frame.iloc[:, i].dropna().iloc[0] for i in range(frame.shape[1])],
        index=frame.columns,
        dtype="float64",
    )
    return _finish_series(frame / first_valid, prices, is_series)


def annualized_return(returns, *, _skip_prices_check=False):
    """Annualized return, the geometric way (also called CAGR): the steady yearly growth rate.

    Formula: ``(prod(1 + r)) ** (252 / n) - 1`` where ``n`` is the number of valid
    returns of that column. In words: compound all the daily returns, then scale that
    growth to a 252-day year. Units: a fraction per year (0.12 = +12% a year).

    This is GEOMETRIC. It is not the arithmetic mean times 252 (which ``sharpe_ratio``
    uses). The two differ whenever returns vary: returns [+10%, -10%] have an arithmetic
    mean of 0, but the money actually shrank (1.1 * 0.9 = 0.99).

    Gaps: NaN returns are dropped per column, and ``n`` counts only the valid ones. Fed
    with ``daily_returns``, the result is exactly ``(P_last_valid / P_first_valid)`` raised
    to ``252 / n``, minus 1, for that ticker's own prices.
    Needs at least 2 valid returns per column and no return below -1 (-100%), otherwise
    ValueError. Prices passed by mistake (median absolute value above 0.5, a heuristic for
    typical stock prices) are refused too. Warns (ShortSampleWarning) below
    ``MIN_RELIABLE_OBS`` valid returns: stretching a few days to a full year makes the
    number wildly sensitive.

    (``_skip_prices_check`` is private, used only by ``risk_summary``, which knows its
    input really is returns. Do not pass it.)
    """
    frame, is_series = _prepare(returns, "returns", expect_returns=not _skip_prices_check)
    counts = _require_obs(frame, "annualized_return", "returns")
    below = (frame < -1.0).any()
    if below.any():
        bad = ", ".join(repr(c) for c in below[below].index)
        raise ValueError(f"annualized_return: a return below -1 (-100%) is impossible, found in {bad}.")
    _warn_short_sample("annualized_return", counts, "returns")
    # prod(1 + r) written as exp(sum(log(1 + r))) so a long series cannot overflow.
    # A return of exactly -1 gives log(0) = -inf and exp(-inf) = 0, so the result is -1
    # (a total loss), which is correct; the divide-by-zero notice is expected there.
    with np.errstate(divide="ignore"):
        log_growth = np.log1p(frame).sum()  # sum skips NaN
    result = np.expm1(TRADING_DAYS_PER_YEAR / counts * log_growth)
    return _finish_scalar(result, is_series)


def annualized_volatility(returns, window=None, *, _skip_prices_check=False):
    """Annualized volatility: how much returns jump around, scaled to one year.

    Formula: ``std(r, ddof=1) * sqrt(252)``. The standard deviation uses the SAMPLE
    formula (divide by n - 1, not n). Multiplying by sqrt(252) turns a daily number into
    a yearly one (volatility grows with the square root of time). Units: a fraction
    per year (0.28 = 28% a year).

    Without ``window``: one number per column from all its valid returns.

    With ``window`` (an int, at least 2): a ROLLING series, computed on each ticker's own
    valid returns. The value on date t is the std of that ticker's last ``window`` valid
    returns up to and including t (the window trails behind, it never looks ahead), so a
    ticker's first ``window - 1`` valid returns give NaN. Dates where the return itself is
    NaN (a holiday) give NaN, but they do not blank out the following days: the window
    just skips over them. The result has the same shape and dates as ``returns``.
    ``window`` must not be longer than the number of rows. A ``window`` below
    ``MIN_RELIABLE_WINDOW`` (20) is allowed but warns, because a std from so few
    points is very noisy. The rolling form does not warn per point.

    Needs at least 2 valid returns per column, otherwise ValueError. Prices passed by
    mistake (median absolute value above 0.5, a heuristic for typical stock prices) are
    refused too. Without ``window`` it warns (ShortSampleWarning) below
    ``MIN_RELIABLE_OBS`` valid returns.

    (``_skip_prices_check`` is private, used only by ``risk_summary``. Do not pass it.)
    """
    if window is not None:
        if isinstance(window, bool) or not isinstance(window, numbers.Integral) or window < 2:
            raise ValueError(f"window must be a whole number of at least 2, got {window!r}.")
        window = int(window)

    frame, is_series = _prepare(returns, "returns", ordered=window is not None, expect_returns=not _skip_prices_check)
    counts = _require_obs(frame, "annualized_volatility", "returns")

    if window is None:
        _warn_short_sample("annualized_volatility", counts, "returns")
        return _finish_scalar(frame.std(ddof=1) * _SQRT_TRADING_DAYS, is_series)

    if window > len(frame):
        raise ValueError(
            f"annualized_volatility: window={window} is longer than the {len(frame)} rows "
            "given, so not a single full window exists."
        )
    if window < MIN_RELIABLE_WINDOW:
        warnings.warn(
            f"annualized_volatility: window={window} is shorter than {MIN_RELIABLE_WINDOW} "
            "observations, so every rolling value is a very noisy estimate.",
            ShortSampleWarning,
            stacklevel=2,
        )
    # Per ticker, on its own valid returns only (dropna), so one ticker's holidays never
    # affect another ticker, and a NaN does not blank out the next `window` days.
    # Trailing window (the pandas default, center=False): the value at a return sees only
    # that return and the ones before it. min_periods=window: a FULL window is needed.
    # reindex puts the values back on the original dates (NaN on the dates with no return).
    parts = []
    for i in range(frame.shape[1]):
        valid = frame.iloc[:, i].dropna()
        parts.append(valid.rolling(window, min_periods=window).std(ddof=1).reindex(frame.index))
    rolling = pd.concat(parts, axis=1)
    rolling.columns = frame.columns
    return _finish_series(rolling * _SQRT_TRADING_DAYS, returns, is_series)


def correlation_matrix(returns, *, _skip_prices_check=False):
    """Pearson correlation between the tickers' returns: do they move together?

    Formula: ``corr(a, b) = cov(a, b) / (std(a) * std(b))`` for every pair of columns.
    Units: a number from -1 (opposite moves) through 0 (unrelated) to +1 (same moves).
    The result is a square DataFrame (tickers x tickers) and symmetric. The diagonal is
    1.0, except for a zero-variance column (see below), whose diagonal is NaN too.

    Needs a DataFrame with at least 2 columns (a Series or a single column raises
    ValueError, a correlation needs two things) and at least 2 valid returns per column.

    Gaps: NaN is handled PAIRWISE (pandas default). Each cell uses only the dates on
    which BOTH of its tickers have a return, so different cells can rest on different
    numbers of days. A pair with fewer than 2 shared dates gives NaN. A column whose
    returns are all identical (zero variance) has an undefined correlation: NaN, in its
    whole row and column including the diagonal.
    Warns (ShortSampleWarning, one warning listing the pairs) if any pair shares fewer
    than ``MIN_RELIABLE_OBS`` dates.

    (``_skip_prices_check`` is private, used only by ``return_correlation``, which knows its
    input really is returns. Do not pass it.)
    """
    frame, is_series = _prepare(returns, "returns", expect_returns=not _skip_prices_check)
    if is_series or frame.shape[1] < 2:
        raise ValueError(
            "correlation_matrix needs a DataFrame with at least 2 columns (tickers), "
            f"got {'a Series' if is_series else 'only one column'}."
        )
    _require_obs(frame, "correlation_matrix", "returns")

    valid = frame.notna().to_numpy().astype(int)
    shared = valid.T @ valid  # shared[i, j] = number of dates where both i and j have a return
    short_pairs = [
        (frame.columns[i], frame.columns[j], int(shared[i, j]))
        for i in range(frame.shape[1])
        for j in range(i + 1, frame.shape[1])
        if shared[i, j] < MIN_RELIABLE_OBS
    ]
    if short_pairs:
        detail = ", ".join(f"{a!r}/{b!r}: {n}" for a, b, n in short_pairs)
        warnings.warn(
            f"correlation_matrix: fewer than {MIN_RELIABLE_OBS} shared valid returns "
            f"(about 3 months of trading days) for {detail}. These correlations are based "
            "on a short sample and may not be reliable.",
            ShortSampleWarning,
            stacklevel=2,
        )
    return frame.corr(method="pearson", min_periods=2)


def max_drawdown(prices):
    """Maximum drawdown: the biggest fall from a previous high to a later low.

    Formula: ``max over t of (1 - P_t / running_max(P)_t)``, where ``running_max`` is the
    highest price seen UP TO that day (not over the whole series). Equivalently it is
    ``-min(P_t / running_max_t - 1)``.

    SIGN: returned as a POSITIVE fraction. 0.25 means "at some point the price was 25%
    below its earlier peak". This matches ``historical_var``, which also returns losses
    as positive numbers. It returns 0.0 for a series that never falls below an earlier
    price.

    Units: a fraction between 0 and 1.

    Gaps: NaN prices are skipped (this works on price levels, like ``cumulative_returns``,
    so it agrees with ``daily_returns``). Needs at least 2 valid prices per column, otherwise ValueError.
    Warns (ShortSampleWarning) below ``MIN_RELIABLE_OBS`` valid prices: a short window
    cannot have seen a real drawdown.
    """
    frame, is_series = _prepare(prices, "prices", ordered=True, positive=True)
    counts = _require_obs(frame, "max_drawdown", "prices")
    _warn_short_sample("max_drawdown", counts, "prices")

    def worst_fall(column: pd.Series) -> float:
        valid = column.dropna()
        return float((1.0 - valid / valid.cummax()).max())

    result = pd.Series(
        [worst_fall(frame.iloc[:, i]) for i in range(frame.shape[1])],
        index=frame.columns,
        dtype="float64",
    )
    return _finish_scalar(result, is_series)


def historical_var(returns, confidence=0.95, *, _skip_prices_check=False):
    """One-day historical Value at Risk: a loss that was not exceeded on ``confidence`` of the days.

    Formula: ``VaR = -quantile(returns, 1 - confidence)``. Sort the past daily returns,
    find the worst ``(1 - confidence)`` share (the lower tail), and take its edge, with
    the sign flipped. Example: at 95% confidence, "VaR = 0.03" means that on 95% of
    past days the loss was 3% or less (on the worst 5% of days it was bigger).

    SIGN: returned as a POSITIVE loss (0.03 = a 3% loss), the same convention as
    ``max_drawdown``. Units: a fraction of the position value over ONE day. It is not
    scaled to longer horizons.

    Quantile method: LINEAR interpolation (the numpy / pandas default). With n sorted
    returns, the quantile sits at position ``(1 - confidence) * (n - 1)`` (counting from
    0), and if that is between two observations we blend them in proportion. Example:
    6 returns, confidence 0.95 puts the quantile 0.25 of the way from the worst to the
    second-worst return. Other methods ("lower", "nearest", ...) give slightly
    different numbers on small samples; the tests pin this one.

    All-gains case: if even the tail days were gains (every return positive), the
    quantile is positive and the VaR comes out NEGATIVE (for example -0.0125). We return
    that as it is rather than clipping it to 0: the function stays exactly
    ``-quantile``, and a negative VaR reads "there was no loss at this confidence level".
    The sign is never flipped silently.

    ``confidence`` must be strictly between 0 and 1, otherwise ValueError. Needs at least
    2 valid returns per column, otherwise ValueError. Prices passed by mistake (median
    absolute value above 0.5, a heuristic for typical stock prices) are refused too. NaN
    returns are dropped per column. (``_skip_prices_check`` is private, used only by
    ``risk_summary``. Do not pass it.)

    Warnings (ShortSampleWarning, in addition to each other):
    - fewer than ``MIN_RELIABLE_OBS`` (60) valid returns, like the other statistics;
    - fewer than ``MIN_TAIL_OBS`` (5) observations in the tail, i.e.
      ``n * (1 - confidence) < 5``. A quantile built on 1-4 days is an anecdote. At 95%
      confidence that means fewer than 100 returns; at 99%, fewer than 500.
    """
    level = _check_confidence(confidence)
    frame, is_series = _prepare(returns, "returns", expect_returns=not _skip_prices_check)
    counts = _require_obs(frame, "historical_var", "returns")
    _warn_short_sample("historical_var", counts, "returns")
    _warn_thin_tail("historical_var", counts, level)
    # DataFrame.quantile skips NaN per column, so each ticker uses its own valid returns.
    quantile = frame.quantile(1.0 - level, interpolation="linear")
    return _finish_scalar(-quantile, is_series)


def sharpe_ratio(returns, risk_free_rate=0.0, *, _skip_prices_check=False):
    """Annualized Sharpe ratio: return earned per unit of risk, above the risk-free rate.

    Formula: ``mean(r - rf_daily) / std(r, ddof=1) * sqrt(252)``, with
    ``rf_daily = risk_free_rate / 252``. In words: the average daily return above the
    risk-free rate, divided by the daily standard deviation (sample formula), scaled
    to a year by sqrt(252). Units: a plain number, no percent (1.0 is decent, 2.0 is
    very good).

    ``risk_free_rate`` is an ANNUAL fraction (0.04 = 4% a year); the default 0.0 gives
    the plain return-per-risk ratio. Negative rates are allowed. This uses the
    ARITHMETIC mean of daily returns, unlike ``annualized_return`` (geometric).

    Zero volatility: if a column's returns are all identical (standard deviation 0), the
    ratio is undefined, and we raise ValueError naming the column rather than return a
    silent NaN or infinity. A perfectly flat price for 60+ days usually means stale or
    broken data, so a loud error is the useful answer.

    Gaps: NaN returns are dropped per column. Needs at least 2 valid returns per column,
    otherwise ValueError. Prices passed by mistake (median absolute value above 0.5, a
    heuristic for typical stock prices) are refused too. (``_skip_prices_check`` is
    private, used only by ``risk_summary``. Do not pass it.) Warns (ShortSampleWarning) below ``MIN_RELIABLE_OBS`` valid
    returns.
    """
    annual_rf = _real_number(risk_free_rate, "risk_free_rate")
    frame, is_series = _prepare(returns, "returns", expect_returns=not _skip_prices_check)
    counts = _require_obs(frame, "sharpe_ratio", "returns")
    spread = frame.std(ddof=1)
    flat = spread < _ZERO_STD
    if flat.any():
        bad = ", ".join(repr(c) for c in flat[flat].index)
        raise ValueError(
            f"sharpe_ratio: the returns of {bad} are all identical (standard deviation 0), "
            "so the Sharpe ratio is undefined. Check the data for stale prices."
        )
    _warn_short_sample("sharpe_ratio", counts, "returns")
    daily_rf = annual_rf / TRADING_DAYS_PER_YEAR
    excess_mean = (frame - daily_rf).mean()
    return _finish_scalar(excess_mean / spread * _SQRT_TRADING_DAYS, is_series)


def risk_summary(prices, confidence=0.95, risk_free_rate=0.0):
    """One table of the headline risk numbers, one row per ticker, ready for the dashboard.

    Input: adjusted close prices (a DataFrame, or a Series for one ticker), oldest first.
    Designed for exchange-traded stocks/ETFs (252 trading days a year). Assets that trade
    7 days a week (e.g. crypto) will be annualized incorrectly (BTC in 2024 returned
    +111.5%, but this shows 67.7%).

    The table is built ONLY by calling the functions in this module, one ticker at a time,
    so every number can be traced back to one of them and a ticker's row never depends on
    the other tickers in the table. Columns:
        annualized_return      ``annualized_return`` of the daily returns (geometric), a fraction
        annualized_volatility  ``annualized_volatility`` of the daily returns, a fraction
        sharpe_ratio           ``sharpe_ratio`` with the given ``risk_free_rate``; a plain
                               number, not a fraction
        max_drawdown           ``max_drawdown`` of the prices, a positive fraction
        var_95                 one-day ``historical_var`` at ``confidence``, a positive
                               fraction; the column is named after the confidence (0.99
                               gives ``var_99``)
        n_obs                  number of valid daily returns calculated for that ticker, a
                               count (0 if none were calculated)

    A ticker that cannot be calculated does not spoil the table. If it has fewer than 2
    valid returns (for example a ticker that is NaN except for its last 2 days), its
    returns are all identical (zero volatility, so no Sharpe ratio), or it has a zero or
    negative price (some providers return 0.0 for a broken day), its row is NaN and one
    ``SkippedTickerWarning`` names the ticker and the reason. For the first two reasons
    ``n_obs`` still shows the real count; for a bad price it is 0 (nothing was calculated).
    The other tickers are calculated as usual. If NO ticker can be calculated, ValueError
    is raised, listing the reasons. The individual functions keep raising for such input.

    The "looks like prices" check of the return functions is not applied here: the returns
    are computed from the prices inside, so they are known to be returns. A real but wild
    ticker (GME in January 2021: +18%, +93%, +135%) is therefore calculated, not skipped.

    Other warnings: instead of up to four warnings per ticker (one from each function),
    it emits at most one of each kind, for the tickers that were calculated: a
    ``GapWarning`` (tickers with bridged returns and their longest bridge, see
    ``daily_returns``), a ``ShortSampleWarning`` for fewer than ``MIN_RELIABLE_OBS`` valid
    returns, and a ``ShortSampleWarning`` for too few observations in the VaR tail. They
    name the ticker and the count.

    Bad ``confidence`` / ``risk_free_rate``, or prices that are not valid input (not
    numeric, infinite, not in date order, ...), raise ValueError as in the functions it calls.
    """
    level = _check_confidence(confidence)
    _real_number(risk_free_rate, "risk_free_rate")  # fail early, not once per ticker
    # positive=False on purpose: a zero/negative price spoils only its own ticker (below).
    frame, _ = _prepare(prices, "prices", ordered=True)  # a Series becomes 1 column

    skipped: dict = {}
    valid_prices = frame.count()
    for label in valid_prices.index[valid_prices < 2]:
        skipped[label] = f"only {valid_prices[label]} valid price(s), so no return can be calculated"
    for label in frame.columns[(frame <= 0).any()]:
        first_bad = frame.index[(frame[label] <= 0).to_numpy()][0]
        when = first_bad.strftime("%Y-%m-%d") if hasattr(first_bad, "strftime") else str(first_bad)
        skipped[label] = f"has a zero or negative price on {when}"  # (overrides "too few prices")
    usable = frame.loc[:, [label not in skipped for label in frame.columns]]

    rows: dict = {}
    n_obs = pd.Series(0, index=frame.columns, dtype="int64")
    with warnings.catch_warnings():
        # The consolidated warnings below replace the per-function ones.
        warnings.simplefilter("ignore", ShortSampleWarning)
        warnings.simplefilter("ignore", GapWarning)
        if usable.shape[1]:
            returns = daily_returns(usable)
            n_obs.loc[usable.columns] = returns.count().to_numpy()
            for label in usable.columns:
                try:
                    rows[label] = [
                        annualized_return(returns[label], _skip_prices_check=True),
                        annualized_volatility(returns[label], _skip_prices_check=True),
                        sharpe_ratio(returns[label], risk_free_rate, _skip_prices_check=True),
                        max_drawdown(usable[label]),
                        historical_var(returns[label], level, _skip_prices_check=True),
                    ]
                except ValueError as error:
                    # Only data problems can land here, because the inputs were validated
                    # above: too few valid returns, or a flat price (Sharpe undefined).
                    # The "looks like prices" check is skipped, see the docstring.
                    skipped[label] = str(error)

    if not rows:
        reasons = "; ".join(f"{label!r}: {reason}" for label, reason in skipped.items())
        raise ValueError(f"risk_summary: no ticker could be calculated. {reasons}")

    names = ["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", f"var_{level * 100:g}"]
    summary = pd.DataFrame.from_dict(rows, orient="index", columns=names).reindex(frame.columns)
    summary["n_obs"] = n_obs
    summary.index.name = "Ticker"

    if skipped:
        detail = "; ".join(f"{label!r}: {skipped[label]}" for label in frame.columns if label in skipped)
        warnings.warn(
            f"risk_summary: could not calculate {len(skipped)} ticker(s); their rows are NaN. {detail}",
            SkippedTickerWarning,
            stacklevel=2,
        )
    calculated = list(rows)
    _warn_gaps("risk_summary", _bridge_stats(usable[calculated]))
    _warn_short_sample("risk_summary", n_obs.loc[calculated], "returns")
    _warn_thin_tail("risk_summary", n_obs.loc[calculated], level)
    return summary


# --------------------------------------------------------------------------- #
# Portfolio and chart functions (Phase 1.5). Added next to the ones above; nothing above changed.
# --------------------------------------------------------------------------- #

#: The weights of a portfolio must add up to 1 within this much (float noise, not a rounding excuse).
_WEIGHT_SUM_TOLERANCE = 1e-9

#: The name of the series ``portfolio_value`` returns.
PORTFOLIO_NAME = "Portfolio"


def _date_text(label) -> str:
    """A date label as ``2026-01-31`` (or plain text for an index that is not dates)."""
    return label.strftime("%Y-%m-%d") if hasattr(label, "strftime") else str(label)


def _clean_weights(weights) -> dict:
    """Check a ``{ticker: weight}`` mapping and return it as a NEW dict of floats.

    Raises ValueError (naming the problem) for: not a dict / Series, empty, a ticker listed
    twice, a weight that is not a finite real number, a negative weight, or weights that
    do not add up to 1 within ``_WEIGHT_SUM_TOLERANCE``. A weight of exactly 0 is fine.
    """
    if isinstance(weights, pd.Series):
        if weights.index.has_duplicates:
            raise ValueError("portfolio_value: weights lists the same ticker more than once.")
        items = list(weights.items())
    elif isinstance(weights, Mapping):
        items = list(weights.items())
    else:
        raise ValueError(
            "portfolio_value: weights must be a dict or a pandas Series mapping ticker -> weight, "
            f"got {type(weights).__name__}."
        )
    if not items:
        raise ValueError("portfolio_value: weights is empty; a portfolio needs at least one holding.")

    cleaned = {ticker: _real_number(weight, f"portfolio_value: the weight of {ticker!r}") for ticker, weight in items}
    negative = [ticker for ticker, weight in cleaned.items() if weight < 0.0]
    if negative:
        detail = ", ".join(f"{ticker!r}: {cleaned[ticker]:g}" for ticker in negative)
        raise ValueError(
            f"portfolio_value: negative weights are not supported ({detail}). Only long positions "
            "for now, so every weight must be 0 or more."
        )
    total = math.fsum(cleaned.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"portfolio_value: the weights must add up to 1 (100%), but they add up to {total:.12g}."
        )
    return cleaned


def _run_lengths(flags: np.ndarray) -> list[int]:
    """Lengths of the runs of consecutive True values, in order. ``[F, T, T, F, T]`` gives ``[2, 1]``."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))
    return [int(n) for n in edges[1::2] - edges[0::2]]


def _warn_shortened_history(caller: str, frame: pd.DataFrame, start: int) -> None:
    """Emit ONE ShortSampleWarning when the portfolio starts later than the first row of the table.

    ``frame`` is the held prices BEFORE trimming and ``start`` is the position of the first
    row where every holding has a price. Nothing is emitted when ``start`` is 0. The tickers
    named are the ones that moved the start: those with no price on the last dropped row
    (take them away and the portfolio could start at least one row earlier). Each is shown with
    how many of the dropped rows it is missing on. Same ``stacklevel`` rule as ``_warn_gaps``.
    """
    if start == 0:
        return
    dropped = frame.iloc[:start]
    blockers = frame.columns[frame.iloc[start - 1].isna().to_numpy()]
    detail = ", ".join(f"{label!r} (no price on {int(dropped[label].isna().sum())} of the dropped rows)" for label in blockers)
    first, last_dropped, new_start, last = (_date_text(frame.index[i]) for i in (0, start - 1, start, -1))
    warnings.warn(
        f"{caller}: the portfolio starts on {new_start}, not on the first date of prices ({first}). "
        f"Dropped because not every holding had a price yet: {_rows(start)} of {len(frame)} "
        f"({first} to {last_dropped}). The start was moved by {detail}. The portfolio covers {_rows(len(frame) - start)} "
        f"({new_start} to {last}), a shorter period than the table. Any comparison with another series "
        "(a benchmark, for example) must use the same dates, otherwise the two sets of numbers describe "
        "different periods.",
        ShortSampleWarning,
        stacklevel=3,
    )


def _warn_valued_at_last_price(caller: str, frame: pd.DataFrame) -> None:
    """Emit ONE GapWarning for holdings that were valued at a last known price. Same ``stacklevel`` rule.

    ``frame`` is the held prices from the common start date on (so its first row is complete).
    Each ticker is shown as ``'AAPL': 19 valued at last price (longest 1 row)``. Two stronger
    sentences can follow: a run longer than ``MAX_HOLIDAY_GAP_ROWS`` rows (longer than any
    holiday, so probably missing data), and a STALE holding, whose last valid price is before
    the last date of the table (stopped trading, delisted, or its data ends early).
    """
    missing = frame.isna().to_numpy()
    detail, long_ones, stale = [], [], []
    for position, label in enumerate(frame.columns):
        flags = missing[:, position]
        count = int(flags.sum())
        if count == 0:
            continue
        runs = _run_lengths(flags)
        detail.append(f"{label!r}: {count} valued at last price (longest {_rows(max(runs))})")
        if max(runs) > MAX_HOLIDAY_GAP_ROWS:
            long_ones.append(f"{label!r} ({_rows(max(runs))})")
        if flags[-1]:  # the last run reaches the end of the table: no price after some date
            last_valid = frame.index[~flags][-1]
            stale.append(f"{label!r} has no price after {_date_text(last_valid)} ({_rows(runs[-1])} to the end of the table)")
    if not detail:
        return
    message = (
        f"{caller}: some dates were valued at a last known price because a holding had no price "
        f"that day ({', '.join(detail)}). That is valuation (the holding keeps its last price until "
        "it trades again), not an invented return. Short runs (a few rows) are normal for exchange "
        "holidays when tickers from different exchanges share a table; otherwise it means missing "
        "data (see data.find_gaps)."
    )
    if long_ones:
        message += (
            f" A run of more than {MAX_HOLIDAY_GAP_ROWS} rows is longer than a holiday: {', '.join(long_ones)}. "
            "This is likely missing data."
        )
    if stale:
        message += (
            f" Stale holding: {'; '.join(stale)}. It stays at that last price until the end, so the "
            "portfolio ignores whatever happened to it afterwards (a delisting, for example)."
        )
    warnings.warn(message, GapWarning, stacklevel=3)


def portfolio_value(prices, weights):
    """Value of 1 unit invested in a portfolio, BUY-AND-HOLD (NO rebalancing): a price-like series.

    BUY-AND-HOLD, NO REBALANCING. The weights are applied ONCE, on the first day: a fraction
    ``w_i`` of the money is used to buy holding ``i`` and nothing is ever bought or sold again.
    A winner therefore grows to a bigger share of the portfolio over time. Why this choice:
    it is the simplest to explain, and it is exactly the "growth of $1" a reader expects.
    A rebalanced portfolio (back to the target weights every day or month) would give a
    different curve, and this function does NOT do that.

    Formula: ``V_t = sum_i( w_i * P_i,t / P_i,start )``, where ``P_i,start`` is the price of
    holding ``i`` on the start date. ``V`` starts at exactly 1.0 (1.12 means +12% since the
    start). Units: a multiple of the starting value, not a percent. Because it is price-like,
    ``risk_summary``, ``max_drawdown``, ``drawdown_series`` and the others take it directly,
    for example ``risk_summary(pd.DataFrame({"Portfolio": value}))``.

    Inputs
        prices   a DataFrame of adjusted closes, one column per ticker, oldest first (like the
                 table from ``data.load_prices``). Extra columns (the benchmark) are ignored.
        weights  a dict or Series ``{ticker: weight}`` of fractions (0.20 = 20%). Every ticker
                 must be a column of ``prices``. Weights must be finite numbers, 0 or more
                 (long-only for now), and add up to 1 within 1e-9 (any tiny float noise is
                 divided out so the series starts at exactly 1.0). A weight of 0 is allowed:
                 that ticker simply does not count (it can even have bad prices).

    Start date: the FIRST date on which EVERY held ticker (weight above 0) has a valid price.
    Rows before it are dropped, so the result can be shorter than ``prices``. Nothing is ever
    filled in before the start date. A holding that starts trading later therefore moves the
    start of the whole portfolio. If no such date exists, or fewer than 2 dates remain from
    it, ValueError.

    SHORTER HISTORY WARNING: when the start date is later than the first row of ``prices``,
    ONE ``ShortSampleWarning`` says so. It names the ticker(s) that moved the start (those with
    no price on the last dropped row), how many rows were dropped and from which date to which
    date, and that any comparison with another series (a benchmark) must use the same dates.
    Without it a late starter would silently make the portfolio cover a shorter period than the
    benchmark next to it. No warning when the portfolio starts on the first row.

    Missing prices after the start (an exchange holiday, an outage): the holding is valued at
    its LAST VALID PRICE on that date. This is valuation (marking to the last known price),
    not an invented return: you cannot trade a closed market, so what you own is worth what
    it was last worth. It differs from the gap policy of ``daily_returns``. That function
    works on ONE ticker, so on a gap day it says "no return known" (NaN) and measures the next
    price against the last valid one. A portfolio is different: its other holdings DO trade
    that day, so the portfolio has a value and a return on that date. The closed holding just
    stays flat, and its whole move lands on the day it trades again (the same bridged move
    ``daily_returns`` would see, only spread differently across tickers). ONE ``GapWarning``
    names each ticker and how many dates were valued this way, and it says more if a run is
    longer than ``MAX_HOLIDAY_GAP_ROWS`` rows or if a holding is STALE (its last valid price
    is before the last date of the table: stopped trading or delisted, so it is held flat to
    the end).

    Output: a float64 Series named "Portfolio" on the dates from the start date on (the
    index keeps its type and name). The inputs are never modified. ValueError for: ``prices``
    that is not a DataFrame, empty or malformed; empty weights; a ticker that is not in
    ``prices`` (even with weight 0, because it is probably a typo); a non-numeric, NaN or
    infinite weight; a negative weight; weights that do not add up to 1; a held price of zero
    or less; an index not in increasing order. No warning about the number of observations
    (it is a path, not an estimate); the only sample-period warning is the shorter-history one
    above.
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError(
            "portfolio_value: prices must be a pandas DataFrame with one column per ticker "
            f"(like the table from data.load_prices), got {type(prices).__name__}."
        )
    cleaned = _clean_weights(weights)
    if prices.columns.has_duplicates:
        raise ValueError("prices has duplicate column names; every ticker must appear once.")
    unknown = [ticker for ticker in cleaned if ticker not in prices.columns]
    if unknown:
        raise ValueError(
            f"portfolio_value: weights name tickers that are not in prices: "
            f"{', '.join(repr(t) for t in unknown)}. The columns of prices are "
            f"{', '.join(repr(c) for c in prices.columns)}."
        )

    # Only the HELD columns are looked at (weight above 0), in the column order of `prices`, so
    # the result depends neither on the order of the weights nor on the tickers that are not held.
    held = [column for column in prices.columns if cleaned.get(column, 0.0) > 0.0]
    frame, _ = _prepare(prices[held], "prices", ordered=True, positive=True)

    # Start date: the first row where every held ticker has a price.
    complete = frame.notna().all(axis=1).to_numpy()
    if not complete.any():
        firsts = []
        for label in frame.columns:
            valid = frame[label].dropna()
            firsts.append(
                f"{label!r}: no valid price" if valid.empty
                else f"{label!r}: {_date_text(valid.index[0])} to {_date_text(valid.index[-1])}"
            )
        raise ValueError(
            "portfolio_value: there is no date on which every holding has a price, so the "
            f"portfolio has no start date. Dates with prices: {'; '.join(firsts)}."
        )
    start = int(np.argmax(complete))
    full, frame = frame, frame.iloc[start:]
    if len(frame) < 2:
        raise ValueError(
            f"portfolio_value: need at least 2 dates from the start date {_date_text(frame.index[0])} "
            f"(the first date on which every holding has a price), but only {len(frame)} remain."
        )

    # Mark to the last known price. The first row is complete, so ffill leaves no NaN, and it
    # can only look BACKWARDS in time, from the start date on: nothing before the start is filled.
    marked = frame.ffill()
    growth = marked.to_numpy() / marked.to_numpy()[0]  # every holding starts at exactly 1.0
    w = np.array([cleaned[column] for column in marked.columns], dtype="float64")
    raw = (growth * w).sum(axis=1)
    # raw[0] is the sum of the weights: 1 up to float noise. Dividing by it makes the start exactly 1.0.
    value = raw / raw[0]

    _warn_shortened_history("portfolio_value", full, start)
    _warn_valued_at_last_price("portfolio_value", frame)
    return pd.Series(value, index=frame.index, name=PORTFOLIO_NAME, dtype="float64")


def drawdown_series(prices):
    """The "underwater" curve: how far below its previous peak the price is on every day.

    Formula: ``d_t = 1 - P_t / running_max(P)_t``, where ``running_max`` is the highest price
    seen UP TO day t (not over the whole series). It is the same quantity ``max_drawdown``
    takes the maximum of: ``drawdown_series(p).max()`` equals ``max_drawdown(p)``, exactly.

    SIGN and units: a POSITIVE fraction between 0 and 1 (0.25 means "25% below the earlier
    peak"), like ``max_drawdown`` and ``historical_var``. It is 0 on a day that sets a new
    peak (or equals it). Example: prices [100, 120, 90, 130] give [0, 0, 0.25, 0].

    Shapes: a Series gives a Series (same name), a DataFrame gives a DataFrame (one curve
    per column), on the same dates as the input.

    Gaps: it is NaN where the price is NaN, and before a column's first valid price. The
    running maximum skips NaN prices (it works on price levels, like ``max_drawdown``).

    Needs at least 2 valid prices per column, otherwise ValueError. No sample-size warning
    (it is a path, not an estimate).
    """
    frame, is_series = _prepare(prices, "prices", ordered=True, positive=True)
    _require_obs(frame, "drawdown_series", "prices")
    # cummax skips NaN: a NaN row stays NaN, and the maximum carries on from before it.
    return _finish_series(1.0 - frame / frame.cummax(), prices, is_series)


def rolling_volatility(prices, window):
    """Trailing annualized volatility of PRICES, for charts: ``daily_returns`` then the rolling std.

    Formula: ``std(last `window` daily returns, ddof=1) * sqrt(252)`` at each date, where the
    returns are ``r_t = P_t / P_prev - 1``. It is exactly
    ``annualized_volatility(daily_returns(prices), window=window)``: the same numbers, the
    same warnings (GapWarning from ``daily_returns``, ShortSampleWarning for a window below
    ``MIN_RELIABLE_WINDOW``), the same errors. Units: a fraction per year (0.28 = 28% a year).

    Past data only: the value at date t uses the returns up to and including t, never later
    ones (see ``annualized_volatility`` for how gaps and the first ``window - 1`` dates work).

    Why it exists: ``annualized_volatility(returns, window=...)`` refuses a ticker whose
    returns "look like prices" (median absolute return above 0.5, such as GME in January 2021),
    which made the rolling-volatility chart vanish for every ticker. Here the returns are
    computed inside, so they are known to be returns and that heuristic is skipped (the same
    reasoning as in ``risk_summary``). ``window`` is required: a whole number of at least 2.
    """
    if window is None:
        raise ValueError(f"window must be a whole number of at least 2, got {window!r}.")
    returns = daily_returns(prices)
    return annualized_volatility(returns, window=window, _skip_prices_check=True)


def return_correlation(prices):
    """Correlation of the tickers' daily returns, from PRICES, for charts.

    Formula: ``corr(a, b) = cov(a, b) / (std(a) * std(b))`` of the daily returns
    ``r_t = P_t / P_prev - 1``, for every pair of columns. It is exactly
    ``correlation_matrix(daily_returns(prices))``: the same numbers, warnings and errors.
    Units: from -1 through 0 to +1, as a square DataFrame (tickers x tickers).

    Why it exists: ``correlation_matrix`` refuses a ticker whose returns "look like prices"
    (median absolute return above 0.5, such as GME in January 2021), which made the
    correlation chart vanish for every ticker. Here the returns are computed inside, so they
    are known to be returns and that heuristic is skipped (the same reasoning as in
    ``risk_summary``). Needs at least 2 columns (tickers): a Series or a 1-column table raises
    ValueError in this function's own name.
    """
    if isinstance(prices, pd.Series) or (isinstance(prices, pd.DataFrame) and prices.shape[1] < 2):
        raise ValueError(
            "return_correlation needs at least 2 tickers (a DataFrame with at least 2 columns), "
            f"got {'a Series' if isinstance(prices, pd.Series) else 'only one column'}."
        )
    returns = daily_returns(prices)
    return correlation_matrix(returns, _skip_prices_check=True)
