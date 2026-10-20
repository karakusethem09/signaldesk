"""SignalDesk data layer: download, clean, check and cache daily stock prices.

This file only deals with getting prices into a tidy table. It does NOT calculate
returns, volatility or any other risk numbers. That lives in ``signaldesk/risk.py``.

The table (a pandas DataFrame) always looks like this:
    - the rows are dates (a DatetimeIndex named "Date"), oldest first
    - there is one column per ticker, named by the ticker symbol, e.g. "AAPL"
    - the values are adjusted close prices (float64). "Adjusted" means the price is
      corrected for stock splits and dividends, which is what you want for returns.

The functions, from the network inward:
    download_prices -> the ONLY function that touches the internet
    clean_prices    -> tidy up a raw table (pure: no network, no files)
    find_gaps       -> report missing values (NaN cells) per ticker (pure, never fills)
    find_missing_days -> report weekdays that are absent from the table (pure)
    load_prices     -> the one the dashboard calls: cache -> download -> clean -> save
"""

from __future__ import annotations

import hashlib
import os
import re
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

__all__ = [
    "PriceDataError",
    "clean_prices",
    "download_prices",
    "find_gaps",
    "find_missing_days",
    "load_prices",
]


class PriceDataError(ValueError):
    """Raised when we asked for prices but did not get usable data back.

    It is a kind of ValueError, so ``except ValueError`` also catches it. The
    dashboard can show ``str(error)`` to the user because the message says what
    went wrong (for example which ticker came back empty).
    """


# --------------------------------------------------------------------------- #
# Small private helpers
# --------------------------------------------------------------------------- #

def _to_timestamp(value, name: str) -> pd.Timestamp:
    """Turn a date-like value ("2026-01-31", a date, a Timestamp) into a midnight Timestamp."""
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{name} {value!r} is not a valid date. Use a form like '2026-01-31'."
        ) from exc
    if pd.isna(stamp):
        raise ValueError(f"{name} is missing. Use a date like '2026-01-31'.")
    if stamp.tzinfo is not None:
        # Drop the timezone but KEEP the local wall-clock date (tz_convert(None) would
        # first move it to UTC, which can change the date, e.g. Tokyo midnight).
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _now_utc() -> pd.Timestamp:
    """The current moment as a UTC timestamp. A separate function so tests can replace it."""
    return pd.Timestamp.now(tz="UTC")


def _today() -> pd.Timestamp:
    """Today's date in NEW YORK, at midnight with no timezone. Tests can replace it.

    Why New York and not this computer's date: yfinance treats ``end`` as midnight in the
    exchange's own timezone. On a computer far east of New York (Tokyo, Sydney) the local
    date moves on while the US session is still open, so "today" would include the
    unfinished US bar, which would then be cached as if it were a final close.
    New York has the latest date on the exchange side of the world, so by the time its
    date has changed, every session for the earlier date (US, European, Asian) has
    closed. Using it is safe for all of them (it is just up to a day late for Asia).
    """
    new_york_now = _now_utc().tz_convert("America/New_York")
    return new_york_now.normalize().tz_localize(None)


def _limit_end_to_today(end) -> tuple[pd.Timestamp, str]:
    """Parse ``end`` and cap it at today's New York date. Returns (end, note).

    ``note`` is empty if nothing changed, otherwise a sentence for error messages saying
    that the end was limited. Both ``load_prices`` and ``find_missing_days`` use this, so
    the table we load and the days we check always cover the same window: prices for
    today and future dates do not exist (yet), so asking for them is capped the same way.
    A bad or missing ``end`` raises the usual ValueError.
    """
    today = _today()
    end_ts = _to_timestamp(end, "end")
    if end_ts <= today:
        return end_ts, ""
    note = (
        f" (end was limited to today, {today:%Y-%m-%d} New York time, because prices "
        "for future dates do not exist yet)"
    )
    return today, note


def _default_cache_dir() -> Path:
    """The default cache folder: ``data/cache`` inside the project, wherever you run from."""
    return Path(__file__).resolve().parent.parent / "data" / "cache"


def _to_date_index(index: pd.Index) -> pd.DatetimeIndex:
    """Turn a table's index into plain daily dates: DatetimeIndex, no timezone, midnight, ns.

    Raises a clear ValueError if the index is numbers (0, 1, 2, ...) instead of dates.
    A timezone-aware index keeps its LOCAL calendar date: Tokyo midnight must stay that
    day, not become the previous day in UTC (``tz_convert(None)`` would do that).
    """
    if len(index) > 0 and pd.api.types.is_numeric_dtype(index):
        raise ValueError("The price table must be indexed by dates, but the index is numeric.")
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()  # daily data: drop any time of day (after the tz step above)
    # Force one fixed time unit. yfinance gives "s" and reading a CSV gives "us";
    # without this a cached table could differ from a fresh one.
    return dates.astype("datetime64[ns]")


def _validate_inputs(tickers, start, end, end_note: str = "") -> tuple[list[str], str, str]:
    """Check the inputs and return them in a standard form: (tickers, start, end).

    - tickers become upper-case, stripped of spaces, with repeats removed (order kept)
    - start and end become "YYYY-MM-DD" strings
    A clear ValueError is raised for an empty ticker list, a bad date, or start >= end.
    ``end_note`` is extra text added to the start >= end message (used when end was clamped).
    """
    if isinstance(tickers, str):  # a lone "AAPL" would otherwise be split into letters
        tickers = [tickers]
    if tickers is None:
        tickers = []

    cleaned: list[str] = []
    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError(f"Invalid ticker {ticker!r}: tickers must be non-empty strings.")
        symbol = ticker.strip().upper()
        if symbol not in cleaned:
            cleaned.append(symbol)
    if not cleaned:
        raise ValueError("The ticker list is empty. Give at least one symbol, e.g. ['AAPL'].")

    start_ts = _to_timestamp(start, "start")
    end_ts = _to_timestamp(end, "end")
    if start_ts >= end_ts:
        raise ValueError(
            f"start ({start_ts:%Y-%m-%d}) must be earlier than end ({end_ts:%Y-%m-%d})."
            f"{end_note}"
        )
    return cleaned, f"{start_ts:%Y-%m-%d}", f"{end_ts:%Y-%m-%d}"


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the "Close" prices out of the raw table yfinance gives back.

    yfinance returns many fields (Open, High, Low, Close, Volume). The columns are a
    two-level MultiIndex of (field, ticker), even for a single ticker. Older versions
    returned plain columns for a single ticker, so we handle that shape too.
    With auto_adjust=True the "Close" field is already the adjusted close.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            if "Close" in raw.columns.get_level_values(level):
                return raw.xs("Close", axis=1, level=level)
        raise PriceDataError("yfinance returned data without a 'Close' field.")

    if "Close" in raw.columns:  # plain columns: only possible for a single ticker
        return raw[["Close"]].rename(columns={"Close": tickers[0]})
    raise PriceDataError("yfinance returned data without a 'Close' field.")


def _require_all_tickers(prices: pd.DataFrame, tickers: list[str]) -> None:
    """Raise PriceDataError if any requested ticker is absent or has only missing values.

    yfinance does not raise for an unknown ticker: it just hands back an empty
    (all-NaN) column. We turn that into a loud error instead of passing it on.
    """
    empty = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if empty:
        raise PriceDataError(
            f"No price data returned for: {', '.join(empty)}. "
            "Check the ticker symbol(s) and the date range, and your internet connection."
        )


def _cache_path(cache_dir, tickers: list[str], start: str, end: str) -> Path:
    """Work out the cache file for a request. Same tickers (any order) + dates = same file.

    Example: prices_AAPL+MSFT_2026-01-01_2026-03-01.csv
    Very long ticker lists are shortened to a hash so the filename stays valid.
    """
    ordered = sorted(tickers)
    slug = "+".join(re.sub(r"[^A-Za-z0-9._-]", "_", t) for t in ordered)
    if len(slug) > 80:
        slug = hashlib.sha1("+".join(ordered).encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"prices_{slug}_{start}_{end}.csv"


def _read_cache(path: Path, tickers: list[str]) -> pd.DataFrame | None:
    """Read a cache file. Return None (with a warning) if it is unusable, so the caller re-downloads."""
    try:
        # float_precision="round_trip" matters: pandas' default fast reader can be off by
        # one in the last binary digit, which would make a cached table differ from a fresh one.
        raw = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
        cached = clean_prices(raw)
    except (OSError, ValueError, KeyError) as exc:
        warnings.warn(f"Ignoring unreadable cache file {path} ({exc}). Downloading again.")
        return None
    if cached.empty or set(cached.columns) != set(tickers):
        warnings.warn(f"Cache file {path} does not match the request. Downloading again.")
        return None
    return cached.reindex(columns=tickers)  # follow the caller's ticker order


def _write_cache(path: Path, prices: pd.DataFrame) -> None:
    """Save prices to the cache as CSV. A failure here warns but never stops the dashboard.

    We write to a temporary file first and then rename it, so a crash half-way
    through cannot leave a broken cache file behind.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(tmp)
        os.replace(tmp, path)
    except OSError as exc:
        warnings.warn(f"Could not write the price cache at {path} ({exc}). Continuing without it.")
        try:
            tmp.unlink(missing_ok=True)  # tidy up a half-written temp file, if there is one
        except OSError:
            pass  # cleanup is best-effort: the warning above is what matters


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #

def download_prices(tickers: list[str], start, end) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance (the only function that uses the network).

    Returns a raw table: one column per ticker, in the order you asked for, with the
    dates as the index. It is NOT cleaned yet (use ``clean_prices`` for that).
    ``start`` is included but ``end`` is excluded (that is how yfinance works).
    Raises ValueError for bad inputs and PriceDataError if any ticker has no data.
    """
    tickers, start_str, end_str = _validate_inputs(tickers, start, end)

    raw = yf.download(
        tickers,
        start=start_str,
        end=end_str,
        auto_adjust=True,  # "Close" is then already the split/dividend adjusted close
        progress=False,
    )
    if raw is None or raw.empty:
        raise PriceDataError(
            f"No price data returned for {', '.join(tickers)} between {start_str} and "
            f"{end_str}. Check the ticker symbol(s), the dates and your internet connection."
        )

    close = _extract_close(raw, tickers)
    _require_all_tickers(close, tickers)
    return close.reindex(columns=tickers)


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Tidy a price table without inventing any numbers. Returns a new table; the input is untouched.

    - rows where EVERY ticker is missing are dropped (for example a market holiday)
    - rows are sorted by date, oldest first
    - times of day are removed (every date becomes midnight), so two timestamps on the
      same day count as the same date; if a date appears twice, the last copy wins
    - the index becomes a timezone-free DatetimeIndex named "Date"; values become float64
    Missing values in a row that still has some data are kept as NaN: we never fill
    gaps (use ``find_gaps`` to see them).

    Dropping an all-missing row is silent here, but it is not invisible: a dropped
    weekday is simply a weekday that is absent from the result, so
    ``find_missing_days`` on the cleaned table will list it.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"clean_prices expects a pandas DataFrame, got {type(prices).__name__}.")

    df = prices.copy()
    df.index = _to_date_index(df.index).rename("Date")

    df.columns = pd.Index([str(c) for c in df.columns])  # plain string names, no columns.name
    df = df.astype("float64")

    # Order matters: drop empty rows BEFORE removing duplicate dates, so an empty
    # duplicate can never win over a duplicate that holds real prices.
    df = df.dropna(how="all")
    df = df.sort_index(kind="stable")  # stable: equal dates keep their original order
    df = df[~df.index.duplicated(keep="last")]
    return df


def find_gaps(prices: pd.DataFrame) -> pd.DataFrame:
    """Report missing prices per ticker. This only reports; it never fills anything.

    Returns one row per ticker with three columns:
        missing_count  how many dates have no price for this ticker
        first_missing  the earliest date with no price (NaT if there is none)
        last_missing   the latest date with no price (NaT if there is none)
    A ticker that started trading late shows up as missing at the start; a gap in
    the middle shows up with a first_missing/last_missing inside the date range.
    Run it on cleaned prices: days when ALL tickers are missing were already dropped.
    """
    counts, firsts, lasts = [], [], []
    for ticker in prices.columns:
        missing_dates = prices.index[prices[ticker].isna().to_numpy()]
        counts.append(len(missing_dates))
        firsts.append(missing_dates.min() if len(missing_dates) else pd.NaT)
        lasts.append(missing_dates.max() if len(missing_dates) else pd.NaT)

    return pd.DataFrame(
        {
            "missing_count": pd.Series(counts, dtype="int64"),
            "first_missing": pd.to_datetime(pd.Series(firsts, dtype="object")).astype("datetime64[ns]"),
            "last_missing": pd.to_datetime(pd.Series(lasts, dtype="object")).astype("datetime64[ns]"),
        }
    ).set_axis(pd.Index([str(c) for c in prices.columns], name="Ticker"), axis=0)


def find_missing_days(prices: pd.DataFrame, start=None, end=None) -> pd.DatetimeIndex:
    """List the weekdays that are absent from the table. A "possible gap" report, not an error.

    By default it looks between the first and last date IN the table. Most of what it
    returns are normal exchange holidays (the US market closes about 9-10 weekdays a year,
    e.g. Good Friday and Thanksgiving). It also shows weekdays whose rows ``clean_prices``
    dropped because every ticker was missing, and days the data provider skipped. Weekends
    are never reported. Look at the dates to decide: a lone Wednesday in the middle of
    March is suspicious, 25 December is not.

    Pass ``start`` and/or ``end`` (the dates you ASKED for) to also catch data that begins
    later or ends earlier than requested, which the default cannot see. ``end`` is
    excluded, like everywhere else in this module: asking for 2026-03-30 to 2026-04-10
    checks the weekdays 03-30 up to and including 04-09. If you give only one of them, the
    other side still uses the table's own first or last date. Rows outside the window are
    ignored. An empty table with both given reports every weekday of the window.

    Like ``load_prices``, an ``end`` after today (New York date) is LIMITED TO TODAY, so
    passing the same start/end you gave ``load_prices`` checks exactly the window that was
    loaded: days that cannot have prices yet are never reported as missing.

    Why it matters: a daily return across a missing day spans two days but would count
    as one observation, which quietly distorts volatility. ``find_gaps`` cannot see this,
    because it only looks at missing values inside rows that exist.
    Run it on cleaned prices; a timezone-aware index keeps its local dates. It reports
    only and never changes the table. Raises ValueError if the index is numbers instead of
    dates, if start is not earlier than end (after the limit to today), or if you give only
    one of start/end and that window does not overlap the table at all (for example a
    start after the table's last date), because then there is nothing to compare.
    Returns a DatetimeIndex named "Date" (empty if nothing is missing).
    """
    dates = _to_date_index(prices.index)
    first = _to_timestamp(start, "start") if start is not None else None
    last, end_note = None, ""
    if end is not None:
        end_ts, end_note = _limit_end_to_today(end)  # same cap as load_prices
        # `end` is exclusive, so the last day that must be present is the day before it.
        last = end_ts - pd.Timedelta(days=1)
        if first is not None and first > last:
            raise ValueError(
                f"start ({first:%Y-%m-%d}) must be earlier than end ({end_ts:%Y-%m-%d}).{end_note}"
            )

    one_sided = (start is None) != (end is None)
    if first is None and len(dates) > 0:
        first = dates.min()
    if last is None and len(dates) > 0:
        last = dates.max()
    if one_sided and first is not None and last is not None and first > last:
        # The side we filled in from the table lies on the wrong side of the given one.
        if start is not None:
            raise ValueError(
                f"start ({first:%Y-%m-%d}) is after the last date in the table "
                f"({last:%Y-%m-%d}), so the window does not overlap the data."
            )
        raise ValueError(
            f"end ({end_ts:%Y-%m-%d}) is not after the first date in the table "
            f"({first:%Y-%m-%d}), so the window does not overlap the data.{end_note}"
        )
    if first is None or last is None:  # nothing to measure against (empty table, no window)
        return pd.DatetimeIndex([], dtype="datetime64[ns]", name="Date")

    expected = pd.bdate_range(first, last)  # every weekday in the span
    return expected.difference(dates).astype("datetime64[ns]").rename("Date")


def load_prices(
    tickers: list[str],
    start,
    end,
    cache_dir=None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Main entry point for the dashboard: get clean adjusted close prices, using a CSV cache.

    1. If a cache file for these tickers + dates exists, read it (no network).
    2. Otherwise download, clean, save to the cache, and return the result.
    A cache read returns exactly the same table as a fresh download would.
    Columns follow the order of ``tickers``.

    ``end`` is excluded (yfinance behaviour) and is LIMITED TO TODAY: an end date in the
    future is replaced by today's date before anything else happens. "Today" means the
    date in New York, not on your computer's clock (see ``_today`` for why). Because
    ``end`` is excluded, the last row is then the last complete trading day before today,
    so we never store today's unfinished intraday price as a "close". It also means the
    cache key can never be a future date that would freeze a half-finished range forever.
    If ``start`` is today or later, there is nothing to load and a ValueError is raised.

    ``cache_dir`` defaults to ``data/cache`` inside the project folder, wherever you
    run the program from; pass a path to use another folder. ``use_cache=False`` forces
    a fresh download and refreshes the cache file.
    Raises ValueError for bad inputs and PriceDataError if a ticker has no data.
    Gaps are kept as NaN: call ``find_gaps`` on the result to see them, and
    ``find_missing_days`` to see weekdays that are absent altogether.
    """
    end_ts, end_note = _limit_end_to_today(end)  # a bad/missing end raises ValueError here
    tickers, start_str, end_str = _validate_inputs(tickers, start, end_ts, end_note)

    if cache_dir is None:
        cache_dir = _default_cache_dir()
    path = _cache_path(cache_dir, tickers, start_str, end_str)

    if use_cache and path.exists():
        cached = _read_cache(path, tickers)
        if cached is not None:
            return cached

    prices = clean_prices(download_prices(tickers, start_str, end_str))
    if prices.empty:
        raise PriceDataError(
            f"No price rows for {', '.join(tickers)} between {start_str} and {end_str}."
        )
    _require_all_tickers(prices, tickers)
    prices = prices.reindex(columns=tickers)

    _write_cache(path, prices)
    return prices
