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

Added for the public website (Phase 1.5):
    load_demo       -> read the committed demo snapshot (data/demo/), fully offline
    refresh_demo_snapshot -> rebuild that snapshot from Yahoo (run by scripts/refresh_demo_data.py)
    prune_cache     -> delete old cache files so the cache folder cannot grow forever
    market_today    -> today's date in New York (use it instead of the computer's clock)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import re
import stat
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

try:  # the "Yahoo is rate-limiting us" error; guarded in case a yfinance version lacks it
    from yfinance.exceptions import YFRateLimitError
except ImportError:  # pragma: no cover
    YFRateLimitError = None

__all__ = [
    "DemoData",
    "PriceDataError",
    "ProviderUnavailableError",
    "clean_prices",
    "download_prices",
    "find_gaps",
    "find_missing_days",
    "load_demo",
    "load_prices",
    "market_today",
    "prune_cache",
    "refresh_demo_snapshot",
]


class PriceDataError(ValueError):
    """Raised when we asked for prices but did not get usable data back.

    It is a kind of ValueError, so ``except ValueError`` also catches it. The
    dashboard can show ``str(error)`` to the user because the message says what
    went wrong (for example which ticker came back empty).
    """


class ProviderUnavailableError(PriceDataError):
    """Raised when NO requested ticker came back with any data.

    It is a kind of PriceDataError, so existing ``except PriceDataError`` code still catches
    it. Yahoo Finance answers three different situations with the same empty table, and we
    cannot tell them apart, so the message names all three: the provider is rate-limiting or
    unavailable (try again later), the symbols do not exist, or the dates contain no trading
    days (a weekend or a market holiday). It is also raised when ``yf.download`` itself
    raises a rate-limit or network error; in that case the provider is the cause.

    When only SOME tickers are missing we raise a plain PriceDataError instead. That does NOT
    prove the provider is healthy: yfinance hides a rate limit (or a local error inside
    yfinance) on a single ticker as "no data for that ticker", exactly like a mistyped
    symbol. So that message names the missing tickers and gives both likely causes: the
    symbol may be wrong, or the data provider may have failed for it.
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


def market_today() -> datetime.date:
    """Today's date in New York, as a plain ``datetime.date``.

    The dashboard uses this for its date pickers instead of ``date.today()``. A server in
    another timezone (Streamlit Cloud runs on UTC) would otherwise show a different "today"
    than the one ``load_prices`` uses to cap the end date. See ``_today`` for the reasoning.
    """
    return _today().date()


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


# DEFENSIVE CODE, not the main path. yfinance 1.7 catches EVERY exception per ticker inside
# yf.download (multi.py, _download_one) and replaces that ticker's data with an empty table, so
# a rate limit or a dropped connection normally does NOT come out of yf.download as an error:
# it arrives as "no data for this ticker" and is handled by raw.empty / _require_all_tickers.
# We still catch these errors in case another yfinance version lets them through. Network
# errors from curl_cffi and requests are OSError subclasses, like the built-in
# ConnectionError and TimeoutError.
_PROVIDER_ERRORS = tuple(
    t for t in (YFRateLimitError, OSError) if t is not None
)
# OSError subclasses that mean "a file or folder on this computer", not "the network".
_LOCAL_FILE_ERRORS = (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)


def _describe_window(start: str, end: str) -> str:
    """" between 2026-10-17 and 2026-10-18" for a request from start up to (excluding) end.

    ``end`` is excluded by the downloader, but a person thinks of the last day as included,
    so the message names the last day that was actually asked for (the day before ``end``).
    """
    last = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return f" on {start}" if last == start else f" between {start} and {last}"


def _no_data_for_any_ticker(tickers: list[str], window: str = "") -> ProviderUnavailableError:
    """The error for "nothing at all came back". ``window`` is an optional " between X and Y".

    There are three honest explanations and we cannot tell them apart, so the message names
    all three: the provider is rate-limiting or down, the symbols do not exist, or the dates
    contain no trading days (a weekend or a market holiday gives an empty table too).
    """
    one = len(tickers) == 1
    asked = "the requested ticker" if one else "any of the requested tickers"
    return ProviderUnavailableError(
        f"No price data came back for {asked} ({', '.join(tickers)})"
        f"{window}. The data provider may be rate-limiting or unavailable right now "
        f"(try again later), the {'symbol' if one else 'symbols'} may not exist, or there "
        "were no trading days in that period (for example a weekend or a market holiday)."
    )


def _require_all_tickers(prices: pd.DataFrame, tickers: list[str], window: str = "") -> None:
    """Raise PriceDataError if any requested ticker is absent or has only missing values.

    yfinance does not raise for an unknown ticker: it just hands back an empty
    (all-NaN) column. We turn that into a loud error instead of passing it on.
    If EVERY ticker is empty the error is the more specific ProviderUnavailableError
    (the provider is probably rate-limiting us, or the dates have no trading days);
    if only some are, it names those. ``window`` is an optional " between X and Y".

    The partial case is honest about its two causes. yfinance hides a rate limit on ONE
    ticker as "no data for that ticker", so we cannot tell it from a mistyped symbol.
    """
    empty = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if len(empty) == len(tickers):
        raise _no_data_for_any_ticker(tickers, window)
    if empty:
        one = len(empty) == 1
        raise PriceDataError(
            f"No price data returned for: {', '.join(empty)}{window}. "
            f"The {'symbol' if one else 'symbols'} may not exist or may not have traded in "
            f"this period, or the data provider may have failed for "
            f"{'it' if one else 'them'} (for example a rate limit). "
            "Check the symbol(s) and the date range, or try again in a minute."
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


# The names ``_cache_path`` can produce, as patterns. ``prune_cache`` deletes ONLY files whose
# whole name matches one of these, so a file somebody else put in the folder (for example
# "prices_my_q3_export.csv") is never touched. The pieces, matching ``_cache_path`` above:
#   prices_<slug>_<start>_<end>.csv   <slug> = tickers joined with "+", every character other
#                                      than A-Z a-z 0-9 . _ - replaced by "_" (or a 16-char
#                                      hash, which these characters also cover)
#   <that name>.<32 hex>.tmp          the temporary file of ``_atomic_write`` (a random id)
#   <that name>.<digits>.tmp          the older temporary name (process id), if one was left
_DATE_PATTERN = r"\d{4}-\d{2}-\d{2}"
_CACHE_FILE_PATTERN = re.compile(rf"prices_[A-Za-z0-9._+\-]+_{_DATE_PATTERN}_{_DATE_PATTERN}\.csv")
_CACHE_TEMP_PATTERN = re.compile(
    rf"prices_[A-Za-z0-9._+\-]+_{_DATE_PATTERN}_{_DATE_PATTERN}\.csv\.(?:[0-9a-f]{{32}}|[0-9]+)\.tmp"
)


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


def _atomic_write(path: Path, write) -> None:
    """Create ``path`` safely: write a temporary file next to it, then rename it into place.

    ``write`` is a function that receives the temporary path and writes the content there.
    The rename (``os.replace``) is all-or-nothing, so a reader sees either the old file or
    the complete new one, never half a file, and a crash cannot leave a broken file behind.

    The temporary name contains a random id, so two writers (threads or processes) never
    share a temporary file, even when they save the same ``path`` at the same moment. The
    temporary file is removed if anything goes wrong. Errors are raised to the caller.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:  # also Ctrl+C: never leave a half-written temp file around
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass  # cleanup is best-effort: the original error is what matters
        raise


def _write_cache(path: Path, prices: pd.DataFrame) -> bool:
    """Save prices to the cache as CSV. A failure here warns but never stops the dashboard.

    We write to a unique temporary file first and then rename it (see ``_atomic_write``).
    Returns True if the file was written, False if the write failed (after the warning).
    Pruning old files is NOT done here: ``load_prices`` does it, and only for the default folder.
    """
    try:
        _atomic_write(path, prices.to_csv)
    except OSError as exc:
        warnings.warn(f"Could not write the price cache at {path} ({exc}). Continuing without it.")
        return False
    return True


STALE_TEMP_SECONDS = 3600  # a leftover cache temp file older than this is from a crashed writer


def prune_cache(cache_dir=None, max_age_days: float | None = 7, max_files: int | None = 200) -> int:
    """Delete old cache files and return how many were deleted.

    Why: every different set of tickers and dates makes a new cache file, and on a public
    server nobody would ever clean them up. Rules, in this order:
      1. a cache file whose last-modified time is older than ``max_age_days`` is deleted;
      2. of the files that are left, the OLDEST ones beyond ``max_files`` are deleted;
      3. a leftover cache temp file (from a writer that crashed) older than an hour is
         deleted. Younger ones are kept: somebody may be writing them right now.
    A "cache file" is a file whose WHOLE name is one that ``_cache_path`` can produce, for
    example ``prices_AAPL+MSFT_2026-01-01_2026-03-01.csv``. It never touches any other file,
    folder or link, even one called ``prices_something.csv``. It is best-effort: if something
    cannot be deleted it only warns, it never raises or stops the dashboard.
    ``None`` switches the age rule or the count rule off. ``cache_dir`` defaults to
    ``data/cache`` in the project. (``load_prices`` calls this by itself only for that
    default folder; for any other folder you call it yourself.)
    """
    if (max_age_days is not None and max_age_days < 0) or (max_files is not None and max_files < 0):
        raise ValueError("max_age_days and max_files must not be negative.")
    folder = Path(_default_cache_dir() if cache_dir is None else cache_dir)

    deleted = 0
    failures: list[str] = []

    def delete(path: Path) -> None:
        nonlocal deleted
        try:
            path.unlink()
            deleted += 1
        except FileNotFoundError:
            pass  # another request pruned it first: fine, the goal is reached
        except OSError as exc:
            failures.append(f"{path.name} ({exc})")

    def regular_files(pattern: re.Pattern) -> list[tuple[float, str, Path]]:
        """(modified time, name, path) of the regular files whose whole name matches ``pattern``
        (no folders, no links)."""
        found = []
        for path in folder.iterdir():
            if pattern.fullmatch(path.name) is None:
                continue
            try:
                info = path.lstat()
            except OSError:
                continue  # vanished while we looked
            if stat.S_ISREG(info.st_mode):
                found.append((info.st_mtime, path.name, path))
        return sorted(found)  # oldest first; equal times fall back to the name

    try:
        if not folder.is_dir():
            return 0
        now = time.time()
        csv_files = regular_files(_CACHE_FILE_PATTERN)

        if max_age_days is not None:
            cutoff = now - max_age_days * 86400
            for item in [f for f in csv_files if f[0] < cutoff]:
                delete(item[2])
            csv_files = [f for f in csv_files if f[0] >= cutoff]
        if max_files is not None and len(csv_files) > max_files:
            for item in csv_files[: len(csv_files) - max_files]:
                delete(item[2])

        for mtime, _name, path in regular_files(_CACHE_TEMP_PATTERN):
            if now - mtime > STALE_TEMP_SECONDS:
                delete(path)
    except OSError as exc:
        failures.append(f"{folder} ({exc})")

    if failures:
        warnings.warn(
            f"Could not clean the price cache: {'; '.join(failures[:3])}. Continuing anyway."
        )
    return deleted


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #

def download_prices(tickers: list[str], start, end) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance (the only function that uses the network).

    Returns a raw table: one column per ticker, in the order you asked for, with the
    dates as the index. It is NOT cleaned yet (use ``clean_prices`` for that).
    ``start`` is included but ``end`` is excluded (that is how yfinance works).
    Raises ValueError for bad inputs and PriceDataError if any ticker has no data.
    If NO ticker has data it raises ProviderUnavailableError, a kind of PriceDataError. Its
    message names every possible cause: the provider is rate-limiting or down, the symbols do
    not exist, or the dates contain no trading days (a weekend, a holiday).

    What yfinance really does (version 1.7): it catches every error PER TICKER and returns an
    empty table for that ticker instead of raising. So a rate limit, a dropped connection, a
    wrong symbol and even a local problem inside yfinance (its own cache database being
    locked, for example) all look the same to us: "no data for this ticker". We cannot tell
    them apart, so when only some tickers are missing the PriceDataError names them and says
    both likely causes. (An error that does come out of yf.download is still turned into a
    ProviderUnavailableError, except errors about files on this computer, which are re-raised.)
    """
    tickers, start_str, end_str = _validate_inputs(tickers, start, end)
    window = _describe_window(start_str, end_str)

    try:
        raw = yf.download(
            tickers,
            start=start_str,
            end=end_str,
            auto_adjust=True,  # "Close" is then already the split/dividend adjusted close
            progress=False,
        )
    except _PROVIDER_ERRORS as exc:
        if isinstance(exc, _LOCAL_FILE_ERRORS):
            raise  # a problem on THIS computer (for example a folder we cannot open), not Yahoo
        # Keep the message plain (it may be shown to a website visitor); the original
        # error stays attached as __cause__ for whoever debugs it.
        raise ProviderUnavailableError(
            f"No price data could be fetched for {', '.join(tickers)}{window}. The data "
            "provider may be rate-limiting or unreachable right now. Please try again later."
        ) from exc
    if raw is None or raw.empty:
        raise _no_data_for_any_ticker(tickers, window)

    close = _extract_close(raw, tickers)
    _require_all_tickers(close, tickers, window)
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
    a fresh download and refreshes the cache file. After a successful write to the DEFAULT
    folder, old cache files there are pruned (see ``prune_cache``); a folder you pass
    yourself is never cleaned automatically, because it may hold your own files.
    Raises ValueError for bad inputs and PriceDataError if a ticker has no data.
    Gaps are kept as NaN: call ``find_gaps`` on the result to see them, and
    ``find_missing_days`` to see weekdays that are absent altogether.
    """
    end_ts, end_note = _limit_end_to_today(end)  # a bad/missing end raises ValueError here
    tickers, start_str, end_str = _validate_inputs(tickers, start, end_ts, end_note)

    using_default_folder = cache_dir is None
    if using_default_folder:
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
    _require_all_tickers(prices, tickers, _describe_window(start_str, end_str))
    prices = prices.reindex(columns=tickers)

    if _write_cache(path, prices) and using_default_folder:
        prune_cache(cache_dir)  # keeps OUR cache folder from growing forever
    return prices


# --------------------------------------------------------------------------- #
# The demo snapshot (the public website's default data, no network needed)
# --------------------------------------------------------------------------- #
#
# Yahoo Finance often blocks or rate-limits servers shared by many users, so the public
# site cannot depend on it. Instead we save one snapshot of prices in git (data/demo/)
# and the site reads that. Three files live there:
#     demo_portfolio.json  hand-written: name, holdings (ticker -> weight), benchmark, years
#     demo_prices.csv      the prices, same CSV format as the cache
#     demo_meta.json       facts about the snapshot: first/last date, rows, tickers, source
# ``refresh_demo_snapshot`` (run by scripts/refresh_demo_data.py) writes the last two from
# the first one. ``load_demo`` reads all three and never touches the network.

DEMO_CONFIG_FILE = "demo_portfolio.json"
DEMO_PRICES_FILE = "demo_prices.csv"
DEMO_META_FILE = "demo_meta.json"


def _default_demo_dir() -> Path:
    """The default demo folder: ``data/demo`` inside the project, wherever you run from."""
    return Path(__file__).resolve().parent.parent / "data" / "demo"


@dataclass(frozen=True, eq=False)
class DemoData:
    """Everything the dashboard needs to show the demo portfolio.

    prices     cleaned adjusted closes, exactly the shape ``load_prices`` returns: a
               DatetimeIndex named "Date", float64, one column per holding (in the order of
               the config) followed by the benchmark column
    holdings   ticker -> weight, as written in the config (the weights are NOT checked here;
               ``risk.portfolio_value`` validates them when they are used)
    benchmark  the benchmark ticker, for example "SPY"
    name       a display name for the portfolio
    as_of      the last date in the data
    start      the first date in the data
    source     where the prices came from, as text for the page footer

    Frozen (you cannot assign to its fields). ``eq=False`` keeps the default "same object"
    comparison, because comparing two DataFrames with ``==`` does not give a yes/no answer.
    """

    prices: pd.DataFrame
    holdings: dict[str, float]
    benchmark: str
    name: str
    as_of: pd.Timestamp
    start: pd.Timestamp
    source: str


def _read_json_object(path: Path) -> dict:
    """Read a JSON file that must contain an object. Any problem becomes a clear ValueError."""
    hint = "Run `python scripts/refresh_demo_data.py` to rebuild the demo files."
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"The demo file {path.name} is missing (looked in {path.parent}). {hint}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"The demo file {path.name} cannot be read ({exc}).") from exc
    try:
        content = json.loads(text)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        raise ValueError(f"The demo file {path.name} is not valid JSON ({exc}).") from exc
    if not isinstance(content, dict):
        raise ValueError(f"The demo file {path.name} must contain a JSON object, not {type(content).__name__}.")
    return content


def _is_number(value) -> bool:
    """True for a real, finite number (not a bool, not NaN, not infinity, not text)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _demo_symbol(value, what: str) -> str:
    """A ticker from the demo config: non-empty text, stripped and upper-case."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{DEMO_CONFIG_FILE}: {what} must be a non-empty ticker symbol, got {value!r}.")
    return value.strip().upper()


def _read_demo_config(folder: Path, require_years: bool = False) -> dict:
    """Read and check demo_portfolio.json. Returns name, holdings, benchmark and years.

    It checks the SHAPE (names are text, weights are numbers), not the weights' math.
    ``years`` is only needed to rebuild the snapshot, so it is optional unless asked for.
    """
    config = _read_json_object(folder / DEMO_CONFIG_FILE)

    name = config.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{DEMO_CONFIG_FILE}: 'name' must be non-empty text.")

    raw_holdings = config.get("holdings")
    if not isinstance(raw_holdings, dict) or not raw_holdings:
        raise ValueError(f"{DEMO_CONFIG_FILE}: 'holdings' must be an object like {{\"AAPL\": 0.5, ...}}.")
    holdings: dict[str, float] = {}
    for raw_symbol, weight in raw_holdings.items():
        symbol = _demo_symbol(raw_symbol, "a holding name")
        if symbol in holdings:
            raise ValueError(f"{DEMO_CONFIG_FILE}: holding {symbol} appears twice.")
        if not _is_number(weight):
            raise ValueError(f"{DEMO_CONFIG_FILE}: the weight of {symbol} must be a number, got {weight!r}.")
        holdings[symbol] = float(weight)

    benchmark = _demo_symbol(config.get("benchmark"), "'benchmark'")
    if benchmark in holdings:
        raise ValueError(f"{DEMO_CONFIG_FILE}: the benchmark {benchmark} must not also be a holding.")

    years = config.get("years")
    if years is None:
        if require_years:
            raise ValueError(f"{DEMO_CONFIG_FILE}: 'years' (how much history to keep) is required.")
    elif not _is_number(years) or years <= 0:
        raise ValueError(f"{DEMO_CONFIG_FILE}: 'years' must be a number above zero, got {years!r}.")

    return {"name": name.strip(), "holdings": holdings, "benchmark": benchmark, "years": years}


def _read_demo_prices(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    """Read demo_prices.csv and check it holds exactly the expected tickers, in that order."""
    hint = "Run `python scripts/refresh_demo_data.py` to rebuild the demo files."
    try:
        # round_trip: the exact same reader setting the cache uses, so prices are bit-exact.
        raw = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
        prices = clean_prices(raw)
    except FileNotFoundError:
        raise ValueError(f"The demo file {path.name} is missing (looked in {path.parent}). {hint}") from None
    except (OSError, ValueError, TypeError, KeyError) as exc:  # includes pandas' parser errors
        raise ValueError(f"The demo file {path.name} cannot be read as a price table ({exc}).") from exc

    if prices.empty:
        raise ValueError(f"The demo file {path.name} contains no price rows.")
    missing = [c for c in expected_columns if c not in prices.columns]
    unexpected = [c for c in prices.columns if c not in expected_columns]
    if missing or unexpected:
        raise ValueError(
            f"The demo file {path.name} does not match {DEMO_CONFIG_FILE}: "
            f"missing columns {missing or 'none'}, unexpected columns {unexpected or 'none'}. {hint}"
        )
    prices = prices.reindex(columns=expected_columns)  # config order, whatever the file's order
    empty = [c for c in prices.columns if prices[c].isna().all()]
    if empty:
        raise ValueError(f"The demo file {path.name} has no prices at all for: {', '.join(empty)}.")
    return prices


def _check_demo_meta(meta: dict, prices: pd.DataFrame) -> str:
    """Make sure demo_meta.json describes the CSV next to it. Returns the ``source`` text.

    If someone refreshed only one of the files (or edited one by hand), the page would show
    a wrong "data as of" date. A loud error is better than a quietly wrong date.
    """
    hint = "The demo files are out of sync. Run `python scripts/refresh_demo_data.py` again."
    source = meta.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{DEMO_META_FILE}: 'source' must be non-empty text.")

    def day(key: str) -> pd.Timestamp:
        value = meta.get(key)
        try:
            if not isinstance(value, str):
                raise ValueError("not text")
            return pd.Timestamp(value).normalize()
        except (ValueError, TypeError):
            raise ValueError(f"{DEMO_META_FILE}: '{key}' must be a date like '2026-01-31', got {value!r}.") from None

    for key, actual in (("as_of", prices.index.max()), ("start", prices.index.min())):
        if day(key) != actual:
            raise ValueError(
                f"{DEMO_META_FILE} says {key} is {day(key):%Y-%m-%d} but {DEMO_PRICES_FILE} says "
                f"{actual:%Y-%m-%d}. {hint}"
            )
    rows = meta.get("rows")
    if isinstance(rows, bool) or rows != len(prices):
        raise ValueError(f"{DEMO_META_FILE} says {rows!r} rows but {DEMO_PRICES_FILE} has {len(prices)}. {hint}")
    tickers = meta.get("tickers")
    if tickers != list(prices.columns):
        raise ValueError(
            f"{DEMO_META_FILE} lists tickers {tickers!r} but {DEMO_PRICES_FILE} has "
            f"{list(prices.columns)!r}. {hint}"
        )
    return source.strip()


def load_demo(demo_dir=None) -> DemoData:
    """Load the committed demo snapshot. Fully offline: it only reads three small files.

    ``demo_dir`` defaults to ``data/demo`` inside the project, wherever you run from.
    The prices come back in the same shape ``load_prices`` returns (DatetimeIndex named
    "Date", float64), with the holdings in config order and the benchmark as the last column.

    Raises ValueError (with a message that says which file and what is wrong) if a file is
    missing or malformed, if the CSV's columns are not exactly the holdings plus the
    benchmark, or if demo_meta.json does not describe the CSV. It does NOT check that the
    weights add up to 1: whoever uses the weights (``risk.portfolio_value``) does that.
    """
    folder = Path(_default_demo_dir() if demo_dir is None else demo_dir)
    config = _read_demo_config(folder)
    meta = _read_json_object(folder / DEMO_META_FILE)
    holdings, benchmark = config["holdings"], config["benchmark"]

    prices = _read_demo_prices(folder / DEMO_PRICES_FILE, list(holdings) + [benchmark])
    source = _check_demo_meta(meta, prices)

    return DemoData(
        prices=prices,
        holdings=dict(holdings),
        benchmark=benchmark,
        name=config["name"],
        as_of=prices.index.max(),
        start=prices.index.min(),
        source=source,
    )


def _demo_window(years: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) for a snapshot of ``years`` years. ``end`` is New York's today, EXCLUDED.

    Excluding today means the last row is the last COMPLETE trading day, never an unfinished
    day (the same rule ``load_prices`` follows).
    """
    end = _today()
    if float(years).is_integer():
        start = end - pd.DateOffset(years=int(years))  # exactly N calendar years back
    else:
        start = end - pd.Timedelta(days=round(years * 365.25))
    return start.normalize(), end


def refresh_demo_snapshot(demo_dir=None) -> dict:
    """Download fresh prices for the demo portfolio and rewrite the snapshot files.

    Reads ``demo_portfolio.json``, downloads the holdings plus the benchmark (always from
    the network, never from the cache), cleans the table and writes ``demo_prices.csv`` and
    ``demo_meta.json``. Returns the metadata it wrote. If the download fails, nothing is
    written and the old snapshot stays as it was. Run it with scripts/refresh_demo_data.py.
    """
    folder = Path(_default_demo_dir() if demo_dir is None else demo_dir)
    config = _read_demo_config(folder, require_years=True)
    tickers = list(config["holdings"]) + [config["benchmark"]]
    start, end = _demo_window(config["years"])

    prices = clean_prices(download_prices(tickers, start, end))
    if prices.empty:
        raise PriceDataError(f"No price rows for {', '.join(tickers)} between {start:%Y-%m-%d} and {end:%Y-%m-%d}.")
    _require_all_tickers(prices, tickers, _describe_window(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"))
    prices = prices.reindex(columns=tickers)

    meta = {
        "as_of": f"{prices.index.max():%Y-%m-%d}",
        "start": f"{prices.index.min():%Y-%m-%d}",
        "rows": int(len(prices)),
        "tickers": tickers,
        "source": f"Yahoo Finance via yfinance {yf.__version__}, adjusted close",
        "created_at": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _atomic_write(folder / DEMO_PRICES_FILE, prices.to_csv)
    _atomic_write(
        folder / DEMO_META_FILE,
        lambda tmp: tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8"),
    )
    return meta
