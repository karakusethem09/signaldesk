"""SignalDesk request limits: keep the public website from being overloaded or abused.

Why this file exists: on the public site anybody can type tickers and pick dates, and every
"live" request goes to Yahoo Finance, which blocks servers that ask too much. So the public
site limits how many tickers, how long a date range, how many live loads per hour, and how
many downloads run at the same time. On your own computer none of that is needed.

Everything here is plain Python with no network, no files and no Streamlit, so it is easy
to test. The dashboard just calls these functions.

The pieces:
    LimitError         -> the one error type; its message is safe to show to a visitor
    normalize_tickers  -> turn "aapl, msft ,AAPL" into ["AAPL", "MSFT"], or refuse bad symbols
    RequestLimits      -> a bundle of the four limits (None = unlimited)
    PUBLIC_LIMITS / LOCAL_LIMITS / active_limits -> pick the bundle from SIGNALDESK_MODE
    check_request      -> refuse a request that has too many tickers or too long a date range
    earliest_allowed_start -> the oldest start date a given end date allows (for date pickers)
    RateLimiter        -> "at most N events per hour per key" (a sliding window)
    DownloadGate       -> "at most N downloads at the same moment"
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import pandas as pd

__all__ = [
    "LOCAL_LIMITS",
    "PUBLIC_LIMITS",
    "DownloadGate",
    "LimitError",
    "RateLimiter",
    "RequestLimits",
    "active_limits",
    "check_request",
    "earliest_allowed_start",
    "normalize_tickers",
]


class LimitError(ValueError):
    """A request was refused by a limit (or a ticker symbol is not acceptable).

    The message is written for a website visitor: plain words, no internals, nothing
    secret. It is a kind of ValueError, so ``except ValueError`` catches it too.
    """


# --------------------------------------------------------------------------- #
# Ticker symbols
# --------------------------------------------------------------------------- #

# One letter/digit/^ first, then up to 14 of: letters, digits, . - = ^ (15 characters max).
# Allows BRK-B, ^GSPC, 7203.T, EURUSD=X. We use re.fullmatch, never `$`, because `$` would
# also accept a trailing newline.
_TICKER_PATTERN = re.compile(r"[A-Z0-9^][A-Z0-9.\-=^]{0,14}")

_SHOWN_SYMBOL_CHARS = 20  # how much of a bad symbol we echo back in a message


def _shown(symbol: str) -> str:
    """A bad symbol cut to a safe length for an error message (it could be 500 characters)."""
    if len(symbol) <= _SHOWN_SYMBOL_CHARS:
        return symbol
    return symbol[:_SHOWN_SYMBOL_CHARS] + "..."


def normalize_tickers(raw) -> list[str]:
    """Turn what a visitor typed into a clean list of ticker symbols.

    ``raw`` is a comma-separated string ("aapl, msft") or a list of strings. Each symbol is
    stripped and upper-cased, empty ones are dropped, and repeats are removed (the first
    one stays, so the order is kept). Every symbol must look like a real ticker, for example
    ``AAPL``, ``BRK-B``, ``^GSPC``, ``7203.T`` or ``EURUSD=X``.

    Raises LimitError naming the first bad symbol (for example ``AAPL!!``, or ``AA PL``
    with a space inside), or if nothing is left at all.
    """
    if raw is None:
        pieces: list = []
    elif isinstance(raw, str):
        pieces = raw.split(",")
    else:
        try:
            pieces = list(raw)
        except TypeError:
            raise LimitError("Please enter tickers as text, for example: AAPL, MSFT.") from None

    symbols: list[str] = []
    for piece in pieces:
        if not isinstance(piece, str):
            raise LimitError("Please enter tickers as text, for example: AAPL, MSFT.")
        text = piece.strip()
        if not text:
            continue
        # Check ASCII BEFORE upper-casing: .upper() turns some other letters into ASCII
        # ones (for example the dotless i), which would let odd symbols slip through.
        if not text.isascii() or _TICKER_PATTERN.fullmatch(text.upper()) is None:
            raise LimitError(
                f"'{_shown(text)}' is not a valid ticker symbol. Use letters and digits "
                "like AAPL, BRK-B, ^GSPC or 7203.T, up to 15 characters, separated by commas."
            )
        symbol = text.upper()
        if symbol not in symbols:
            symbols.append(symbol)

    if not symbols:
        raise LimitError("Please enter at least one ticker symbol, for example: AAPL, MSFT.")
    return symbols


# --------------------------------------------------------------------------- #
# The limits themselves
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RequestLimits:
    """A bundle of limits. ``None`` means "no limit" for that item.

    max_tickers              most tickers in one request
    max_years                longest date range in one request, in years
    max_live_loads_per_hour  how many live (non-demo) loads one visitor may do per hour
    max_concurrent_downloads how many downloads may run at the same moment (all visitors)
    A limit that is set must be above zero.
    """

    max_tickers: int | None = None
    max_years: float | None = None
    max_live_loads_per_hour: int | None = None
    max_concurrent_downloads: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_tickers", "max_years", "max_live_loads_per_hour", "max_concurrent_downloads"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ValueError(f"{name} must be above zero or None (unlimited), got {value!r}.")


# The public website: strict. 10 tickers, 10 years, 10 live loads per hour, 1 download at a time.
PUBLIC_LIMITS = RequestLimits(10, 10, 10, 1)
# Your own computer: nothing is limited.
LOCAL_LIMITS = RequestLimits(None, None, None, None)

MODE_VARIABLE = "SIGNALDESK_MODE"


def active_limits(env=None) -> RequestLimits:
    """Pick the limits from the environment variable ``SIGNALDESK_MODE``.

    Only the exact value "local" (any capitals) switches the limits off. Anything else,
    including a typo or nothing at all, gives PUBLIC_LIMITS. That is on purpose: if you
    forget to set the variable on the server, the site must stay protected, not open.
    ``env`` lets tests pass a plain dict instead of the real environment.
    """
    environment = os.environ if env is None else env
    mode = environment.get(MODE_VARIABLE)
    if isinstance(mode, str) and mode.strip().lower() == "local":
        return LOCAL_LIMITS
    return PUBLIC_LIMITS


def _as_day(value, name: str) -> pd.Timestamp:
    """A date-like value as a midnight Timestamp with no timezone (the local calendar day)."""
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise LimitError(f"The {name} date is not a valid date.") from None
    if pd.isna(stamp):
        raise LimitError(f"The {name} date is missing.")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _number(value: float) -> str:
    """10.0 -> "10", 2.5 -> "2.5" (for messages)."""
    return f"{value:g}"


def earliest_allowed_start(end, max_years: float) -> pd.Timestamp:
    """The earliest start date allowed for a request that ends (EXCLUDED) at ``end``.

    Years are CALENDAR years counted back from the last day of the request, which is the
    day before ``end`` (``end`` is excluded, exactly as in ``load_prices``). So with a limit
    of 10 years and a last day of 2026-10-20, the earliest start is 2016-10-20: "10 years"
    means from the same day ten years ago. Counting calendar years (not days / 365.25)
    matters because ten calendar years are 3652 or 3653 days depending on leap days, so a
    day-based limit would sometimes refuse an ordinary "last 10 years" choice.
    A limit that is not a whole number of years (2.5) is counted as 365.25 days per year,
    rounded UP to whole days.
    """
    last_day = _as_day(end, "end") - pd.Timedelta(days=1)
    if float(max_years).is_integer():
        return last_day - pd.DateOffset(years=int(max_years))
    return last_day - pd.Timedelta(days=math.ceil(max_years * 365.25))


def check_request(tickers, start, end, limits: RequestLimits) -> None:
    """Refuse a request that is too big. Returns None if it is fine, raises LimitError if not.

    ``tickers`` is what the visitor typed: a comma-separated string or a list of symbols. It
    goes through ``normalize_tickers`` here, so repeats count once and a bad symbol is
    refused the same way everywhere (an already-normalized list works too).
    ``start`` and ``end`` are dates, with ``end`` EXCLUDED, exactly as you would hand them
    to ``load_prices``. The date limit counts calendar years back from the last day of the
    request (see ``earliest_allowed_start``); a start before that is refused.
    A request whose start is not before its end is not a limit problem: ``load_prices``
    rejects it with its own message, so it passes here.
    Limits that are None (unlimited) are not checked at all.
    """
    if limits.max_tickers is not None:
        count = len(normalize_tickers(tickers))
        if count > limits.max_tickers:
            raise LimitError(
                f"Up to {limits.max_tickers} tickers per request on the public site "
                f"(you asked for {count}). Please remove some tickers."
            )

    if limits.max_years is not None:
        start_day = _as_day(start, "start")
        earliest = earliest_allowed_start(end, limits.max_years)
        if start_day < earliest:
            last_day = _as_day(end, "end") - pd.Timedelta(days=1)
            raise LimitError(
                f"Up to {_number(limits.max_years)} years per request on the public site. "
                f"For a range ending on {last_day:%Y-%m-%d} the earliest start date is "
                f"{earliest:%Y-%m-%d} (you chose {start_day:%Y-%m-%d}). "
                "Please choose a later start date."
            )


# --------------------------------------------------------------------------- #
# Rate limiting and download gate
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Allow at most ``max_events`` events per ``window_seconds`` for each key (a sliding window).

    Example: ``RateLimiter(10, 3600)`` lets each key do 10 things in any rolling hour.
    An event counts for exactly ``window_seconds`` after it happened, then it is forgotten.
    Refused attempts are NOT recorded, so asking again and again does not extend the wait.

    - ``max_events=None`` means no limit: ``allow`` always says yes and stores nothing.
    - It is thread-safe: the dashboard serves many visitors from one process.
    - Memory stays small: a key is forgotten as soon as all its events have expired.
    - ``clock`` is a function returning seconds; tests pass a fake one.
    """

    def __init__(self, max_events: int | None, window_seconds: float = 3600, clock=time.monotonic):
        if max_events is not None and not max_events > 0:
            raise ValueError(f"max_events must be above zero or None (unlimited), got {max_events!r}.")
        if not window_seconds > 0:
            raise ValueError(f"window_seconds must be above zero, got {window_seconds!r}.")
        self._max = max_events
        self._window = float(window_seconds)
        self._clock = clock
        self._events: dict = {}  # key -> deque of event times, oldest first
        self._lock = threading.Lock()
        self._sweep_every = self._window / 10  # how often we drop keys nobody uses any more
        self._last_sweep = clock()

    # --- internal helpers (the caller must hold the lock) -------------------

    def _live_events(self, key, now: float) -> deque | None:
        """The key's events that still count, or None. Forgets the key if none are left."""
        events = self._events.get(key)
        if events is None:
            return None
        while events and now - events[0] >= self._window:
            events.popleft()
        if not events:
            del self._events[key]
            return None
        return events

    def _sweep(self, now: float) -> None:
        """Now and then, forget every key whose events have all expired (not just this one)."""
        if now - self._last_sweep < self._sweep_every:
            return
        self._last_sweep = now
        for key in list(self._events):
            self._live_events(key, now)

    # --- public methods -----------------------------------------------------

    def allow(self, key) -> bool:
        """Ask to do one thing. True = go ahead (it is recorded), False = limit reached."""
        if self._max is None:
            return True
        with self._lock:
            now = self._clock()
            self._sweep(now)
            events = self._live_events(key, now)
            if events is not None and len(events) >= self._max:
                return False
            if events is None:
                events = self._events.setdefault(key, deque())
            events.append(now)
            return True

    def remaining(self, key) -> int | None:
        """How many more events this key may do right now. None if there is no limit."""
        if self._max is None:
            return None
        with self._lock:
            now = self._clock()
            self._sweep(now)
            events = self._live_events(key, now)
            return self._max - (len(events) if events is not None else 0)

    def retry_after(self, key) -> float:
        """Seconds until this key may do one more event (0.0 if it may do one right now)."""
        if self._max is None:
            return 0.0
        with self._lock:
            now = self._clock()
            self._sweep(now)
            events = self._live_events(key, now)
            if events is None or len(events) < self._max:
                return 0.0
            return max(0.0, events[0] + self._window - now)

    @property
    def key_count(self) -> int:
        """How many keys are being remembered right now (used by tests to check memory)."""
        with self._lock:
            return len(self._events)


BUSY_MESSAGE = (
    "The data provider is busy serving another request. Please try again in a few seconds."
)


class DownloadGate:
    """Lets at most ``max_concurrent`` downloads run at the same moment.

    Use it like this::

        gate = DownloadGate(1)
        with gate.slot(timeout=5):
            ...download...

    If every slot stays busy for longer than ``timeout`` seconds, ``slot`` raises LimitError
    with a friendly "busy" message. The slot is always given back, even if the download
    fails. ``max_concurrent=None`` means no gate: ``slot`` never waits.
    Create ONE gate for the whole server (all visitors must share it), not one per visitor.
    """

    def __init__(self, max_concurrent: int | None):
        if max_concurrent is not None and not max_concurrent > 0:
            raise ValueError(
                f"max_concurrent must be above zero or None (no gate), got {max_concurrent!r}."
            )
        self._semaphore = None if max_concurrent is None else threading.BoundedSemaphore(max_concurrent)

    @contextmanager
    def slot(self, timeout: float | None = 5.0):
        """Wait up to ``timeout`` seconds (None = forever) for a free slot, then hold it."""
        if self._semaphore is None:
            yield
            return
        if not self._semaphore.acquire(timeout=timeout):
            raise LimitError(BUSY_MESSAGE)
        try:
            yield
        finally:
            self._semaphore.release()
