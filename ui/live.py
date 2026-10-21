"""The live-data pipeline for custom mode: short-term memory, two rate limits, download gate.

Plain Python with no Streamlit, so it is easy to test. ``ui/state.py`` creates ONE of each
shared object (the memory, the two limiters, the gate) per server process and hands them in.

There are TWO hourly limits, and both must have room for a real download:
    - the per-visitor limit, counted per browser session. A page reload is a new session with a
      fresh allowance, so on its own it cannot stop a flood;
    - the site-wide limit, one counter for ALL visitors (the fixed key ``SITE_KEY``). Every live
      load leaves from the one server address that all visitors share, and Yahoo blocks an address
      that asks too much, which would stop the live feature for everybody.

The order of events when a visitor presses "Run analysis":

    1. The request was already checked (``ui/inputs.py``) and passed ``limits.check_request``.
    2. Ask the short-term memory. A hit costs nothing: no download, no budget of either limit.
    3. Ask both limits whether there is room (this only looks, it spends nothing).
    4. Wait for a download slot (``limits.DownloadGate``). A busy server spends nothing either.
    5. Look in the memory once more (another visitor may have loaded the same data while we
       waited), look at both limits once more, spend one unit of EACH, and only now call
       ``data.load_prices``. A refusal by one limit never spends the other's budget.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd

from signaldesk import data, limits

PRICE_CACHE_TTL_SECONDS = 6 * 60 * 60  # the same request is answered from memory for 6 hours
PRICE_CACHE_MAX_ENTRIES = 64  # a small cap, so the memory cannot grow without limit
GATE_TIMEOUT_SECONDS = 20  # a 10-ticker x 10-year download can take a while
RATE_WINDOW_SECONDS = 60 * 60  # "live analyses per hour"
SITE_KEY = "__site__"  # the one key of the site-wide limiter (it counts every visitor together)


class RateLimited(Exception):
    """The visitor has used up their live analyses for now. ``retry_after`` is in seconds."""

    def __init__(self, retry_after: float):
        super().__init__(f"rate limited, retry in {retry_after:.0f} seconds")
        self.retry_after = retry_after


class SiteRateLimited(RateLimited):
    """The WHOLE SITE has used up its live analyses for this hour (not just this visitor)."""


# Spending a unit of both limits must be one step that nobody can slip into. The gate usually
# keeps downloads one at a time, but a gate that allows several would otherwise let two visitors
# both pass the "is there room" look and then spend the last unit twice.
_SPEND_LOCK = threading.Lock()


class PriceCache:
    """A small in-memory cache of price tables, with a time limit and a size limit.

    Thread-safe: many visitors share one cache. Entries older than ``ttl_seconds`` are
    forgotten, and when there are more than ``max_entries`` the oldest ones go first.
    ``clock`` returns seconds; tests pass a fake one.
    """

    def __init__(
        self,
        ttl_seconds: float = PRICE_CACHE_TTL_SECONDS,
        max_entries: int = PRICE_CACHE_MAX_ENTRIES,
        clock=time.monotonic,
    ):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock
        self._items: OrderedDict = OrderedDict()  # key -> (stored_at, table), oldest first
        self._lock = threading.Lock()

    def get(self, key) -> pd.DataFrame | None:
        """The table stored under ``key`` (a copy), or None if it is missing or too old."""
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            stored_at, table = item
            if self._clock() - stored_at >= self._ttl:  # not math: date (how old an entry is, in seconds)
                del self._items[key]
                return None
            return table.copy()

    def put(self, key, table: pd.DataFrame) -> None:
        """Store a copy of ``table``. Expired entries are dropped, then the oldest if still too many."""
        with self._lock:
            now = self._clock()
            for old_key in [k for k, (stored_at, _) in self._items.items() if now - stored_at >= self._ttl]:  # not math: date
                del self._items[old_key]
            self._items.pop(key, None)
            self._items[key] = (now, table.copy())
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


@dataclass(frozen=True)
class LiveLoad:
    """The prices of a live request and where they came from."""

    prices: pd.DataFrame
    from_cache: bool  # True: answered from the short-term memory, nothing was downloaded


def cache_key(request) -> tuple:
    """What makes two requests the same: tickers, dates, and TODAY (New York), so a new day starts fresh."""
    return (tuple(request.tickers), request.start, request.end_exclusive, data.market_today())


def _check_room(visitor: str, limiter: limits.RateLimiter, site_limiter: limits.RateLimiter) -> None:
    """Raise if the site or the visitor has no room left. Only looks: nothing is spent.

    The site is asked first: if the whole site is full, waiting out the visitor's own limit would not help.
    """
    site_wait = site_limiter.retry_after(SITE_KEY)
    if site_wait > 0:
        raise SiteRateLimited(site_wait)
    visitor_wait = limiter.retry_after(visitor)
    if visitor_wait > 0:
        raise RateLimited(visitor_wait)


def _spend_one_of_each(visitor: str, limiter: limits.RateLimiter, site_limiter: limits.RateLimiter) -> None:
    """Look once more and spend one unit of the site's budget AND the visitor's, or neither."""
    with _SPEND_LOCK:
        _check_room(visitor, limiter, site_limiter)  # nothing can change between this look and the spending
        site_limiter.allow(SITE_KEY)
        limiter.allow(visitor)


def load_live_prices(
    request,
    *,
    visitor: str,
    cache: PriceCache,
    limiter: limits.RateLimiter,
    site_limiter: limits.RateLimiter,
    gate: limits.DownloadGate,
    gate_timeout: float = GATE_TIMEOUT_SECONDS,
) -> LiveLoad:
    """Get the prices for a checked request.

    Raises SiteRateLimited or RateLimited (no room), LimitError (the server stayed busy) or a data error.

    Only a real call of ``data.load_prices`` uses up budget, one unit of the visitor's and one of the
    site's. A memory hit, a busy server and a refusal use up none. The call has no ``cache_dir``, so
    the data layer's default folder is used (and pruned) as usual.
    """
    key = cache_key(request)
    hit = cache.get(key)
    if hit is not None:
        return LiveLoad(hit, from_cache=True)

    _check_room(visitor, limiter, site_limiter)

    with gate.slot(timeout=gate_timeout):  # LimitError if the server stays busy: nothing was used up
        hit = cache.get(key)  # someone else may have loaded the same data while we waited
        if hit is not None:
            return LiveLoad(hit, from_cache=True)
        _spend_one_of_each(visitor, limiter, site_limiter)
        prices = data.load_prices(request.tickers, request.start, request.end_exclusive)

    cache.put(key, prices)
    return LiveLoad(prices, from_cache=False)
