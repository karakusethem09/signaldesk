"""Tests for signaldesk/limits.py. Pure Python: no network, no files, no real clock.

Time is faked with a tiny clock object we move by hand, so every expected value below
can be checked with a pencil.
"""

import threading
import time
from dataclasses import FrozenInstanceError, fields
from datetime import date, datetime

import pandas as pd
import pytest

from signaldesk import limits
from signaldesk.limits import (
    LOCAL_LIMITS,
    PUBLIC_LIMITS,
    DownloadGate,
    LimitError,
    RateLimiter,
    RequestLimits,
    active_limits,
    check_request,
    normalize_tickers,
)


class FakeClock:
    """A clock that only moves when we say so."""

    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --------------------------------------------------------------------------- #
# LimitError
# --------------------------------------------------------------------------- #

def test_limit_error_is_a_value_error():
    assert issubclass(LimitError, ValueError)
    with pytest.raises(ValueError):
        raise LimitError("nope")


# --------------------------------------------------------------------------- #
# normalize_tickers
# --------------------------------------------------------------------------- #

def test_normalize_splits_strips_uppercases_and_keeps_order():
    assert normalize_tickers(" aapl,Msft , nvda ") == ["AAPL", "MSFT", "NVDA"]


def test_normalize_drops_empty_pieces_and_repeats_keeping_the_first():
    assert normalize_tickers("AAPL,, ,msft,aapl,MSFT,") == ["AAPL", "MSFT"]


def test_normalize_accepts_a_list_and_a_tuple():
    assert normalize_tickers(["aapl", " msft ", "AAPL", ""]) == ["AAPL", "MSFT"]
    assert normalize_tickers(("brk-b",)) == ["BRK-B"]


@pytest.mark.parametrize(
    "symbol",
    ["AAPL", "BRK-B", "^GSPC", "7203.T", "EURUSD=X", "A", "0700.HK", "^VIX", "BF.B", "A" * 15],
)
def test_normalize_accepts_real_world_symbols(symbol):
    assert normalize_tickers(symbol.lower()) == [symbol]


@pytest.mark.parametrize(
    "bad",
    [
        "AAPL!!",
        "AA PL",  # a space inside
        "AAPL MSFT",  # two symbols with a space instead of a comma
        "A" * 16,  # one character too long
        "A" * 500,
        "-AAPL",  # may not start with - . or =
        ".AAPL",
        "=X",
        "AAPL;DROP",
        "AAPL/../x",
        "AAPL\nMSFT",
        "$AAPL",
        "école",  # non-ASCII letters
        "ＡＡＰＬ",  # full-width AAPL
        "ıbm",  # dotless i: .upper() would turn it into the ASCII I
    ],
)
def test_normalize_rejects_bad_symbols_and_names_them(bad):
    with pytest.raises(LimitError) as caught:
        normalize_tickers(f"AAPL, {bad}")
    shown = bad if len(bad) <= 20 else bad[:20] + "..."
    assert f"'{shown}' is not a valid ticker" in str(caught.value)


def test_normalize_error_message_names_the_symbol():
    with pytest.raises(LimitError, match=r"'AAPL!!' is not a valid ticker"):
        normalize_tickers(["MSFT", "AAPL!!"])


def test_normalize_error_message_shortens_a_very_long_symbol():
    with pytest.raises(LimitError) as caught:
        normalize_tickers("A" * 500)
    assert len(str(caught.value)) < 250  # the 500 characters are not echoed back
    assert "AAAAAAAAAAAAAAAAAAAA..." in str(caught.value)


def test_normalize_rejects_a_trailing_newline_inside_a_symbol():
    # `$` in a regex would let "AAPL\n" through; fullmatch must not.
    with pytest.raises(LimitError):
        normalize_tickers(["AAPL\nX"])


@pytest.mark.parametrize("empty", ["", "   ", ",", " , ,, ", [], [""], ["  ", ""], None])
def test_normalize_rejects_an_empty_result(empty):
    with pytest.raises(LimitError, match="at least one ticker"):
        normalize_tickers(empty)


@pytest.mark.parametrize("not_text", [[1, 2], ["AAPL", None], 42, [["AAPL"]]])
def test_normalize_rejects_things_that_are_not_text(not_text):
    with pytest.raises(LimitError, match="as text"):
        normalize_tickers(not_text)


# --------------------------------------------------------------------------- #
# RequestLimits, presets, active_limits
# --------------------------------------------------------------------------- #

def test_presets_have_the_agreed_values():
    assert PUBLIC_LIMITS == RequestLimits(10, 10, 10, 1, 30)
    assert (
        PUBLIC_LIMITS.max_tickers,
        PUBLIC_LIMITS.max_years,
        PUBLIC_LIMITS.max_live_loads_per_hour,
        PUBLIC_LIMITS.max_concurrent_downloads,
        PUBLIC_LIMITS.max_site_live_loads_per_hour,
    ) == (10, 10, 10, 1, 30)
    assert LOCAL_LIMITS == RequestLimits(None, None, None, None, None)


def test_the_site_wide_live_load_limit_defaults_to_none():
    assert RequestLimits().max_site_live_loads_per_hour is None
    assert RequestLimits(max_tickers=5).max_site_live_loads_per_hour is None
    assert RequestLimits(10, 10, 10, 1).max_site_live_loads_per_hour is None


def test_the_public_preset_caps_the_whole_site_at_30_live_loads_an_hour():
    assert PUBLIC_LIMITS.max_site_live_loads_per_hour == 30
    # the site cap is larger than one visitor's own budget, and it is a separate field
    assert PUBLIC_LIMITS.max_site_live_loads_per_hour > PUBLIC_LIMITS.max_live_loads_per_hour


def test_the_local_preset_has_no_site_wide_limit():
    assert LOCAL_LIMITS.max_site_live_loads_per_hour is None
    assert all(value is None for value in vars(LOCAL_LIMITS).values())  # every limit is off


def test_request_limits_still_accepts_the_old_four_positional_values():
    old_style = RequestLimits(10, 10, 2, 1)  # how code written before the fifth field builds one
    assert (old_style.max_tickers, old_style.max_years) == (10, 10)
    assert (old_style.max_live_loads_per_hour, old_style.max_concurrent_downloads) == (2, 1)
    assert old_style.max_site_live_loads_per_hour is None
    assert old_style != PUBLIC_LIMITS  # the fifth value is what tells them apart


def test_request_limits_fifth_positional_value_is_the_site_wide_limit():
    assert RequestLimits(1, 2, 3, 4, 5).max_site_live_loads_per_hour == 5
    assert RequestLimits(max_site_live_loads_per_hour=7) == RequestLimits(None, None, None, None, 7)
    assert [f.name for f in fields(RequestLimits)] == [
        "max_tickers",
        "max_years",
        "max_live_loads_per_hour",
        "max_concurrent_downloads",
        "max_site_live_loads_per_hour",
    ]


def test_request_limits_is_frozen():
    with pytest.raises(FrozenInstanceError):
        PUBLIC_LIMITS.max_tickers = 99


@pytest.mark.parametrize(
    "field",
    [
        "max_tickers",
        "max_years",
        "max_live_loads_per_hour",
        "max_concurrent_downloads",
        "max_site_live_loads_per_hour",
    ],
)
@pytest.mark.parametrize("bad", [0, -1, float("nan")])
def test_request_limits_rejects_a_set_limit_that_is_not_above_zero(field, bad):
    with pytest.raises(ValueError, match=field):
        RequestLimits(**{field: bad})


def test_request_limits_default_to_unlimited():
    assert RequestLimits() == LOCAL_LIMITS


def test_active_limits_is_public_when_nothing_is_set():
    assert active_limits({}) is PUBLIC_LIMITS


@pytest.mark.parametrize("mode", ["local", "LOCAL", "Local", " local ", "local\n"])
def test_active_limits_local(mode):
    assert active_limits({"SIGNALDESK_MODE": mode}) is LOCAL_LIMITS


@pytest.mark.parametrize(
    "mode", ["public", "PUBLIC", "", " ", "prod", "locall", "loca", "local-ish", "true", "1", "off", None, 0, False]
)
def test_active_limits_anything_else_is_public(mode):
    # Includes typos and empty values: forgetting to set it must never remove the limits.
    assert active_limits({"SIGNALDESK_MODE": mode}) is PUBLIC_LIMITS


def test_active_limits_ignores_other_variables():
    assert active_limits({"MODE": "local", "SIGNALDESK_MOD": "local"}) is PUBLIC_LIMITS


def test_active_limits_reads_the_real_environment_by_default(monkeypatch):
    monkeypatch.delenv("SIGNALDESK_MODE", raising=False)
    assert active_limits() is PUBLIC_LIMITS
    monkeypatch.setenv("SIGNALDESK_MODE", "local")
    assert active_limits() is LOCAL_LIMITS
    monkeypatch.setenv("SIGNALDESK_MODE", "banana")
    assert active_limits() is PUBLIC_LIMITS


# --------------------------------------------------------------------------- #
# check_request
# --------------------------------------------------------------------------- #

def test_check_request_accepts_a_small_request():
    assert check_request(["AAPL", "MSFT"], "2024-01-01", "2026-01-01", PUBLIC_LIMITS) is None


def test_check_request_ticker_count_boundary():
    ten = [f"T{i}" for i in range(10)]
    assert check_request(ten, "2024-01-01", "2025-01-01", PUBLIC_LIMITS) is None  # exactly 10 is fine

    eleven = ten + ["T10"]
    with pytest.raises(LimitError) as caught:
        check_request(eleven, "2024-01-01", "2025-01-01", PUBLIC_LIMITS)
    assert str(caught.value).startswith(
        "Up to 10 tickers per request on the public site (you asked for 11)"
    )


def test_check_request_message_for_twelve_tickers():
    twelve = [f"T{i}" for i in range(12)]
    with pytest.raises(LimitError, match=r"Up to 10 tickers per request on the public site \(you asked for 12\)"):
        check_request(twelve, "2024-01-01", "2025-01-01", PUBLIC_LIMITS)


def test_check_request_counts_each_distinct_symbol_once():
    # 12 entries but only 3 different symbols once case and spaces are ignored.
    repeated = ["AAPL", "aapl", " AAPL ", "MSFT", "msft"] * 3 + ["NVDA", "NVDA"]
    assert check_request(repeated, "2024-01-01", "2025-01-01", PUBLIC_LIMITS) is None


def test_check_request_normalizes_what_the_visitor_typed_itself():
    thirty = ",".join(f"T{i}" for i in range(30))  # one comma-separated string, 30 symbols
    with pytest.raises(LimitError, match=r"Up to 10 tickers per request on the public site \(you asked for 30\)"):
        check_request(thirty, "2025-01-01", "2025-06-01", PUBLIC_LIMITS)

    # case, spaces and repeats are handled by normalize_tickers, so they count once
    assert check_request(" aapl, msft ,AAPL,, Msft", "2025-01-01", "2025-06-01", RequestLimits(max_tickers=2)) is None
    with pytest.raises(LimitError, match=r"you asked for 3"):
        check_request(" aapl, msft ,nvda,AAPL", "2025-01-01", "2025-06-01", RequestLimits(max_tickers=2))


def test_check_request_still_accepts_an_already_normalized_list():
    ready = limits.normalize_tickers("aapl, msft, nvda")
    assert check_request(ready, "2025-01-01", "2025-06-01", RequestLimits(max_tickers=3)) is None
    with pytest.raises(LimitError, match=r"you asked for 3"):
        check_request(ready, "2025-01-01", "2025-06-01", RequestLimits(max_tickers=2))


def test_check_request_refuses_a_bad_symbol_the_same_way_normalize_does():
    with pytest.raises(LimitError, match=r"'AAPL!!' is not a valid ticker symbol"):
        check_request("AAPL, AAPL!!", "2025-01-01", "2025-06-01", PUBLIC_LIMITS)
    with pytest.raises(LimitError, match="at least one ticker"):
        check_request(" , ", "2025-01-01", "2025-06-01", PUBLIC_LIMITS)


def test_check_request_a_bare_string_is_one_ticker_not_letters():
    assert check_request("AAPL", "2024-01-01", "2025-01-01", RequestLimits(max_tickers=1)) is None


# The end passed to check_request is EXCLUDED (like load_prices). A person picks an INCLUSIVE end
# date, and the dashboard adds one day. Years are calendar years back from the last day.

def excl(inclusive_end: str) -> pd.Timestamp:
    """The exclusive end for an inclusive end date a person picked."""
    return pd.Timestamp(inclusive_end) + pd.Timedelta(days=1)


def test_check_request_ten_years_with_three_leap_days_is_allowed():
    # 2016-02-01 to 2026-02-01 is ten calendar years but 3653 days (leap days in 2016, 2020, 2024).
    # A rule of "days / 365.25 <= 10" would refuse this, which is the bug this test guards.
    assert (pd.Timestamp("2026-02-01") - pd.Timestamp("2016-02-01")).days == 3653
    assert check_request(["AAPL"], "2016-02-01", excl("2026-02-01"), PUBLIC_LIMITS) is None


def test_check_request_one_day_longer_than_ten_calendar_years_is_refused():
    with pytest.raises(LimitError) as caught:
        check_request(["AAPL"], "2016-01-31", excl("2026-02-01"), PUBLIC_LIMITS)
    assert str(caught.value) == (
        "Up to 10 years per request on the public site. For a range ending on 2026-02-01 the "
        "earliest start date is 2016-02-01 (you chose 2016-01-31). Please choose a later start date."
    )


def test_check_request_a_person_picking_ten_years_back_to_the_same_day_is_allowed():
    assert check_request(["AAPL"], "2016-10-20", excl("2026-10-20"), PUBLIC_LIMITS) is None
    with pytest.raises(LimitError, match=r"earliest start date is 2016-10-20 \(you chose 2016-10-19"):
        check_request(["AAPL"], "2016-10-19", excl("2026-10-20"), PUBLIC_LIMITS)


def test_check_request_a_10y_preset_built_from_the_exclusive_end_is_allowed():
    end = pd.Timestamp("2026-10-20")  # exclusive: what is handed to load_prices
    assert check_request(["AAPL"], end - pd.DateOffset(years=10), end, PUBLIC_LIMITS) is None


def test_check_request_every_ten_year_window_is_allowed_on_every_day_of_a_decade():
    # The old rule refused 100% of these. Check every end date from 2024 to 2033 (covers every
    # position in the 4-year leap cycle, including 29 February and the days around it).
    days_checked = 0
    for end in pd.date_range("2024-01-01", "2033-12-31"):  # exclusive ends
        last_day = end - pd.Timedelta(days=1)
        from_exclusive_end = end - pd.DateOffset(years=10)  # a "10Y" preset (app convention)
        from_last_day = last_day - pd.DateOffset(years=10)  # "same day ten years ago", inclusive

        assert check_request(["AAPL"], from_exclusive_end, end, PUBLIC_LIMITS) is None, end
        assert check_request(["AAPL"], from_last_day, end, PUBLIC_LIMITS) is None, end

        # The limit is exactly the inclusive-end window: one day earlier is refused.
        with pytest.raises(LimitError):
            check_request(["AAPL"], from_last_day - pd.Timedelta(days=1), end, PUBLIC_LIMITS)
        # The preset starts up to two days after the exact limit (one day because the end is
        # excluded, one more when the ten years contain a 29 February), so 3 days too early
        # is refused on every day.
        with pytest.raises(LimitError):
            check_request(["AAPL"], from_exclusive_end - pd.Timedelta(days=3), end, PUBLIC_LIMITS)
        days_checked += 1

    assert days_checked == 3653  # the loop really ran for every day (2024-01-01 .. 2033-12-31)


def test_check_request_leap_day_end_uses_the_calendar_clip():
    # Ten years before 29 February 2028 is 28 February 2018 (2018 has no 29 February).
    end = excl("2028-02-29")
    assert check_request(["AAPL"], "2018-02-28", end, PUBLIC_LIMITS) is None
    with pytest.raises(LimitError, match="earliest start date is 2018-02-28"):
        check_request(["AAPL"], "2018-02-27", end, PUBLIC_LIMITS)


def test_check_request_a_range_shorter_than_the_limit_is_always_fine():
    assert check_request(["AAPL"], "2026-01-01", excl("2026-06-30"), PUBLIC_LIMITS) is None
    assert check_request(["AAPL"], "2026-06-30", excl("2026-06-30"), PUBLIC_LIMITS) is None  # a single day


def test_check_request_twelve_years_is_refused_with_a_friendly_message():
    with pytest.raises(LimitError) as caught:
        check_request(["AAPL"], "2014-01-01", "2026-01-01", PUBLIC_LIMITS)  # about 12 years
    assert str(caught.value) == (
        "Up to 10 years per request on the public site. For a range ending on 2025-12-31 the "
        "earliest start date is 2015-12-31 (you chose 2014-01-01). Please choose a later start date."
    )


def test_earliest_allowed_start_whole_years_use_the_calendar():
    assert limits.earliest_allowed_start("2026-10-21", 10) == pd.Timestamp("2016-10-20")
    assert limits.earliest_allowed_start("2026-10-21", 10.0) == pd.Timestamp("2016-10-20")
    assert limits.earliest_allowed_start("2026-10-21", 1) == pd.Timestamp("2025-10-20")
    assert limits.earliest_allowed_start("2028-03-01", 10) == pd.Timestamp("2018-02-28")  # last day is 29 Feb
    assert limits.earliest_allowed_start("2026-10-21", 10).tzinfo is None


def test_earliest_allowed_start_fractional_years_count_365_25_days_rounded_up():
    # last day 2026-01-01; 2.5 * 365.25 = 913.125 -> 914 days back
    assert limits.earliest_allowed_start("2026-01-02", 2.5) == pd.Timestamp("2026-01-01") - pd.Timedelta(days=914)
    assert pd.Timestamp("2026-01-01") - pd.Timedelta(days=914) == pd.Timestamp("2023-07-02")
    with pytest.raises(LimitError, match=r"Up to 2\.5 years.*earliest start date is 2023-07-02"):
        check_request(["AAPL"], "2023-07-01", "2026-01-02", RequestLimits(max_years=2.5))  # 915 days back
    assert check_request(["AAPL"], "2023-07-02", "2026-01-02", RequestLimits(max_years=2.5)) is None  # 914


def test_check_request_checks_tickers_before_years():
    with pytest.raises(LimitError, match="tickers"):
        check_request([f"T{i}" for i in range(11)], "2000-01-01", "2026-01-01", PUBLIC_LIMITS)


@pytest.mark.parametrize(
    "start, end",
    [
        ("2020-01-01", "2024-01-01"),
        (date(2020, 1, 1), date(2024, 1, 1)),
        (datetime(2020, 1, 1, 9, 30), datetime(2024, 1, 1, 16, 0)),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01")),
        (pd.Timestamp("2020-01-01", tz="Asia/Tokyo"), pd.Timestamp("2024-01-01", tz="Asia/Tokyo")),
    ],
)
def test_check_request_accepts_the_usual_date_types(start, end):
    limits_two_years = RequestLimits(max_years=2)
    # last day 2023-12-31, so two years back is 2021-12-31; 2020-01-01 is earlier than that
    with pytest.raises(LimitError, match=r"earliest start date is 2021-12-31 \(you chose 2020-01-01\)"):
        check_request(["AAPL"], start, end, limits_two_years)


def test_check_request_shows_a_fractional_limit_without_trailing_zeros():
    with pytest.raises(LimitError, match=r"Up to 2\.5 years"):
        check_request(["AAPL"], "2020-01-01", "2024-01-01", RequestLimits(max_years=2.5))


def test_check_request_local_limits_allow_anything():
    huge = [f"T{i}" for i in range(500)]
    assert check_request(huge, "1990-01-01", "2026-01-01", LOCAL_LIMITS) is None


def test_check_request_only_checks_the_limits_that_are_set():
    only_tickers = RequestLimits(max_tickers=2)
    assert check_request(["A", "B"], "1990-01-01", "2026-01-01", only_tickers) is None  # years unlimited
    only_years = RequestLimits(max_years=1)
    assert check_request([f"T{i}" for i in range(50)], "2025-01-01", "2025-06-01", only_years) is None


def test_check_request_leaves_a_backwards_range_to_load_prices():
    # start after end is not a limit problem; load_prices gives its own message for it.
    assert check_request(["AAPL"], "2026-01-01", "2020-01-01", PUBLIC_LIMITS) is None


@pytest.mark.parametrize("bad_date", ["not-a-date", None, ""])
def test_check_request_bad_dates_are_a_limit_error_with_a_plain_message(bad_date):
    with pytest.raises(LimitError, match="date"):
        check_request(["AAPL"], bad_date, "2026-01-01", PUBLIC_LIMITS)


# --------------------------------------------------------------------------- #
# RateLimiter
# --------------------------------------------------------------------------- #

def make_limiter(max_events=3, window=3600, clock=None):
    clock = clock or FakeClock()
    return RateLimiter(max_events, window, clock=clock), clock


def test_rate_limiter_allows_three_refuses_the_fourth_then_allows_after_an_hour():
    limiter, clock = make_limiter(3, 3600)

    assert [limiter.allow("a") for _ in range(4)] == [True, True, True, False]

    clock.advance(3599)  # one second too early: the first event is still inside the window
    assert limiter.allow("a") is False

    clock.advance(1)  # exactly 3600 s after the first three events
    assert limiter.allow("a") is True


def test_rate_limiter_keys_are_independent():
    limiter, _ = make_limiter(1)
    assert limiter.allow("alice") is True
    assert limiter.allow("alice") is False
    assert limiter.allow("bob") is True  # bob has his own allowance


def test_rate_limiter_window_slides_instead_of_resetting_all_at_once():
    limiter, clock = make_limiter(2, 100)
    assert limiter.allow("k")  # t=0
    clock.advance(60)
    assert limiter.allow("k")  # t=60
    assert limiter.allow("k") is False  # full

    clock.advance(40)  # t=100: the t=0 event expires, the t=60 one is still counted
    assert limiter.allow("k") is True  # slot freed by the first event
    assert limiter.allow("k") is False  # t=60 and t=100 are in the window

    clock.advance(60)  # t=160: the t=60 event expires
    assert limiter.allow("k") is True


def test_rate_limiter_refused_attempts_are_not_recorded():
    limiter, clock = make_limiter(1, 100)
    assert limiter.allow("k")
    for _ in range(50):  # hammering does not push the release time further away
        clock.advance(1)
        assert limiter.allow("k") is False
    clock.advance(50)  # t=100 in total since the one recorded event
    assert limiter.allow("k") is True


def test_rate_limiter_remaining_counts_down_and_back_up():
    limiter, clock = make_limiter(3, 3600)
    assert limiter.remaining("a") == 3  # a key we have never seen
    limiter.allow("a")
    assert limiter.remaining("a") == 2
    limiter.allow("a")
    limiter.allow("a")
    assert limiter.remaining("a") == 0
    assert limiter.allow("a") is False
    assert limiter.remaining("a") == 0  # a refused attempt does not make it negative
    clock.advance(3600)
    assert limiter.remaining("a") == 3


def test_rate_limiter_retry_after():
    limiter, clock = make_limiter(2, 3600)
    assert limiter.retry_after("a") == 0.0  # nothing used yet
    limiter.allow("a")  # t=0
    clock.advance(1000)
    limiter.allow("a")  # t=1000
    assert limiter.retry_after("a") == pytest.approx(2600.0)  # until the t=0 event expires

    clock.advance(600)  # t=1600
    assert limiter.retry_after("a") == pytest.approx(2000.0)

    clock.advance(2000)  # t=3600: the first event just expired
    assert limiter.retry_after("a") == 0.0
    assert limiter.allow("a") is True


def test_rate_limiter_retry_after_is_zero_while_slots_are_free():
    limiter, _ = make_limiter(3)
    limiter.allow("a")
    limiter.allow("a")
    assert limiter.retry_after("a") == 0.0  # one slot is still free


def test_rate_limiter_with_no_max_always_allows_and_stores_nothing():
    limiter, clock = make_limiter(None)
    assert all(limiter.allow("a") for _ in range(1000))
    assert limiter.remaining("a") is None
    assert limiter.retry_after("a") == 0.0
    assert limiter.key_count == 0


def test_rate_limiter_queries_do_not_create_keys():
    limiter, _ = make_limiter(3)
    limiter.remaining("ghost")
    limiter.retry_after("ghost")
    assert limiter.key_count == 0


def test_rate_limiter_forgets_a_key_once_all_its_events_expired():
    limiter, clock = make_limiter(3, 3600)
    limiter.allow("a")
    assert limiter.key_count == 1
    clock.advance(3600)
    assert limiter.remaining("a") == 3  # looking at "a" cleans it up
    assert limiter.key_count == 0


def test_rate_limiter_memory_stays_bounded_for_keys_that_never_come_back():
    limiter, clock = make_limiter(3, 3600)
    for visitor in range(100):  # 100 different one-time visitors
        limiter.allow(f"visitor-{visitor}")
    assert limiter.key_count == 100

    clock.advance(3600)  # an hour later all of them have expired...
    limiter.allow("someone-new")  # ...and any later call cleans the lot, not just its own key
    assert limiter.key_count == 1


def test_rate_limiter_keeps_keys_that_still_have_live_events_when_cleaning():
    limiter, clock = make_limiter(3, 3600)
    limiter.allow("old")  # t=0
    clock.advance(3000)
    limiter.allow("recent")  # t=3000
    clock.advance(700)  # t=3700: "old" expired, "recent" not yet
    limiter.allow("trigger")
    assert limiter.key_count == 2  # recent + trigger
    assert limiter.remaining("recent") == 2


def test_rate_limiter_works_with_the_real_clock():
    limiter = RateLimiter(2, 3600)  # default clock: time.monotonic
    assert limiter.allow("a") and limiter.allow("a")
    assert limiter.allow("a") is False
    assert 3500 < limiter.retry_after("a") <= 3600


def test_rate_limiter_default_clock_is_monotonic():
    assert RateLimiter.__init__.__defaults__[1] is time.monotonic


@pytest.mark.parametrize("bad_max", [0, -1])
def test_rate_limiter_rejects_a_max_that_is_not_above_zero(bad_max):
    with pytest.raises(ValueError, match="max_events"):
        RateLimiter(bad_max)


@pytest.mark.parametrize("bad_window", [0, -5])
def test_rate_limiter_rejects_a_window_that_is_not_above_zero(bad_window):
    with pytest.raises(ValueError, match="window_seconds"):
        RateLimiter(3, bad_window)


def test_rate_limiter_is_thread_safe_exactly_max_events_get_through():
    # 8 threads x 10 tries for ONE key with a limit of 5: exactly 5 may succeed.
    limiter = RateLimiter(5, 3600)

    # Make a race certain: every look at the key's events pauses for 1 ms. Without the lock all
    # 8 threads read "0 events so far" at the same moment and all 8 are let through.
    real_live_events = limiter._live_events

    def slow_live_events(key, now):
        events = real_live_events(key, now)
        time.sleep(0.001)
        return events

    limiter._live_events = slow_live_events

    start_together = threading.Barrier(8)
    wins = []

    def hammer():
        start_together.wait()
        wins.append(sum(1 for _ in range(10) if limiter.allow("shared")))

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(wins) == 5
    assert limiter.remaining("shared") == 0


def test_rate_limiter_is_thread_safe_across_many_keys():
    # Each of 8 keys has its own limit of 3; 8 threads each hammer every key.
    limiter = RateLimiter(3, 3600)
    real_live_events = limiter._live_events

    def slow_live_events(key, now):
        events = real_live_events(key, now)
        time.sleep(0.0005)
        return events

    limiter._live_events = slow_live_events
    start_together = threading.Barrier(8)
    wins = []

    def hammer():
        start_together.wait()
        wins.append(sum(1 for key in range(8) for _ in range(6) if limiter.allow(f"k{key}")))

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(wins) == 8 * 3  # every key got exactly its 3


# --------------------------------------------------------------------------- #
# DownloadGate
# --------------------------------------------------------------------------- #

BUSY = "The data provider is busy serving another request. Please try again in a few seconds."


def test_gate_lets_one_download_in_and_gives_the_slot_back():
    gate = DownloadGate(1)
    with gate.slot(timeout=0.1):
        pass
    with gate.slot(timeout=0.1):  # the slot was released, so this works again
        pass


def test_gate_second_download_gets_the_friendly_busy_error():
    gate = DownloadGate(1)
    with gate.slot(timeout=0.1):
        with pytest.raises(LimitError) as caught:
            with gate.slot(timeout=0.05):
                raise AssertionError("must not get a slot while the first one is held")
    assert str(caught.value) == BUSY


def test_gate_releases_the_slot_when_the_download_fails():
    gate = DownloadGate(1)
    with pytest.raises(RuntimeError, match="download blew up"):
        with gate.slot(timeout=0.1):
            raise RuntimeError("download blew up")
    with gate.slot(timeout=0.1):  # not stuck
        pass


def test_gate_a_failed_wait_does_not_release_someone_elses_slot():
    # If a waiter that timed out released a slot it never held, a second download could sneak in.
    gate = DownloadGate(1)
    with gate.slot(timeout=0.1):
        with pytest.raises(LimitError):
            with gate.slot(timeout=0.01):
                pass
        with pytest.raises(LimitError):  # still full: no slot was created by the failed wait
            with gate.slot(timeout=0.01):
                pass


def test_gate_with_two_slots_allows_exactly_two_at_once():
    gate = DownloadGate(2)
    with gate.slot(timeout=0.1):
        with gate.slot(timeout=0.1):
            with pytest.raises(LimitError):
                with gate.slot(timeout=0.01):
                    pass


def test_gate_waits_for_a_slot_that_frees_up_in_time():
    gate = DownloadGate(1)
    holding = threading.Event()
    release = threading.Event()

    def hold():
        with gate.slot(timeout=1):
            holding.set()
            release.wait(2)

    holder = threading.Thread(target=hold)
    holder.start()
    assert holding.wait(2)
    threading.Timer(0.05, release.set).start()  # the holder lets go shortly
    with gate.slot(timeout=2):  # waits for it instead of failing
        pass
    holder.join()


def test_gate_none_means_no_gate():
    gate = DownloadGate(None)
    with gate.slot(timeout=0):
        with gate.slot(timeout=0):
            with gate.slot(timeout=0):
                pass  # any number at once


def test_gate_never_runs_more_than_max_concurrent_downloads_at_once():
    gate = DownloadGate(2)
    lock = threading.Lock()
    state = {"running": 0, "peak": 0, "done": 0, "busy": 0}

    def work():
        try:
            with gate.slot(timeout=5):
                with lock:
                    state["running"] += 1
                    state["peak"] = max(state["peak"], state["running"])
                time.sleep(0.01)
                with lock:
                    state["running"] -= 1
                    state["done"] += 1
        except LimitError:
            with lock:
                state["busy"] += 1

    threads = [threading.Thread(target=work) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["peak"] == 2  # both slots used, never a third
    assert state["done"] == 12 and state["busy"] == 0


@pytest.mark.parametrize("bad", [0, -1])
def test_gate_rejects_a_max_that_is_not_above_zero(bad):
    with pytest.raises(ValueError, match="max_concurrent"):
        DownloadGate(bad)


def test_module_exports_are_importable():
    for name in limits.__all__:
        assert hasattr(limits, name)
