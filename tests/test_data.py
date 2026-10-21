"""Tests for signaldesk/data.py. None of these tests use the network.

How we avoid the network: ``load_prices`` calls ``download_prices``, and
``download_prices`` calls ``yf.download``. In the tests we use pytest's
``monkeypatch`` to swap those for small fake functions that return hand-made
tables (and count how often they were called).
"""

import datetime
import os
import socket
import sqlite3
import threading
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signaldesk import data
from signaldesk.data import (
    PriceDataError,
    clean_prices,
    download_prices,
    find_gaps,
    find_missing_days,
    load_prices,
)


# --------------------------------------------------------------------------- #
# Helpers: small hand-made data
# --------------------------------------------------------------------------- #

def make_frame(rows: dict, columns=("AAPL", "MSFT")) -> pd.DataFrame:
    """Build a price table from {"2026-01-02": [aapl, msft], ...}."""
    return pd.DataFrame(
        list(rows.values()),
        index=pd.to_datetime(list(rows.keys())),
        columns=list(columns),
    )


def messy_prices() -> pd.DataFrame:
    """Unsorted, with a duplicate date, an all-NaN row, and a partial-NaN row."""
    # A plain list (not a dict) because a dict cannot hold the same date twice.
    dates = ["2026-01-06", "2026-01-02", "2026-01-05", "2026-01-07", "2026-01-02"]
    values = [
        [102.123456789012, 205.5],
        [100.1, 200.2],  # first copy of 2026-01-02: must lose to the later copy
        [np.nan, np.nan],  # all missing -> must be dropped
        [np.nan, 207.7],  # partial -> must be kept
        [100.15, 200.25],  # second (later) copy of 2026-01-02
    ]
    return pd.DataFrame(values, index=pd.to_datetime(dates), columns=["AAPL", "MSFT"])


class FakeDownloader:
    """Stands in for download_prices. Records every call so tests can count them."""

    def __init__(self, frame_factory=messy_prices):
        self.calls = []
        self.frame_factory = frame_factory

    def __call__(self, tickers, start, end):
        self.calls.append((list(tickers), start, end))
        frame = self.frame_factory()
        # Pretend we downloaded exactly the tickers requested (in the requested order).
        return frame.reindex(columns=list(tickers))


REAL_TODAY_FUNCTION = data._today  # the real one, captured before any test replaces it
TODAY = pd.Timestamp("2026-06-15")  # a Monday


@pytest.fixture(autouse=True)
def pinned_today(monkeypatch):
    """Freeze "today" for every test, so no test depends on the real clock.

    load_prices limits `end` to today (see the D1 tests below), so without this a test
    would behave differently depending on the day it is run.
    """
    monkeypatch.setattr(data, "_today", lambda: TODAY)
    return TODAY


@pytest.fixture
def downloader(monkeypatch):
    fake = FakeDownloader()
    monkeypatch.setattr(data, "download_prices", fake)
    return fake


def yfinance_style(tickers, rows, with_extra_fields=True) -> pd.DataFrame:
    """Imitate the raw table yf.download returns: MultiIndex columns (field, ticker)."""
    index = pd.DatetimeIndex(pd.to_datetime(list(rows.keys())), name="Date")
    pieces = {}
    for field in ("Close", "High", "Low", "Open", "Volume") if with_extra_fields else ("Close",):
        for i, ticker in enumerate(tickers):
            values = [row[i] for row in rows.values()]
            # Make non-Close fields obviously different, so a mix-up would be caught.
            pieces[(field, ticker)] = values if field == "Close" else [v * 1000 for v in values]
    frame = pd.DataFrame(pieces, index=index)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["Price", "Ticker"])
    return frame


# --------------------------------------------------------------------------- #
# clean_prices
# --------------------------------------------------------------------------- #

def test_clean_sorts_dates_oldest_first():
    cleaned = clean_prices(messy_prices())
    assert cleaned.index.is_monotonic_increasing
    assert list(cleaned.index.strftime("%Y-%m-%d")) == ["2026-01-02", "2026-01-06", "2026-01-07"]


def test_clean_drops_duplicate_dates_keeping_the_last_one():
    raw = pd.DataFrame(
        {"AAPL": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-02"]),
    )
    cleaned = clean_prices(raw)
    assert len(cleaned) == 2
    assert cleaned.loc["2026-01-02", "AAPL"] == 3.0  # the later copy of the duplicate


def test_clean_duplicate_date_in_messy_data_keeps_the_later_copy():
    cleaned = clean_prices(messy_prices())
    assert cleaned.loc["2026-01-02", "AAPL"] == 100.15
    assert cleaned.loc["2026-01-02", "MSFT"] == 200.25


def test_clean_drops_rows_where_all_tickers_are_missing():
    cleaned = clean_prices(messy_prices())
    assert pd.Timestamp("2026-01-05") not in cleaned.index


def test_clean_keeps_partial_nan_rows_and_does_not_fill_them():
    cleaned = clean_prices(messy_prices())
    row = cleaned.loc["2026-01-07"]
    assert np.isnan(row["AAPL"])  # still missing: we never fill
    assert row["MSFT"] == 207.7


def test_clean_empty_duplicate_does_not_hide_a_real_price():
    # The later copy of 2026-01-02 is all-NaN. The earlier one has real prices and must survive.
    raw = pd.DataFrame(
        {"AAPL": [10.0, np.nan], "MSFT": [20.0, np.nan]},
        index=pd.to_datetime(["2026-01-02", "2026-01-02"]),
    )
    cleaned = clean_prices(raw)
    assert len(cleaned) == 1
    assert cleaned.loc["2026-01-02", "AAPL"] == 10.0


def test_clean_output_format():
    raw = pd.DataFrame(
        {"AAPL": [1, 2], "MSFT": [3, 4]},  # integer values
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]).tz_localize("America/New_York"),
    )
    raw.columns.name = "Ticker"
    cleaned = clean_prices(raw)
    expected_index = pd.DatetimeIndex(["2026-01-02", "2026-01-05"], name="Date").astype(
        "datetime64[ns]"
    )
    pd.testing.assert_index_equal(cleaned.index, expected_index, exact=True)  # exact dates
    assert cleaned.index.tz is None
    assert (cleaned.dtypes == "float64").all()
    assert cleaned.columns.name is None
    assert list(cleaned.columns) == ["AAPL", "MSFT"]


@pytest.mark.parametrize(
    "timezone",
    [
        "America/New_York",  # behind UTC
        "Asia/Tokyo",  # ahead of UTC: local midnight is the PREVIOUS day in UTC
        "Australia/Sydney",
        "UTC",
    ],
)
def test_clean_keeps_the_local_exchange_date_for_timezone_aware_indexes(timezone):
    # An exchange's daily bar is stamped with its LOCAL midnight. Removing the timezone
    # must keep that local date. (Converting to UTC first would turn Tokyo's Monday
    # 2026-01-05 00:00 into Sunday 2026-01-04 15:00, i.e. the wrong day.)
    local_dates = ["2026-01-05", "2026-01-06", "2026-01-07"]  # Mon, Tue, Wed
    raw = pd.DataFrame(
        {"AAPL": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(local_dates).tz_localize(timezone),
    )
    cleaned = clean_prices(raw)

    expected_index = pd.DatetimeIndex(local_dates, name="Date").astype("datetime64[ns]")
    pd.testing.assert_index_equal(cleaned.index, expected_index, exact=True)
    assert cleaned["AAPL"].tolist() == [1.0, 2.0, 3.0]


def test_clean_removes_times_of_day_so_same_day_timestamps_are_duplicates():
    raw = pd.DataFrame(
        {"AAPL": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2026-03-02 00:00", "2026-03-02 16:00", "2026-03-03 09:30"]),
    )
    cleaned = clean_prices(raw)

    expected_index = pd.DatetimeIndex(["2026-03-02", "2026-03-03"], name="Date").astype(
        "datetime64[ns]"
    )
    pd.testing.assert_index_equal(cleaned.index, expected_index, exact=True)
    assert cleaned["AAPL"].tolist() == [2.0, 3.0]  # the later 2026-03-02 timestamp wins


def test_clean_accepts_string_dates_in_the_index():
    raw = pd.DataFrame({"AAPL": [2.0, 1.0]}, index=["2026-01-05", "2026-01-02"])
    cleaned = clean_prices(raw)
    assert isinstance(cleaned.index, pd.DatetimeIndex)
    assert cleaned.index[0] == pd.Timestamp("2026-01-02")


def test_clean_does_not_change_its_input():
    raw = messy_prices()
    snapshot = raw.copy()
    clean_prices(raw)
    pd.testing.assert_frame_equal(raw, snapshot)


def test_clean_is_idempotent():
    once = clean_prices(messy_prices())
    pd.testing.assert_frame_equal(clean_prices(once), once, check_exact=True)


def test_clean_handles_an_empty_table():
    cleaned = clean_prices(pd.DataFrame({"AAPL": []}, index=pd.DatetimeIndex([])))
    assert cleaned.empty
    assert list(cleaned.columns) == ["AAPL"]


def test_clean_rejects_bad_input():
    with pytest.raises(TypeError):
        clean_prices([1, 2, 3])
    with pytest.raises(ValueError, match="dates"):
        clean_prices(pd.DataFrame({"AAPL": [1.0, 2.0]}))  # plain 0,1,2... index, not dates


# --------------------------------------------------------------------------- #
# find_gaps
# --------------------------------------------------------------------------- #

def gappy_prices() -> pd.DataFrame:
    """Known gaps: AAPL missing on 3 dates, MSFT on 1, GOOG on none."""
    return pd.DataFrame(
        {
            "AAPL": [np.nan, 11.0, np.nan, np.nan, 14.0],
            "MSFT": [20.0, 21.0, 22.0, np.nan, 24.0],
            "GOOG": [30.0, 31.0, 32.0, 33.0, 34.0],
        },
        index=pd.to_datetime(
            ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        ),
    )


def test_find_gaps_counts_and_dates():
    gaps = find_gaps(gappy_prices())

    assert list(gaps.index) == ["AAPL", "MSFT", "GOOG"]
    assert list(gaps.columns) == ["missing_count", "first_missing", "last_missing"]
    assert gaps["missing_count"].tolist() == [3, 1, 0]

    assert gaps.loc["AAPL", "first_missing"] == pd.Timestamp("2026-01-02")
    assert gaps.loc["AAPL", "last_missing"] == pd.Timestamp("2026-01-07")
    assert gaps.loc["MSFT", "first_missing"] == pd.Timestamp("2026-01-07")
    assert gaps.loc["MSFT", "last_missing"] == pd.Timestamp("2026-01-07")
    assert pd.isna(gaps.loc["GOOG", "first_missing"])
    assert pd.isna(gaps.loc["GOOG", "last_missing"])


def test_find_gaps_dtypes_are_stable():
    for frame in (gappy_prices(), gappy_prices()[["GOOG"]]):  # with and without any gap dates
        gaps = find_gaps(frame)
        assert str(gaps["missing_count"].dtype) == "int64"
        assert str(gaps["first_missing"].dtype) == "datetime64[ns]"
        assert str(gaps["last_missing"].dtype) == "datetime64[ns]"


def test_find_gaps_only_reports_and_never_fills():
    prices = gappy_prices()
    snapshot = prices.copy()
    find_gaps(prices)
    pd.testing.assert_frame_equal(prices, snapshot)  # the input still has its NaNs


def test_find_gaps_on_complete_and_empty_data():
    complete = find_gaps(gappy_prices()[["GOOG"]])
    assert complete["missing_count"].tolist() == [0]

    empty = find_gaps(pd.DataFrame())
    assert empty.empty
    assert list(empty.columns) == ["missing_count", "first_missing", "last_missing"]


# --------------------------------------------------------------------------- #
# find_missing_days: weekdays that are absent from the table
# --------------------------------------------------------------------------- #

def as_dates(*dates: str) -> pd.DatetimeIndex:
    """The expected result type of find_missing_days: ns dates, named "Date"."""
    return pd.DatetimeIndex(list(dates), name="Date").astype("datetime64[ns]")


def test_find_missing_days_sees_a_missing_wednesday_that_find_gaps_cannot():
    # Wednesday 2026-03-04 has no row at all, for the only ticker.
    prices = pd.DataFrame(
        {"AAPL": [1.0, 2.0, 4.0, 5.0]},
        index=pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-05", "2026-03-06"]),
    )
    assert find_gaps(prices)["missing_count"].tolist() == [0]  # NaN cells only: sees nothing

    missing = find_missing_days(prices)

    pd.testing.assert_index_equal(missing, as_dates("2026-03-04"), exact=True)
    assert missing[0].day_name() == "Wednesday"


def test_find_missing_days_covers_a_row_that_clean_prices_dropped():
    # Two tickers, both NaN on Wednesday 2026-03-04: clean_prices drops that row...
    raw = pd.DataFrame(
        {"AAPL": [1.0, 2.0, np.nan, 4.0, 5.0], "MSFT": [10.0, 20.0, np.nan, 40.0, 50.0]},
        index=pd.to_datetime(
            ["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"]
        ),
    )
    assert find_missing_days(raw).empty  # before cleaning the row exists, so nothing is absent

    cleaned = clean_prices(raw)

    assert pd.Timestamp("2026-03-04") not in cleaned.index  # dropped by clean_prices
    # ...and it is not invisible: the dropped weekday is now reported as missing.
    pd.testing.assert_index_equal(find_missing_days(cleaned), as_dates("2026-03-04"), exact=True)


def test_find_missing_days_ignores_weekends_but_reports_every_missing_weekday():
    # Fri 2026-03-06 -> Mon 2026-03-09 spans a weekend only: nothing missing.
    over_weekend = pd.DataFrame(
        {"AAPL": [1.0, 2.0]}, index=pd.to_datetime(["2026-03-06", "2026-03-09"])
    )
    assert find_missing_days(over_weekend).empty

    # Mon 2026-03-02 -> Mon 2026-03-09 with only the two Mondays: Tue-Fri are absent
    # (Sat 7th and Sun 8th are NOT reported).
    sparse = pd.DataFrame(
        {"AAPL": [1.0, 2.0]}, index=pd.to_datetime(["2026-03-02", "2026-03-09"])
    )
    pd.testing.assert_index_equal(
        find_missing_days(sparse),
        as_dates("2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"),
        exact=True,
    )


def test_find_missing_days_edge_cases_and_format():
    complete = pd.DataFrame(
        {"AAPL": [1.0, 2.0, 3.0]}, index=pd.bdate_range("2026-03-02", periods=3)
    )
    for table in (complete, complete.iloc[:1], complete.iloc[:0], pd.DataFrame()):
        missing = find_missing_days(table)  # nothing missing / one row / no rows at all
        pd.testing.assert_index_equal(missing, as_dates(), exact=True)  # empty, same type and name


def test_find_missing_days_with_a_requested_window_reports_missing_edges():
    # We ASKED for 2026-03-30 .. 2026-04-10 (end excluded) but data only covers 04-02 .. 04-08.
    prices = pd.DataFrame(
        {"AAPL": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=pd.to_datetime(
            ["2026-04-02", "2026-04-03", "2026-04-06", "2026-04-07", "2026-04-08"]
        ),
    )

    # Without the window, the old behaviour: only gaps BETWEEN the first and last row.
    assert find_missing_days(prices).empty

    missing = find_missing_days(prices, start="2026-03-30", end="2026-04-10")

    pd.testing.assert_index_equal(
        missing, as_dates("2026-03-30", "2026-03-31", "2026-04-01", "2026-04-09"), exact=True
    )
    assert pd.Timestamp("2026-04-10") not in missing  # end is exclusive, like everywhere else


def test_find_missing_days_window_can_have_only_a_start_or_only_an_end():
    prices = pd.DataFrame(
        {"AAPL": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2026-04-02", "2026-04-03", "2026-04-06"]),
    )
    only_start = find_missing_days(prices, start="2026-03-30")  # the end side uses the table
    pd.testing.assert_index_equal(
        only_start, as_dates("2026-03-30", "2026-03-31", "2026-04-01"), exact=True
    )
    only_end = find_missing_days(prices, end="2026-04-09")  # last day checked is 04-08
    pd.testing.assert_index_equal(only_end, as_dates("2026-04-07", "2026-04-08"), exact=True)


def test_find_missing_days_window_edge_cases():
    prices = pd.DataFrame(
        {"AAPL": [1.0, 2.0, 3.0]}, index=pd.bdate_range("2026-04-06", periods=3)  # Mon-Wed
    )
    # Window exactly matching the data: nothing missing. The end day itself is excluded.
    assert find_missing_days(prices, start="2026-04-06", end="2026-04-09").empty
    # A window narrower than the data ignores the rows outside it.
    assert find_missing_days(prices, start="2026-04-07", end="2026-04-08").empty
    # An empty table with a window: every weekday of the window is missing.
    pd.testing.assert_index_equal(
        find_missing_days(prices.iloc[:0], start="2026-04-06", end="2026-04-08"),
        as_dates("2026-04-06", "2026-04-07"),
        exact=True,
    )
    # Nonsense windows are rejected.
    with pytest.raises(ValueError, match="earlier than"):
        find_missing_days(prices, start="2026-04-08", end="2026-04-08")
    with pytest.raises(ValueError, match="not a valid date"):
        find_missing_days(prices, start="not-a-date")


# --- D13: find_missing_days limits `end` to today, exactly like load_prices -----------

def test_find_missing_days_limits_a_future_end_to_today(pinned_today):
    assert pinned_today == pd.Timestamp("2026-06-15")  # a Monday
    # Data for the weekdays 06-01 .. 06-12, except Wednesday 06-10. The last complete
    # trading day before Monday 06-15 is Friday 06-12.
    dates = pd.bdate_range("2026-06-01", "2026-06-12").difference(
        pd.DatetimeIndex(["2026-06-10"])
    )
    prices = pd.DataFrame({"AAPL": np.arange(len(dates)) + 100.0}, index=dates)

    missing = find_missing_days(prices, start="2026-06-01", end="2026-12-31")  # end in the future

    pd.testing.assert_index_equal(missing, as_dates("2026-06-10"), exact=True)
    assert not (missing >= pinned_today).any()  # nothing on or after today is "missing"

    # Every end from today onwards is the same window, so gives the same answer.
    for same_window_end in ("2026-06-15", "2026-06-16", "2027-01-01"):
        pd.testing.assert_index_equal(
            find_missing_days(prices, start="2026-06-01", end=same_window_end), missing
        )
    # An end in the past is not changed: it still reports the days up to that end.
    pd.testing.assert_index_equal(
        find_missing_days(prices.iloc[:0], start="2026-06-08", end="2026-06-11"),
        as_dates("2026-06-08", "2026-06-09", "2026-06-10"),
        exact=True,
    )


def test_load_prices_then_find_missing_days_with_the_same_dates_finds_nothing_false(
    monkeypatch, tmp_path
):
    # The reviewer's scenario. Clock: 2026-10-20 16:00 UTC (noon in New York), so New
    # York's "today" is 2026-10-20 and the last complete trading day is Monday 10-19.
    monkeypatch.setattr(data, "_today", REAL_TODAY_FUNCTION)  # undo the autouse pin
    monkeypatch.setattr(data, "_now_utc", at_utc("2026-10-20 16:00"))

    def complete_october():  # every weekday 10-01 .. 10-19: 13 rows, nothing missing
        dates = pd.bdate_range("2026-10-01", "2026-10-19")
        return pd.DataFrame({"AAPL": np.arange(len(dates)) + 100.0}, index=dates)

    monkeypatch.setattr(data, "download_prices", FakeDownloader(complete_october))

    prices = load_prices(["AAPL"], "2026-10-01", "2026-12-31", cache_dir=tmp_path)
    assert len(prices) == 13
    assert prices.index.max() == pd.Timestamp("2026-10-19")

    # Same start/end as the load. Without the limit this reported 52 "missing" days
    # (2026-10-20 .. 2026-12-30), none of which can have prices yet.
    missing = find_missing_days(prices, start="2026-10-01", end="2026-12-31")
    assert len(missing) == 0
    assert list(missing) == []


@pytest.mark.parametrize("start", ["2026-06-15", "2026-07-01"])  # today, and the future
def test_find_missing_days_start_on_or_after_today_raises_like_load_prices(start):
    prices = pd.DataFrame({"AAPL": [1.0]}, index=pd.to_datetime(["2026-06-12"]))
    with pytest.raises(ValueError, match=r"earlier than end.*limited to today"):
        find_missing_days(prices, start=start, end="2026-12-31")


def test_find_missing_days_start_not_before_an_unclamped_end_has_no_clamp_note():
    prices = pd.DataFrame({"AAPL": [1.0]}, index=pd.to_datetime(["2026-06-01"]))
    with pytest.raises(ValueError, match="earlier than") as error:
        find_missing_days(prices, start="2026-06-10", end="2026-06-10")
    assert "limited to today" not in str(error.value)


# --- D14: a one-sided window that misses the data entirely is an error ---------------

def april_table() -> pd.DataFrame:
    """Five rows: 2026-04-02 .. 2026-04-08."""
    return pd.DataFrame(
        {"AAPL": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=pd.to_datetime(
            ["2026-04-02", "2026-04-03", "2026-04-06", "2026-04-07", "2026-04-08"]
        ),
    )


def test_find_missing_days_start_after_all_the_data_raises():
    with pytest.raises(ValueError, match=r"start \(2026-05-01\).*last date.*does not overlap"):
        find_missing_days(april_table(), start="2026-05-01")
    with pytest.raises(ValueError, match="does not overlap"):
        find_missing_days(april_table(), start="2026-04-09")  # the day after the last row


def test_find_missing_days_end_before_all_the_data_raises():
    with pytest.raises(ValueError, match=r"end \(2026-03-25\).*first date.*does not overlap"):
        find_missing_days(april_table(), end="2026-03-25")
    with pytest.raises(ValueError, match="does not overlap"):
        find_missing_days(april_table(), end="2026-04-02")  # end is excluded: window stops 04-01


def test_find_missing_days_one_sided_windows_that_just_touch_the_data_are_fine():
    assert find_missing_days(april_table(), start="2026-04-08").empty  # starts on the last row
    assert find_missing_days(april_table(), end="2026-04-03").empty  # window ends on the first row


def test_find_missing_days_two_sided_window_with_no_data_still_reports_the_days():
    # Both sides given: the caller said exactly what they expected, so "no data in the
    # window" is a report (every weekday missing), not an error.
    missing = find_missing_days(april_table(), start="2026-03-02", end="2026-03-06")
    pd.testing.assert_index_equal(
        missing, as_dates("2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"), exact=True
    )


def test_find_missing_days_handles_a_timezone_aware_index_by_keeping_local_dates():
    # Tokyo-midnight stamps for Mon 2026-03-02 and Wed 2026-03-04 (UTC would say Sun and Tue).
    prices = pd.DataFrame(
        {"AAPL": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2026-03-02", "2026-03-04"]).tz_localize("Asia/Tokyo"),
    )
    pd.testing.assert_index_equal(find_missing_days(prices), as_dates("2026-03-03"), exact=True)


def test_find_missing_days_rejects_an_index_of_numbers_instead_of_dates():
    with pytest.raises(ValueError, match="dates"):
        find_missing_days(pd.DataFrame({"AAPL": [1.0, 2.0, 3.0]}))  # index is 0, 1, 2


def test_find_missing_days_ignores_times_of_day_and_does_not_change_its_input():
    prices = pd.DataFrame(
        {"AAPL": [1.0, 2.0]},
        index=pd.to_datetime(["2026-03-02 16:00", "2026-03-04 16:00"]),
    )
    snapshot = prices.copy()
    pd.testing.assert_index_equal(find_missing_days(prices), as_dates("2026-03-03"), exact=True)
    pd.testing.assert_frame_equal(prices, snapshot)


# --------------------------------------------------------------------------- #
# download_prices (yf.download is replaced by a fake)
# --------------------------------------------------------------------------- #

def patch_yf(monkeypatch, result):
    """Replace yf.download with a fake that returns `result` and records its arguments."""
    seen = {}

    def fake_download(tickers, **kwargs):
        seen["tickers"] = tickers
        seen["kwargs"] = kwargs
        return result

    monkeypatch.setattr(data.yf, "download", fake_download)
    return seen


def test_download_returns_close_prices_one_column_per_ticker_in_order(monkeypatch):
    raw = yfinance_style(
        ["AAPL", "MSFT"],  # yfinance sorts columns alphabetically
        {"2026-01-02": [270.0, 470.0], "2026-01-05": [266.5, 469.9]},
    )
    seen = patch_yf(monkeypatch, raw)

    prices = download_prices(["MSFT", "AAPL"], "2026-01-01", "2026-01-10")

    assert list(prices.columns) == ["MSFT", "AAPL"]  # caller's order, not yfinance's
    assert prices.loc["2026-01-02", "AAPL"] == 270.0  # the Close field, not High/Open/...
    assert prices.loc["2026-01-05", "MSFT"] == 469.9
    assert seen["kwargs"]["auto_adjust"] is True  # asks for split/dividend adjusted prices
    assert seen["kwargs"]["start"] == "2026-01-01"
    assert seen["kwargs"]["end"] == "2026-01-10"


def test_download_single_ticker_with_multiindex_columns(monkeypatch):
    # In yfinance 1.7 even ONE ticker comes back with (field, ticker) columns.
    raw = yfinance_style(["AAPL"], {"2026-01-02": [270.0], "2026-01-05": [266.5]})
    patch_yf(monkeypatch, raw)

    prices = download_prices(["AAPL"], "2026-01-01", "2026-01-10")

    assert list(prices.columns) == ["AAPL"]
    assert prices["AAPL"].tolist() == [270.0, 266.5]


def test_download_single_ticker_with_plain_columns(monkeypatch):
    # Older yfinance versions returned flat columns (Open, Close, ...) for one ticker.
    raw = pd.DataFrame(
        {"Open": [1.0, 2.0], "Close": [270.0, 266.5]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )
    patch_yf(monkeypatch, raw)

    prices = download_prices("aapl", "2026-01-01", "2026-01-10")  # a bare string, lower case

    assert list(prices.columns) == ["AAPL"]
    assert prices["AAPL"].tolist() == [270.0, 266.5]


def test_download_unknown_ticker_raises_and_names_it(monkeypatch):
    # yfinance does not raise for a bad ticker; it returns an all-NaN column.
    raw = yfinance_style(
        ["AAPL", "ZZZZ"],
        {"2026-01-02": [270.0, np.nan], "2026-01-05": [266.5, np.nan]},
    )
    patch_yf(monkeypatch, raw)

    with pytest.raises(PriceDataError, match="ZZZZ"):
        download_prices(["AAPL", "ZZZZ"], "2026-01-01", "2026-01-10")


def test_download_ticker_missing_from_result_raises(monkeypatch):
    raw = yfinance_style(["AAPL"], {"2026-01-02": [270.0]})
    patch_yf(monkeypatch, raw)

    with pytest.raises(PriceDataError, match="MSFT"):
        download_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-10")


@pytest.mark.parametrize("empty_result", [pd.DataFrame(), None])
def test_download_no_data_at_all_raises(monkeypatch, empty_result):
    patch_yf(monkeypatch, empty_result)
    with pytest.raises(PriceDataError, match="No price data"):
        download_prices(["AAPL"], "2026-01-01", "2026-01-10")


def test_download_data_without_close_field_raises(monkeypatch):
    raw = pd.DataFrame({"Open": [1.0]}, index=pd.to_datetime(["2026-01-02"]))
    patch_yf(monkeypatch, raw)
    with pytest.raises(PriceDataError, match="Close"):
        download_prices(["AAPL"], "2026-01-01", "2026-01-10")


# --------------------------------------------------------------------------- #
# Input validation (shared by download_prices and load_prices)
# --------------------------------------------------------------------------- #

BAD_INPUTS = [
    ([], "2026-01-01", "2026-02-01", "empty"),
    (None, "2026-01-01", "2026-02-01", "empty"),
    ([""], "2026-01-01", "2026-02-01", "Invalid ticker"),
    ([42], "2026-01-01", "2026-02-01", "Invalid ticker"),
    (["AAPL"], "2026-02-01", "2026-01-01", "earlier than"),  # start after end
    (["AAPL"], "2026-01-01", "2026-01-01", "earlier than"),  # start == end
    # Same calendar day, only the times differ: dates are normalised to midnight, so
    # these are equal and must be rejected (they are NOT "10:00 is before 15:00").
    (["AAPL"], "2026-01-01 10:00", "2026-01-01 15:00", "earlier than"),
    (["AAPL"], "not-a-date", "2026-01-01", "not a valid date"),
    (["AAPL"], "2026-01-01", None, "missing"),
]


@pytest.mark.parametrize("tickers, start, end, message", BAD_INPUTS)
def test_load_prices_rejects_bad_inputs_before_doing_any_work(
    tickers, start, end, message, downloader, tmp_path
):
    with pytest.raises(ValueError, match=message):
        load_prices(tickers, start, end, cache_dir=tmp_path)
    assert downloader.calls == []  # never reached the downloader
    assert list(tmp_path.iterdir()) == []  # and wrote nothing


@pytest.mark.parametrize("tickers, start, end, message", BAD_INPUTS)
def test_download_prices_rejects_bad_inputs_before_touching_the_network(
    tickers, start, end, message, monkeypatch
):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("yf.download was called")

    monkeypatch.setattr(data.yf, "download", must_not_be_called)
    with pytest.raises(ValueError, match=message):
        download_prices(tickers, start, end)


# --------------------------------------------------------------------------- #
# load_prices: caching
# --------------------------------------------------------------------------- #

def test_first_call_downloads_cleans_and_writes_a_cache_file(downloader, tmp_path):
    prices = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)

    assert len(downloader.calls) == 1
    assert downloader.calls[0] == (["AAPL", "MSFT"], "2026-01-01", "2026-03-01")
    # the result is cleaned: sorted, no all-NaN row, no duplicate date
    assert list(prices.index.strftime("%Y-%m-%d")) == ["2026-01-02", "2026-01-06", "2026-01-07"]
    # exactly one cache file, with the deterministic name
    files = list(tmp_path.glob("*.csv"))
    assert [f.name for f in files] == ["prices_AAPL+MSFT_2026-01-01_2026-03-01.csv"]
    assert list(tmp_path.glob("*.tmp")) == []  # no temp file left behind


def test_second_call_reads_the_cache_and_does_not_download(downloader, tmp_path):
    fresh = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    assert len(downloader.calls) == 1

    cached = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)

    assert len(downloader.calls) == 1  # still 1: the downloader was NOT called again
    # identical: same values, index type/name/unit, dtypes and column order
    pd.testing.assert_frame_equal(cached, fresh, check_exact=True)
    assert type(cached.index) is type(fresh.index)
    assert cached.index.dtype == fresh.index.dtype
    assert cached.index.name == fresh.index.name == "Date"
    assert cached.columns.dtype == fresh.columns.dtype
    assert list(cached.columns) == list(fresh.columns)
    assert (cached.dtypes == fresh.dtypes).all()


def test_cached_result_keeps_gaps_and_full_float_precision(downloader, tmp_path):
    fresh = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    cached = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)

    assert np.isnan(cached.loc["2026-01-07", "AAPL"])  # the gap survived the CSV round trip
    assert cached.loc["2026-01-06", "AAPL"] == 102.123456789012 == fresh.loc["2026-01-06", "AAPL"]


def test_cache_round_trip_is_bit_exact_for_realistic_prices(monkeypatch, tmp_path):
    # Regression test found by the live smoke test: pandas' default (fast) CSV reader
    # can be off by one in the last binary digit, e.g. 467.70281982421875 came back as
    # 467.7028198242187. Yahoo prices are float32-derived values like this, so we use
    # many of them, plus random floats, to make sure a cache read is EXACTLY a fresh result.
    rng = np.random.default_rng(42)
    random_prices = rng.uniform(1, 1000, size=(300, 2))
    float32_prices = random_prices.astype("float32").astype("float64")  # Yahoo-style values
    dates = pd.bdate_range("2025-01-01", periods=300)
    table = pd.DataFrame(
        np.concatenate([random_prices[:, :1], float32_prices[:, 1:]], axis=1),
        index=dates,
        columns=["AAPL", "MSFT"],
    )
    table.iloc[7, 0] = np.nan  # include a gap too
    monkeypatch.setattr(data, "download_prices", FakeDownloader(lambda: table))

    fresh = load_prices(["AAPL", "MSFT"], "2025-01-01", "2026-06-01", cache_dir=tmp_path)
    cached = load_prices(["AAPL", "MSFT"], "2025-01-01", "2026-06-01", cache_dir=tmp_path)

    assert np.array_equal(cached.to_numpy(), fresh.to_numpy(), equal_nan=True)  # bit for bit
    pd.testing.assert_frame_equal(cached, fresh, check_exact=True)


def test_cache_hit_follows_the_callers_ticker_order(downloader, tmp_path):
    first = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    flipped = load_prices(["MSFT", "AAPL"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)

    assert len(downloader.calls) == 1  # same cache file, no second download
    assert list(first.columns) == ["AAPL", "MSFT"]
    assert list(flipped.columns) == ["MSFT", "AAPL"]
    pd.testing.assert_frame_equal(flipped, first[["MSFT", "AAPL"]], check_exact=True)


def test_fresh_download_also_follows_the_callers_ticker_order(downloader, tmp_path):
    prices = load_prices(["MSFT", "AAPL"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    assert list(prices.columns) == ["MSFT", "AAPL"]


def test_tickers_are_normalised_so_case_and_repeats_do_not_matter(downloader, tmp_path):
    first = load_prices(["aapl", " MSFT ", "AAPL"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    second = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)

    assert list(first.columns) == ["AAPL", "MSFT"]
    assert len(downloader.calls) == 1
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_different_dates_or_tickers_use_a_different_cache_file(downloader, tmp_path):
    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-04-01", cache_dir=tmp_path)  # new end
    load_prices(["AAPL"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)  # new tickers

    assert len(downloader.calls) == 3
    assert len(list(tmp_path.glob("*.csv"))) == 3


def test_date_formats_are_standardised_for_the_cache(downloader, tmp_path):
    load_prices(["AAPL"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    load_prices(["AAPL"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-01"), cache_dir=tmp_path)
    assert len(downloader.calls) == 1


def test_timezone_aware_start_and_end_keep_their_local_date(downloader, tmp_path):
    # Tokyo midnight must stay "2026-01-05", not become 2026-01-04 (its date in UTC).
    start = pd.Timestamp("2026-01-05", tz="Asia/Tokyo")
    end = pd.Timestamp("2026-02-02", tz="Asia/Tokyo")
    load_prices(["AAPL"], start, end, cache_dir=tmp_path)
    assert downloader.calls == [(["AAPL"], "2026-01-05", "2026-02-02")]


# --- D1: a future `end` is limited to today ---------------------------------

def test_future_end_is_limited_to_today_for_the_download_and_the_cache_name(
    downloader, tmp_path, pinned_today
):
    assert pinned_today == pd.Timestamp("2026-06-15")

    load_prices(["AAPL"], "2026-05-01", "2026-12-31", cache_dir=tmp_path)

    assert downloader.calls == [(["AAPL"], "2026-05-01", "2026-06-15")]  # end clamped to today
    names = [f.name for f in tmp_path.glob("*.csv")]
    assert names == ["prices_AAPL_2026-05-01_2026-06-15.csv"]  # NOT ..._2026-12-31.csv


def test_all_future_end_dates_share_one_cache_entry(downloader, tmp_path):
    load_prices(["AAPL"], "2026-05-01", "2026-12-31", cache_dir=tmp_path)
    load_prices(["AAPL"], "2026-05-01", "2027-03-01", cache_dir=tmp_path)  # different future end
    load_prices(["AAPL"], "2026-05-01", "2026-06-15", cache_dir=tmp_path)  # end == today

    assert len(downloader.calls) == 1  # all three are the same request
    assert len(list(tmp_path.glob("*.csv"))) == 1


def test_a_cache_entry_made_for_a_future_end_goes_stale_tomorrow_not_forever(
    downloader, tmp_path, monkeypatch
):
    load_prices(["AAPL"], "2026-05-01", "2026-12-31", cache_dir=tmp_path)
    monkeypatch.setattr(data, "_today", lambda: pd.Timestamp("2026-06-16"))  # the next day

    load_prices(["AAPL"], "2026-05-01", "2026-12-31", cache_dir=tmp_path)

    assert [call[2] for call in downloader.calls] == ["2026-06-15", "2026-06-16"]  # fresh data
    assert len(list(tmp_path.glob("*.csv"))) == 2


def test_end_today_or_in_the_past_is_not_changed(downloader, tmp_path):
    load_prices(["AAPL"], "2026-05-01", "2026-06-14", cache_dir=tmp_path)  # yesterday
    load_prices(["AAPL"], "2026-05-01", "2026-06-15", cache_dir=tmp_path)  # today
    load_prices(["AAPL"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)  # long ago
    assert [call[2] for call in downloader.calls] == ["2026-06-14", "2026-06-15", "2026-03-01"]


@pytest.mark.parametrize("start", ["2026-06-15", "2026-07-01"])  # today, and the future
def test_start_on_or_after_today_raises_and_explains_the_clamp(downloader, tmp_path, start):
    with pytest.raises(ValueError, match=r"earlier than end.*limited to today"):
        load_prices(["AAPL"], start, "2026-12-31", cache_dir=tmp_path)
    assert downloader.calls == []
    assert list(tmp_path.iterdir()) == []


def test_start_equal_to_a_past_end_still_gives_the_normal_error_without_the_clamp_note(
    downloader, tmp_path
):
    with pytest.raises(ValueError, match="earlier than") as error:
        load_prices(["AAPL"], "2026-03-01", "2026-03-01", cache_dir=tmp_path)
    assert "limited to today" not in str(error.value)  # end was not clamped here


def at_utc(moment: str):
    """A replacement for data._now_utc that always returns this UTC moment."""
    return lambda: pd.Timestamp(moment, tz="UTC")


@pytest.mark.parametrize(
    "now_utc, expected_today",
    [
        # Summer time (EDT, UTC-4): New York is 4 hours behind UTC.
        ("2026-10-20 16:00", "2026-10-20"),  # noon in New York, US session still open
        ("2026-10-21 03:00", "2026-10-20"),  # 23:00 in New York: still the 20th there
        ("2026-10-21 03:59:59", "2026-10-20"),  # one second before New York midnight
        ("2026-10-21 04:00", "2026-10-21"),  # exactly New York midnight
        ("2026-10-21 04:30", "2026-10-21"),  # 00:30 in New York: now the 21st
        # Winter time (EST, UTC-5): the cut-over moves to 05:00 UTC.
        ("2026-12-15 04:59", "2026-12-14"),
        ("2026-12-15 05:00", "2026-12-15"),
    ],
)
def test_today_is_the_new_york_date(monkeypatch, now_utc, expected_today):
    monkeypatch.setattr(data, "_now_utc", at_utc(now_utc))

    today = REAL_TODAY_FUNCTION()

    assert today == pd.Timestamp(expected_today)
    assert isinstance(today, pd.Timestamp)
    assert today.tzinfo is None  # a plain date, no timezone
    assert today == today.normalize()  # at midnight


def test_today_ignores_the_computers_own_date(monkeypatch):
    # At 16:00 UTC it is already 01:00 on the 21st in Tokyo, but still noon on the
    # 20th in New York. A computer in Tokyo must still get the 20th.
    monkeypatch.setattr(data, "_now_utc", at_utc("2026-10-20 16:00"))
    tokyo_date = data._now_utc().tz_convert("Asia/Tokyo").normalize().tz_localize(None)
    assert tokyo_date == pd.Timestamp("2026-10-21")  # what a Tokyo computer's clock says

    assert REAL_TODAY_FUNCTION() == pd.Timestamp("2026-10-20")  # but we use New York's date


def test_real_clock_helpers_return_sensible_values():
    now = data._now_utc()
    assert str(now.tzinfo) == "UTC"
    assert abs(pd.Timestamp.now(tz="UTC") - now) < pd.Timedelta(minutes=1)

    today = REAL_TODAY_FUNCTION()  # uses the real clock
    assert today.tzinfo is None and today == today.normalize()
    assert pd.Timedelta(0) <= now.tz_localize(None) - today < pd.Timedelta(days=2)


def test_load_prices_on_a_tokyo_clock_does_not_ask_for_the_unfinished_us_day(
    downloader, tmp_path, monkeypatch
):
    # The regression behind D9: with a Tokyo clock at 16:00 UTC on 2026-10-20, using the
    # computer's date would send end=2026-10-21, which includes the unfinished US bar of
    # the 20th and caches it as a close. With New York's date, end stays 2026-10-20.
    monkeypatch.setattr(data, "_today", REAL_TODAY_FUNCTION)  # undo the autouse pin
    monkeypatch.setattr(data, "_now_utc", at_utc("2026-10-20 16:00"))

    load_prices(["AAPL"], "2026-10-01", "2026-12-31", cache_dir=tmp_path)

    assert downloader.calls == [(["AAPL"], "2026-10-01", "2026-10-20")]
    assert [f.name for f in tmp_path.glob("*.csv")] == ["prices_AAPL_2026-10-01_2026-10-20.csv"]


# --- D8: default cache folder is anchored to the project --------------------

def test_default_cache_dir_is_inside_the_project_whatever_the_working_directory(
    monkeypatch, tmp_path
):
    before = data._default_cache_dir()
    monkeypatch.chdir(tmp_path)  # pretend the program was started from somewhere else
    after = data._default_cache_dir()

    assert before == after
    assert after.is_absolute()
    assert after.parts[-2:] == ("data", "cache")
    project_root = after.parent.parent
    assert (project_root / "signaldesk" / "data.py").is_file()


def test_load_prices_uses_the_default_cache_dir_when_none_is_given(
    downloader, monkeypatch, tmp_path
):
    default_dir = tmp_path / "default_cache"
    monkeypatch.setattr(data, "_default_cache_dir", lambda: default_dir)

    load_prices(["AAPL"], "2026-01-01", "2026-03-01")  # no cache_dir argument
    load_prices(["AAPL"], "2026-01-01", "2026-03-01")

    assert len(list(default_dir.glob("*.csv"))) == 1
    assert len(downloader.calls) == 1  # second call was a cache hit


def test_explicit_cache_dir_overrides_the_default(downloader, monkeypatch, tmp_path):
    default_dir = tmp_path / "default_cache"
    monkeypatch.setattr(data, "_default_cache_dir", lambda: default_dir)
    chosen = tmp_path / "chosen"

    load_prices(["AAPL"], "2026-01-01", "2026-03-01", cache_dir=chosen)

    assert len(list(chosen.glob("*.csv"))) == 1
    assert not default_dir.exists()


def test_relative_cache_dir_is_still_relative_to_the_working_directory(
    downloader, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    load_prices(["AAPL"], "2026-01-01", "2026-03-01", cache_dir="my_cache")
    assert len(list((tmp_path / "my_cache").glob("*.csv"))) == 1


def test_use_cache_false_forces_a_download_and_refreshes_the_cache(monkeypatch, tmp_path):
    old = FakeDownloader()
    monkeypatch.setattr(data, "download_prices", old)
    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)

    def newer_prices():
        return make_frame({"2026-01-02": [1.0, 2.0], "2026-01-05": [3.0, 4.0]})

    new = FakeDownloader(newer_prices)
    monkeypatch.setattr(data, "download_prices", new)

    refreshed = load_prices(
        ["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path, use_cache=False
    )
    assert len(new.calls) == 1
    assert refreshed["AAPL"].tolist() == [1.0, 3.0]

    # and the cache now holds the refreshed data
    again = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    assert len(new.calls) == 1
    pd.testing.assert_frame_equal(again, refreshed, check_exact=True)


def test_cache_directory_is_created_if_missing(downloader, tmp_path):
    cache_dir = tmp_path / "does" / "not" / "exist"
    load_prices(["AAPL"], "2026-01-01", "2026-03-01", cache_dir=cache_dir)
    assert len(list(cache_dir.glob("*.csv"))) == 1


def test_unreadable_cache_file_is_ignored_with_a_warning(downloader, tmp_path):
    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    cache_file = next(tmp_path.glob("*.csv"))
    cache_file.write_text("")  # corrupt it: an empty file

    with pytest.warns(UserWarning, match="cache"):
        prices = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)

    assert len(downloader.calls) == 2  # re-downloaded
    assert len(prices) == 3
    # and the file was repaired, so the next call is a clean cache hit
    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    assert len(downloader.calls) == 2


def test_cache_file_with_wrong_tickers_is_ignored_with_a_warning(downloader, tmp_path):
    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    cache_file = next(tmp_path.glob("*.csv"))
    pd.read_csv(cache_file, index_col=0)[["AAPL"]].to_csv(cache_file)  # drop MSFT

    with pytest.warns(UserWarning, match="does not match"):
        prices = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)

    assert len(downloader.calls) == 2
    assert list(prices.columns) == ["AAPL", "MSFT"]


def test_failed_download_raises_and_writes_no_cache(monkeypatch, tmp_path):
    def failing_download(tickers, start, end):
        raise PriceDataError("No price data returned for ZZZZ.")

    monkeypatch.setattr(data, "download_prices", failing_download)

    with pytest.raises(PriceDataError, match="ZZZZ"):
        load_prices(["ZZZZ"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_with_no_usable_rows_raises_and_writes_no_cache(monkeypatch, tmp_path):
    all_nan = lambda: make_frame({"2026-01-02": [np.nan, np.nan]})  # noqa: E731
    monkeypatch.setattr(data, "download_prices", FakeDownloader(all_nan))

    with pytest.raises(PriceDataError):
        load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_missing_ticker_from_the_downloader_is_an_error_not_a_silent_nan_column(
    monkeypatch, tmp_path
):
    only_aapl = lambda tickers, start, end: make_frame(  # noqa: E731
        {"2026-01-02": [1.0, 2.0]}
    )[["AAPL"]]
    monkeypatch.setattr(data, "download_prices", only_aapl)

    with pytest.raises(PriceDataError, match="MSFT"):
        load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)


def test_unwritable_cache_only_warns(downloader, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, so nothing can be created inside me")

    with pytest.warns(UserWarning, match="Could not write"):
        prices = load_prices(["AAPL"], "2026-01-01", "2026-03-01", cache_dir=blocker / "cache")

    # The data still came back (2 rows: for AAPL alone, 2026-01-07 is an all-NaN row and is dropped).
    assert prices.index.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-06"]


def test_gaps_in_loaded_prices_can_be_reported_with_find_gaps(downloader, tmp_path):
    prices = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    gaps = find_gaps(prices)
    assert gaps["missing_count"].to_dict() == {"AAPL": 1, "MSFT": 0}
    assert gaps.loc["AAPL", "first_missing"] == pd.Timestamp("2026-01-07")


# --------------------------------------------------------------------------- #
# _cache_path
# --------------------------------------------------------------------------- #

def test_cache_path_is_deterministic_and_ignores_ticker_order(tmp_path):
    a = data._cache_path(tmp_path, ["MSFT", "AAPL"], "2026-01-01", "2026-03-01")
    b = data._cache_path(tmp_path, ["AAPL", "MSFT"], "2026-01-01", "2026-03-01")
    assert a == b
    assert a.parent == tmp_path
    assert a.name == "prices_AAPL+MSFT_2026-01-01_2026-03-01.csv"


def test_cache_path_is_filename_safe_and_short_for_long_ticker_lists(tmp_path):
    odd = data._cache_path(tmp_path, ["^GSPC", "EURUSD=X", "A/B"], "2026-01-01", "2026-03-01")
    assert "/" not in odd.name and "=" not in odd.name and "^" not in odd.name

    many = [f"TICK{i:03d}" for i in range(60)]
    long_path = data._cache_path(tmp_path, many, "2026-01-01", "2026-03-01")
    assert len(long_path.name) < 100
    assert long_path == data._cache_path(tmp_path, list(reversed(many)), "2026-01-01", "2026-03-01")
    assert long_path != data._cache_path(tmp_path, many[:-1], "2026-01-01", "2026-03-01")


# =========================================================================== #
# Phase 1.5 additions: provider errors, safe cache writes, pruning, market_today
# =========================================================================== #

# --------------------------------------------------------------------------- #
# ProviderUnavailableError: "nothing came back at all"
# --------------------------------------------------------------------------- #

def patch_yf_raising(monkeypatch, error):
    """Replace yf.download with a fake that raises ``error``."""

    def fake_download(tickers, **kwargs):
        raise error

    monkeypatch.setattr(data.yf, "download", fake_download)


def test_provider_unavailable_error_is_a_price_data_error_and_a_value_error():
    assert issubclass(data.ProviderUnavailableError, PriceDataError)
    assert issubclass(data.ProviderUnavailableError, ValueError)
    assert "ProviderUnavailableError" in data.__all__


@pytest.mark.parametrize("empty_result", [pd.DataFrame(), None])
def test_download_with_no_data_for_any_ticker_raises_provider_unavailable(monkeypatch, empty_result):
    patch_yf(monkeypatch, empty_result)

    with pytest.raises(data.ProviderUnavailableError) as caught:
        download_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-10")

    message = str(caught.value)
    assert "(AAPL, MSFT)" in message  # names every ticker we asked for
    assert "between 2026-01-01 and 2026-01-09" in message  # the last day ASKED for (the 10th is excluded)
    assert "rate-limiting or unavailable" in message  # the provider may be the problem...
    assert "symbols may not exist" in message  # ...or the symbols may be...
    assert "no trading days" in message  # ...or the dates may contain no trading day at all
    assert "try again later" in message


def test_download_with_every_column_all_nan_raises_provider_unavailable(monkeypatch):
    # Rows exist but every ticker is empty: the typical answer of a rate-limiting Yahoo.
    raw = yfinance_style(
        ["AAPL", "MSFT"], {"2026-01-02": [np.nan, np.nan], "2026-01-05": [np.nan, np.nan]}
    )
    patch_yf(monkeypatch, raw)

    with pytest.raises(data.ProviderUnavailableError, match=r"\(AAPL, MSFT\)"):
        download_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-10")


def test_download_with_a_single_missing_ticker_is_provider_unavailable_too(monkeypatch):
    # With one ticker, "every ticker is empty" is true, so the message covers both causes.
    raw = yfinance_style(["ZZZZ"], {"2026-01-02": [np.nan]})
    patch_yf(monkeypatch, raw)

    with pytest.raises(data.ProviderUnavailableError, match="symbol may not exist"):  # singular: one symbol
        download_prices(["ZZZZ"], "2026-01-01", "2026-01-10")


def test_download_with_only_some_tickers_missing_is_a_plain_price_data_error(monkeypatch):
    raw = yfinance_style(
        ["AAPL", "ZZZZ"], {"2026-01-02": [270.0, np.nan], "2026-01-05": [266.5, np.nan]}
    )
    patch_yf(monkeypatch, raw)

    with pytest.raises(PriceDataError, match="ZZZZ") as caught:
        download_prices(["AAPL", "ZZZZ"], "2026-01-01", "2026-01-10")
    assert not isinstance(caught.value, data.ProviderUnavailableError)  # the provider works
    assert "AAPL" not in str(caught.value)  # only the missing ticker is named


def test_download_with_a_ticker_absent_from_the_result_is_a_plain_price_data_error(monkeypatch):
    patch_yf(monkeypatch, yfinance_style(["AAPL"], {"2026-01-02": [270.0]}))

    with pytest.raises(PriceDataError, match="MSFT") as caught:
        download_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-10")
    assert not isinstance(caught.value, data.ProviderUnavailableError)


def test_download_with_a_missing_close_field_is_still_a_plain_price_data_error(monkeypatch):
    patch_yf(monkeypatch, pd.DataFrame({"Open": [1.0]}, index=pd.to_datetime(["2026-01-02"])))

    with pytest.raises(PriceDataError, match="Close") as caught:
        download_prices(["AAPL"], "2026-01-01", "2026-01-10")
    assert not isinstance(caught.value, data.ProviderUnavailableError)


def test_existing_except_price_data_error_code_still_catches_the_new_error(monkeypatch):
    patch_yf(monkeypatch, pd.DataFrame())
    try:
        download_prices(["AAPL"], "2026-01-01", "2026-01-10")
    except PriceDataError as error:
        assert isinstance(error, data.ProviderUnavailableError)
    else:
        raise AssertionError("expected an error")


def provider_errors():
    from yfinance.exceptions import YFRateLimitError

    return [
        YFRateLimitError(),
        ConnectionError("connection reset"),
        TimeoutError("timed out"),
        OSError("network is unreachable"),
    ]


@pytest.mark.parametrize("index", range(4))
def test_rate_limit_and_network_errors_become_provider_unavailable(monkeypatch, index):
    original = provider_errors()[index]
    patch_yf_raising(monkeypatch, original)

    with pytest.raises(data.ProviderUnavailableError) as caught:
        download_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-10")

    message = str(caught.value)
    assert "AAPL, MSFT" in message
    assert "rate-limiting or unreachable" in message
    assert caught.value.__cause__ is original  # the real error is kept for debugging...
    assert str(original) not in message  # ...but not shown in the visitor-facing text


def test_curl_cffi_network_errors_are_treated_as_provider_unavailable(monkeypatch):
    # yfinance 1.7 talks to Yahoo through curl_cffi; its errors are OSError subclasses.
    from curl_cffi.requests.exceptions import ConnectionError as CurlConnectionError

    patch_yf_raising(monkeypatch, CurlConnectionError("Could not resolve host"))
    with pytest.raises(data.ProviderUnavailableError):
        download_prices(["AAPL"], "2026-01-01", "2026-01-10")


@pytest.mark.parametrize(
    "local_error",
    [PermissionError("denied"), FileNotFoundError("no such folder"), NotADirectoryError("x")],
)
def test_problems_on_this_computer_are_not_blamed_on_the_provider(monkeypatch, local_error):
    patch_yf_raising(monkeypatch, local_error)
    with pytest.raises(type(local_error)):
        download_prices(["AAPL"], "2026-01-01", "2026-01-10")


@pytest.mark.parametrize("bug", [RuntimeError("a real bug"), KeyError("x"), AssertionError("oops")])
def test_unexpected_errors_are_not_swallowed(monkeypatch, bug):
    patch_yf_raising(monkeypatch, bug)
    with pytest.raises(type(bug)):
        download_prices(["AAPL"], "2026-01-01", "2026-01-10")


def test_load_prices_passes_the_provider_error_through_and_writes_no_cache(monkeypatch, tmp_path):
    patch_yf(monkeypatch, pd.DataFrame())

    with pytest.raises(data.ProviderUnavailableError, match="rate-limiting"):
        load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_load_prices_with_a_rate_limit_error_writes_no_cache(monkeypatch, tmp_path):
    patch_yf_raising(monkeypatch, provider_errors()[0])

    with pytest.raises(PriceDataError):
        load_prices(["AAPL"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_load_prices_with_some_tickers_missing_stays_a_plain_price_data_error(monkeypatch, tmp_path):
    raw = yfinance_style(["AAPL", "ZZZZ"], {"2026-01-02": [270.0, np.nan], "2026-01-05": [266.5, np.nan]})
    patch_yf(monkeypatch, raw)

    with pytest.raises(PriceDataError, match="ZZZZ") as caught:
        load_prices(["AAPL", "ZZZZ"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    assert not isinstance(caught.value, data.ProviderUnavailableError)


# --------------------------------------------------------------------------- #
# W3: a date range without trading days (weekend, holiday) is not a provider problem
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "start, end, expected",
    [
        ("2026-10-17", "2026-10-19", " between 2026-10-17 and 2026-10-18"),  # Saturday + Sunday
        ("2026-10-17", "2026-10-18", " on 2026-10-17"),  # a single day
        ("2026-01-01", "2026-01-10", " between 2026-01-01 and 2026-01-09"),
        ("2025-12-30", "2026-01-02", " between 2025-12-30 and 2026-01-01"),  # across a year end
        ("2024-02-28", "2024-03-01", " between 2024-02-28 and 2024-02-29"),  # across a leap day
    ],
)
def test_describe_window_names_the_last_day_that_was_asked_for(start, end, expected):
    assert data._describe_window(start, end) == expected


@pytest.mark.parametrize(
    "start, end",
    [
        ("2026-10-17", "2026-10-19"),  # Saturday to Sunday
        ("2026-09-07", "2026-09-08"),  # Labor Day, a Monday when the US market is closed
    ],
)
def test_a_range_without_trading_days_names_that_cause_and_the_dates(monkeypatch, start, end):
    patch_yf(monkeypatch, pd.DataFrame())  # what Yahoo really returns for such a range

    with pytest.raises(data.ProviderUnavailableError) as caught:
        download_prices(["AAPL"], start, end)

    message = str(caught.value)
    last_day = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert start in message and last_day in message  # the dates are named
    assert "no trading days" in message and "weekend" in message
    assert "rate-limiting or unavailable" in message  # the other causes are still named
    assert "symbol may not exist" in message


def test_the_all_nan_path_names_the_dates_too(monkeypatch):
    # Rows exist but the only ticker is empty: this goes through _require_all_tickers.
    raw = yfinance_style(["AAPL"], {"2026-10-16": [np.nan]})
    patch_yf(monkeypatch, raw)

    with pytest.raises(data.ProviderUnavailableError) as caught:
        download_prices(["AAPL"], "2026-10-17", "2026-10-19")
    assert "between 2026-10-17 and 2026-10-18" in str(caught.value)
    assert "no trading days" in str(caught.value)


def test_the_partly_missing_message_also_names_the_dates(monkeypatch):
    raw = yfinance_style(["AAPL", "ZZZZ"], {"2026-01-02": [270.0, np.nan]})
    patch_yf(monkeypatch, raw)
    with pytest.raises(PriceDataError, match=r"ZZZZ between 2026-01-01 and 2026-01-09"):
        download_prices(["AAPL", "ZZZZ"], "2026-01-01", "2026-01-10")


def test_load_prices_for_a_weekend_gives_the_no_trading_days_message(monkeypatch, tmp_path):
    patch_yf(monkeypatch, pd.DataFrame())
    # (the pinned "today" is 2026-06-15; 30 and 31 May 2026 are a Saturday and a Sunday)
    with pytest.raises(data.ProviderUnavailableError, match=r"between 2026-05-30 and 2026-05-31.*no trading days"):
        load_prices(["AAPL"], "2026-05-30", "2026-06-01", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# W2: what REAL yfinance does with errors. These tests run the real yf.download
# (threads, per-ticker error catching, table building) and replace only
# Ticker.history, the one method that talks to Yahoo. Every attempt to open a
# network connection is refused and recorded, and the test fails if there was one.
# --------------------------------------------------------------------------- #

@pytest.fixture
def real_yf(monkeypatch):
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield
    assert attempts == []  # nothing even tried to reach the network


def history_frame() -> pd.DataFrame:
    """What Ticker.history returns: one table per ticker, timezone-aware exchange dates."""
    index = pd.DatetimeIndex(["2026-01-02", "2026-01-05"]).tz_localize("America/New_York")
    return pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [3.0, 4.0], "Low": [0.5, 1.5], "Close": [270.0, 266.5], "Volume": [10, 20]},
        index=index,
    )


def patch_history(monkeypatch, behaviour):
    """Replace Ticker.history by ``behaviour(ticker)``, which returns a table or raises."""

    def fake_history(self, *args, **kwargs):
        return behaviour(self.ticker)

    monkeypatch.setattr(data.yf.Ticker, "history", fake_history)


# (error class, constructor arguments): built fresh for every use, never shared between tests
PROVIDER_FAILURES = [
    pytest.param("yfinance.exceptions:YFRateLimitError", (), id="YFRateLimitError"),
    pytest.param("builtins:ConnectionError", ("connection reset",), id="ConnectionError"),
    pytest.param("builtins:TimeoutError", ("timed out",), id="TimeoutError"),
    pytest.param("curl_cffi.requests.exceptions:ConnectionError", ("Could not resolve host",), id="curl_cffi"),
]


def make_failure(spec: str, args: tuple) -> Exception:
    import importlib

    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)(*args)


def test_real_download_works_offline_with_a_patched_history(real_yf, monkeypatch):
    # The healthy path through the real yf.download: this proves the setup below is realistic.
    patch_history(monkeypatch, lambda ticker: history_frame())

    result = download_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-10")

    assert list(result.columns) == ["AAPL", "MSFT"]
    assert list(result["AAPL"]) == [270.0, 266.5]


@pytest.mark.parametrize("spec, args", PROVIDER_FAILURES)
def test_real_download_with_every_ticker_failing_is_provider_unavailable(real_yf, monkeypatch, spec, args):
    def fail(ticker):
        raise make_failure(spec, args)

    patch_history(monkeypatch, fail)

    with pytest.raises(data.ProviderUnavailableError) as caught:
        download_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-10")

    message = str(caught.value)
    assert "(AAPL, MSFT)" in message
    assert "rate-limiting or unavailable" in message


@pytest.mark.parametrize("spec, args", PROVIDER_FAILURES)
def test_real_download_with_one_of_two_tickers_failing_admits_both_causes(real_yf, monkeypatch, spec, args):
    # yfinance turns the failure into "no data for MSFT". We cannot tell a rate limit from a
    # mistyped symbol, so the message must not blame the visitor alone.
    def sometimes_fail(ticker):
        if ticker == "MSFT":
            raise make_failure(spec, args)
        return history_frame()

    patch_history(monkeypatch, sometimes_fail)

    with pytest.raises(PriceDataError) as caught:
        download_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-10")

    message = str(caught.value)
    assert not isinstance(caught.value, data.ProviderUnavailableError)  # the provider answered for AAPL
    assert "MSFT" in message and "AAPL" not in message  # only the missing ticker is named
    assert "may not exist" in message  # the symbol may be wrong...
    assert "data provider may have failed" in message and "rate limit" in message  # ...or the provider failed
    assert "try again in a minute" in message


def test_real_download_with_one_failing_ticker_through_load_prices_writes_no_cache(real_yf, monkeypatch, tmp_path):
    patch_history(monkeypatch, lambda t: (_ for _ in ()).throw(ConnectionError("x")) if t == "MSFT" else history_frame())
    with pytest.raises(PriceDataError, match="data provider may have failed"):
        load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_local_error_inside_yfinance_also_surfaces_as_missing_data(real_yf, monkeypatch):
    # yfinance keeps its own small database on this computer. If that fails (this is a real
    # message seen live), yfinance swallows the error like any other and reports "no data".
    # We cannot see the cause, so for one ticker it is the same message as a rate limit.
    def locked(ticker):
        if ticker == "MSFT":
            raise sqlite3.OperationalError("database is locked")
        return history_frame()

    patch_history(monkeypatch, locked)
    with pytest.raises(PriceDataError, match="MSFT") as caught:
        download_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-10")
    assert not isinstance(caught.value, data.ProviderUnavailableError)

    def always_locked(ticker):
        raise sqlite3.OperationalError("database is locked")

    patch_history(monkeypatch, always_locked)
    with pytest.raises(data.ProviderUnavailableError):
        download_prices(["AAPL", "MSFT"], "2026-01-01", "2026-01-10")


def test_real_download_of_a_weekend_is_the_no_trading_days_message(real_yf, monkeypatch):
    patch_history(monkeypatch, lambda ticker: pd.DataFrame())  # Yahoo: an empty table for a weekend

    with pytest.raises(data.ProviderUnavailableError) as caught:
        download_prices(["AAPL"], "2026-10-17", "2026-10-19")

    message = str(caught.value)
    assert "between 2026-10-17 and 2026-10-18" in message and "no trading days" in message


# --------------------------------------------------------------------------- #
# D7: every cache write uses its own temporary file
# --------------------------------------------------------------------------- #

CACHE_NAME = "prices_AAPL+MSFT_2026-01-01_2026-03-01.csv"


def constant_frame(value: float, rows: int = 5) -> pd.DataFrame:
    """A cleaned-looking table where every cell is ``value`` (so any mix-up shows at once)."""
    index = pd.bdate_range("2026-01-01", periods=rows, name="Date")
    # clean_prices gives the exact shape a cache read gives back (same index type and unit).
    return clean_prices(pd.DataFrame({"AAPL": value, "MSFT": value + 0.25}, index=index))


def test_each_cache_write_uses_a_different_temp_file(monkeypatch, tmp_path):
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        return real_replace(src, dst)

    monkeypatch.setattr(data.os, "replace", spy)
    path = tmp_path / CACHE_NAME

    for i in range(5):
        data._write_cache(path, constant_frame(float(i)))

    assert len(seen) == 5
    assert len(set(seen)) == 5  # five writes, five different temp names
    assert all(name.startswith(CACHE_NAME) and name.endswith(".tmp") for name in seen)
    assert [f.name for f in tmp_path.iterdir()] == [CACHE_NAME]  # and nothing left behind


def test_the_temp_file_is_in_the_same_folder_as_the_cache_file(monkeypatch, tmp_path):
    folders = []
    real_replace = os.replace

    def spy(src, dst):
        folders.append(Path(src).parent)
        return real_replace(src, dst)

    monkeypatch.setattr(data.os, "replace", spy)
    data._write_cache(tmp_path / CACHE_NAME, constant_frame(1.0))
    assert folders == [tmp_path]  # same folder, so the rename is atomic


def test_many_threads_writing_the_same_cache_key_leave_one_valid_file(tmp_path):
    path = tmp_path / CACHE_NAME
    threads_n, writes_n, rows = 8, 20, 300  # 160 writes of a 300-row table
    start_together = threading.Barrier(threads_n)
    errors = []

    def frame_for(thread: int, write: int) -> pd.DataFrame:
        return constant_frame(thread * 1000 + write + 0.5, rows)

    def worker(thread: int) -> None:
        start_together.wait()
        for write in range(writes_n):
            try:
                data._write_cache(path, frame_for(thread, write))
            except BaseException as exc:  # noqa: BLE001 - we want to see anything at all
                errors.append(exc)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pool = [threading.Thread(target=worker, args=(t,)) for t in range(threads_n)]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()

    assert errors == []
    assert [str(w.message) for w in caught] == []  # no "Could not write the price cache"

    cached = data._read_cache(path, ["AAPL", "MSFT"])
    assert cached is not None  # the file parses and has the right columns
    value = cached.iloc[0, 0]
    candidates = {t * 1000 + w + 0.5: (t, w) for t in range(threads_n) for w in range(writes_n)}
    assert value in candidates  # the first cell is one of the values somebody wrote...
    pd.testing.assert_frame_equal(cached, frame_for(*candidates[value]), check_exact=True)
    # ...and the WHOLE table belongs to that one write (no mix of two writers).
    assert list(tmp_path.glob("*.tmp")) == []
    assert [f.name for f in tmp_path.iterdir()] == [CACHE_NAME]


def test_a_cache_write_that_fails_half_way_leaves_no_temp_file(monkeypatch, tmp_path):
    def half_a_write(self, path_or_buf, *args, **kwargs):
        Path(path_or_buf).write_text("Date,AAPL\n2026-01-0")  # the disk fills up here
        raise OSError("No space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_csv", half_a_write)

    with pytest.warns(UserWarning, match="Could not write"):
        data._write_cache(tmp_path / CACHE_NAME, constant_frame(1.0))
    assert list(tmp_path.iterdir()) == []


def test_a_failed_rename_leaves_no_temp_file_and_keeps_the_old_cache(monkeypatch, tmp_path):
    path = tmp_path / CACHE_NAME
    data._write_cache(path, constant_frame(1.0))
    before = path.read_bytes()

    def broken_replace(src, dst):
        raise PermissionError("denied")

    monkeypatch.setattr(data.os, "replace", broken_replace)
    with pytest.warns(UserWarning, match="Could not write"):
        data._write_cache(path, constant_frame(2.0))

    assert path.read_bytes() == before  # the old file is intact
    assert [f.name for f in tmp_path.iterdir()] == [CACHE_NAME]


def test_an_interrupt_during_a_cache_write_still_cleans_up_the_temp_file(monkeypatch, tmp_path):
    def interrupted(self, path_or_buf, *args, **kwargs):
        Path(path_or_buf).write_text("partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_csv", interrupted)
    with pytest.raises(KeyboardInterrupt):
        data._write_cache(tmp_path / CACHE_NAME, constant_frame(1.0))
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# D12: prune_cache keeps the cache folder from growing forever
# --------------------------------------------------------------------------- #

DAY = 86400


def make_file(folder: Path, name: str, age_seconds: float = 0, text: str = "x") -> Path:
    """Create a file whose last-modified time is ``age_seconds`` ago."""
    path = folder / name
    path.write_text(text)
    moment = time.time() - age_seconds
    os.utime(path, (moment, moment))
    return path


def names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def temp_name(cache_name: str) -> str:
    """A temp file name the way ``_atomic_write`` makes one: cache name + random id + .tmp."""
    return f"{cache_name}.{uuid.uuid4().hex}.tmp"


def test_prune_deletes_price_files_older_than_max_age_days(tmp_path):
    make_file(tmp_path, "prices_OLD_2020-01-01_2020-02-01.csv", 8 * DAY)
    make_file(tmp_path, "prices_NEW_2020-01-01_2020-02-01.csv", 6 * DAY)
    make_file(tmp_path, "prices_TODAY_2020-01-01_2020-02-01.csv", 0)

    deleted = data.prune_cache(tmp_path)

    assert deleted == 1
    assert names(tmp_path) == [
        "prices_NEW_2020-01-01_2020-02-01.csv",
        "prices_TODAY_2020-01-01_2020-02-01.csv",
    ]


def test_prune_age_boundary_is_strict(tmp_path):
    make_file(tmp_path, "prices_A_2020-01-01_2020-02-01.csv", 7 * DAY - 60)  # a minute younger
    make_file(tmp_path, "prices_B_2020-01-01_2020-02-01.csv", 7 * DAY + 60)  # a minute older
    assert data.prune_cache(tmp_path, max_age_days=7) == 1
    assert names(tmp_path) == ["prices_A_2020-01-01_2020-02-01.csv"]


def test_prune_custom_max_age(tmp_path):
    make_file(tmp_path, "prices_A_2020-01-01_2020-02-01.csv", 2 * DAY)
    make_file(tmp_path, "prices_B_2020-01-01_2020-02-01.csv", 0.5 * DAY)
    assert data.prune_cache(tmp_path, max_age_days=1) == 1
    assert names(tmp_path) == ["prices_B_2020-01-01_2020-02-01.csv"]


def test_prune_keeps_the_newest_max_files_and_deletes_the_oldest(tmp_path):
    for i in range(5):  # f0 is the oldest, f4 the newest, all well inside the age limit
        make_file(tmp_path, f"prices_f{i}_2020-01-01_2020-02-01.csv", (5 - i) * 3600)

    deleted = data.prune_cache(tmp_path, max_files=3)

    assert deleted == 2
    assert names(tmp_path) == [f"prices_f{i}_2020-01-01_2020-02-01.csv" for i in (2, 3, 4)]


def test_prune_exactly_max_files_deletes_nothing(tmp_path):
    for i in range(3):
        make_file(tmp_path, f"prices_f{i}_2020-01-01_2020-02-01.csv", i * 3600)
    assert data.prune_cache(tmp_path, max_files=3) == 0
    assert len(names(tmp_path)) == 3


def test_prune_applies_the_age_rule_first_then_the_count_rule(tmp_path):
    make_file(tmp_path, "prices_ancient_2020-01-01_2020-02-01.csv", 30 * DAY)
    for i in range(4):
        make_file(tmp_path, f"prices_f{i}_2020-01-01_2020-02-01.csv", (4 - i) * 3600)

    # 1 file is too old; of the 4 left, the oldest 2 go because max_files is 2.
    assert data.prune_cache(tmp_path, max_files=2) == 3
    assert names(tmp_path) == [f"prices_f{i}_2020-01-01_2020-02-01.csv" for i in (2, 3)]


def test_prune_never_touches_other_files_or_folders(tmp_path):
    old = 400 * DAY
    keep = [
        make_file(tmp_path, "notes.txt", old),
        make_file(tmp_path, "other.csv", old),
        make_file(tmp_path, "prices_wrong_extension.txt", old),
        make_file(tmp_path, "myprices_A_2020-01-01_2020-02-01.csv", old),
        make_file(tmp_path, "something.tmp", old),  # an old .tmp that is not ours
        make_file(tmp_path, ".gitkeep", old),
        # lookalikes of OUR names that are not names _cache_path can make (a user's own files)
        make_file(tmp_path, "prices_my_q3_export.csv", old),
        make_file(tmp_path, "prices_2024.csv", old),
        make_file(tmp_path, "prices_AAPL_2020-01-01.csv", old),  # only one date
        make_file(tmp_path, "prices_AAPL_2020-01-01_2020-02-01.csv.bak", old),
        make_file(tmp_path, "prices_AAPL_2020-01-01_2020-02-01.csv.notes.tmp", old),  # tmp, but no id
        make_file(tmp_path, "prices_AAPL_2020-01-01_2020-02-01.csv.tmp", old),
        make_file(tmp_path, "prices_AAPL_2020-01-01_2020-02-01.csv.beef.tmp", old),  # id is not 32 hex
        make_file(tmp_path, "Prices_AAPL_2020-01-01_2020-02-01.csv", old),  # capital P
        make_file(tmp_path, "prices_AAPL_2020-01-01_2020-02-01.CSV", old),
        make_file(tmp_path, "prices__2020-01-01_2020-02-01.csv", old),  # empty ticker part
        make_file(tmp_path, "prices_AAPL MSFT_2020-01-01_2020-02-01.csv", old),  # a space
    ]
    folder = tmp_path / "prices_folder.csv"  # a FOLDER whose name looks like a cache file
    folder.mkdir()
    os.utime(folder, (time.time() - old, time.time() - old))
    make_file(tmp_path, "prices_X_2020-01-01_2020-02-01.csv", old)  # the one that must go

    deleted = data.prune_cache(tmp_path, max_age_days=1, max_files=0)

    assert deleted == 1
    assert all(path.exists() for path in keep)
    assert folder.is_dir()
    assert not (tmp_path / "prices_X_2020-01-01_2020-02-01.csv").exists()


def test_prune_leaves_a_users_own_prices_file_alone_even_when_it_is_ancient(tmp_path):
    mine = make_file(tmp_path, "prices_my_export.csv", 30 * DAY)
    assert data.prune_cache(tmp_path) == 0
    assert data.prune_cache(tmp_path, max_age_days=0, max_files=0) == 0
    assert mine.exists()


# The names prune_cache recognises are derived from what _cache_path really produces.
CACHE_NAME_REQUESTS = [
    ["AAPL"],
    ["AAPL", "MSFT"],
    ["MSFT", "AAPL"],
    ["^GSPC"],
    ["EURUSD=X"],
    ["BRK-B"],
    ["7203.T"],
    ["^GSPC", "EURUSD=X", "BRK-B", "7203.T", "AAPL"],
    [f"T{i:03d}" for i in range(40)],  # long: the name becomes a hash
    ["BF.B", "A", "0700.HK", "^VIX", "GC=F"],
]


@pytest.mark.parametrize("tickers", CACHE_NAME_REQUESTS, ids=lambda t: "+".join(t)[:30])
def test_every_name_cache_path_makes_is_recognised_by_prune(tmp_path, tickers):
    path = data._cache_path(tmp_path, tickers, "2026-01-01", "2026-03-01")
    assert data._CACHE_FILE_PATTERN.fullmatch(path.name), path.name
    assert data._CACHE_TEMP_PATTERN.fullmatch(temp_name(path.name)), path.name
    assert data._CACHE_TEMP_PATTERN.fullmatch(f"{path.name}.12345.tmp")  # the older, process-id style

    old = make_file(tmp_path, path.name, 30 * DAY)
    assert data.prune_cache(tmp_path) == 1  # and so it is pruned when it is old
    assert not old.exists()


@pytest.mark.parametrize(
    "name",
    [
        "prices_my_q3_export.csv",
        "prices_2024.csv",
        "notes.csv",
        "prices_.csv",
        "prices_AAPL_2026-01-01.csv",
        "prices_AAPL_2026-1-1_2026-3-1.csv",
        "prices_AAPL_2026-01-01_2026-03-01.csv.old",
        "prices_AAPL_2026-01-01_2026-03-01.csv.tmp",
        "xprices_AAPL_2026-01-01_2026-03-01.csv",
        "prices_AAPL_2026-01-01_2026-03-01.csv\n",
        "../prices_AAPL_2026-01-01_2026-03-01.csv",
    ],
)
def test_names_that_cache_path_cannot_make_are_not_recognised(name):
    assert data._CACHE_FILE_PATTERN.fullmatch(name) is None
    assert data._CACHE_TEMP_PATTERN.fullmatch(name) is None


def test_prune_does_not_follow_or_delete_links(tmp_path):
    if os.utime not in os.supports_follow_symlinks:
        pytest.skip("this platform cannot set the time of a link itself")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = make_file(elsewhere, "precious.txt", 400 * DAY)
    cache = tmp_path / "cache"
    cache.mkdir()
    link = cache / "prices_link_2020-01-01_2020-02-01.csv"
    link.symlink_to(target)
    old = time.time() - 400 * DAY
    os.utime(link, (old, old), follow_symlinks=False)

    assert data.prune_cache(cache, max_age_days=1) == 0
    assert link.is_symlink() and target.exists()


def test_prune_deletes_stale_temp_files_but_not_fresh_ones(tmp_path):
    make_file(tmp_path, "prices_A_2020-01-01_2020-02-01.csv.1234.tmp", 2 * 3600)  # old pid-style name
    make_file(tmp_path, temp_name("prices_B_2020-01-01_2020-02-01.csv"), 3601)  # just over an hour
    fresh = make_file(tmp_path, temp_name("prices_C_2020-01-01_2020-02-01.csv"), 3000)  # being written now

    deleted = data.prune_cache(tmp_path)

    assert deleted == 2
    assert names(tmp_path) == [fresh.name]


def test_prune_temp_files_do_not_count_towards_max_files(tmp_path):
    make_file(tmp_path, "prices_A_2020-01-01_2020-02-01.csv", 100)
    make_file(tmp_path, temp_name("prices_A_2020-01-01_2020-02-01.csv"), 100)  # fresh temp
    assert data.prune_cache(tmp_path, max_files=1) == 0


def test_prune_returns_zero_for_a_missing_or_empty_folder(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning for a folder that simply does not exist
        assert data.prune_cache(tmp_path / "does_not_exist") == 0
        assert data.prune_cache(tmp_path) == 0


def test_prune_none_switches_a_rule_off(tmp_path):
    make_file(tmp_path, "prices_A_2020-01-01_2020-02-01.csv", 400 * DAY)
    make_file(tmp_path, "prices_B_2020-01-01_2020-02-01.csv", 0)
    assert data.prune_cache(tmp_path, max_age_days=None, max_files=None) == 0
    assert data.prune_cache(tmp_path, max_age_days=None, max_files=1) == 1  # only the count rule
    assert names(tmp_path) == ["prices_B_2020-01-01_2020-02-01.csv"]


@pytest.mark.parametrize("kwargs", [{"max_age_days": -1}, {"max_files": -1}])
def test_prune_rejects_negative_limits(tmp_path, kwargs):
    with pytest.raises(ValueError, match="negative"):
        data.prune_cache(tmp_path, **kwargs)


def test_prune_default_folder_is_the_default_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "_default_cache_dir", lambda: tmp_path)
    make_file(tmp_path, "prices_A_2020-01-01_2020-02-01.csv", 30 * DAY)
    assert data.prune_cache() == 1


def test_prune_default_limits_are_7_days_and_200_files(tmp_path):
    make_file(tmp_path, "prices_A_2020-01-01_2020-02-01.csv", 6.9 * DAY)  # kept by the age rule
    for i in range(200):
        make_file(tmp_path, f"prices_n{i:03d}_2020-01-01_2020-02-01.csv", 10 + i)
    assert data.prune_cache(tmp_path) == 1  # 201 files: exactly the oldest one goes
    assert len(list(tmp_path.glob("prices_*.csv"))) == 200
    assert not (tmp_path / "prices_A_2020-01-01_2020-02-01.csv").exists()


def test_prune_is_best_effort_when_a_file_cannot_be_deleted(monkeypatch, tmp_path):
    victim = make_file(tmp_path, "prices_A_2020-01-01_2020-02-01.csv", 30 * DAY)

    def refuse(self, missing_ok=False):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(Path, "unlink", refuse)
    with pytest.warns(UserWarning, match="Could not clean the price cache"):
        assert data.prune_cache(tmp_path) == 0  # no exception, nothing counted
    assert victim.exists()


def test_prune_is_best_effort_when_the_folder_cannot_be_listed(monkeypatch, tmp_path):
    def refuse(self):
        raise PermissionError("cannot list")

    monkeypatch.setattr(Path, "iterdir", refuse)
    with pytest.warns(UserWarning, match="Could not clean the price cache"):
        assert data.prune_cache(tmp_path) == 0


def test_prune_ignores_a_file_that_another_request_already_deleted(monkeypatch, tmp_path):
    make_file(tmp_path, "prices_A_2020-01-01_2020-02-01.csv", 30 * DAY)
    real_unlink = Path.unlink

    def somebody_else_got_there_first(self, missing_ok=False):
        real_unlink(self)
        raise FileNotFoundError(self.name)

    monkeypatch.setattr(Path, "unlink", somebody_else_got_there_first)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # this is normal, not worth a warning
        assert data.prune_cache(tmp_path) == 0


def test_write_cache_itself_does_not_prune_and_says_whether_it_wrote(monkeypatch, tmp_path):
    stale = make_file(tmp_path, "prices_STALE_2020-01-01_2020-02-01.csv", 30 * DAY)

    assert data._write_cache(tmp_path / CACHE_NAME, constant_frame(1.0)) is True
    assert stale.exists()  # pruning is the caller's decision (see load_prices below)

    monkeypatch.setattr(data.os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("no")))
    with pytest.warns(UserWarning, match="Could not write"):
        assert data._write_cache(tmp_path / "prices_OTHER_2026-01-01_2026-03-01.csv", constant_frame(1.0)) is False
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.fixture
def default_cache(monkeypatch, tmp_path):
    """Make tmp_path/default_cache the DEFAULT cache folder, so tests never touch the real one."""
    folder = tmp_path / "default_cache"
    folder.mkdir()
    monkeypatch.setattr(data, "_default_cache_dir", lambda: folder)
    return folder


def test_load_prices_keeps_the_default_cache_folder_tidy(downloader, default_cache):
    stale = make_file(default_cache, "prices_STALE_2020-01-01_2020-02-01.csv", 30 * DAY)
    recent = make_file(default_cache, "prices_RECENT_2020-01-01_2020-02-01.csv", 1 * DAY)

    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01")  # cache_dir=None: the default folder

    assert not stale.exists()
    assert recent.exists()
    assert (default_cache / CACHE_NAME).exists()  # the file just written is never pruned


def test_load_prices_never_prunes_a_folder_the_caller_chose(downloader, default_cache, tmp_path):
    mine = tmp_path / "my_folder"
    mine.mkdir()
    # Even a file with a name that matches our scheme and is ancient stays: it is not our folder.
    old_scheme = make_file(mine, "prices_STALE_2020-01-01_2020-02-01.csv", 30 * DAY)
    old_default = make_file(default_cache, "prices_STALE_2020-01-01_2020-02-01.csv", 30 * DAY)

    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=mine)

    assert old_scheme.exists()
    assert old_default.exists()  # and the default folder was not cleaned either: it was not used
    assert (mine / CACHE_NAME).exists()


def test_a_users_own_prices_file_survives_load_prices_in_any_folder(downloader, default_cache, tmp_path):
    mine = tmp_path / "my_folder"
    mine.mkdir()
    export = make_file(mine, "prices_my_export.csv", 30 * DAY)
    export_default = make_file(default_cache, "prices_my_export.csv", 30 * DAY)
    q3 = make_file(default_cache, "prices_my_q3_export.csv", 30 * DAY)

    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=mine)
    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01")

    assert export.exists() and export_default.exists() and q3.exists()
    assert data.prune_cache(mine) == 0 and data.prune_cache(default_cache) == 0
    assert export.exists() and export_default.exists() and q3.exists()


def test_prune_cache_can_still_be_called_by_hand_on_a_folder_the_caller_chose(downloader, tmp_path):
    mine = tmp_path / "my_folder"
    mine.mkdir()
    stale = make_file(mine, "prices_STALE_2020-01-01_2020-02-01.csv", 30 * DAY)
    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01", cache_dir=mine)
    assert stale.exists()
    assert data.prune_cache(mine) == 1  # the explicit call does the work
    assert not stale.exists()


def test_a_failed_cache_write_does_not_prune_the_default_folder(downloader, default_cache, monkeypatch):
    stale = make_file(default_cache, "prices_STALE_2020-01-01_2020-02-01.csv", 30 * DAY)
    monkeypatch.setattr(data.os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("no")))

    with pytest.warns(UserWarning, match="Could not write"):
        prices = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01")
    assert len(prices) == 3
    assert stale.exists()


def test_a_cache_hit_does_not_prune(downloader, default_cache):
    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01")  # writes the cache file
    stale = make_file(default_cache, "prices_STALE_2020-01-01_2020-02-01.csv", 30 * DAY)

    load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01")  # served from the cache: no write

    assert len(downloader.calls) == 1
    assert stale.exists()


def test_a_pruning_problem_never_breaks_load_prices(downloader, default_cache, monkeypatch):
    make_file(default_cache, "prices_STALE_2020-01-01_2020-02-01.csv", 30 * DAY)

    def refuse(self, missing_ok=False):
        raise PermissionError("nope")

    monkeypatch.setattr(Path, "unlink", refuse)
    with pytest.warns(UserWarning, match="Could not clean"):
        prices = load_prices(["AAPL", "MSFT"], "2026-01-01", "2026-03-01")
    assert len(prices) == 3  # the prices still came back


def test_the_temp_names_written_by_the_cache_match_the_prune_pattern(monkeypatch, tmp_path):
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        return real_replace(src, dst)

    monkeypatch.setattr(data.os, "replace", spy)
    data._write_cache(tmp_path / CACHE_NAME, constant_frame(1.0))
    assert len(seen) == 1 and data._CACHE_TEMP_PATTERN.fullmatch(seen[0])


# --------------------------------------------------------------------------- #
# market_today: today's date in New York
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "now_utc, expected",
    [
        ("2026-10-20 16:00", datetime.date(2026, 10, 20)),
        ("2026-10-21 03:59:59", datetime.date(2026, 10, 20)),  # one second before New York midnight
        ("2026-10-21 04:00", datetime.date(2026, 10, 21)),  # New York midnight (summer time)
        ("2026-12-15 04:59", datetime.date(2026, 12, 14)),  # winter time: cut-over at 05:00 UTC
        ("2026-12-15 05:00", datetime.date(2026, 12, 15)),
    ],
)
def test_market_today_is_the_new_york_date(monkeypatch, now_utc, expected):
    monkeypatch.setattr(data, "_today", REAL_TODAY_FUNCTION)  # undo the autouse pin
    monkeypatch.setattr(data, "_now_utc", at_utc(now_utc))

    result = data.market_today()

    assert result == expected
    assert type(result) is datetime.date  # a plain date: not a Timestamp, not a datetime


def test_market_today_ignores_the_servers_own_timezone(monkeypatch):
    # 16:00 UTC is already the 21st in Tokyo and still the 20th in New York.
    monkeypatch.setattr(data, "_today", REAL_TODAY_FUNCTION)
    monkeypatch.setattr(data, "_now_utc", at_utc("2026-10-20 16:00"))
    assert data._now_utc().tz_convert("Asia/Tokyo").date() == datetime.date(2026, 10, 21)
    assert data.market_today() == datetime.date(2026, 10, 20)


def test_market_today_agrees_with_the_internal_today(pinned_today):
    # With "today" pinned (autouse fixture), market_today follows it: one source of truth.
    assert data.market_today() == pinned_today.date() == datetime.date(2026, 6, 15)


def test_market_today_with_the_real_clock_is_within_a_day_of_the_utc_date(monkeypatch):
    monkeypatch.setattr(data, "_today", REAL_TODAY_FUNCTION)  # undo the autouse pin
    utc_date = pd.Timestamp.now(tz="UTC").date()
    assert abs((data.market_today() - utc_date).days) <= 1
