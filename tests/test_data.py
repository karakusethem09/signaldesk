"""Tests for signaldesk/data.py. None of these tests use the network.

How we avoid the network: ``load_prices`` calls ``download_prices``, and
``download_prices`` calls ``yf.download``. In the tests we use pytest's
``monkeypatch`` to swap those for small fake functions that return hand-made
tables (and count how often they were called).
"""

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
