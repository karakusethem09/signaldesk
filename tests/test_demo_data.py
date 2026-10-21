"""Tests for the demo snapshot: ``load_demo``, ``refresh_demo_snapshot`` and the refresh script.

Nothing here uses the network. An always-on fixture (``no_network``) replaces ``yf.download``
with a function that fails the test if it is ever called. Tests that need a "download" use
a small fake ``download_prices`` instead. Every file the tests write goes into ``tmp_path``;
the committed snapshot in ``data/demo/`` is only ever READ.

The file has three parts:
    1. load_demo on small made-up demo folders (happy path and every kind of broken folder)
    2. the committed snapshot itself (the real files the website will load)
    3. refresh_demo_snapshot and scripts/refresh_demo_data.py with a fake download
"""

import dataclasses
import importlib.util
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signaldesk import data
from signaldesk.data import DemoData, PriceDataError, ProviderUnavailableError, load_demo

PROJECT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = PROJECT / "data" / "demo"
SCRIPT_PATH = PROJECT / "scripts" / "refresh_demo_data.py"

CONFIG_FILE, PRICES_FILE, META_FILE = "demo_portfolio.json", "demo_prices.csv", "demo_meta.json"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Make any attempt to reach Yahoo fail the test on the spot."""

    def boom(*args, **kwargs):
        raise AssertionError("the network was used: yf.download was called")

    monkeypatch.setattr(data.yf, "download", boom)


def test_the_no_network_guard_really_trips():
    # If this guard did nothing, every "offline" test below would prove nothing.
    with pytest.raises(AssertionError, match="network was used"):
        data.yf.download(["AAPL"])


# --------------------------------------------------------------------------- #
# Helpers: a small, valid demo folder that individual tests then break on purpose
# --------------------------------------------------------------------------- #

HOLDINGS = {"AAPL": 0.5, "MSFT": 0.3, "JPM": 0.2}
BENCHMARK = "SPY"
SOURCE = "Yahoo Finance via yfinance 1.7.0, adjusted close"


def small_prices(columns=("AAPL", "MSFT", "JPM", "SPY"), rows: int = 6) -> pd.DataFrame:
    """Made-up cleaned prices. The values are awkward on purpose (1/3 has no exact decimal)."""
    index = pd.bdate_range("2026-01-05", periods=rows, name="Date")
    values = {
        column: [(k + 1) * 100 / 3 + day * 0.1 for day in range(rows)]
        for k, column in enumerate(columns)
    }
    return data.clean_prices(pd.DataFrame(values, index=index))


def meta_for(prices: pd.DataFrame, **overrides) -> dict:
    """The demo_meta.json that correctly describes ``prices`` (change fields with overrides)."""
    meta = {
        "as_of": f"{prices.index.max():%Y-%m-%d}",
        "start": f"{prices.index.min():%Y-%m-%d}",
        "rows": len(prices),
        "tickers": list(prices.columns),
        "source": SOURCE,
        "created_at": "2026-01-13T10:00:00Z",
    }
    meta.update(overrides)
    return meta


def config_dict(**overrides) -> dict:
    config = {"name": "Test portfolio", "holdings": dict(HOLDINGS), "benchmark": BENCHMARK, "years": 5}
    config.update(overrides)
    return config


def write_demo(folder: Path, config=None, prices=None, meta=None) -> Path:
    """Write a demo folder. ``config``/``meta`` may be a dict (written as JSON) or raw text."""
    folder.mkdir(parents=True, exist_ok=True)
    prices = small_prices() if prices is None else prices
    config = config_dict() if config is None else config
    meta = meta_for(prices) if meta is None else meta

    for name, content in ((CONFIG_FILE, config), (META_FILE, meta)):
        text = content if isinstance(content, str) else json.dumps(content, indent=2)
        (folder / name).write_text(text, encoding="utf-8")
    prices.to_csv(folder / PRICES_FILE)  # the same CSV format the cache uses
    return folder


@pytest.fixture
def demo_dir(tmp_path):
    return write_demo(tmp_path / "demo")


# --------------------------------------------------------------------------- #
# 1. load_demo: the happy path
# --------------------------------------------------------------------------- #

def test_load_demo_happy_path(demo_dir):
    demo = load_demo(demo_dir)
    expected = small_prices()

    assert isinstance(demo, DemoData)
    assert demo.name == "Test portfolio"
    assert demo.holdings == HOLDINGS
    assert all(type(weight) is float for weight in demo.holdings.values())
    assert demo.benchmark == "SPY"
    assert demo.source == SOURCE
    assert demo.as_of == pd.Timestamp("2026-01-12")  # the last of the six business days
    assert demo.start == pd.Timestamp("2026-01-05")
    assert isinstance(demo.as_of, pd.Timestamp) and isinstance(demo.start, pd.Timestamp)
    # bit-for-bit equal to what was written, so the CSV round trip loses nothing
    pd.testing.assert_frame_equal(demo.prices, expected, check_exact=True)


def test_load_demo_prices_have_the_load_prices_shape(demo_dir):
    prices = load_demo(demo_dir).prices

    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.index.name == "Date"
    assert prices.index.tz is None
    assert str(prices.index.dtype) == "datetime64[ns]"
    assert prices.index.is_monotonic_increasing and not prices.index.has_duplicates
    assert list(prices.columns) == ["AAPL", "MSFT", "JPM", "SPY"]  # holdings in config order, then benchmark
    assert prices.columns.name is None
    assert set(prices.dtypes.astype(str)) == {"float64"}


def test_load_demo_gives_the_same_table_load_prices_gives(demo_dir, tmp_path, monkeypatch):
    # Put the demo CSV in the cache under the name load_prices expects, then ask load_prices
    # for it. Both tables must be identical, down to the index type, names and dtypes.
    monkeypatch.setattr(data, "_today", lambda: pd.Timestamp("2026-06-15"))
    demo = load_demo(demo_dir)
    tickers = list(demo.prices.columns)
    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.copyfile(demo_dir / PRICES_FILE, data._cache_path(cache, tickers, "2026-01-05", "2026-01-13"))

    via_load_prices = data.load_prices(tickers, "2026-01-05", "2026-01-13", cache_dir=cache)

    pd.testing.assert_frame_equal(demo.prices, via_load_prices, check_exact=True)


def test_load_demo_reorders_csv_columns_into_config_order(tmp_path):
    shuffled = small_prices()[["SPY", "JPM", "AAPL", "MSFT"]]  # the CSV has its columns in another order
    folder = write_demo(tmp_path / "demo", prices=shuffled, meta=meta_for(small_prices()))
    prices = load_demo(folder).prices
    assert list(prices.columns) == ["AAPL", "MSFT", "JPM", "SPY"]
    pd.testing.assert_frame_equal(prices, small_prices(), check_exact=True)


def test_load_demo_is_frozen_and_returns_a_fresh_object_each_time(demo_dir):
    demo = load_demo(demo_dir)
    with pytest.raises(dataclasses.FrozenInstanceError):
        demo.benchmark = "QQQ"
    assert dataclasses.is_dataclass(DemoData)
    assert [f.name for f in dataclasses.fields(DemoData)] == [
        "prices", "holdings", "benchmark", "name", "as_of", "start", "source",
    ]
    again = load_demo(demo_dir)
    again.holdings["AAPL"] = 0.99  # changing one copy must not leak into the next load
    assert load_demo(demo_dir).holdings["AAPL"] == 0.5


def test_load_demo_accepts_a_string_path_and_is_deterministic(demo_dir):
    first, second = load_demo(str(demo_dir)), load_demo(demo_dir)
    pd.testing.assert_frame_equal(first.prices, second.prices, check_exact=True)
    assert (first.holdings, first.as_of, first.start) == (second.holdings, second.as_of, second.start)


def test_load_demo_normalises_symbols_names_and_integer_weights(tmp_path):
    config = config_dict(
        name="  Padded name  ", holdings={" aapl ": 1, "msft": 0, "jpm": 0.0}, benchmark=" spy "
    )
    folder = write_demo(tmp_path / "demo", config=config)
    demo = load_demo(folder)
    assert demo.name == "Padded name"
    assert demo.holdings == {"AAPL": 1.0, "MSFT": 0.0, "JPM": 0.0}
    assert all(type(w) is float for w in demo.holdings.values())  # int 1 became 1.0
    assert demo.benchmark == "SPY"


def test_load_demo_checks_weights_only_as_numbers_not_their_sum(tmp_path):
    # Whether the weights add up to 1 is risk.portfolio_value's job, not the loader's.
    config = config_dict(holdings={"AAPL": 0.7, "MSFT": -0.2, "JPM": 3})
    demo = load_demo(write_demo(tmp_path / "demo", config=config))
    assert demo.holdings == {"AAPL": 0.7, "MSFT": -0.2, "JPM": 3.0}


def test_load_demo_does_not_need_years_in_the_config(tmp_path):
    config = config_dict()
    del config["years"]  # only the refresh script needs it
    assert load_demo(write_demo(tmp_path / "demo", config=config)).benchmark == "SPY"


def test_load_demo_works_from_any_working_directory_by_default(tmp_path, monkeypatch):
    # The default folder is anchored to the project, like the default cache folder.
    assert data._default_demo_dir() == PROJECT / "data" / "demo"
    monkeypatch.chdir(tmp_path)  # an unrelated folder with no data/demo inside
    from_default = load_demo()
    from_explicit = load_demo(SNAPSHOT_DIR)
    pd.testing.assert_frame_equal(from_default.prices, from_explicit.prices, check_exact=True)


def test_load_demo_never_touches_the_network_or_the_downloaders(demo_dir, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("load_demo must not download anything")

    monkeypatch.setattr(data, "download_prices", forbidden)
    monkeypatch.setattr(data, "load_prices", forbidden)
    assert len(load_demo(demo_dir).prices) == 6  # (yf.download is already guarded by no_network)


# --------------------------------------------------------------------------- #
# 1b. load_demo: broken folders give a clear ValueError
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("filename", [CONFIG_FILE, PRICES_FILE, META_FILE])
def test_load_demo_missing_file_names_the_file(demo_dir, filename):
    (demo_dir / filename).unlink()
    with pytest.raises(ValueError, match=rf"{re.escape(filename)} is missing") as caught:
        load_demo(demo_dir)
    assert "refresh_demo_data.py" in str(caught.value)  # tells you how to fix it


def test_load_demo_missing_folder(tmp_path):
    with pytest.raises(ValueError, match="is missing"):
        load_demo(tmp_path / "no_such_folder")


def test_load_demo_a_folder_where_a_file_should_be(demo_dir):
    (demo_dir / META_FILE).unlink()
    (demo_dir / META_FILE).mkdir()
    with pytest.raises(ValueError, match="cannot be read"):
        load_demo(demo_dir)


@pytest.mark.parametrize("filename", [CONFIG_FILE, META_FILE])
@pytest.mark.parametrize("broken", ["{not json", "", '{"name": "x",}', "nope", '{"a": 1} trailing'])
def test_load_demo_malformed_json(demo_dir, filename, broken):
    (demo_dir / filename).write_text(broken)
    with pytest.raises(ValueError, match=rf"{re.escape(filename)} is not valid JSON"):
        load_demo(demo_dir)


@pytest.mark.parametrize("filename", [CONFIG_FILE, META_FILE])
@pytest.mark.parametrize("not_an_object", ["[1, 2]", '"text"', "42", "null"])
def test_load_demo_json_that_is_not_an_object(demo_dir, filename, not_an_object):
    (demo_dir / filename).write_text(not_an_object)
    with pytest.raises(ValueError, match=rf"{re.escape(filename)} must contain a JSON object"):
        load_demo(demo_dir)


def test_load_demo_json_file_with_invalid_bytes(demo_dir):
    (demo_dir / CONFIG_FILE).write_bytes(b"\xff\xfe\x00 not utf-8 \x80")
    with pytest.raises(ValueError, match=CONFIG_FILE):
        load_demo(demo_dir)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"name": None}, "'name'"),
        ({"name": ""}, "'name'"),
        ({"name": 5}, "'name'"),
        ({"holdings": None}, "'holdings'"),
        ({"holdings": {}}, "'holdings'"),
        ({"holdings": ["AAPL", "MSFT"]}, "'holdings'"),
        ({"benchmark": None}, "'benchmark'"),
        ({"benchmark": ""}, "'benchmark'"),
        ({"benchmark": 500}, "'benchmark'"),
        ({"benchmark": "AAPL"}, "must not also be a holding"),
        ({"holdings": {"AAPL": 0.5, "aapl": 0.5}}, "AAPL appears twice"),
        ({"holdings": {"": 1.0}}, "holding name"),
        ({"years": 0}, "'years'"),
        ({"years": -3}, "'years'"),
        ({"years": "five"}, "'years'"),
    ],
)
def test_load_demo_config_with_missing_or_wrong_fields(tmp_path, overrides, message):
    config = config_dict(**overrides)
    folder = write_demo(tmp_path / "demo", config=config)
    with pytest.raises(ValueError, match=message) as caught:
        load_demo(folder)
    assert CONFIG_FILE in str(caught.value)


@pytest.mark.parametrize("missing_key", ["name", "holdings", "benchmark"])
def test_load_demo_config_with_a_key_left_out(tmp_path, missing_key):
    config = config_dict()
    del config[missing_key]
    with pytest.raises(ValueError, match=missing_key):
        load_demo(write_demo(tmp_path / "demo", config=config))


@pytest.mark.parametrize(
    "bad_weight",
    ["0.5", None, True, False, [0.5], {"w": 0.5}, float("nan"), float("inf"), float("-inf"), ""],
)
def test_load_demo_non_numeric_weights(tmp_path, bad_weight):
    holdings = {"AAPL": 0.5, "MSFT": bad_weight, "JPM": 0.2}  # json.dumps writes NaN/Infinity as such
    folder = write_demo(tmp_path / "demo", config=config_dict(holdings=holdings))
    with pytest.raises(ValueError, match=r"the weight of MSFT must be a number"):
        load_demo(folder)


# columns that do not match "holdings + benchmark"

def test_load_demo_csv_missing_a_holding_column(tmp_path):
    prices = small_prices(("AAPL", "JPM", "SPY"))  # MSFT is gone
    folder = write_demo(tmp_path / "demo", prices=prices)
    with pytest.raises(ValueError, match=r"missing columns \['MSFT'\]"):
        load_demo(folder)


def test_load_demo_csv_missing_the_benchmark_column(tmp_path):
    folder = write_demo(tmp_path / "demo", prices=small_prices(("AAPL", "MSFT", "JPM")))
    with pytest.raises(ValueError, match=r"missing columns \['SPY'\]"):
        load_demo(folder)


def test_load_demo_csv_with_an_extra_column(tmp_path):
    folder = write_demo(tmp_path / "demo", prices=small_prices(("AAPL", "MSFT", "JPM", "SPY", "TSLA")))
    with pytest.raises(ValueError, match=r"unexpected columns \['TSLA'\]"):
        load_demo(folder)


def test_load_demo_csv_with_a_renamed_column_names_both_sides(tmp_path):
    folder = write_demo(tmp_path / "demo", prices=small_prices(("AAPL", "MSFT", "JNJ", "SPY")))
    with pytest.raises(ValueError, match=r"missing columns \['JPM'\], unexpected columns \['JNJ'\]"):
        load_demo(folder)


def test_load_demo_config_changed_but_csv_not(tmp_path):
    config = config_dict(holdings={"AAPL": 0.5, "MSFT": 0.3, "JPM": 0.1, "NVDA": 0.1})
    folder = write_demo(tmp_path / "demo", config=config)
    with pytest.raises(ValueError, match=r"missing columns \['NVDA'\]"):
        load_demo(folder)


def test_load_demo_csv_with_a_duplicated_column(demo_dir):
    lines = (demo_dir / PRICES_FILE).read_text().splitlines()
    lines[0] = "Date,AAPL,AAPL,JPM,SPY"  # pandas turns the second one into "AAPL.1"
    (demo_dir / PRICES_FILE).write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=r"AAPL\.1"):
        load_demo(demo_dir)


def test_load_demo_a_column_with_no_prices_at_all(tmp_path):
    prices = small_prices()
    prices["MSFT"] = np.nan
    folder = write_demo(tmp_path / "demo", prices=prices)
    with pytest.raises(ValueError, match=r"no prices at all for: MSFT"):
        load_demo(folder)


# a CSV that is unreadable in other ways

@pytest.mark.parametrize(
    "content, message",
    [
        ("", "cannot be read as a price table"),  # an empty file
        ("Date,AAPL,MSFT,JPM,SPY\n", "no price rows"),  # a header and nothing else
        ("just some words\n", "no price rows"),
        ("Date,AAPL,MSFT,JPM,SPY\n2026-01-05,1,2,3,abc\n", "cannot be read as a price table"),  # text price
        ("Date,AAPL,MSFT,JPM,SPY\nnot-a-date,1,2,3,4\n", "cannot be read as a price table"),
        ("Date,AAPL,MSFT,JPM,SPY\n0,1,2,3,4\n1,1,2,3,4\n", "cannot be read as a price table"),  # numbers, not dates
        # a row that is one cell short is not a parse error: the cell is NaN, so SPY has no prices
        ("Date,AAPL,MSFT,JPM,SPY\n2026-01-05,1,2,3\n", "no prices at all for: SPY"),
    ],
)
@pytest.mark.filterwarnings("ignore:Could not infer format")  # pandas, for the text-instead-of-dates case
def test_load_demo_unreadable_prices_csv(demo_dir, content, message):
    (demo_dir / PRICES_FILE).write_text(content)
    with pytest.raises(ValueError, match=message) as caught:
        load_demo(demo_dir)
    assert PRICES_FILE in str(caught.value)


def test_load_demo_prices_csv_with_invalid_bytes(demo_dir):
    (demo_dir / PRICES_FILE).write_bytes(b"\xff\xfe\x00\x80 \xc3\x28 garbage")
    with pytest.raises(ValueError, match=PRICES_FILE):
        load_demo(demo_dir)


def test_load_demo_error_messages_for_prices_csv_say_what_is_wrong(demo_dir):
    (demo_dir / PRICES_FILE).write_text("Date,AAPL,MSFT,JPM,SPY\n")
    with pytest.raises(ValueError, match="contains no price rows"):
        load_demo(demo_dir)


# demo_meta.json that does not describe the CSV next to it

@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"rows": 5}, r"says 5 rows but demo_prices.csv has 6"),
        ({"rows": "6"}, "rows but"),
        ({"rows": True}, "rows but"),
        ({"rows": None}, "rows but"),
        ({"as_of": "2026-01-09"}, r"as_of is 2026-01-09 but demo_prices.csv says 2026-01-12"),
        ({"start": "2026-01-06"}, r"start is 2026-01-06 but demo_prices.csv says 2026-01-05"),
        ({"as_of": "yesterday"}, "'as_of' must be a date"),
        ({"as_of": None}, "'as_of' must be a date"),
        ({"as_of": 20260112}, "'as_of' must be a date"),
        ({"start": "garbage"}, "'start' must be a date"),
        ({"tickers": ["AAPL", "MSFT", "JPM"]}, "lists tickers"),
        ({"tickers": ["SPY", "AAPL", "MSFT", "JPM"]}, "lists tickers"),
        ({"tickers": "AAPL"}, "lists tickers"),
        ({"source": ""}, "'source' must be non-empty text"),
        ({"source": "   "}, "'source' must be non-empty text"),
        ({"source": None}, "'source' must be non-empty text"),
        ({"source": 7}, "'source' must be non-empty text"),
    ],
)
def test_load_demo_meta_that_disagrees_with_the_csv(tmp_path, overrides, message):
    prices = small_prices()
    folder = write_demo(tmp_path / "demo", prices=prices, meta=meta_for(prices, **overrides))
    with pytest.raises(ValueError, match=message) as caught:
        load_demo(folder)
    assert META_FILE in str(caught.value)


@pytest.mark.parametrize("missing_key", ["as_of", "start", "rows", "tickers", "source"])
def test_load_demo_meta_with_a_key_left_out(tmp_path, missing_key):
    prices = small_prices()
    meta = meta_for(prices)
    del meta[missing_key]
    with pytest.raises(ValueError, match=META_FILE):
        load_demo(write_demo(tmp_path / "demo", prices=prices, meta=meta))


def test_load_demo_meta_rows_true_is_not_taken_for_one_row(tmp_path):
    # In Python True == 1, so without an explicit check "rows": true would pass for a 1-row CSV.
    prices = small_prices(rows=1)
    assert len(prices) == 1
    folder = write_demo(tmp_path / "demo", prices=prices, meta=meta_for(prices, rows=True))
    with pytest.raises(ValueError, match=r"says True rows but demo_prices.csv has 1"):
        load_demo(folder)
    # and the honest value works
    assert len(load_demo(write_demo(tmp_path / "ok", prices=prices)).prices) == 1


def test_load_demo_does_not_need_created_at_in_the_meta(tmp_path):
    prices = small_prices()
    meta = meta_for(prices)
    del meta["created_at"]  # it is information for humans; nothing depends on it
    assert load_demo(write_demo(tmp_path / "demo", prices=prices, meta=meta)).as_of == prices.index.max()


# --------------------------------------------------------------------------- #
# 2. The committed snapshot itself (data/demo/): these are the real files
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def snapshot() -> DemoData:
    return load_demo(SNAPSHOT_DIR)


def snapshot_config() -> dict:
    return json.loads((SNAPSHOT_DIR / CONFIG_FILE).read_text(encoding="utf-8"))


def snapshot_meta() -> dict:
    return json.loads((SNAPSHOT_DIR / META_FILE).read_text(encoding="utf-8"))


def test_snapshot_config_is_the_agreed_demo_portfolio():
    assert snapshot_config() == {
        "name": "Demo: diversified US large caps",
        "holdings": {"AAPL": 0.20, "MSFT": 0.20, "NVDA": 0.15, "JPM": 0.15, "XOM": 0.15, "JNJ": 0.15},
        "benchmark": "SPY",
        "years": 5,
    }
    # JSON objects keep their order, so this is the order of the file, not just the set
    assert list(snapshot_config()["holdings"]) == ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ"]


def test_snapshot_loads_with_the_agreed_holdings(snapshot):
    assert snapshot.name == "Demo: diversified US large caps"
    assert snapshot.benchmark == "SPY"
    assert snapshot.holdings == {"AAPL": 0.20, "MSFT": 0.20, "NVDA": 0.15, "JPM": 0.15, "XOM": 0.15, "JNJ": 0.15}
    assert sum(snapshot.holdings.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(weight > 0 for weight in snapshot.holdings.values())
    assert snapshot.source.strip() != ""


def test_snapshot_has_seven_columns_in_config_order(snapshot):
    assert list(snapshot.prices.columns) == ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "SPY"]
    assert list(snapshot.prices.columns) == list(snapshot_config()["holdings"]) + [snapshot_config()["benchmark"]]


def test_snapshot_has_about_five_years_of_rows(snapshot):
    assert len(snapshot.prices) >= 1200
    years_covered = (snapshot.as_of - snapshot.start).days / 365.25
    assert abs(years_covered - snapshot_config()["years"]) < 0.05  # "years": 5 means about 5 years


def test_snapshot_shape_and_types(snapshot):
    prices = snapshot.prices
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.index.name == "Date"
    assert prices.index.tz is None
    assert str(prices.index.dtype) == "datetime64[ns]"
    assert set(prices.dtypes.astype(str)) == {"float64"}
    assert prices.index.is_monotonic_increasing and not prices.index.has_duplicates
    assert (prices.index == prices.index.normalize()).all()  # dates only, no times of day
    assert prices.columns.name is None


def test_snapshot_as_of_and_start_are_the_first_and_last_dates(snapshot):
    assert snapshot.as_of == snapshot.prices.index[-1]
    assert snapshot.start == snapshot.prices.index[0]
    assert snapshot.start < snapshot.as_of


def test_snapshot_prices_are_sane(snapshot):
    prices = snapshot.prices
    assert not prices.isna().all().any()  # no column is entirely empty
    assert np.isfinite(prices.to_numpy(dtype=float)[~prices.isna().to_numpy()]).all()
    assert (prices.dropna() > 0).all().all()  # an adjusted close is always above zero
    # A few missing cells would be tolerable; a lot would distort every risk number.
    gaps = data.find_gaps(prices)
    assert (gaps["missing_count"] <= 0.01 * len(prices)).all()
    # No price jumps by more than 3x or falls by more than 70% in one day (that would be a data error).
    daily_ratio = (prices / prices.shift(1)).iloc[1:]  # (the first row has no day before it)
    assert ((daily_ratio > 0.3) & (daily_ratio < 3.0) | daily_ratio.isna()).all().all()


def test_snapshot_missing_days_are_only_weekday_holidays(snapshot):
    missing = data.find_missing_days(snapshot.prices)

    assert isinstance(missing, pd.DatetimeIndex)
    assert (missing.dayofweek < 5).all()  # weekends are never reported
    assert (snapshot.prices.index.dayofweek < 5).all()  # and there are no weekend rows either
    # About 9-11 US market holidays a year; 5 years should be nowhere near, say, 80 days.
    assert len(missing) <= 11 * 6
    per_year = pd.Series(missing.year).value_counts()
    assert (per_year <= 11).all()
    # A holiday is a single day. Two missing weekdays in a row (Friday + Monday count as in a
    # row) would look like the provider lost data, not like a holiday.
    days = missing.to_numpy().astype("datetime64[D]")
    weekdays_between_neighbours = np.busday_count(days[:-1], days[1:])  # 1 = next weekday
    assert not (weekdays_between_neighbours == 1).any()


def test_snapshot_has_the_well_known_market_holidays_missing(snapshot):
    # Look at each calendar year that lies completely inside the data: Christmas Day and
    # Independence Day (when on a weekday) must be absent, because the US market is closed.
    missing = set(data.find_missing_days(snapshot.prices))
    full_years = [
        year
        for year in range(snapshot.start.year, snapshot.as_of.year + 1)
        if pd.Timestamp(year, 1, 1) >= snapshot.start and pd.Timestamp(year, 12, 31) <= snapshot.as_of
    ]
    assert full_years, "the snapshot should contain at least one complete calendar year"
    for year in full_years:
        for holiday in (pd.Timestamp(year, 12, 25), pd.Timestamp(year, 7, 4)):
            if holiday.dayofweek < 5:
                assert holiday in missing, f"{holiday:%Y-%m-%d} should be a market holiday"
                assert holiday not in snapshot.prices.index


def test_snapshot_meta_agrees_with_the_csv():
    # Read the CSV as plain text (not through load_demo) so the check is independent of the loader.
    lines = (SNAPSHOT_DIR / PRICES_FILE).read_text(encoding="utf-8").strip().splitlines()
    header, rows = lines[0].split(","), lines[1:]
    meta = snapshot_meta()

    assert meta["rows"] == len(rows)
    assert meta["start"] == rows[0].split(",")[0]
    assert meta["as_of"] == rows[-1].split(",")[0]
    assert meta["tickers"] == header[1:]
    assert header[0] == "Date"


def test_snapshot_meta_fields(snapshot):
    meta = snapshot_meta()

    assert set(meta) == {"as_of", "start", "rows", "tickers", "source", "created_at"}
    assert meta["rows"] == len(snapshot.prices)
    assert pd.Timestamp(meta["start"]) == snapshot.start
    assert pd.Timestamp(meta["as_of"]) == snapshot.as_of
    assert meta["source"] == snapshot.source
    assert "Yahoo" in meta["source"]
    # created_at is a UTC time written as ISO text, for example 2026-10-20T18:32:44Z
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", meta["created_at"])
    created = pd.Timestamp(meta["created_at"])
    assert created.tzinfo is not None and str(created.tzinfo) == "UTC"
    # In New York's calendar (the market's), the snapshot was made AFTER the last day it holds.
    assert created.tz_convert("America/New_York").normalize().tz_localize(None) > snapshot.as_of


def test_snapshot_csv_floats_round_trip_exactly(snapshot):
    # The CSV stores each price with full precision, so reading it back changes no digit:
    # every cell's text, read as a float, must give back the same number as the loaded table.
    lines = (SNAPSHOT_DIR / PRICES_FILE).read_text(encoding="utf-8").strip().splitlines()[1:]
    text_values = np.array(
        [[float(cell) if cell else np.nan for cell in line.split(",")[1:]] for line in lines]
    )
    np.testing.assert_array_equal(text_values, snapshot.prices.to_numpy())


def test_snapshot_gives_the_same_table_load_prices_gives(snapshot, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_today", lambda: snapshot.as_of + pd.Timedelta(days=1))
    tickers = list(snapshot.prices.columns)
    start, end = f"{snapshot.start:%Y-%m-%d}", f"{snapshot.as_of + pd.Timedelta(days=1):%Y-%m-%d}"
    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.copyfile(SNAPSHOT_DIR / PRICES_FILE, data._cache_path(cache, tickers, start, end))

    via_load_prices = data.load_prices(tickers, start, end, cache_dir=cache)

    pd.testing.assert_frame_equal(snapshot.prices, via_load_prices, check_exact=True)


def test_snapshot_loads_with_the_default_folder_too(snapshot):
    pd.testing.assert_frame_equal(load_demo().prices, snapshot.prices, check_exact=True)


def test_snapshot_folder_holds_exactly_the_three_files():
    # (hidden files such as .DS_Store, which macOS adds by itself, do not count)
    visible = sorted(p.name for p in SNAPSHOT_DIR.iterdir() if not p.name.startswith("."))
    assert visible == sorted([CONFIG_FILE, PRICES_FILE, META_FILE])


# --------------------------------------------------------------------------- #
# 3. refresh_demo_snapshot and the script (the download is faked)
# --------------------------------------------------------------------------- #

NOW_UTC = pd.Timestamp("2026-10-20 18:32:44", tz="UTC")  # 14:32 in New York, still the 20th


@pytest.fixture
def clock(monkeypatch):
    """Freeze the clock. New York's date is then 2026-10-20."""
    monkeypatch.setattr(data, "_now_utc", lambda: NOW_UTC)


def raw_download(columns) -> pd.DataFrame:
    """What the fake download hands back: a sorted raw table with one holiday-like empty row."""
    index = pd.bdate_range("2026-10-05", periods=8, name="Date")
    values = {
        column: [(k + 1) * 100 / 3 + day / 7 for day in range(8)] for k, column in enumerate(columns)
    }
    frame = pd.DataFrame(values, index=index)
    frame.loc[index[3]] = np.nan  # every ticker missing on one weekday: clean_prices drops it
    return frame


class FakeDownload:
    """Stands in for download_prices. Records every call; can be told to fail instead."""

    def __init__(self, error=None, frame_for=raw_download):
        self.calls = []
        self.error = error
        self.frame_for = frame_for

    def __call__(self, tickers, start, end):
        self.calls.append((list(tickers), pd.Timestamp(start), pd.Timestamp(end)))
        if self.error is not None:
            raise self.error
        return self.frame_for(list(tickers))


@pytest.fixture
def fake_download(monkeypatch):
    fake = FakeDownload()
    monkeypatch.setattr(data, "download_prices", fake)
    return fake


def test_refresh_writes_the_two_files_and_returns_the_meta(demo_dir, clock, fake_download):
    # Start from a folder that has a config and a stale snapshot: refresh replaces the snapshot.
    (demo_dir / PRICES_FILE).write_text("stale")
    (demo_dir / META_FILE).write_text("{}")

    meta = data.refresh_demo_snapshot(demo_dir)

    assert (demo_dir / CONFIG_FILE).exists()
    assert json.loads((demo_dir / META_FILE).read_text()) == meta
    assert set(meta) == {"as_of", "start", "rows", "tickers", "source", "created_at"}
    assert meta["tickers"] == ["AAPL", "MSFT", "JPM", "SPY"]  # holdings in config order, then the benchmark
    assert meta["rows"] == 7  # 8 downloaded rows, one all-NaN row cleaned away
    assert meta["start"] == "2026-10-05"
    assert meta["as_of"] == "2026-10-14"
    assert meta["created_at"] == "2026-10-20T18:32:44Z"  # UTC, ISO
    assert meta["source"].startswith("Yahoo Finance via yfinance ")
    assert data.yf.__version__ in meta["source"]


def test_refresh_output_loads_back_exactly(demo_dir, clock, fake_download):
    data.refresh_demo_snapshot(demo_dir)

    demo = load_demo(demo_dir)  # also proves the three files agree with each other

    expected = data.clean_prices(raw_download(["AAPL", "MSFT", "JPM", "SPY"]))
    pd.testing.assert_frame_equal(demo.prices, expected, check_exact=True)
    assert demo.as_of == pd.Timestamp("2026-10-14")
    assert demo.holdings == HOLDINGS


def test_refresh_asks_for_the_holdings_plus_the_benchmark_over_years_ending_today(demo_dir, clock, fake_download):
    data.refresh_demo_snapshot(demo_dir)

    assert len(fake_download.calls) == 1
    tickers, start, end = fake_download.calls[0]
    assert tickers == ["AAPL", "MSFT", "JPM", "SPY"]
    assert start == pd.Timestamp("2021-10-20")  # "years": 5 -> five calendar years back
    assert end == pd.Timestamp("2026-10-20")  # New York's today; excluded by the downloader


def test_refresh_uses_the_new_york_date_not_the_utc_date(demo_dir, monkeypatch, fake_download):
    # 02:00 UTC on the 21st is still the evening of the 20th in New York.
    monkeypatch.setattr(data, "_now_utc", lambda: pd.Timestamp("2026-10-21 02:00", tz="UTC"))
    data.refresh_demo_snapshot(demo_dir)
    assert fake_download.calls[0][2] == pd.Timestamp("2026-10-20")


@pytest.mark.parametrize(
    "today, years, expected_start",
    [
        ("2026-10-20", 5, "2021-10-20"),
        ("2026-10-20", 1, "2025-10-20"),
        ("2026-10-20", 10, "2016-10-20"),
        ("2024-02-29", 1, "2023-02-28"),  # no 29 February last year: the calendar clips to the 28th
        ("2026-10-20", 2.5, "2024-04-20"),  # 913 days back (2.5 * 365.25, rounded)
        ("2026-10-20", 0.5, "2026-04-20"),  # 183 days back
    ],
)
def test_demo_window(monkeypatch, today, years, expected_start):
    monkeypatch.setattr(data, "_now_utc", lambda: pd.Timestamp(f"{today} 16:00", tz="UTC"))
    start, end = data._demo_window(years)
    assert end == pd.Timestamp(today)
    assert start == pd.Timestamp(expected_start)
    assert start.tzinfo is None and end.tzinfo is None


def test_refresh_never_uses_the_cache_or_load_prices(demo_dir, tmp_path, clock, fake_download, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(data, "_default_cache_dir", lambda: cache)

    def forbidden(*args, **kwargs):
        raise AssertionError("refresh must download fresh data, not go through load_prices/the cache")

    monkeypatch.setattr(data, "load_prices", forbidden)
    data.refresh_demo_snapshot(demo_dir)
    assert not cache.exists()


def test_refresh_leaves_no_temp_files_behind(demo_dir, clock, fake_download):
    data.refresh_demo_snapshot(demo_dir)
    assert sorted(p.name for p in demo_dir.iterdir()) == sorted([CONFIG_FILE, PRICES_FILE, META_FILE])


def test_refresh_does_not_change_the_config(demo_dir, clock, fake_download):
    before = (demo_dir / CONFIG_FILE).read_bytes()
    data.refresh_demo_snapshot(demo_dir)
    assert (demo_dir / CONFIG_FILE).read_bytes() == before


def test_refresh_is_repeatable(demo_dir, clock, fake_download):
    data.refresh_demo_snapshot(demo_dir)
    first = (demo_dir / PRICES_FILE).read_bytes(), (demo_dir / META_FILE).read_bytes()
    data.refresh_demo_snapshot(demo_dir)
    assert ((demo_dir / PRICES_FILE).read_bytes(), (demo_dir / META_FILE).read_bytes()) == first


@pytest.mark.parametrize(
    "error",
    [
        ProviderUnavailableError("Yahoo is not answering"),
        PriceDataError("No price data returned for: ZZZZ."),
        ValueError("bad dates"),
    ],
)
def test_a_failed_download_changes_nothing(demo_dir, clock, monkeypatch, error):
    before = {name: (demo_dir / name).read_bytes() for name in (CONFIG_FILE, PRICES_FILE, META_FILE)}
    monkeypatch.setattr(data, "download_prices", FakeDownload(error=error))

    with pytest.raises(type(error)):
        data.refresh_demo_snapshot(demo_dir)

    assert {name: (demo_dir / name).read_bytes() for name in before} == before
    assert sorted(p.name for p in demo_dir.iterdir()) == sorted(before)  # and no temp files


def test_refresh_refuses_a_download_with_no_rows(demo_dir, clock, monkeypatch):
    monkeypatch.setattr(data, "download_prices", FakeDownload(frame_for=lambda cols: pd.DataFrame(columns=cols)))
    before = (demo_dir / PRICES_FILE).read_bytes()
    with pytest.raises(PriceDataError, match="No price rows"):
        data.refresh_demo_snapshot(demo_dir)
    assert (demo_dir / PRICES_FILE).read_bytes() == before


def test_refresh_refuses_a_download_where_one_ticker_is_empty(demo_dir, clock, monkeypatch):
    def one_empty(columns):
        frame = raw_download(columns)
        frame["MSFT"] = np.nan
        return frame

    monkeypatch.setattr(data, "download_prices", FakeDownload(frame_for=one_empty))
    before = (demo_dir / PRICES_FILE).read_bytes()
    with pytest.raises(PriceDataError, match="MSFT"):
        data.refresh_demo_snapshot(demo_dir)
    assert (demo_dir / PRICES_FILE).read_bytes() == before


@pytest.mark.parametrize("years", [None, 0, -1, "5", True, float("nan")])
def test_refresh_needs_a_sensible_years_value_and_does_not_download_without_it(demo_dir, clock, fake_download, years):
    config = config_dict(years=years)
    if years is None:
        del config["years"]
    write_demo(demo_dir, config=config)
    with pytest.raises(ValueError, match="years"):
        data.refresh_demo_snapshot(demo_dir)
    assert fake_download.calls == []


def test_refresh_with_a_missing_config_is_a_clear_error(tmp_path, clock, fake_download):
    with pytest.raises(ValueError, match="demo_portfolio.json is missing"):
        data.refresh_demo_snapshot(tmp_path)
    assert fake_download.calls == []


# --- the script ---------------------------------------------------------------

def load_script():
    """Import scripts/refresh_demo_data.py (it is not a package, so load it by path)."""
    spec = importlib.util.spec_from_file_location("refresh_demo_data_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exists_and_is_a_thin_wrapper():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "refresh_demo_snapshot" in source  # the logic lives in signaldesk/data.py
    assert "yf.download" not in source and "import yfinance" not in source  # it never talks to Yahoo itself
    assert len(source.splitlines()) < 100


def test_script_refreshes_and_prints_a_summary(demo_dir, clock, fake_download, capsys):
    script = load_script()

    exit_code = script.main(["--demo-dir", str(demo_dir)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert str(demo_dir / PRICES_FILE) in out and str(demo_dir / META_FILE) in out
    assert "AAPL, MSFT, JPM, SPY" in out
    assert "rows      : 7" in out
    assert "2026-10-05 to 2026-10-14" in out
    assert "find_gaps" in out and "missing_count" in out  # the gaps table
    assert "find_missing_days" in out
    assert "2026-10-08 Thu" in out  # the weekday whose all-NaN row was cleaned away
    assert load_demo(demo_dir).as_of == pd.Timestamp("2026-10-14")  # and the files are valid


def test_script_uses_the_project_demo_folder_by_default(tmp_path, clock, fake_download, monkeypatch, capsys):
    # (Redirected to a temporary folder so this test can never overwrite the committed snapshot.)
    folder = write_demo(tmp_path / "redirected")
    monkeypatch.setattr(data, "_default_demo_dir", lambda: folder)
    assert load_script().main([]) == 0
    assert load_demo(folder).as_of == pd.Timestamp("2026-10-14")
    capsys.readouterr()


def test_script_reports_a_failed_download_and_exits_with_1(demo_dir, clock, monkeypatch, capsys):
    monkeypatch.setattr(data, "download_prices", FakeDownload(error=ProviderUnavailableError("Yahoo says no")))
    before = (demo_dir / PRICES_FILE).read_bytes()

    exit_code = load_script().main(["--demo-dir", str(demo_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Could not refresh the demo data" in captured.err and "Yahoo says no" in captured.err
    assert captured.out == ""
    assert (demo_dir / PRICES_FILE).read_bytes() == before


@pytest.mark.parametrize("error", [PermissionError("read-only file system"), OSError("No space left on device")])
def test_script_reports_a_write_problem_instead_of_a_traceback(demo_dir, clock, fake_download, monkeypatch, capsys, error):
    def refuse(src, dst):
        raise error

    monkeypatch.setattr(data.os, "replace", refuse)  # the download works, saving the files does not
    before = (demo_dir / PRICES_FILE).read_bytes()

    exit_code = load_script().main(["--demo-dir", str(demo_dir)])  # must not raise

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Could not refresh the demo data" in captured.err and str(error) in captured.err
    assert (demo_dir / PRICES_FILE).read_bytes() == before
    assert sorted(p.name for p in demo_dir.iterdir()) == sorted([CONFIG_FILE, PRICES_FILE, META_FILE])  # no temp files


def test_script_reports_a_broken_config_and_exits_with_1(tmp_path, clock, fake_download, capsys):
    (tmp_path / CONFIG_FILE).write_text("{broken")
    assert load_script().main(["--demo-dir", str(tmp_path)]) == 1
    assert "not valid JSON" in capsys.readouterr().err
    assert fake_download.calls == []
