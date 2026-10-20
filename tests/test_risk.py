"""Tests for signaldesk/risk.py. Fully offline and deterministic: no network, no data.py.

Every expected number below is worked out by hand in a comment next to it, using tiny
tables built right in the test. The tests are written so that a WRONG formula fails:
for example a volatility that is not annualized, a VaR with the wrong sign, or a rolling
window that peeks into the future would each break at least one test here.

Warnings: risk.py warns (ShortSampleWarning) when a number rests on few observations, and
(GapWarning) when returns span missing prices. Our hand-checkable examples are tiny and
gappy on purpose. So the whole file ignores those two warnings by default (the lines
below), and the tests that are ABOUT a warning switch it on explicitly with ``pytest.warns``
(to prove it fires) or ``warnings.simplefilter("error", ...)`` (to prove it does not).
SkippedTickerWarning is NOT ignored: a test that triggers it must say so.
"""

import copy
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from signaldesk import risk
from signaldesk.risk import (
    MAX_HOLIDAY_GAP_ROWS,
    MIN_RELIABLE_OBS,
    GapWarning,
    ShortSampleWarning,
    SkippedTickerWarning,
    annualized_return,
    annualized_volatility,
    correlation_matrix,
    cumulative_returns,
    daily_returns,
    historical_var,
    max_drawdown,
    risk_summary,
    sharpe_ratio,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore::signaldesk.risk.ShortSampleWarning"),
    pytest.mark.filterwarnings("ignore::signaldesk.risk.GapWarning"),
]

NAN = float("nan")


# --------------------------------------------------------------------------- #
# Helpers: small hand-made data
# --------------------------------------------------------------------------- #

def dates(n: int) -> pd.DatetimeIndex:
    """n business days, like the data layer's index (DatetimeIndex, ascending, unique)."""
    return pd.bdate_range("2026-01-05", periods=n, name="Date")


def make_series(values, name="AAPL") -> pd.Series:
    return pd.Series(values, index=dates(len(values)), name=name, dtype="float64")


def make_frame(columns: dict) -> pd.DataFrame:
    """Build a table from {"AAPL": [...], "MSFT": [...]} (all lists the same length)."""
    n = len(next(iter(columns.values())))
    return pd.DataFrame({k: pd.Series(v, dtype="float64").to_numpy() for k, v in columns.items()}, index=dates(n))


def alternating(n: int, amount: float = 0.01) -> np.ndarray:
    """A fixed return pattern: +amount, -amount, +amount, ... (n values, no randomness)."""
    return np.where(np.arange(n) % 2 == 0, amount, -amount)


def prices_from_returns(returns, name="AAPL", start=100.0) -> pd.Series:
    """Prices that produce exactly these returns: start, start*(1+r1), ..."""
    values = start * np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(returns))])
    return pd.Series(values, index=dates(len(values)), name=name)


def long_prices() -> pd.DataFrame:
    """251 rows (250 returns) for two tickers. MSFT starts 10 days late and has one gap (row 100)."""
    rng = np.random.default_rng(0)  # seeded, so every run gives the same numbers
    aapl = prices_from_returns(alternating(250), "AAPL")
    msft = prices_from_returns(rng.normal(0.0, 0.01, 250), "MSFT", start=50.0)
    prices = pd.concat([aapl, msft], axis=1)
    prices.iloc[:10, 1] = np.nan  # a later start
    prices.iloc[100, 1] = np.nan  # a gap in the middle
    return prices


def messages_of(category, function, *args, **kwargs) -> list[str]:
    """Call a function and return the text of every warning of this category it emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        function(*args, **kwargs)
    return [str(w.message) for w in caught if issubclass(w.category, category)]


def short_sample_messages(function, *args, **kwargs) -> list[str]:
    """The text of every ShortSampleWarning the call emitted."""
    return messages_of(ShortSampleWarning, function, *args, **kwargs)


def assert_no_short_sample_warning(function, *args, **kwargs):
    """Fail if the call emits a ShortSampleWarning or a GapWarning (they become errors)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShortSampleWarning)
        warnings.simplefilter("error", GapWarning)
        return function(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Module constants
# --------------------------------------------------------------------------- #

def test_constants_and_warning_class():
    assert risk.TRADING_DAYS_PER_YEAR == 252
    assert MIN_RELIABLE_OBS == 60
    assert risk.MIN_TAIL_OBS == 5
    assert MAX_HOLIDAY_GAP_ROWS == 5
    assert issubclass(ShortSampleWarning, UserWarning)
    assert issubclass(GapWarning, UserWarning)
    assert issubclass(SkippedTickerWarning, UserWarning)


# --------------------------------------------------------------------------- #
# daily_returns
# --------------------------------------------------------------------------- #

def test_daily_returns_known_values():
    # prices [100, 110, 99]: 110/100 - 1 = +0.10 ; 99/110 - 1 = -0.10
    prices = make_series([100, 110, 99])
    result = daily_returns(prices)
    assert result.tolist() == pytest.approx([0.10, -0.10], abs=1e-12)
    # The first row has no previous price and is dropped; the other dates are kept.
    assert result.index.equals(prices.index[1:])
    assert result.name == "AAPL"


def test_daily_returns_dataframe_is_per_column():
    prices = make_frame({"AAPL": [100, 110, 99], "MSFT": [50, 50, 55]})
    result = daily_returns(prices)
    # MSFT: 50/50 - 1 = 0 ; 55/50 - 1 = +0.10
    assert list(result.columns) == ["AAPL", "MSFT"]
    assert result["AAPL"].tolist() == pytest.approx([0.10, -0.10], abs=1e-12)
    assert result["MSFT"].tolist() == pytest.approx([0.0, 0.10], abs=1e-12)


def test_daily_returns_gap_row_is_nan_and_the_next_return_uses_the_last_valid_price():
    # [100, NaN, 110]: the NaN row has no return; 110 is measured against 100, the last
    # valid price: 110/100 - 1 = +0.10. (The old policy gave [NaN, NaN].)
    result = daily_returns(make_series([100, NAN, 110]))
    assert len(result) == 2
    assert math.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.10, abs=1e-12)


def test_daily_returns_gap_of_several_rows_and_compounding():
    # [100, NaN, NaN, 120, 90]: row 3 is measured against 100 (+0.20, a bridged return
    # over two NaN rows), row 4 against 120 (90/120 - 1 = -0.25).
    result = daily_returns(make_series([100, NAN, NAN, 120, 90]))
    assert result.isna().tolist() == [True, True, False, False]
    assert result.dropna().tolist() == pytest.approx([0.20, -0.25], abs=1e-12)


def test_daily_returns_each_column_is_measured_against_itself():
    # AAPL has a holiday on day 2, MSFT does not. MSFT's returns must be the plain ones,
    # whatever AAPL looks like.
    prices = make_frame({"AAPL": [100, NAN, 110, 121], "MSFT": [50, 55, 44, 44]})
    result = daily_returns(prices)
    assert result["AAPL"].isna().tolist() == [True, False, False]
    assert result["AAPL"].dropna().tolist() == pytest.approx([0.10, 0.10], abs=1e-12)
    assert result["MSFT"].tolist() == pytest.approx([0.10, -0.20, 0.0], abs=1e-12)


def test_daily_returns_warns_once_naming_each_ticker_and_its_bridged_count():
    # AAPL: two bridged returns (rows 2 and 5); MSFT: one (row 3); NEW: none.
    prices = make_frame(
        {
            "AAPL": [100, NAN, 101, 102, NAN, NAN, 103],
            "MSFT": [50, 51, NAN, 52, 53, 54, 55],
            "NEW": [NAN, NAN, 10, 11, 12, 13, 14],
        }
    )
    # Longest bridge: AAPL skips 1 NaN row (row 1) and then 2 NaN rows (rows 4-5) -> longest 2 rows;
    # MSFT skips 1 NaN row -> longest 1 row.
    messages = messages_of(GapWarning, daily_returns, prices)
    assert len(messages) == 1  # one warning per call, not per ticker
    assert "'AAPL': 2 bridged (longest 2 rows)" in messages[0]
    assert "'MSFT': 1 bridged (longest 1 row)" in messages[0]
    assert "NEW" not in messages[0]
    assert "holidays" in messages[0] and "find_gaps" in messages[0]
    assert "longer than a holiday" not in messages[0]  # the stronger wording is only for long gaps
    with pytest.warns(GapWarning):
        daily_returns(prices)


def gap_table(gap_rows: int) -> pd.DataFrame:
    """A 'normal' ticker with 3 one-row holidays, and a ticker with ONE gap of `gap_rows` rows."""
    holidays = [100.0 + i for i in range(30)]
    for row in (5, 12, 20):
        holidays[row] = NAN
    outage = [50.0 + i for i in range(30)]
    outage[10 : 10 + gap_rows] = [NAN] * gap_rows
    return make_frame({"AAPL": holidays, "XOM": outage})


def test_gap_warning_stronger_wording_appears_only_for_a_gap_longer_than_five_rows():
    # A gap of exactly MAX_HOLIDAY_GAP_ROWS (5) rows is still "holiday-like": no strong sentence.
    five = messages_of(GapWarning, daily_returns, gap_table(5))
    assert len(five) == 1
    assert "'AAPL': 3 bridged (longest 1 row)" in five[0]
    assert "'XOM': 1 bridged (longest 5 rows)" in five[0]
    assert "longer than a holiday" not in five[0] and "missing data" in five[0]
    # 6 rows is longer than any holiday: the stronger sentence appears and names the ticker
    # and the number of rows, but not the ticker with only 1-row holidays.
    six = messages_of(GapWarning, daily_returns, gap_table(6))
    assert len(six) == 1
    assert "'XOM': 1 bridged (longest 6 rows)" in six[0]
    assert "longer than a holiday: 'XOM' (6 rows)" in six[0]
    assert "likely missing data" in six[0] and "multi-week" in six[0]
    assert "'AAPL' (" not in six[0]
    # a 40-row outage says 40
    outage = make_frame({"XOM": [1.0] + [NAN] * 40 + [2.0]})
    assert "'XOM' (40 rows)" in messages_of(GapWarning, daily_returns, outage)[0]


def test_gap_warning_without_any_gap_in_the_middle_stays_silent_whatever_the_edges():
    # a 40-row hole at the START (a late starter) is not a bridge, so no warning at all
    assert messages_of(GapWarning, daily_returns, make_frame({"NEW": [NAN] * 40 + [1.0, 2.0, 3.0]})) == []


def test_daily_returns_leading_and_trailing_nan_are_not_bridges():
    # a late start (leading NaN), an early end (trailing NaN), and no gap in the middle
    prices = make_frame({"A": [NAN, NAN, 100, 101, 102, NAN], "B": [10, 11, 12, 13, NAN, NAN]})
    assert messages_of(GapWarning, daily_returns, prices) == []
    assert_no_short_sample_warning(daily_returns, prices)  # i.e. no GapWarning either
    result = daily_returns(prices)
    # A: the first valid price (row 2) has no return, so rows 1-2 are NaN; trailing row is NaN
    assert result["A"].isna().tolist() == [True, True, False, False, True]
    assert result["A"].dropna().tolist() == pytest.approx([0.01, 1 / 101], abs=1e-12)


def test_daily_returns_clean_data_does_not_warn():
    assert_no_short_sample_warning(daily_returns, make_series([100, 110, 99]))


def test_daily_returns_ticker_that_starts_later():
    prices = make_frame({"AAPL": [100, 110, 99, 99], "NEW": [NAN, NAN, 50, 55]})
    result = daily_returns(prices)
    # NEW: row1 NaN/NaN -> NaN ; row2 50/NaN -> NaN ; row3 55/50 - 1 = +0.10
    assert result["NEW"].isna().tolist() == [True, True, False]
    assert result["NEW"].iloc[2] == pytest.approx(0.10, abs=1e-12)
    # AAPL: +0.10, -0.10, 0.0 are unaffected by NEW's missing start.
    assert result["AAPL"].tolist() == pytest.approx([0.10, -0.10, 0.0], abs=1e-12)


def test_daily_returns_accepts_integer_prices():
    result = daily_returns(pd.Series([100, 110, 99], index=dates(3)))
    assert result.tolist() == pytest.approx([0.10, -0.10], abs=1e-12)


# --------------------------------------------------------------------------- #
# cumulative_returns
# --------------------------------------------------------------------------- #

def test_cumulative_returns_known_values():
    # P_t / P_first : [100, 110, 99, 120] -> [1.0, 1.10, 0.99, 1.20]
    result = cumulative_returns(make_series([100, 110, 99, 120]))
    assert result.tolist() == pytest.approx([1.0, 1.10, 0.99, 1.20], abs=1e-12)
    assert len(result) == 4  # unlike daily_returns, no row is dropped
    assert result.name == "AAPL"


def test_cumulative_returns_late_start_and_gap():
    prices = make_frame({"AAPL": [100, NAN, 110, 99], "NEW": [NAN, 50, 55, 44]})
    result = cumulative_returns(prices)
    # AAPL: gap stays NaN, the base is the first valid price (100): [1.0, NaN, 1.10, 0.99]
    assert result["AAPL"].tolist()[0] == pytest.approx(1.0)
    assert math.isnan(result["AAPL"].iloc[1])
    assert result["AAPL"].iloc[2:].tolist() == pytest.approx([1.10, 0.99], abs=1e-12)
    # NEW: NaN before its first valid price, then 1.0 on its first day: [NaN, 1.0, 1.10, 0.88]
    assert math.isnan(result["NEW"].iloc[0])
    assert result["NEW"].iloc[1:].tolist() == pytest.approx([1.0, 1.10, 0.88], abs=1e-12)


# --------------------------------------------------------------------------- #
# annualized_return
# --------------------------------------------------------------------------- #

def test_annualized_return_is_geometric_not_arithmetic():
    # returns [+0.10, -0.10]: the arithmetic mean is 0, so "mean * 252" would say 0.
    # But the money shrank: 1.10 * 0.90 = 0.99, and n = 2.
    # (0.99) ** (252 / 2) - 1 = 0.99 ** 126 - 1 = -0.71814...
    result = annualized_return(make_series([0.10, -0.10]))
    assert result == pytest.approx(0.99**126 - 1, rel=1e-9)
    assert abs(result) > 0.5  # nowhere near the arithmetic answer of 0


def test_annualized_return_over_exactly_one_year():
    # 252 returns of +0.1% each: the full-year growth is 1.001 ** 252, so n = 252 and the
    # exponent is 252/252 = 1: (1.001 ** 252) - 1 = 0.28650...
    result = annualized_return(make_series([0.001] * 252))
    assert result == pytest.approx(1.001**252 - 1, rel=1e-9)


def test_annualized_return_scales_to_a_year():
    # 126 returns of +1% (half a year): growth 1.01 ** 126, exponent 252/126 = 2,
    # so the annual rate is (1.01 ** 126) ** 2 - 1 = 1.01 ** 252 - 1 = 11.2...
    result = annualized_return(make_series([0.01] * 126))
    assert result == pytest.approx(1.01**252 - 1, rel=1e-9)


def test_annualized_return_drops_nan_and_counts_only_valid_returns():
    # [0.01, NaN, 0.01] has n = 2 valid returns: (1.01 * 1.01) ** (252/2) - 1 = 1.01 ** 252 - 1
    result = annualized_return(make_series([0.01, NAN, 0.01]))
    assert result == pytest.approx(1.01**252 - 1, rel=1e-9)


def test_annualized_return_dataframe_and_total_loss():
    returns = make_frame({"AAPL": [0.10, -0.10, NAN], "WIPED": [-1.0, 0.1, 0.1]})
    result = annualized_return(returns)
    assert list(result.index) == ["AAPL", "WIPED"]
    assert result["AAPL"] == pytest.approx(0.99**126 - 1, rel=1e-9)
    # A -100% day wipes everything out: the product is 0, so the result is exactly -1.
    assert result["WIPED"] == -1.0


def test_annualized_return_rejects_a_return_below_minus_one():
    with pytest.raises(ValueError, match="below -1"):
        annualized_return(make_series([0.01, 0.02, -1.5]))


# --------------------------------------------------------------------------- #
# annualized_volatility
# --------------------------------------------------------------------------- #

def test_volatility_known_value():
    # returns [0.10, -0.10]: mean 0, squared deviations 0.01 + 0.01 = 0.02.
    # sample variance (ddof=1) = 0.02 / (2 - 1) = 0.02 -> std = 0.141421...
    # annualized: 0.141421... * sqrt(252) = 0.1 * sqrt(504) = 2.24499...
    # (not annualized would be 0.1414; ddof=0 would give 0.1 * sqrt(252) = 1.5874)
    result = annualized_volatility(make_series([0.10, -0.10]))
    assert result == pytest.approx(0.1 * math.sqrt(504), rel=1e-12)
    assert isinstance(result, float)


def test_volatility_dataframe_is_per_column():
    returns = make_frame({"A": [0.1, -0.1, 0.1, -0.1], "B": [0.02, 0.0, 0.02, 0.0]})
    # A: mean 0, sum of squares 4 * 0.01 = 0.04, / (4 - 1) -> std = 0.1 * sqrt(4/3)
    # B: mean 0.01, deviations +-0.01, sum of squares 0.0004, / 3 -> std = 0.01 * sqrt(4/3)
    # annualized: std * sqrt(252): A = 0.1 * sqrt(336) = 1.8330..., B = 0.01 * sqrt(336) = 0.18330...
    result = annualized_volatility(returns)
    assert list(result.index) == ["A", "B"]
    assert result["A"] == pytest.approx(0.1 * math.sqrt(336), rel=1e-12)
    assert result["B"] == pytest.approx(0.01 * math.sqrt(336), rel=1e-12)


def test_volatility_drops_nan_per_column():
    # [0.10, NaN, -0.10] has the same 2 valid returns as [0.10, -0.10].
    assert annualized_volatility(make_series([0.10, NAN, -0.10])) == pytest.approx(0.1 * math.sqrt(504), rel=1e-12)
    # NEW starts late: only its 4 valid returns count, so it equals B above.
    returns = make_frame({"OLD": [0.1, -0.1, 0.1, -0.1, 0.1], "NEW": [NAN, 0.02, 0.0, 0.02, 0.0]})
    result = annualized_volatility(returns)
    assert result["NEW"] == pytest.approx(0.01 * math.sqrt(336), rel=1e-12)


def test_rolling_volatility_known_values_use_only_the_past():
    returns = make_series([0.00, 0.01, 0.02, 0.04, 0.06])
    result = annualized_volatility(returns, window=3)
    # window 3, std with ddof=1, then * sqrt(252). Each value uses the 3 returns ending that day.
    # t0, t1: not enough history -> NaN
    # t2: (0.00, 0.01, 0.02): evenly spaced by 0.01 -> std = 0.01
    # t3: (0.01, 0.02, 0.04): mean 0.07/3, sum of squares 7/15000, / 2 -> std = sqrt(7/30000) = 0.015275...
    # t4: (0.02, 0.04, 0.06): evenly spaced by 0.02 -> std = 0.02
    assert len(result) == 5
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(0.01 * math.sqrt(252), rel=1e-9)
    assert result.iloc[3] == pytest.approx(math.sqrt(7 / 30000) * math.sqrt(252), rel=1e-9)
    assert result.iloc[4] == pytest.approx(0.02 * math.sqrt(252), rel=1e-9)
    assert result.index.equals(returns.index)
    assert result.name == "AAPL"


def test_rolling_volatility_is_not_centered():
    # A CENTERED window would give t1 a value (it would use t0, t1, t2) and give t2 the
    # value of (0.01, 0.02, 0.04) = 0.015275 instead of 0.01. The trailing window must not.
    result = annualized_volatility(make_series([0.00, 0.01, 0.02, 0.04, 0.06]), window=3)
    assert math.isnan(result.iloc[1])
    assert result.iloc[2] != pytest.approx(math.sqrt(7 / 30000) * math.sqrt(252), rel=1e-6)


def test_rolling_volatility_has_no_look_ahead():
    # Change only the LAST two returns. Anything computed from the past must not move.
    original = make_series([0.00, 0.01, 0.02, 0.04, 0.06])
    changed = original.copy()
    changed.iloc[3:] = [0.50, -0.30]
    before = annualized_volatility(original, window=3)
    after = annualized_volatility(changed, window=3)
    pd.testing.assert_series_equal(before.iloc[:3], after.iloc[:3])  # t0..t2 unchanged
    assert before.iloc[3] != pytest.approx(after.iloc[3])  # t3 does use the changed value


def test_rolling_volatility_skips_missing_returns_and_does_not_blank_the_next_days():
    # window 2 over [0.01, 0.02, NaN, 0.04, 0.05, 0.06]. The NaN row has no value itself,
    # but the window just skips it (it uses the ticker's own valid returns):
    #   t1: (0.01, 0.02)  -> |0.01| / sqrt(2) * sqrt(252) = 0.01 * sqrt(126)
    #   t3: (0.02, 0.04)  -> 0.02 * sqrt(126)   (the pair spans the NaN row)
    #   t4: (0.04, 0.05)  -> 0.01 * sqrt(126)
    #   t5: (0.05, 0.06)  -> 0.01 * sqrt(126)
    # t0 has no full window and t2 is the NaN row itself: NaN.
    returns = make_series([0.01, 0.02, NAN, 0.04, 0.05, 0.06])
    result = annualized_volatility(returns, window=2)
    unit = 0.01 * math.sqrt(126)
    assert result.isna().tolist() == [True, False, True, False, False, False]
    assert result.dropna().tolist() == pytest.approx([unit, 2 * unit, unit, unit], rel=1e-9)


def test_rolling_volatility_holidays_do_not_blank_out_a_long_window():
    # 120 returns with 3 NaN "holidays". The ticker has 117 valid returns, so with window 60
    # it has 117 - 59 = 58 values (all the full windows), not "a NaN kills 60 days" (which
    # would leave far fewer).
    values = alternating(120, 0.01) * np.linspace(1.0, 2.0, 120)
    values[[30, 60, 90]] = NAN
    result = annualized_volatility(make_series(values), window=60)
    assert result.notna().sum() == 117 - 59
    assert result.iloc[[30, 60, 90]].isna().all()  # the holiday dates themselves stay NaN
    assert result.iloc[:31].isna().all()  # 60 valid returns are not there before the date 63


def test_rolling_volatility_of_a_ticker_does_not_depend_on_the_other_tickers():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.01, 100)
    b = rng.normal(0, 0.02, 100)
    a[[10, 40]] = NAN  # AAPL's holidays
    b[[25, 70, 71]] = NAN  # another exchange's holidays
    table = make_frame({"AAPL": a, "OTHER": b})
    together = annualized_volatility(table, window=30)
    alone = annualized_volatility(table[["AAPL"]], window=30)
    pd.testing.assert_series_equal(together["AAPL"], alone["AAPL"])
    pd.testing.assert_series_equal(together["AAPL"], annualized_volatility(table["AAPL"], window=30))


def test_rolling_volatility_dataframe_keeps_shape():
    returns = make_frame({"A": [0.1, -0.1, 0.1, -0.1], "B": [0.0, 0.0, 0.01, 0.03]})
    result = annualized_volatility(returns, window=2)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["A", "B"]
    assert result.index.equals(returns.index)
    # A, window 2: (0.1, -0.1) -> |0.2| / sqrt(2) * sqrt(252) = 0.2 * sqrt(126) at every full window
    assert result["A"].iloc[1:].tolist() == pytest.approx([0.2 * math.sqrt(126)] * 3, rel=1e-9)
    # B, window 2, last window (0.01, 0.03): |0.02| / sqrt(2) * sqrt(252) = 0.02 * sqrt(126)
    assert result["B"].iloc[3] == pytest.approx(0.02 * math.sqrt(126), rel=1e-9)
    assert result["B"].iloc[1] == pytest.approx(0.0, abs=1e-12)


def test_rolling_volatility_accepts_numpy_integer_window():
    returns = make_series([0.00, 0.01, 0.02, 0.04, 0.06])
    pd.testing.assert_series_equal(annualized_volatility(returns, window=np.int64(3)), annualized_volatility(returns, window=3))


@pytest.mark.parametrize("window", [0, 1, -3, 2.5, 3.0, "5", True, [3]])
def test_volatility_rejects_a_bad_window(window):
    with pytest.raises(ValueError, match="window"):
        annualized_volatility(make_series([0.01, 0.02, 0.03, 0.04]), window=window)


def test_volatility_rejects_a_window_longer_than_the_data():
    with pytest.raises(ValueError, match="window=10"):
        annualized_volatility(make_series([0.01, 0.02, 0.03]), window=10)


def test_rolling_volatility_warns_for_a_short_window_only():
    returns = make_series(alternating(30))
    with pytest.warns(ShortSampleWarning, match="window=19"):
        annualized_volatility(returns, window=19)
    # window = 20 is fine, and the rolling form never warns per point, even with 30 rows (< 60).
    assert_no_short_sample_warning(annualized_volatility, returns, window=20)


# --------------------------------------------------------------------------- #
# correlation_matrix
# --------------------------------------------------------------------------- #

def test_correlation_perfect_relationships():
    # B = 2 * A moves exactly with A (+1). C is A reversed: when A rises, C falls (-1).
    returns = make_frame(
        {
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.02, 0.04, 0.06, 0.08],
            "C": [0.04, 0.03, 0.02, 0.01],
        }
    )
    result = correlation_matrix(returns)
    expected = np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    assert result.to_numpy() == pytest.approx(expected, abs=1e-12)
    assert list(result.index) == ["A", "B", "C"]
    assert list(result.columns) == ["A", "B", "C"]
    assert result.equals(result.T)  # symmetric


def test_correlation_unrelated_is_zero():
    # A = (+1, -1, +1, -1), B = (+1, +1, -1, -1) (times 0.01): both have mean 0 and
    # sum(A*B) = 1 - 1 - 1 + 1 = 0 (times 0.0001), so cov = 0 and corr = 0.
    returns = make_frame({"A": [0.01, -0.01, 0.01, -0.01], "B": [0.01, 0.01, -0.01, -0.01]})
    assert correlation_matrix(returns).loc["A", "B"] == pytest.approx(0.0, abs=1e-12)


def test_correlation_known_non_trivial_value():
    # A = [1, 2, 3, 4], B = [2, 1, 4, 3] (times 0.01, so they look like returns; the scale
    # does not change a correlation): both mean 2.5.
    # deviations A: -1.5 -0.5 +0.5 +1.5 ; B: -0.5 -1.5 +1.5 +0.5
    # sum of products = 0.75 + 0.75 + 0.75 + 0.75 = 3 ; sums of squares: 5 and 5
    # corr = 3 / sqrt(5 * 5) = 0.6
    returns = make_frame({"A": [0.01, 0.02, 0.03, 0.04], "B": [0.02, 0.01, 0.04, 0.03]})
    assert correlation_matrix(returns).loc["A", "B"] == pytest.approx(0.6, abs=1e-12)


def test_correlation_nan_is_handled_pairwise():
    # C has no value on the first day. Pairwise: A/B still use all 4 days (corr 0.6 as above),
    # whereas dropping every row with any NaN would use only 3 days and give 0.6547.
    returns = make_frame({"A": [0.01, 0.02, 0.03, 0.04], "B": [0.02, 0.01, 0.04, 0.03], "C": [NAN, 0.01, 0.02, 0.03]})
    result = correlation_matrix(returns)
    assert result.loc["A", "B"] == pytest.approx(0.6, abs=1e-12)
    # A/C use days 2-4: A = [2, 3, 4], C = [1, 2, 3] -> exactly +1
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-12)
    # B/C use days 2-4: B = [1, 4, 3], C = [1, 2, 3]: means 8/3 and 2,
    # sum of products 2, sums of squares 14/3 and 2 -> corr = 2 / sqrt(28/3) = 0.65465...
    assert result.loc["B", "C"] == pytest.approx(2 / math.sqrt(28 / 3), abs=1e-12)


def test_correlation_pair_without_enough_shared_days_is_nan():
    # A and B share only day 3, and one point cannot define a correlation.
    returns = make_frame({"A": [0.01, 0.02, 0.03, NAN, NAN], "B": [NAN, NAN, 0.03, 0.04, 0.05]})
    result = correlation_matrix(returns)
    assert math.isnan(result.loc["A", "B"])
    assert result.loc["A", "A"] == pytest.approx(1.0)


def test_correlation_of_a_constant_column_is_nan_including_its_diagonal():
    returns = make_frame({"FLAT": [0.01, 0.01, 0.01, 0.01], "A": [0.01, 0.02, 0.03, 0.05]})
    result = correlation_matrix(returns)
    assert math.isnan(result.loc["FLAT", "A"])
    assert math.isnan(result.loc["FLAT", "FLAT"])  # the diagonal is NaN too, not 1.0
    assert result.loc["A", "A"] == pytest.approx(1.0)


def test_correlation_needs_a_dataframe_with_two_columns():
    with pytest.raises(ValueError, match="at least 2 columns"):
        correlation_matrix(make_series([0.01, 0.02, 0.03]))
    with pytest.raises(ValueError, match="at least 2 columns"):
        correlation_matrix(make_frame({"A": [0.01, 0.02, 0.03]}))


def test_correlation_needs_two_valid_returns_per_column():
    returns = make_frame({"A": [0.01, 0.02, 0.03], "B": [NAN, NAN, 0.03]})
    with pytest.raises(ValueError, match="'B' has 1"):
        correlation_matrix(returns)


def test_correlation_warns_for_a_pair_with_few_shared_days():
    returns = make_frame({"AAPL": alternating(10), "MSFT": alternating(10, 0.02)})
    with pytest.warns(ShortSampleWarning, match=r"'AAPL'/'MSFT': 10"):
        correlation_matrix(returns)


def test_correlation_warns_when_overlap_is_short_even_if_columns_are_long():
    # Both columns have 80 valid returns, but they only share 40 of those days.
    a = np.concatenate([alternating(80), np.full(40, NAN)])
    b = np.concatenate([np.full(40, NAN), alternating(80, 0.02)])
    with pytest.warns(ShortSampleWarning, match=r"'A'/'B': 40"):
        correlation_matrix(make_frame({"A": a, "B": b}))


# --------------------------------------------------------------------------- #
# max_drawdown
# --------------------------------------------------------------------------- #

def test_max_drawdown_uses_the_running_peak_not_the_global_max():
    # prices [100, 50, 200, 150]. Running max: 100, 100, 200, 200.
    # P / running max - 1: 0, -0.50, 0, -0.25 -> the worst fall is 50%.
    # (Using the global max, 200, would wrongly say 50 / 200 - 1 = -75%.)
    result = max_drawdown(make_series([100, 50, 200, 150]))
    assert result == pytest.approx(0.50, abs=1e-12)


def test_max_drawdown_is_a_positive_fraction():
    # [100, 90, 80]: the peak is 100, the low is 80: 80 / 100 - 1 = -0.20, returned as +0.20
    result = max_drawdown(make_series([100, 90, 80]))
    assert result == pytest.approx(0.20, abs=1e-12)
    assert result > 0


def test_max_drawdown_the_low_must_come_after_the_peak():
    # [100, 80, 120]: the price falls 20% from 100 to 80 and then recovers. The 80 is
    # BEFORE the 120 peak, so (120 - 80) / 120 = 33% would be wrong. The answer is 0.20.
    assert max_drawdown(make_series([100, 80, 120])) == pytest.approx(0.20, abs=1e-12)


def test_max_drawdown_is_zero_for_a_series_that_never_falls():
    for prices in ([100, 101, 105], [100, 100, 100]):
        result = max_drawdown(make_series(prices))
        assert result == 0.0
        assert math.copysign(1.0, result) == 1.0  # a plain 0.0, not -0.0


def test_max_drawdown_skips_gaps():
    # [100, NaN, 80, 120, 60] -> valid prices [100, 80, 120, 60]; running max 100, 100, 120, 120;
    # the worst is 60 / 120 - 1 = -0.50.
    assert max_drawdown(make_series([100, NAN, 80, 120, 60])) == pytest.approx(0.50, abs=1e-12)


def test_max_drawdown_dataframe_is_per_column():
    prices = make_frame({"AAPL": [100, 50, 200, 150], "NEW": [NAN, 10, 12, 9]})
    result = max_drawdown(prices)
    # NEW: valid [10, 12, 9], peak 12, low 9: 9 / 12 - 1 = -0.25
    assert list(result.index) == ["AAPL", "NEW"]
    assert result["AAPL"] == pytest.approx(0.50, abs=1e-12)
    assert result["NEW"] == pytest.approx(0.25, abs=1e-12)


# --------------------------------------------------------------------------- #
# historical_var
# --------------------------------------------------------------------------- #

# Six returns in a shuffled order. Sorted: -0.10, -0.06, -0.02, +0.02, +0.04, +0.06.
SIX_RETURNS = [0.02, -0.10, 0.06, -0.02, 0.04, -0.06]


def test_var_known_value_pins_linear_interpolation():
    # confidence 0.95 -> look at the lowest 5%. With n = 6 the quantile sits at position
    # 0.05 * (6 - 1) = 0.25 (counting from 0), a quarter of the way from the worst return
    # (-0.10) to the second worst (-0.06):  -0.10 + 0.25 * 0.04 = -0.09.
    # VaR = -(-0.09) = +0.09 (a positive loss).
    # Other methods would differ: "lower" -> 0.10, "higher" -> 0.06, "midpoint" -> 0.08.
    result = historical_var(make_series(SIX_RETURNS), confidence=0.95)
    assert result == pytest.approx(0.09, abs=1e-12)


def test_var_is_a_positive_loss_from_the_lower_tail():
    result = historical_var(make_series(SIX_RETURNS), confidence=0.95)
    assert result > 0  # not -0.09
    # The UPPER tail (quantile 0.95) would give 0.04 + 0.75 * 0.02 = 0.055; it is not that.
    assert result != pytest.approx(0.055)


def test_var_exact_position_needs_no_interpolation():
    # confidence 0.8 -> quantile 0.2 -> position 0.2 * 5 = 1.0 -> the second worst, -0.06 -> VaR 0.06
    assert historical_var(make_series(SIX_RETURNS), confidence=0.8) == pytest.approx(0.06, abs=1e-12)


def test_var_on_a_uniform_grid_at_two_confidence_levels():
    # 101 returns: -0.050, -0.049, ..., +0.050 (steps of 0.001), shuffled with a fixed seed.
    # Positions 0.05 * 100 = 5 and 0.01 * 100 = 1, so the quantiles are the 6th and 2nd worst:
    # -0.050 + 5 * 0.001 = -0.045 (VaR 95% = 0.045) and -0.050 + 1 * 0.001 = -0.049 (VaR 99% = 0.049).
    grid = np.random.default_rng(0).permutation(np.arange(-50, 51) / 1000)
    returns = make_series(grid)
    assert historical_var(returns, 0.95) == pytest.approx(0.045, abs=1e-9)
    assert historical_var(returns, 0.99) == pytest.approx(0.049, abs=1e-9)


def test_var_default_confidence_is_95():
    returns = make_series(SIX_RETURNS)
    assert historical_var(returns) == historical_var(returns, confidence=0.95)


def test_var_grows_with_confidence():
    returns = make_series(SIX_RETURNS)
    assert historical_var(returns, 0.99) > historical_var(returns, 0.95) > historical_var(returns, 0.80)


def test_var_when_every_return_is_a_gain_is_negative_and_not_clipped():
    # returns +0.01 ... +0.06: the quantile at position 0.25 is 0.01 + 0.25 * 0.01 = +0.0125,
    # so VaR = -0.0125: "no loss at this confidence level". We do not clip it to 0.
    returns = make_series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    assert historical_var(returns, 0.95) == pytest.approx(-0.0125, abs=1e-12)


def test_var_dataframe_drops_nan_per_column():
    returns = make_frame(
        {
            "AAPL": SIX_RETURNS,
            "NEW": [NAN, NAN, 0.04, -0.02, 0.01, -0.08],
        }
    )
    result = historical_var(returns, 0.95)
    # NEW has 4 valid returns, sorted -0.08, -0.02, +0.01, +0.04; position 0.05 * 3 = 0.15:
    # -0.08 + 0.15 * 0.06 = -0.071 -> VaR 0.071
    assert list(result.index) == ["AAPL", "NEW"]
    assert result["AAPL"] == pytest.approx(0.09, abs=1e-12)
    assert result["NEW"] == pytest.approx(0.071, abs=1e-12)


def test_var_series_with_nan_equals_the_series_without_it():
    with_gap = make_series([0.02, NAN, -0.10, 0.06, -0.02, 0.04, -0.06])
    assert historical_var(with_gap) == pytest.approx(0.09, abs=1e-12)


def test_var_does_not_depend_on_the_order_of_the_returns():
    ordered = make_series(sorted(SIX_RETURNS))
    assert historical_var(ordered) == pytest.approx(historical_var(make_series(SIX_RETURNS)), abs=1e-15)


@pytest.mark.parametrize("confidence", [0, 1, 1.5, -0.1, NAN, float("inf"), "0.95", True, None])
def test_var_rejects_a_bad_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        historical_var(make_series(SIX_RETURNS), confidence=confidence)


def test_var_warns_for_a_short_sample():
    with pytest.warns(ShortSampleWarning, match=r"'AAPL': 6"):
        historical_var(make_series(SIX_RETURNS))


def test_var_warns_when_the_tail_has_fewer_than_five_observations():
    # n = 80 is above MIN_RELIABLE_OBS (60) but 80 * 0.05 = 4 observations in the tail (< 5).
    messages = short_sample_messages(historical_var, make_series(alternating(80)), 0.95)
    assert len(messages) == 1
    assert "loss tail" in messages[0] and "'AAPL': n=80" in messages[0]


def test_var_tail_warning_boundary_is_100_observations_at_95_percent():
    # 100 * (1 - 0.95) = 5 observations in the tail: enough. 99 * 0.05 = 4.95: not enough.
    assert_no_short_sample_warning(historical_var, make_series(alternating(100)), 0.95)
    assert len(short_sample_messages(historical_var, make_series(alternating(99)), 0.95)) == 1


def test_var_tail_warning_is_not_triggered_by_float_noise():
    # 50 * (1 - 0.9) is 4.999999999999999 in floating point, but it means exactly 5.
    messages = short_sample_messages(historical_var, make_series(alternating(50)), 0.9)
    assert not any("loss tail" in m for m in messages)
    # (it is below MIN_RELIABLE_OBS, so the ordinary short-sample warning does appear)
    assert any("fewer than 60" in m for m in messages)


def test_var_tail_warning_scales_with_confidence():
    # At 99% the tail needs n >= 500: 250 returns are plenty for 95% but not for 99%.
    returns = make_series(alternating(250))
    assert_no_short_sample_warning(historical_var, returns, 0.95)
    assert any("loss tail" in m for m in short_sample_messages(historical_var, returns, 0.99))


# --------------------------------------------------------------------------- #
# sharpe_ratio
# --------------------------------------------------------------------------- #

def test_sharpe_known_value():
    # returns [0.02, 0, 0.02, 0]: mean 0.01, deviations +-0.01 -> sum of squares 0.0004,
    # sample variance 0.0004 / 3 -> std = 0.01 * sqrt(4/3) = 0.011547...
    # daily ratio = 0.01 / (0.01 * sqrt(4/3)) = sqrt(3/4) = 0.8660...
    # annualized: * sqrt(252) -> sqrt(189) = 13.7477...  (not annualized would be 0.866)
    result = sharpe_ratio(make_series([0.02, 0.0, 0.02, 0.0]))
    assert result == pytest.approx(math.sqrt(189), rel=1e-12)
    assert isinstance(result, float)


def test_sharpe_subtracts_the_annual_risk_free_rate_per_day():
    # rf = 0.252 a year -> 0.252 / 252 = 0.001 a day. Excess mean = 0.01 - 0.001 = 0.009.
    # The std is unchanged by a constant shift: 0.01 * sqrt(4/3).
    # Sharpe = 0.009 / (0.01 * sqrt(4/3)) * sqrt(252) = 12.373...
    returns = make_series([0.02, 0.0, 0.02, 0.0])
    result = sharpe_ratio(returns, risk_free_rate=0.252)
    expected = 0.009 / (0.01 * math.sqrt(4 / 3)) * math.sqrt(252)
    assert result == pytest.approx(expected, rel=1e-12)
    assert result < sharpe_ratio(returns)  # forgetting to subtract rf would give 13.75


def test_sharpe_negative_when_returns_are_below_the_risk_free_rate():
    # mean -0.01 with the same spread -> -sqrt(189)
    assert sharpe_ratio(make_series([-0.02, 0.0, -0.02, 0.0])) == pytest.approx(-math.sqrt(189), rel=1e-12)


def test_sharpe_dataframe_and_nan_handling():
    returns = make_frame({"A": [0.02, 0.0, 0.02, 0.0], "NEW": [NAN, 0.02, 0.0, 0.02]})
    result = sharpe_ratio(returns)
    # NEW's valid returns are 0.02, 0.0, 0.02: mean 0.02/3 * 2 = 0.013333, deviations
    # +0.006667, -0.013333, +0.006667 -> sum of squares 0.00026667, / 2 -> std 0.011547
    # Sharpe = 0.013333 / 0.011547 * sqrt(252) = sqrt(4/3) * sqrt(252) = sqrt(336) = 18.33...
    assert list(result.index) == ["A", "NEW"]
    assert result["A"] == pytest.approx(math.sqrt(189), rel=1e-12)
    assert result["NEW"] == pytest.approx(math.sqrt(336), rel=1e-9)


@pytest.mark.parametrize("flat", [[0.01] * 5, [0.07] * 13, [0.0] * 4])
def test_sharpe_raises_on_zero_volatility(flat):
    # [0.07] * 13 has a floating-point std of about 3e-17 (not exactly 0): still "zero".
    with pytest.raises(ValueError, match="undefined"):
        sharpe_ratio(make_series(flat))


def test_sharpe_zero_volatility_error_names_the_ticker():
    returns = make_frame({"A": [0.02, 0.0, 0.02, 0.0], "STALE": [0.01, 0.01, 0.01, 0.01]})
    with pytest.raises(ValueError, match="'STALE'"):
        sharpe_ratio(returns)


@pytest.mark.parametrize("rate", [NAN, float("inf"), "0.04", True, None])
def test_sharpe_rejects_a_bad_risk_free_rate(rate):
    with pytest.raises(ValueError, match="risk_free_rate"):
        sharpe_ratio(make_series([0.02, 0.0, 0.02, 0.0]), risk_free_rate=rate)


# --------------------------------------------------------------------------- #
# risk_summary
# --------------------------------------------------------------------------- #

SUMMARY_COLUMNS = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "var_95",
    "n_obs",
]


def test_risk_summary_known_values():
    # prices [100, 110, 99] -> returns [+0.10, -0.10]
    # annualized_return: 0.99 ** 126 - 1 ; volatility: 0.1 * sqrt(504)
    # sharpe: the mean is 0, so 0 ; max_drawdown: peak 110, low 99 -> 0.10
    # var_95: quantile at position 0.05 -> -0.10 + 0.05 * 0.20 = -0.09 -> 0.09 ; n_obs: 2
    summary = risk_summary(make_frame({"AAPL": [100, 110, 99]}))
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert list(summary.index) == ["AAPL"]
    assert summary.index.name == "Ticker"
    row = summary.loc["AAPL"]
    assert row["annualized_return"] == pytest.approx(0.99**126 - 1, rel=1e-9)
    assert row["annualized_volatility"] == pytest.approx(0.1 * math.sqrt(504), rel=1e-9)
    assert row["sharpe_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert row["max_drawdown"] == pytest.approx(0.10, abs=1e-12)
    assert row["var_95"] == pytest.approx(0.09, abs=1e-9)
    assert row["n_obs"] == 2
    assert pd.api.types.is_integer_dtype(summary["n_obs"])


def test_risk_summary_values_equal_the_individual_functions():
    prices = long_prices()
    returns = daily_returns(prices)
    summary = risk_summary(prices, confidence=0.95, risk_free_rate=0.03)
    assert list(summary.index) == ["AAPL", "MSFT"]
    expected = {
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate=0.03),
        "max_drawdown": max_drawdown(prices),
        "var_95": historical_var(returns, confidence=0.95),
        "n_obs": returns.count(),
    }
    for column, series in expected.items():
        assert summary[column].tolist() == pytest.approx(series.tolist(), rel=1e-12, abs=1e-15), column
    # MSFT starts 10 days late (the first 10 prices are NaN) and has one missing price on row
    # 100. It has 251 - 10 - 1 = 240 valid prices, and every valid price after the first
    # one has a return (the one after the gap is bridged): 239 valid returns.
    # AAPL has all 250.
    assert summary["n_obs"].tolist() == [250, 239]


def test_risk_summary_risk_free_rate_changes_only_the_sharpe_ratio():
    prices = long_prices()
    plain = risk_summary(prices)
    with_rf = risk_summary(prices, risk_free_rate=0.05)
    assert (with_rf["sharpe_ratio"] < plain["sharpe_ratio"]).all()
    pd.testing.assert_frame_equal(plain.drop(columns="sharpe_ratio"), with_rf.drop(columns="sharpe_ratio"))


def test_risk_summary_var_column_is_named_after_the_confidence():
    prices = long_prices()
    assert "var_99" in risk_summary(prices, confidence=0.99).columns
    assert "var_95" not in risk_summary(prices, confidence=0.99).columns
    assert "var_97.5" in risk_summary(prices, confidence=0.975).columns
    summary = risk_summary(prices, confidence=0.99)
    assert summary["var_99"].tolist() == pytest.approx(historical_var(daily_returns(prices), 0.99).tolist())


def test_risk_summary_accepts_a_series():
    summary = risk_summary(make_series([100, 110, 99], name="AAPL"))
    assert list(summary.index) == ["AAPL"]
    assert summary.loc["AAPL", "n_obs"] == 2


def test_risk_summary_emits_one_consolidated_warning_per_problem_not_one_per_function():
    # 2 tickers, 9 returns each: short sample AND thin VaR tail. Without consolidation the 4
    # statistics would emit 4-5 warnings per ticker; here there are exactly two in total.
    prices = pd.concat([prices_from_returns(alternating(9), "AAPL"), prices_from_returns(alternating(9, 0.02), "MSFT")], axis=1)
    messages = short_sample_messages(risk_summary, prices)
    assert len(messages) == 2
    short = next(m for m in messages if "fewer than 60" in m)
    tail = next(m for m in messages if "loss tail" in m)
    assert "'AAPL': 9" in short and "'MSFT': 9" in short
    assert "'AAPL': n=9" in tail and "'MSFT': n=9" in tail


def test_risk_summary_tail_only_warning_between_60_and_100_returns():
    messages = short_sample_messages(risk_summary, prices_from_returns(alternating(80)).to_frame())
    assert len(messages) == 1 and "loss tail" in messages[0]


def test_risk_summary_tail_warning_uses_the_given_confidence():
    # 250 returns: at 99% there are 250 * 0.01 = 2.5 observations in the tail (< 5), so it
    # warns and says 0.99; at 95% there are 12.5, so there is no warning at all.
    prices = prices_from_returns(alternating(250)).to_frame()
    messages = short_sample_messages(risk_summary, prices, confidence=0.99)
    assert len(messages) == 1
    assert "loss tail" in messages[0] and "confidence=0.99" in messages[0] and "'AAPL': n=250" in messages[0]
    assert short_sample_messages(risk_summary, prices, confidence=0.95) == []


def test_risk_summary_rejects_bad_confidence_and_risk_free_rate():
    with pytest.raises(ValueError, match="confidence"):
        risk_summary(long_prices(), confidence=1.0)
    with pytest.raises(ValueError, match="risk_free_rate"):
        risk_summary(long_prices(), risk_free_rate=NAN)


def test_risk_summary_gap_warning_is_one_consolidated_warning():
    # AAPL has one bridged return (row 2), MSFT has none; the 4 calculations inside
    # must not each repeat the warning.
    prices = prices_from_returns(alternating(100), "AAPL").to_frame()
    prices.iloc[50, 0] = NAN
    prices["MSFT"] = prices_from_returns(alternating(100, 0.02), "MSFT")
    messages = messages_of(GapWarning, risk_summary, prices)
    assert len(messages) == 1
    assert messages[0].startswith("risk_summary:") and "'AAPL': 1 bridged (longest 1 row)" in messages[0]
    assert "MSFT" not in messages[0] and "longer than a holiday" not in messages[0]
    assert messages_of(GapWarning, risk_summary, prices[["MSFT"]]) == []


def test_risk_summary_gap_warning_gets_the_stronger_wording_for_a_long_outage():
    # XOM is missing 40 rows in a row (rows 30-69): one bridged return, longest bridge 40 rows.
    prices = prices_from_returns(alternating(100), "XOM").to_frame()
    prices.iloc[30:70, 0] = NAN
    messages = messages_of(GapWarning, risk_summary, prices)
    assert len(messages) == 1
    assert messages[0].startswith("risk_summary:")
    assert "'XOM': 1 bridged (longest 40 rows)" in messages[0]
    assert "longer than a holiday: 'XOM' (40 rows)" in messages[0] and "likely missing data" in messages[0]


# --- a ticker that cannot be calculated does not spoil the table -------------------------

def good_and_bad_prices(bad: pd.Series) -> pd.DataFrame:
    """100 good AAPL returns next to one problem ticker (given as 101 prices)."""
    return pd.concat([prices_from_returns(alternating(100), "AAPL"), bad], axis=1)


def test_risk_summary_a_ticker_with_one_valid_return_gets_a_nan_row_and_a_warning():
    # The reviewer's case: NVDA is NaN except for its last 2 rows -> 1 valid return.
    nvda = pd.Series(NAN, index=dates(101), name="NVDA")
    nvda.iloc[-2:] = [100.0, 101.0]
    prices = good_and_bad_prices(nvda)
    with pytest.warns(SkippedTickerWarning, match=r"'NVDA'"):
        summary = risk_summary(prices)
    messages = messages_of(SkippedTickerWarning, risk_summary, prices)
    assert len(messages) == 1
    assert "'NVDA'" in messages[0] and "at least 2 valid returns" in messages[0] and "'AAPL'" not in messages[0]
    # NVDA: NaN in every metric column, but its real count of valid returns is shown.
    assert list(summary.index) == ["AAPL", "NVDA"]
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert summary.loc["NVDA", SUMMARY_COLUMNS[:-1]].isna().all()
    assert summary.loc["NVDA", "n_obs"] == 1
    # AAPL is calculated as usual: the same as when it is alone.
    alone = risk_summary(prices[["AAPL"]])
    pd.testing.assert_series_equal(summary.loc["AAPL"], alone.loc["AAPL"])
    assert summary.loc["AAPL", "n_obs"] == 100


def test_risk_summary_a_ticker_with_fewer_than_two_valid_prices_gets_a_nan_row():
    one_price = pd.Series(NAN, index=dates(101), name="ONE")
    one_price.iloc[-1] = 5.0
    never = pd.Series(NAN, index=dates(101), name="NEVER")
    with pytest.warns(SkippedTickerWarning) as record:
        summary = risk_summary(pd.concat([prices_from_returns(alternating(100), "AAPL"), one_price, never], axis=1))
    skipped = [w for w in record if issubclass(w.category, SkippedTickerWarning)]
    assert len(skipped) == 1  # one warning listing both
    assert "'ONE': only 1 valid price" in str(skipped[0].message)
    assert "'NEVER': only 0 valid price" in str(skipped[0].message)
    assert summary["n_obs"].tolist() == [100, 0, 0]
    assert summary.loc[["ONE", "NEVER"], SUMMARY_COLUMNS[:-1]].isna().all().all()
    assert summary.loc["AAPL"].notna().all()


def test_risk_summary_a_flat_ticker_gets_a_nan_row_and_a_warning_with_the_reason():
    flat = pd.Series(50.0, index=dates(101), name="FLAT")
    prices = good_and_bad_prices(flat)
    messages = messages_of(SkippedTickerWarning, risk_summary, prices)
    assert len(messages) == 1
    assert "'FLAT'" in messages[0] and "identical" in messages[0]
    with pytest.warns(SkippedTickerWarning):
        summary = risk_summary(prices)
    assert summary.loc["FLAT", SUMMARY_COLUMNS[:-1]].isna().all()
    assert summary.loc["FLAT", "n_obs"] == 100  # it has 100 valid returns; they are just all 0.0
    assert summary.loc["AAPL"].notna().all()


def test_risk_summary_raises_when_no_ticker_can_be_calculated():
    nvda = pd.Series(NAN, index=dates(101), name="NVDA")
    nvda.iloc[-2:] = [100.0, 101.0]
    with pytest.raises(ValueError, match=r"no ticker could be calculated.*'NVDA'.*at least 2 valid returns"):
        risk_summary(nvda.to_frame())
    with pytest.raises(ValueError, match=r"no ticker could be calculated.*'FLAT'.*identical"):
        risk_summary(pd.Series(50.0, index=dates(101), name="FLAT"))
    with pytest.raises(ValueError, match=r"no ticker could be calculated.*'AAPL'.*only 1 valid price"):
        risk_summary(make_series([100.0], name="AAPL"))
    # every ticker bad -> still ValueError (naming all of them)
    both = pd.concat([nvda, pd.Series(50.0, index=dates(101), name="FLAT")], axis=1)
    with pytest.raises(ValueError, match=r"'NVDA'.*'FLAT'"):
        risk_summary(both)


def test_the_individual_functions_still_raise_for_the_same_tickers():
    nvda = pd.Series(NAN, index=dates(101), name="NVDA")
    nvda.iloc[-2:] = [100.0, 101.0]
    returns = daily_returns(nvda)
    assert returns.count() == 1
    for function in (annualized_return, annualized_volatility, historical_var, sharpe_ratio):
        with pytest.raises(ValueError, match="'NVDA' has 1"):
            function(returns)
    with pytest.raises(ValueError, match="identical"):
        sharpe_ratio(daily_returns(pd.Series(50.0, index=dates(101), name="FLAT")))


# --- R1: a ticker's numbers do not depend on the other tickers or their holidays -------------

def calendar_prices() -> pd.DataFrame:
    """AAPL (4 'US holidays', two of them in a row) beside a ticker with a different calendar
    (6 missing days of its own, and a later start)."""
    rng = np.random.default_rng(7)
    aapl = prices_from_returns(rng.normal(0.0005, 0.012, 200), "AAPL")
    other = prices_from_returns(rng.normal(0.0, 0.02, 200), "7203.T", start=30.0)
    aapl.iloc[[20, 21, 90, 150]] = NAN
    other.iloc[[5, 6, 55, 120, 121, 122]] = NAN
    other.iloc[:3] = NAN
    return pd.concat([aapl, other], axis=1)


def test_risk_summary_row_is_identical_alone_or_beside_a_ticker_with_another_calendar():
    prices = calendar_prices()
    together = risk_summary(prices)
    for ticker in prices.columns:
        alone = risk_summary(prices[[ticker]])
        # exactly equal, not just close: nothing about the other ticker may leak in
        pd.testing.assert_series_equal(together.loc[ticker], alone.loc[ticker], check_exact=True)
    # ... and so are the building blocks
    pd.testing.assert_series_equal(daily_returns(prices)["AAPL"], daily_returns(prices[["AAPL"]])["AAPL"], check_exact=True)
    # The summary's n_obs is the number of valid prices minus one (each valid price except the
    # first has a return, bridged or not).
    assert together["n_obs"].tolist() == (prices.count() - 1).tolist()


def test_risk_summary_return_is_the_growth_between_first_and_last_valid_price():
    # The reviewer's R1 numbers in miniature: the table's CAGR must be the ticker's own
    # P_last_valid / P_first_valid annualized over its own number of returns, however many
    # holidays (NaN rows) the ticker has and whoever else is in the table.
    prices = calendar_prices()
    summary = risk_summary(prices)
    for ticker in prices.columns:
        valid = prices[ticker].dropna()
        n_returns = len(valid) - 1
        expected = (valid.iloc[-1] / valid.iloc[0]) ** (252 / n_returns) - 1
        assert summary.loc[ticker, "annualized_return"] == pytest.approx(expected, rel=1e-9)
        assert summary.loc[ticker, "n_obs"] == n_returns


# --------------------------------------------------------------------------- #
# Sample-size warnings (ShortSampleWarning) across the functions
# --------------------------------------------------------------------------- #

def _statistic_calls(n: int):
    """One (function, args) per statistic, all fed n valid observations named AAPL."""
    returns = make_series(alternating(n))
    prices = prices_from_returns(alternating(n - 1))  # n prices
    two_columns = make_frame({"AAPL": alternating(n), "MSFT": alternating(n, 0.02)})
    return {
        "annualized_return": (annualized_return, (returns,)),
        "annualized_volatility": (annualized_volatility, (returns,)),
        "sharpe_ratio": (sharpe_ratio, (returns,)),
        "historical_var": (historical_var, (returns,)),
        "max_drawdown": (max_drawdown, (prices,)),
        "correlation_matrix": (correlation_matrix, (two_columns,)),
    }


@pytest.mark.parametrize("name", ["annualized_return", "annualized_volatility", "sharpe_ratio", "historical_var", "max_drawdown", "correlation_matrix"])
def test_statistics_warn_below_the_minimum_sample(name):
    function, args = _statistic_calls(MIN_RELIABLE_OBS - 1)[name]
    with pytest.warns(ShortSampleWarning, match=r"'AAPL'.*59"):
        function(*args)


@pytest.mark.parametrize("name", ["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "correlation_matrix"])
def test_statistics_do_not_warn_at_the_minimum_sample(name):
    # exactly MIN_RELIABLE_OBS (60) valid observations: no warning
    function, args = _statistic_calls(MIN_RELIABLE_OBS)[name]
    assert_no_short_sample_warning(function, *args)


def test_var_does_not_warn_with_a_long_enough_sample():
    # 250 valid returns: above 60, and 250 * 0.05 = 12.5 observations in the 95% tail
    assert_no_short_sample_warning(historical_var, make_series(alternating(250)))
    # at 500 returns even the 99% tail has 5 observations
    assert_no_short_sample_warning(historical_var, make_series(alternating(500)), 0.99)


def test_nothing_warns_on_a_long_clean_dataset():
    prices = prices_from_returns(alternating(250)).to_frame()
    prices["MSFT"] = prices_from_returns(np.random.default_rng(1).normal(0, 0.01, 250), "MSFT")
    returns = daily_returns(prices)
    for function, args in [
        (daily_returns, (prices,)),
        (cumulative_returns, (prices,)),
        (annualized_return, (returns,)),
        (annualized_volatility, (returns,)),
        (annualized_volatility, (returns, 60)),
        (correlation_matrix, (returns,)),
        (max_drawdown, (prices,)),
        (historical_var, (returns,)),
        (sharpe_ratio, (returns,)),
        (risk_summary, (prices,)),
    ]:
        assert_no_short_sample_warning(function, *args)


def test_the_short_sample_warning_counts_only_valid_observations():
    # 100 rows but only 30 valid returns: it is the valid count that matters.
    values = np.concatenate([np.full(70, NAN), alternating(30)])
    with pytest.warns(ShortSampleWarning, match=r"'AAPL': 30"):
        annualized_volatility(make_series(values))


def test_warning_names_every_short_column():
    returns = make_frame({"AAPL": alternating(100), "NEW": np.concatenate([np.full(90, NAN), alternating(10)])})
    messages = short_sample_messages(annualized_volatility, returns)
    assert len(messages) == 1  # one warning per call, not per column
    assert "'NEW': 10" in messages[0] and "AAPL" not in messages[0]


# --------------------------------------------------------------------------- #
# Input validation (ValueError)
# --------------------------------------------------------------------------- #

PRICE_FUNCTIONS = [daily_returns, cumulative_returns, max_drawdown, risk_summary]
RETURN_FUNCTIONS = [annualized_return, annualized_volatility, historical_var, sharpe_ratio]
ALL_FUNCTIONS = PRICE_FUNCTIONS + RETURN_FUNCTIONS + [correlation_matrix]


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_non_pandas_input_is_rejected(function):
    for bad in ([100.0, 101.0, 102.0], np.array([100.0, 101.0, 102.0]), None, "AAPL", 5.0):
        with pytest.raises(ValueError, match="pandas"):
            function(bad)


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_empty_input_is_rejected(function):
    with pytest.raises(ValueError, match="empty|at least 2"):
        function(pd.DataFrame())
    with pytest.raises(ValueError, match="empty|at least 2"):
        function(pd.DataFrame({"AAPL": []}, dtype="float64"))
    with pytest.raises(ValueError, match="empty|at least 2"):
        function(pd.Series([], dtype="float64"))


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_non_numeric_input_is_rejected(function):
    text = pd.DataFrame({"AAPL": ["a", "b", "c"], "MSFT": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="not numeric"):
        function(text)
    mixed = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0], "MSFT": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="'MSFT'"):
        function(mixed)
    with pytest.raises(ValueError, match="not numeric"):
        function(pd.Series([True, False, True]))


@pytest.mark.parametrize("function", [daily_returns, cumulative_returns, max_drawdown])
def test_price_functions_need_two_prices_and_name_the_ticker(function):
    with pytest.raises(ValueError, match="'AAPL' has 1"):
        function(make_series([100.0], name="AAPL"))
    # One column is fine and one is not: the message must name the bad one.
    two = make_frame({"AAPL": [100.0, 101.0, 102.0], "NEW": [NAN, NAN, 5.0]})
    with pytest.raises(ValueError, match="'NEW' has 1"):
        function(two)
    with pytest.raises(ValueError, match="'AAPL' has 0"):
        function(make_series([NAN, NAN, NAN], name="AAPL"))


@pytest.mark.parametrize("function", RETURN_FUNCTIONS)
def test_return_functions_need_two_returns_and_name_the_ticker(function):
    with pytest.raises(ValueError, match="'AAPL' has 1"):
        function(make_series([0.01], name="AAPL"))
    two = make_frame({"AAPL": [0.01, 0.02, 0.03], "NEW": [NAN, NAN, 0.05]})
    with pytest.raises(ValueError, match="'NEW' has 1"):
        function(two)
    with pytest.raises(ValueError, match="'AAPL' has 0"):
        function(make_series([NAN, NAN, NAN], name="AAPL"))


def test_fewer_than_two_data_points_unnamed_series_still_gives_a_clear_message():
    with pytest.raises(ValueError, match="'series' has 1"):
        annualized_volatility(pd.Series([0.01], index=dates(1)))


def test_returns_need_two_valid_values_even_if_the_prices_had_two():
    # prices [100, NaN, 110] have 2 valid prices and give ONE valid return (+0.10, bridged).
    # One return is not enough for a standard deviation, so the statistics refuse.
    returns = daily_returns(make_series([100, NAN, 110]))
    assert returns.count() == 1
    for function in RETURN_FUNCTIONS:
        with pytest.raises(ValueError, match="'AAPL' has 1"):
            function(returns)


@pytest.mark.parametrize("function", [daily_returns, cumulative_returns, max_drawdown])
@pytest.mark.parametrize("bad_price", [0.0, -5.0])
def test_prices_must_be_positive(function, bad_price):
    with pytest.raises(ValueError, match="greater than 0"):
        function(make_series([100.0, bad_price, 105.0]))


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
def test_infinite_values_are_rejected(function):
    inf = make_frame({"AAPL": [100.0, 101.0, 102.0], "MSFT": [1.0, float("inf"), 3.0]})
    with pytest.raises(ValueError, match="infinite.*'MSFT'"):
        function(inf)


@pytest.mark.parametrize("function", [daily_returns, cumulative_returns, max_drawdown])
def test_time_order_functions_require_an_ascending_unique_index(function):
    prices = make_series([100.0, 101.0, 102.0])
    with pytest.raises(ValueError, match="increasing order"):
        function(prices.iloc[::-1])  # newest first
    duplicated = prices.copy()
    duplicated.index = duplicated.index[[0, 0, 1]]
    with pytest.raises(ValueError, match="increasing order"):
        function(duplicated)


def test_rolling_volatility_requires_an_ordered_index_but_plain_volatility_does_not():
    returns = make_series([0.01, 0.02, 0.03, 0.04])
    shuffled = returns.iloc[[2, 0, 3, 1]]
    with pytest.raises(ValueError, match="increasing order"):
        annualized_volatility(shuffled, window=2)
    # A single number does not depend on the order, so (for example) a bootstrap sample is fine.
    assert annualized_volatility(shuffled) == pytest.approx(annualized_volatility(returns), rel=1e-12)


def test_duplicate_column_names_are_rejected():
    frame = pd.DataFrame([[100.0, 100.0], [101.0, 102.0]], columns=["A", "A"], index=dates(2))
    with pytest.raises(ValueError, match="duplicate"):
        daily_returns(frame)


# --------------------------------------------------------------------------- #
# Cross-cutting: purity, determinism, return types
# --------------------------------------------------------------------------- #

def test_functions_do_not_modify_their_input():
    prices = long_prices()
    returns = daily_returns(prices)
    calls = [
        (daily_returns, prices),
        (cumulative_returns, prices),
        (max_drawdown, prices),
        (risk_summary, prices),
        (annualized_return, returns),
        (annualized_volatility, returns),
        (correlation_matrix, returns),
        (historical_var, returns),
        (sharpe_ratio, returns),
    ]
    for function, data in calls:
        before = copy.deepcopy(data)
        function(data)
        pd.testing.assert_frame_equal(data, before)
    # also with a window, and for a Series input
    before = returns.copy()
    annualized_volatility(returns, window=20)
    pd.testing.assert_frame_equal(returns, before)
    series = prices["AAPL"]
    before_series = series.copy()
    for function in (daily_returns, cumulative_returns, max_drawdown):
        function(series)
    pd.testing.assert_series_equal(series, before_series)


def test_functions_are_deterministic():
    prices = long_prices()
    returns = daily_returns(prices)
    pd.testing.assert_frame_equal(risk_summary(prices), risk_summary(prices))
    pd.testing.assert_frame_equal(correlation_matrix(returns), correlation_matrix(returns))
    pd.testing.assert_frame_equal(annualized_volatility(returns, window=30), annualized_volatility(returns, window=30))
    pd.testing.assert_series_equal(historical_var(returns), historical_var(returns))


def test_series_in_gives_float_and_dataframe_in_gives_series_by_ticker():
    returns = daily_returns(long_prices())
    for function in (annualized_return, annualized_volatility, historical_var, sharpe_ratio):
        assert type(function(returns["AAPL"])) is float
        by_ticker = function(returns)
        assert isinstance(by_ticker, pd.Series)
        assert list(by_ticker.index) == ["AAPL", "MSFT"]
        assert by_ticker["AAPL"] == pytest.approx(function(returns["AAPL"]), rel=1e-12)
    prices = long_prices()
    assert type(max_drawdown(prices["AAPL"])) is float
    assert list(max_drawdown(prices).index) == ["AAPL", "MSFT"]


def test_time_series_functions_keep_the_shape_of_the_input():
    prices = long_prices()
    assert isinstance(daily_returns(prices["AAPL"]), pd.Series)
    assert isinstance(daily_returns(prices), pd.DataFrame)
    assert daily_returns(prices).shape == (250, 2)
    assert cumulative_returns(prices).shape == prices.shape
    assert cumulative_returns(prices["AAPL"]).index.equals(prices.index)
    assert daily_returns(prices["MSFT"]).name == "MSFT"


def test_series_and_dataframe_inputs_agree():
    prices = long_prices()
    from_frame = daily_returns(prices)
    for ticker in prices.columns:
        pd.testing.assert_series_equal(daily_returns(prices[ticker]), from_frame[ticker])


# --------------------------------------------------------------------------- #
# Consistency: the returns, the cumulative growth and the CAGR tell the same story
# --------------------------------------------------------------------------- #

def test_annualized_return_matches_the_last_cumulative_return_hand_example():
    # prices [100, NaN, 110, 121]: returns [NaN, +0.10 (bridged over the NaN), +0.10], so n = 2.
    # cumulative_returns ends at 121 / 100 = 1.21 = 1.10 * 1.10: the same growth over the same
    # 2 returns. CAGR = 1.21 ** (252 / 2) - 1.
    prices = make_series([100, NAN, 110, 121])
    returns = daily_returns(prices)
    assert returns.count() == 2
    growth = cumulative_returns(prices).dropna().iloc[-1]
    assert growth == pytest.approx(1.21, abs=1e-12)
    assert annualized_return(returns) == pytest.approx(1.21**126 - 1, rel=1e-9)
    assert annualized_return(returns) == pytest.approx(growth ** (252 / returns.count()) - 1, rel=1e-12)


def test_annualized_return_is_consistent_with_cumulative_returns_on_gappy_data():
    prices = long_prices()  # MSFT: late start and a gap
    returns = daily_returns(prices)
    cagr = annualized_return(returns)
    for ticker in prices.columns:
        growth = cumulative_returns(prices)[ticker].dropna().iloc[-1]  # = P_last_valid / P_first_valid
        n = returns[ticker].count()
        assert n == prices[ticker].count() - 1  # one return per valid price after the first
        assert cagr[ticker] == pytest.approx(growth ** (252 / n) - 1, rel=1e-9)
    # ... and the compounded returns are exactly that growth
    for ticker in prices.columns:
        compounded = (1.0 + returns[ticker].dropna()).prod()
        assert compounded == pytest.approx(cumulative_returns(prices)[ticker].dropna().iloc[-1], rel=1e-9)


def test_a_missing_price_on_a_big_day_does_not_drop_the_return():
    # Prices 100, 100, 115, 115: the +15% day. If the 115 on that day is missing, the +15% move
    # is not lost: it shows up in the next return (bridged), so the total growth is unchanged.
    complete = make_series([100, 100, 115, 115])
    gappy = make_series([100, 100, NAN, 115])
    assert (1.0 + daily_returns(complete)).prod() == pytest.approx(1.15, abs=1e-12)
    assert (1.0 + daily_returns(gappy)).prod() == pytest.approx(1.15, abs=1e-12)
    assert cumulative_returns(gappy).dropna().iloc[-1] == pytest.approx(1.15, abs=1e-12)


# --------------------------------------------------------------------------- #
# Prices passed where returns are expected
# --------------------------------------------------------------------------- #

PRICES_LIKE = make_series([100.0, 101.5, 99.8, 102.3, 103.0], name="AAPL")


@pytest.mark.parametrize(
    "call",
    [
        lambda x: annualized_return(x),
        lambda x: annualized_volatility(x),
        lambda x: annualized_volatility(x, window=3),
        lambda x: historical_var(x),
        lambda x: sharpe_ratio(x),
    ],
    ids=["annualized_return", "annualized_volatility", "rolling_volatility", "historical_var", "sharpe_ratio"],
)
def test_return_functions_refuse_a_price_series(call):
    with pytest.raises(ValueError, match="looks like prices, not returns; call daily_returns first"):
        call(PRICES_LIKE)
    # the message names the offending column of a table; the return-like column is fine
    mixed = make_frame({"RET": [0.01, -0.02, 0.015, 0.0, 0.01], "PX": [100.0, 101.5, 99.8, 102.3, 103.0]})
    with pytest.raises(ValueError, match="'PX' looks like prices"):
        call(mixed)


def test_correlation_matrix_refuses_prices():
    prices = make_frame({"A": [100.0, 101.0, 102.5, 101.0], "B": [50.0, 49.0, 51.0, 52.0]})
    with pytest.raises(ValueError, match="looks like prices, not returns"):
        correlation_matrix(prices)
    correlation_matrix(daily_returns(prices))  # the right way works


def test_the_prices_guard_does_not_trip_on_real_looking_returns():
    # typical daily returns, plus one huge outlier day (+90%): the MEDIAN is what counts
    outlier = make_series([0.01, -0.02, 0.9, 0.015, -0.01, 0.0, 0.02])
    assert annualized_volatility(outlier) > 0
    # ... and one truly extreme day (+400%, a short squeeze). Absolute values: 0.01, 0.02, 4.0,
    # 0.015, 0.01, 0.0, 0.02 -> their MEAN is 4.075 / 7 = 0.58 (above 0.5, so a mean-based guard
    # would wrongly refuse this series), but their MEDIAN (sorted: 0, 0.01, 0.01, 0.015, 0.02,
    # 0.02, 4.0) is 0.015. The guard must use the median.
    squeeze = make_series([0.01, -0.02, 4.0, 0.015, -0.01, 0.0, 0.02])
    assert squeeze.abs().mean() > 0.5 > squeeze.abs().median()
    assert annualized_volatility(squeeze) > 0
    assert annualized_return(squeeze) > 0
    # sorted returns -0.02, -0.01, 0, 0.01, 0.015, 0.02, 4.0: quantile at position 0.05 * 6 = 0.3
    # is -0.02 + 0.3 * 0.01 = -0.017, so the VaR is 0.017
    assert historical_var(squeeze) == pytest.approx(0.017, abs=1e-9)
    assert sharpe_ratio(squeeze) > 0
    assert correlation_matrix(pd.concat([squeeze, squeeze * 0.5], axis=1, keys=["A", "B"])).loc["A", "B"] == pytest.approx(1.0)
    # even a wild but sub-0.5 median passes: [0.5, -0.5, ...] has a median absolute value of
    # exactly 0.5, which is not "above 0.5"
    assert annualized_volatility(make_series([0.5, -0.5, 0.5, -0.5])) > 0
    with pytest.raises(ValueError, match="looks like prices"):
        annualized_volatility(make_series([0.51, -0.51, 0.51, -0.51]))
    # the real pipeline: prices -> daily_returns -> statistics
    returns = daily_returns(long_prices())
    assert (annualized_volatility(returns) > 0).all()


# --------------------------------------------------------------------------- #
# Output names: no accidental Series names
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("function", [annualized_return, annualized_volatility, historical_var, sharpe_ratio])
def test_scalar_metrics_on_a_dataframe_return_an_unnamed_series(function):
    # historical_var used to come back named np.float64(0.0500...) (the quantile)
    result = function(daily_returns(long_prices()))
    assert result.name is None
    assert list(result.index) == ["AAPL", "MSFT"]


def test_max_drawdown_on_a_dataframe_returns_an_unnamed_series():
    assert max_drawdown(long_prices()).name is None


def test_series_names_follow_the_input_for_time_series_outputs():
    prices = long_prices()
    assert daily_returns(prices["AAPL"]).name == "AAPL"
    assert cumulative_returns(prices["AAPL"]).name == "AAPL"
    assert annualized_volatility(daily_returns(prices["AAPL"]), window=30).name == "AAPL"
    unnamed = make_series([100, 110, 99], name=None)
    assert daily_returns(unnamed).name is None
    assert cumulative_returns(unnamed).name is None


def test_risk_summary_and_correlation_labels():
    summary = risk_summary(long_prices())
    assert summary.index.name == "Ticker" and summary.columns.name is None
    corr = correlation_matrix(daily_returns(long_prices()))
    assert list(corr.index) == list(corr.columns) == ["AAPL", "MSFT"]


# --------------------------------------------------------------------------- #
# risk_summary knows its input is returns (R8) and survives a bad price (R12)
# --------------------------------------------------------------------------- #

def gme_like_prices() -> pd.DataFrame:
    """A tiny window of a squeeze: returns +18%, +93%, +135% (median 0.93, "looks like prices")."""
    gme = prices_from_returns([0.18, 0.93, 1.35], "GME", start=40.0)
    calm = prices_from_returns([0.01, -0.02, 0.015], "AAPL")
    return pd.concat([gme, calm], axis=1)


def test_risk_summary_calculates_a_wild_ticker_instead_of_skipping_it():
    prices = gme_like_prices()
    returns = daily_returns(prices)
    assert returns["GME"].abs().median() > 0.5  # the "looks like prices" heuristic WOULD fire
    summary = risk_summary(prices)  # and a SkippedTickerWarning would fail the test: it is not ignored
    row = summary.loc["GME"]
    assert row.notna().all() and row["n_obs"] == 3
    # GME's numbers, by hand: prices 40 -> 47.2 -> 91.096 -> 214.0756; growth 5.35189 over 3 returns
    growth = 40.0 * 1.18 * 1.93 * 2.35 / 40.0
    assert row["annualized_return"] == pytest.approx(growth ** (252 / 3) - 1, rel=1e-9)
    assert row["max_drawdown"] == 0.0  # it only went up
    # std of (0.18, 0.93, 1.35): mean 0.82, sum of squares 0.4096 + 0.0121 + 0.2809 = 0.7026,
    # / 2 = 0.3513 -> std 0.59270 -> * sqrt(252) = 9.4091...
    assert row["annualized_volatility"] == pytest.approx(math.sqrt(0.3513) * math.sqrt(252), rel=1e-9)
    # AAPL next to it is unaffected by GME
    alone = risk_summary(prices[["AAPL"]])
    pd.testing.assert_series_equal(summary.loc["AAPL"], alone.loc["AAPL"], check_exact=True)


def test_the_public_functions_still_refuse_the_same_returns():
    returns = daily_returns(gme_like_prices())[["GME"]]
    for function in (annualized_return, annualized_volatility, historical_var, sharpe_ratio):
        with pytest.raises(ValueError, match="'GME' looks like prices, not returns"):
            function(returns)
        with pytest.raises(ValueError, match="looks like prices"):
            function(returns["GME"])


def test_risk_summary_with_only_a_wild_ticker_still_gives_a_row():
    summary = risk_summary(gme_like_prices()["GME"])
    assert list(summary.index) == ["GME"] and summary.loc["GME"].notna().all()


def test_risk_summary_a_ticker_with_a_zero_price_gets_a_nan_row_and_the_date():
    # The reviewer's case: a data provider returns 0.0 for one day (SHIB-USD, August 2020).
    shib = prices_from_returns(alternating(100, 0.02), "SHIB-USD", start=5.0)
    zero_day = shib.index[40]
    shib.iloc[40] = 0.0
    prices = pd.concat([prices_from_returns(alternating(100), "AAPL"), shib], axis=1)
    messages = messages_of(SkippedTickerWarning, risk_summary, prices)
    assert len(messages) == 1
    assert "'SHIB-USD': has a zero or negative price on " + zero_day.strftime("%Y-%m-%d") in messages[0]
    assert "AAPL" not in messages[0]
    with pytest.warns(SkippedTickerWarning):
        summary = risk_summary(prices)
    assert list(summary.index) == ["AAPL", "SHIB-USD"]
    assert summary.loc["SHIB-USD", SUMMARY_COLUMNS[:-1]].isna().all()
    assert summary.loc["SHIB-USD", "n_obs"] == 0  # nothing was calculated for it
    assert summary.loc["AAPL"].notna().all() and summary.loc["AAPL", "n_obs"] == 100
    # AAPL is exactly what it is when alone
    pd.testing.assert_series_equal(summary.loc["AAPL"], risk_summary(prices[["AAPL"]]).loc["AAPL"], check_exact=True)
    # the individual functions keep raising for the same prices
    for function in (daily_returns, cumulative_returns, max_drawdown):
        with pytest.raises(ValueError, match="greater than 0"):
            function(prices)


def test_risk_summary_a_negative_price_is_skipped_too_and_the_first_bad_date_is_named():
    bad = make_series([10.0, 11.0, -1.0, 12.0, 0.0, 13.0], name="NEG")
    good = prices_from_returns(alternating(5), "OK")  # 6 prices
    prices = pd.concat([good, bad], axis=1)
    messages = messages_of(SkippedTickerWarning, risk_summary, prices)
    assert len(messages) == 1
    assert f"'NEG': has a zero or negative price on {bad.index[2]:%Y-%m-%d}" in messages[0]  # the FIRST bad day
    with pytest.warns(SkippedTickerWarning):
        summary = risk_summary(prices)
    assert summary.loc["NEG", SUMMARY_COLUMNS[:-1]].isna().all()
    assert summary.loc["OK", "n_obs"] == 5


def test_risk_summary_raises_when_the_only_ticker_has_a_non_positive_price():
    with pytest.raises(ValueError, match=r"no ticker could be calculated.*'SHIB-USD'.*zero or negative price on"):
        risk_summary(make_series([5.0, 0.0, 6.0, 7.0], name="SHIB-USD"))


# --------------------------------------------------------------------------- #
# Documented limits (the Phase 2 agents and the dashboard read these docstrings)
# --------------------------------------------------------------------------- #

def test_the_252_day_limit_is_documented():
    for doc in (risk.__doc__, risk_summary.__doc__):
        assert "7 days a week" in doc and "crypto" in doc and "252 trading days a year" in doc


def test_the_prices_guard_limit_is_documented():
    assert "HEURISTIC" in risk.__doc__ and "below about 0.5" in risk.__doc__
