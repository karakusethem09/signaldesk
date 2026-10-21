"""Unit tests for the small pieces of the ui package: input checks, text formatting, the analysis step
(with its warning capture and its lock), the live-data pipeline and the chart builders.

None of these need a running page, a browser or the network. test_app.py and test_app_custom.py test
the pieces together, through the page.
"""

import json
import threading
import time
import warnings
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import pytest
from app_helpers import (
    BENCHMARK,
    DEMO_WEIGHTS,
    HOLDINGS,
    ROOT,
    FakeLoader,
    assert_no_chart_math,
    make_prices,
)

from signaldesk import data, limits, risk
from ui import analysis, charts, formatting, inputs, live, state

pytestmark = [
    pytest.mark.usefixtures("offline"),  # these tests never need the network: any attempt fails the test
    pytest.mark.filterwarnings("ignore::signaldesk.risk.ShortSampleWarning"),
    pytest.mark.filterwarnings("ignore::signaldesk.risk.GapWarning"),
    pytest.mark.filterwarnings("ignore::signaldesk.risk.SkippedTickerWarning"),
]


def test_every_test_in_this_file_runs_with_the_network_guard_on(request):
    """The guard is applied by this module's pytestmark; taking it out of there must fail a test."""
    assert "offline" in request.fixturenames


def test_the_offline_guard_covers_this_file_too(offline):
    """If the guard did nothing, "these tests never touch the network" would prove nothing."""
    import socket

    with pytest.raises(AssertionError, match="Python socket"):
        socket.create_connection(("127.0.0.1", 1), timeout=1)
    from curl_cffi import requests as curl_requests

    with pytest.raises(AssertionError, match="libcurl"):
        curl_requests.get("http://127.0.0.1:1/", timeout=3)  # nothing listens there either way
    assert [kind for kind, _ in offline] == ["socket", "curl"]
    offline.clear()  # expected: do not let the fixture's final check fail this test


# --------------------------------------------------------------------------- #
# inputs.py: the holdings table, the benchmark, the dates
# --------------------------------------------------------------------------- #

def table(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=[inputs.TICKER_COLUMN, inputs.WEIGHT_COLUMN])


def test_the_prefilled_editor_equals_the_demo_portfolio_after_risk_py_normalizes_it():
    config = json.loads((ROOT / "data" / "demo" / "demo_portfolio.json").read_text())
    fractions = risk.normalize_weights(dict(inputs.DEFAULT_HOLDINGS))

    assert list(fractions) == list(config["holdings"])  # same tickers, same order
    for ticker, weight in config["holdings"].items():
        assert fractions[ticker] == pytest.approx(weight, abs=1e-12), ticker
    assert inputs.DEFAULT_BENCHMARK == config["benchmark"]
    editor = state.default_holdings_table()
    assert editor[inputs.TICKER_COLUMN].tolist() == [t for t, _ in inputs.DEFAULT_HOLDINGS]
    assert editor[inputs.WEIGHT_COLUMN].tolist() == [float(w) for _, w in inputs.DEFAULT_HOLDINGS]


def test_read_holdings_cleans_the_symbols_and_keeps_the_typed_percentages():
    result = inputs.read_holdings(table((" aapl ", 60), ("brk-b", 40.5)))
    assert result == {"AAPL": 60.0, "BRK-B": 40.5}
    assert list(result) == ["AAPL", "BRK-B"]
    assert all(type(w) is float for w in result.values())


@pytest.mark.parametrize(
    "rows, message",
    [
        ([("AAPL", None)], "Fill in a weight for every ticker"),
        ([("AAPL", np.nan)], "Fill in a weight for every ticker"),
        ([("AAPL", pd.NA)], "Fill in a weight for every ticker"),
        ([(None, 10)], "Fill in a ticker for every weight"),
        ([("  ", 10)], "Fill in a ticker for every weight"),
        ([(np.nan, 10)], "Fill in a ticker for every weight"),
        ([], "Add at least one holding"),
        ([(None, None)], "Add at least one holding"),
        ([("AAPL", -0.5)], "must be between 0 and 100"),
        ([("AAPL", 100.0001)], "must be between 0 and 100"),
        ([("AAPL", float("inf"))], "must be between 0 and 100"),
        ([("AAPL", 1e308)], "must be between 0 and 100"),
        ([("AAPL", "abc")], "must be a number"),
        ([("AAPL", 10), ("aapl", 20)], "listed more than once"),
    ],
)
def test_read_holdings_refuses_with_plain_words(rows, message):
    with pytest.raises(inputs.InputProblem, match=message) as caught:
        inputs.read_holdings(table(*rows))
    text = str(caught.value)
    assert "nan" not in text.lower().replace("finance", "") and "np." not in text and "float" not in text


def test_read_holdings_lets_a_bad_symbol_raise_the_limit_modules_own_error():
    with pytest.raises(limits.LimitError, match="not a valid ticker symbol"):
        inputs.read_holdings(table(("AA PL", 10)))
    with pytest.raises(limits.LimitError, match="not a valid ticker symbol"):
        inputs.read_holdings(table(("AAPL, MSFT", 10)))  # two symbols in one cell


def test_the_weight_bounds_are_inclusive():
    assert inputs.read_holdings(table(("AAPL", 0), ("MSFT", 100))) == {"AAPL": 0.0, "MSFT": 100.0}


def test_to_fractions_is_risk_pys_normalize_weights():
    assert inputs.to_fractions({"A": 50.0, "B": 25.0, "C": 25.0}) == {"A": 0.5, "B": 0.25, "C": 0.25}
    assert inputs.to_fractions({"A": 1.0, "B": 3.0}) == {"A": 0.25, "B": 0.75}  # the proportions count, not the total
    with pytest.raises(inputs.InputProblem, match="All the weights are zero"):
        inputs.to_fractions({"A": 0.0, "B": 0.0})


def test_read_benchmark():
    assert inputs.read_benchmark(" spy ", ["AAPL"]) == "SPY"
    assert inputs.read_benchmark("^gspc", ["AAPL"]) == "^GSPC"
    for blank in (None, "", "   "):
        with pytest.raises(inputs.InputProblem, match="Enter a benchmark ticker"):
            inputs.read_benchmark(blank, ["AAPL"])
    with pytest.raises(inputs.InputProblem, match="just one benchmark"):
        inputs.read_benchmark("SPY, QQQ", ["AAPL"])
    with pytest.raises(inputs.InputProblem, match="also one of your holdings"):
        inputs.read_benchmark("aapl", ["AAPL"])
    with pytest.raises(limits.LimitError, match="not a valid ticker symbol"):
        inputs.read_benchmark("S&P", ["AAPL"])


def test_build_live_request_checks_everything_before_any_download():
    rows = table(("aapl", 60), ("msft", 40))
    request = inputs.build_live_request(rows, "spy", date(2025, 1, 6), date(2025, 12, 31), limits.PUBLIC_LIMITS)

    assert request.tickers == ["AAPL", "MSFT", "SPY"]
    assert request.weights == pytest.approx({"AAPL": 0.6, "MSFT": 0.4}, abs=1e-12)
    assert request.benchmark == "SPY"
    assert request.start == date(2025, 1, 6)
    assert request.end_exclusive == date(2026, 1, 1)  # the typed end is included

    with pytest.raises(inputs.InputProblem, match="Fill in the start date"):
        inputs.build_live_request(rows, "spy", None, date(2025, 12, 31), limits.PUBLIC_LIMITS)
    with pytest.raises(inputs.InputProblem, match="start date must be before the end date"):
        inputs.build_live_request(rows, "spy", date(2025, 12, 31), date(2025, 12, 31), limits.PUBLIC_LIMITS)
    with pytest.raises(limits.LimitError, match="Up to 10 years"):
        inputs.build_live_request(rows, "spy", date(2010, 1, 1), date(2025, 12, 31), limits.PUBLIC_LIMITS)
    many = table(*[(f"T{n}", 10) for n in range(10)])
    with pytest.raises(limits.LimitError, match="Up to 10 tickers"):
        inputs.build_live_request(many, "spy", date(2025, 1, 6), date(2025, 12, 31), limits.PUBLIC_LIMITS)
    # No limits on your own computer ...
    request = inputs.build_live_request(many, "spy", date(2005, 1, 3), date(2025, 12, 31), limits.LOCAL_LIMITS)
    assert len(request.tickers) == 11
    # ... but a bad symbol is still refused, because check_request skips that check in local mode.
    with pytest.raises(limits.LimitError, match="not a valid ticker symbol"):
        inputs.build_live_request(table(("A!", 10)), "spy", date(2025, 1, 6), date(2025, 12, 31), limits.LOCAL_LIMITS)


def test_dates_presets_and_the_inclusive_end():
    assert inputs.end_exclusive_from(date(2026, 10, 20)) == date(2026, 10, 21)
    end_exclusive = date(2026, 10, 21)  # the last day shown is 2026-10-20
    for label, first_day in [("1Y", "2025-10-20"), ("3Y", "2023-10-20"), ("5Y", "2021-10-20"), ("10Y", "2016-10-20")]:
        start = inputs.preset_start(end_exclusive, label)
        assert f"{start:%Y-%m-%d}" == first_day
        assert start == limits.earliest_allowed_start(end_exclusive, inputs.PRESET_YEARS[label])
    assert inputs.resolve_dates("3Y", date(2020, 1, 1), date(2020, 2, 2), date(2026, 10, 20)) == (
        date(2023, 10, 20),
        date(2026, 10, 20),
    )
    assert inputs.resolve_dates(inputs.CUSTOM_DATES, date(2020, 1, 1), date(2020, 2, 2), date(2026, 10, 20)) == (
        date(2020, 1, 1),
        date(2020, 2, 2),
    )
    assert inputs.resolve_dates(inputs.CUSTOM_DATES, None, None, date(2026, 10, 20)) == (None, None)
    assert inputs.DEMO_PRESETS == ("1Y", "3Y", "5Y") and inputs.LIVE_PRESETS == ("1Y", "3Y", "5Y", "10Y")


# --------------------------------------------------------------------------- #
# formatting.py
# --------------------------------------------------------------------------- #

def summary_frame(**columns) -> pd.DataFrame:
    base = {
        "annualized_return": [0.1, 0.2],
        "annualized_volatility": [0.2, 0.3],
        "sharpe_ratio": [0.5, 0.6],
        "max_drawdown": [0.1, 0.2],
        "var_95": [0.02, 0.03],
        "n_obs": [100, 100],
    }
    base.update(columns)
    return pd.DataFrame(base, index=pd.Index(["A", "B"], name="Ticker"))


def test_format_percent():
    assert formatting.format_percent(0.256) == "25.6%"
    assert formatting.format_percent(-0.03) == "-3.0%"
    assert formatting.format_percent(27.081) == "2,708.1%"  # thousands separators
    assert formatting.format_percent(999.9) == "99,990.0%"
    assert formatting.format_percent(1000) == ">100,000%"
    assert formatting.format_percent(float("nan")) == "—"
    # A 3-day sample stretched to a year (GME-like): never a 60-digit number.
    assert formatting.format_percent(1.6e61) == ">100,000%"
    assert formatting.format_percent(0.0) == "0.0%"  # a percent sign is shown once, not twice


def test_format_ratio():
    assert formatting.format_ratio(1.2345) == "1.23"
    assert formatting.format_ratio(-0.5) == "-0.50"
    assert formatting.format_ratio(float("nan")) == "—"


@pytest.mark.parametrize(
    "seconds, words",
    [
        (0, "a moment"),
        (-5, "a moment"),
        (0.2, "less than a minute"),
        (59, "less than a minute"),
        (60, "about a minute"),
        (61, "about 2 minutes"),
        (719.2, "about 12 minutes"),
        (3540, "about 59 minutes"),
        (3599.9, "about 1 hour"),
        (3600, "about 1 hour"),
        (3601, "about 1 hour and 1 minute"),
        (3721, "about 1 hour and 3 minutes"),
        (7200, "about 2 hours"),
        (90000, "more than a day"),
    ],
)
def test_describe_wait_gives_plain_words(seconds, words):
    assert formatting.describe_wait(seconds) == words


def test_the_var_column_is_found_by_its_prefix():
    summary = summary_frame(var_95=None).rename(columns={"var_95": "var_99"})
    assert formatting.var_column(summary) == "var_99"
    assert formatting.var_label(summary) == "1-day VaR 99% (loss)"
    table_ = formatting.metrics_for_display(summary_frame().rename(columns={"var_95": "var_99"}))
    assert "1-day VaR 99% (loss)" in table_.columns
    assert table_["1-day VaR 99% (loss)"].tolist() == [0.02, 0.03]  # values are not changed


def test_metrics_for_display_flags_below_60_returns_but_not_60():
    summary = pd.DataFrame(
        {
            "annualized_return": [0.1] * 3,
            "annualized_volatility": [0.2] * 3,
            "sharpe_ratio": [0.5] * 3,
            "max_drawdown": [0.1] * 3,
            "var_95": [0.02] * 3,
            "n_obs": [59, 60, 61],
        },
        index=pd.Index(["A", "B", "C"], name="Ticker"),
    )
    notes = formatting.metrics_for_display(summary)["Note"].tolist()
    assert [bool(note) for note in notes] == [True, False, False]
    assert notes[0] == f"⚠ short sample (under {risk.MIN_RELIABLE_OBS} returns)"


def test_metrics_for_display_marks_skipped_rows_and_does_not_flag_them_as_short():
    summary = summary_frame(
        annualized_return=[0.1, np.nan],
        annualized_volatility=[0.2, np.nan],
        sharpe_ratio=[0.5, np.nan],
        max_drawdown=[0.1, np.nan],
        var_95=[0.02, np.nan],
        n_obs=[100, 1],
    )
    assert formatting.metrics_for_display(summary)["Note"].tolist() == ["", "skipped (see warning above)"]


def test_style_metrics_formats_percent_ratio_and_count_and_caps_huge_numbers():
    summary = pd.DataFrame(
        {
            "annualized_return": [0.256, 1.6e61],
            "annualized_volatility": [0.2, 0.3],
            "sharpe_ratio": [0.5, np.nan],
            "max_drawdown": [0.1234, 0.2],
            "var_95": [0.02, 0.03],
            "n_obs": [1500, 3],
        },
        index=pd.Index(["A", "B"], name="Ticker"),
    )
    html = formatting.style_metrics(formatting.metrics_for_display(summary)).to_html()
    for text in ("25.6%", ">100,000%", "12.3%", "0.50", "1,500"):
        assert text in html, text


def test_style_weights_shows_percent_text():
    html = formatting.style_weights({"AAPL": 0.2, "MSFT": 0.8}).to_html()
    assert "20.0%" in html and "80.0%" in html


# --------------------------------------------------------------------------- #
# analysis.py: warnings, the lock, the analysis
# --------------------------------------------------------------------------- #

def test_skipped_rows_looks_at_the_metrics_not_at_n_obs():
    summary = pd.DataFrame(
        {
            "annualized_return": [0.1, np.nan, np.nan],
            "annualized_volatility": [0.2, np.nan, np.nan],
            "sharpe_ratio": [0.5, np.nan, np.nan],
            "max_drawdown": [0.1, np.nan, np.nan],
            "var_95": [0.02, np.nan, np.nan],
            "n_obs": [100, 1, 0],  # a skipped ticker can have n_obs of 1 or 0
        },
        index=pd.Index(["A", "B", "C"], name="Ticker"),
    )
    assert analysis.skipped_rows(summary).tolist() == [False, True, True]


def test_capture_risk_warnings_keeps_ours_and_passes_all_others_on():
    bucket = []
    with pytest.warns(FutureWarning, match="old pandas thing") as passed_on:
        with analysis.capture_risk_warnings(bucket):
            warnings.warn("old pandas thing", FutureWarning)
            warnings.warn("too short", risk.ShortSampleWarning)

    assert [str(w.message) for w in bucket] == ["too short"]
    assert not any(issubclass(w.category, risk.ShortSampleWarning) for w in passed_on)  # ours stay off the terminal


def test_capture_risk_warnings_still_delivers_when_the_calculation_raises():
    bucket = []
    with pytest.warns(DeprecationWarning, match="going away"):
        with pytest.raises(ValueError, match="boom"):
            with analysis.capture_risk_warnings(bucket):
                warnings.warn("going away", DeprecationWarning)
                warnings.warn("too short", risk.ShortSampleWarning)
                raise ValueError("boom")
    assert [str(w.message) for w in bucket] == ["too short"]


def test_capture_risk_warnings_records_repeats_and_all_three_kinds():
    bucket = []
    with analysis.capture_risk_warnings(bucket):
        for _ in range(2):  # the very same line twice: a "show once" filter would keep only the first
            warnings.warn("same", risk.ShortSampleWarning)
        warnings.warn("gap", risk.GapWarning)
        warnings.warn("skip", risk.SkippedTickerWarning)
    assert [(w.category, str(w.message)) for w in bucket] == [
        (risk.ShortSampleWarning, "same"),
        (risk.ShortSampleWarning, "same"),
        (risk.GapWarning, "gap"),
        (risk.SkippedTickerWarning, "skip"),
    ]


def test_capture_holds_the_lock_while_the_block_runs_and_gives_it_back():
    assert not analysis._RISK_LOCK.locked()
    with analysis.capture_risk_warnings([]):
        assert analysis._RISK_LOCK.locked()
    assert not analysis._RISK_LOCK.locked()
    with pytest.raises(ValueError):  # also when the block fails
        with analysis.capture_risk_warnings([]):
            raise ValueError("boom")
    assert not analysis._RISK_LOCK.locked()
    assert isinstance(analysis._RISK_LOCK, type(threading.Lock()))  # one plain, module-level threading.Lock


def test_dedupe_warnings_keeps_one_returns_gap_note_one_portfolio_gap_note_and_drops_exact_repeats():
    def record(category, message):
        return warnings.WarningMessage(message, category, "x.py", 1)

    caught = [
        record(risk.GapWarning, "risk_summary: gaps in 'A'"),
        record(risk.GapWarning, "daily_returns: gaps in 'A'"),  # same gaps, other wording
        record(risk.GapWarning, "portfolio_value: valued 'A' at its last price"),  # a different thing: kept
        record(risk.GapWarning, "portfolio_value: valued 'B' at its last price"),  # but only once
        record(risk.ShortSampleWarning, "short B"),
        record(risk.ShortSampleWarning, "short B"),  # exact repeat
        record(risk.ShortSampleWarning, "short C"),
    ]
    assert analysis.dedupe_warnings(caught) == [
        (risk.GapWarning, "risk_summary: gaps in 'A'"),
        (risk.GapWarning, "portfolio_value: valued 'A' at its last price"),
        (risk.ShortSampleWarning, "short B"),
        (risk.ShortSampleWarning, "short C"),
    ]


def analyze_default(frame, weights=None, benchmark=BENCHMARK, window=60, rf=0.0):
    weights = DEMO_WEIGHTS if weights is None else weights
    return analysis.analyze(frame[list(weights) + [benchmark]], weights, benchmark, window, rf)


def test_analyze_puts_every_column_on_the_portfolios_dates_and_uses_risk_pys_numbers():
    frame = make_prices()
    frame.loc[frame.index[:100], "XOM"] = np.nan  # the portfolio starts 100 days late
    result = analyze_default(frame)

    value = risk.portfolio_value(frame[HOLDINGS], DEMO_WEIGHTS)
    assert result.value.equals(value) and len(value) == 200
    assert result.frame.index.equals(value.index)  # holdings, Portfolio and benchmark: the SAME dates
    assert list(result.frame.columns) == HOLDINGS + ["Portfolio", BENCHMARK]
    assert result.frame[BENCHMARK].equals(frame[BENCHMARK].loc[value.index])  # the benchmark is selected, not changed
    expected_kpi = risk.risk_summary(pd.DataFrame({"Portfolio": value, BENCHMARK: frame[BENCHMARK].loc[value.index]}), 0.95, 0.0)
    pd.testing.assert_frame_equal(result.kpi, expected_kpi)
    pd.testing.assert_frame_equal(result.summary, risk.risk_summary(result.frame, 0.95, 0.0))
    assert result.n_returns == 199
    assert result.skipped == [] and result.usable == list(result.frame.columns)
    assert result.problems == {}
    # The late starter's warning reaches the caller as data (category, message).
    assert [category for category, _ in result.warnings] == [risk.ShortSampleWarning]
    assert "the portfolio starts on" in result.warnings[0][1]


def test_analyze_calls_the_price_taking_chart_functions(monkeypatch):
    seen = {}
    real_rolling, real_corr = risk.rolling_volatility, risk.return_correlation
    monkeypatch.setattr(risk, "rolling_volatility", lambda prices, window: (seen.update(rolling=(list(prices.columns), window)), real_rolling(prices, window))[1])
    monkeypatch.setattr(risk, "return_correlation", lambda prices: (seen.update(corr=list(prices.columns)), real_corr(prices))[1])
    analyze_default(make_prices(), window=45)

    assert seen["rolling"] == (HOLDINGS + ["Portfolio", BENCHMARK], 45)
    assert seen["corr"] == HOLDINGS + [BENCHMARK]  # prices, not returns; no "Portfolio"


def test_analyze_refuses_a_benchmark_that_is_also_a_holding():
    with pytest.raises(ValueError, match="also one of the holdings"):
        analysis.analyze(make_prices(), {"AAPL": 0.5, "MSFT": 0.5}, "AAPL", 60, 0.0)


def test_analyze_lets_risk_pys_errors_through_for_the_page_to_show():
    frame = make_prices()
    frame.loc[frame.index[:150], "XOM"] = np.nan
    frame.loc[frame.index[150:], "JNJ"] = np.nan
    with pytest.raises(ValueError, match="no date on which every holding has a price"):
        analyze_default(frame)


def test_analyze_notes_a_window_longer_than_the_data_instead_of_raising():
    result = analyze_default(make_prices(50), window=60)
    assert result.rolling_vol is None and result.n_returns == 49 and "rolling" not in result.problems


def test_two_threads_analysing_at_the_same_time_each_get_only_their_own_warnings(monkeypatch):
    """The lock keeps two ANALYSES from overlapping, so neither takes the other's warnings.

    This is the guarantee the lock gives. It is not a promise that every warning always arrives: pandas and
    Streamlit enter ``catch_warnings`` outside the lock too (see the docstring of ui/analysis.py).

    Thread A's data has a late-starting holding (portfolio_value warns: shorter history). Thread B's data has a
    gap (it warns: valued at a last price). The two threads are made to meet INSIDE their analysis, at the very
    first risk.py call, so that without the lock their ``catch_warnings`` blocks would overlap.
    """
    late = make_prices()
    late.loc[late.index[:100], "XOM"] = np.nan
    gap = make_prices()
    gap.loc[gap.index[100:103], "MSFT"] = np.nan

    meeting = threading.Barrier(2, timeout=0.4)
    already_met = set()
    real_value = risk.portfolio_value

    def portfolio_value_after_meeting(prices, weights):
        me = threading.get_ident()
        if me not in already_met:
            already_met.add(me)
            try:
                meeting.wait()  # with the lock the other thread cannot get in, so this times out: fine
            except threading.BrokenBarrierError:
                pass
        return real_value(prices, weights)

    monkeypatch.setattr(risk, "portfolio_value", portfolio_value_after_meeting)

    results = {}
    errors = []

    def work(name, frame):
        try:
            results[name] = analyze_default(frame)
        except Exception as error:  # pragma: no cover - would fail the asserts below
            errors.append(error)

    for _ in range(3):  # a few rounds: the barrier is spent after the first round, so later rounds race for real
        results.clear()
        threads = [threading.Thread(target=work, args=("late", late)), threading.Thread(target=work, args=("gap", gap))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not errors

        kinds_late = {category for category, _ in results["late"].warnings}
        kinds_gap = {category for category, _ in results["gap"].warnings}
        assert kinds_late == {risk.ShortSampleWarning}, results["late"].warnings
        assert kinds_gap == {risk.GapWarning}, results["gap"].warnings
        assert "'XOM'" in results["late"].warnings[0][1]
        assert all("'MSFT'" in message for _, message in results["gap"].warnings)
        already_met.clear()
        meeting.reset()


def test_the_lock_makes_a_second_analysis_wait_for_the_first():
    order = []
    release_first = threading.Event()

    def first():
        with analysis.capture_risk_warnings([]):
            order.append("first inside")
            release_first.wait(timeout=5)
            order.append("first leaving")

    def second():
        with analysis.capture_risk_warnings([]):
            order.append("second inside")

    t1 = threading.Thread(target=first)
    t1.start()
    while "first inside" not in order:
        time.sleep(0.005)
    t2 = threading.Thread(target=second)
    t2.start()
    time.sleep(0.15)
    assert order == ["first inside"]  # the second one is still waiting at the door
    release_first.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert order == ["first inside", "first leaving", "second inside"]


# --------------------------------------------------------------------------- #
# live.py: the short-term memory and the pipeline
# --------------------------------------------------------------------------- #

class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def tiny_frame(value=1.0) -> pd.DataFrame:
    return pd.DataFrame({"AAPL": [value, value + 1]}, index=pd.bdate_range("2026-01-05", periods=2, name="Date"))


def test_price_cache_returns_copies_and_forgets_after_the_time_limit():
    clock = FakeClock()
    cache = live.PriceCache(ttl_seconds=6 * 3600, max_entries=64, clock=clock)
    cache.put("k", tiny_frame())
    got = cache.get("k")
    got.iloc[0, 0] = 999.0  # changing what we got must not change what is stored
    assert cache.get("k").iloc[0, 0] == 1.0
    clock.now += 6 * 3600 - 1
    assert cache.get("k") is not None  # just under 6 hours: still there
    clock.now += 1
    assert cache.get("k") is None  # exactly 6 hours: gone
    assert len(cache) == 0
    assert live.PRICE_CACHE_TTL_SECONDS == 6 * 3600 and live.PRICE_CACHE_MAX_ENTRIES == 64


def test_price_cache_keeps_its_own_copy_of_what_it_was_given():
    cache = live.PriceCache(ttl_seconds=3600, max_entries=8, clock=FakeClock())
    original = tiny_frame()
    cache.put("k", original)
    original.iloc[0, 0] = 999.0  # the caller changes its table after storing it
    assert cache.get("k").iloc[0, 0] == 1.0


def test_price_cache_keeps_at_most_max_entries_and_drops_the_oldest_first():
    clock = FakeClock()
    cache = live.PriceCache(ttl_seconds=3600, max_entries=3, clock=clock)
    for name in "abcd":
        cache.put(name, tiny_frame())
        clock.now += 1
    assert len(cache) == 3
    assert cache.get("a") is None and all(cache.get(name) is not None for name in "bcd")
    cache.put("b", tiny_frame(5.0))  # storing a key again refreshes it (it becomes the newest)
    cache.put("e", tiny_frame())
    assert cache.get("c") is None and cache.get("b").iloc[0, 0] == 5.0


def test_price_cache_drops_expired_entries_when_storing():
    clock = FakeClock()
    cache = live.PriceCache(ttl_seconds=100, max_entries=64, clock=clock)
    cache.put("old", tiny_frame())
    clock.now += 101
    cache.put("new", tiny_frame())
    assert len(cache) == 1


def test_price_cache_is_safe_to_use_from_many_threads():
    cache = live.PriceCache(ttl_seconds=3600, max_entries=8)

    def hammer(n):
        for i in range(200):
            cache.put((n, i % 10), tiny_frame())
            cache.get((n, (i + 1) % 10))

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert len(cache) <= 8


def request_for(tickers=("AAPL", "SPY"), start=date(2025, 1, 6), end_exclusive=date(2025, 12, 31)):
    return inputs.LiveRequest(
        weights={"AAPL": 1.0}, benchmark="SPY", tickers=list(tickers), start=start, end_exclusive=end_exclusive
    )


class Rig:
    """A cache, two limiters with a fake clock, a gate and a fake loader, wired together for the pipeline tests."""

    def __init__(self, monkeypatch, budget=2, concurrent=1, loader_error=None, site_budget=None):
        self.clock = FakeClock()
        self.cache = live.PriceCache(clock=self.clock)
        self.limiter = limits.RateLimiter(budget, 3600, clock=self.clock)  # per visitor
        self.site_limiter = limits.RateLimiter(site_budget, 3600, clock=self.clock)  # the whole site (None = no cap)
        self.gate = limits.DownloadGate(concurrent)
        self.loader = FakeLoader(make_prices(columns=["AAPL", "MSFT", "SPY"]), loader_error)
        monkeypatch.setattr(data, "load_prices", self.loader)
        monkeypatch.setattr(data, "market_today", lambda: date(2026, 10, 20))

    def load(self, request=None, visitor="v1", **kwargs):
        return live.load_live_prices(
            request or request_for(),
            visitor=visitor,
            cache=self.cache,
            limiter=self.limiter,
            site_limiter=self.site_limiter,
            gate=self.gate,
            **kwargs,
        )

    def site_remaining(self):
        return self.site_limiter.remaining(live.SITE_KEY)


def new_request(n: int):
    """A request that differs from every other one (so the memory cannot answer it)."""
    return request_for(end_exclusive=date(2025, 12, 31) - timedelta(days=n))


def test_a_real_load_downloads_once_and_the_repeat_comes_from_memory(monkeypatch):
    rig = Rig(monkeypatch)
    first = rig.load()
    again = rig.load()

    assert first.from_cache is False and again.from_cache is True
    assert rig.loader.calls == [(["AAPL", "SPY"], date(2025, 1, 6), date(2025, 12, 31))]
    assert rig.loader.options == [{"cache_dir": None, "use_cache": True}]
    pd.testing.assert_frame_equal(first.prices, again.prices)
    assert rig.limiter.remaining("v1") == 1  # only the real call was counted


def test_cache_hits_never_use_the_budget_even_many_of_them(monkeypatch):
    rig = Rig(monkeypatch, budget=2)
    rig.load()
    for _ in range(25):
        assert rig.load().from_cache is True
    assert rig.limiter.remaining("v1") == 1 and len(rig.loader.calls) == 1


def test_a_request_from_memory_is_still_served_when_the_budget_is_used_up(monkeypatch):
    rig = Rig(monkeypatch, budget=1)
    rig.load()
    with pytest.raises(live.RateLimited) as caught:
        rig.load(request_for(end_exclusive=date(2025, 12, 30)))  # a NEW request: refused
    assert 0 < caught.value.retry_after <= 3600
    assert rig.load().from_cache is True  # the old one: served
    assert len(rig.loader.calls) == 1


def test_the_budget_comes_back_after_the_hour(monkeypatch):
    rig = Rig(monkeypatch, budget=1)
    rig.load()
    with pytest.raises(live.RateLimited) as caught:
        rig.load(request_for(end_exclusive=date(2025, 12, 30)))
    rig.clock.now += caught.value.retry_after + 1
    assert rig.load(request_for(end_exclusive=date(2025, 12, 30))).from_cache is False


def test_each_visitor_has_their_own_budget(monkeypatch):
    rig = Rig(monkeypatch, budget=1)
    rig.load(visitor="alice")
    rig.load(request_for(end_exclusive=date(2025, 12, 30)), visitor="bob")  # not blocked by alice
    with pytest.raises(live.RateLimited):
        rig.load(request_for(end_exclusive=date(2025, 12, 29)), visitor="alice")


def test_a_different_ticker_list_date_or_day_is_a_different_request(monkeypatch):
    rig = Rig(monkeypatch, budget=10)
    rig.load()
    rig.load(request_for(tickers=("MSFT", "SPY")))
    rig.load(request_for(tickers=("SPY", "AAPL")))  # same tickers, other order: columns would differ
    rig.load(request_for(start=date(2025, 1, 7)))
    rig.load(request_for(end_exclusive=date(2025, 12, 30)))
    assert len(rig.loader.calls) == 5
    monkeypatch.setattr(data, "market_today", lambda: date(2026, 10, 21))  # a new day: start fresh
    assert rig.load().from_cache is False
    assert len(rig.loader.calls) == 6


def test_the_download_runs_inside_a_gate_slot_with_the_given_timeout(monkeypatch):
    rig = Rig(monkeypatch)
    events = []

    class RecordingGate:
        def slot(self, timeout=5.0):
            gate = self

            class Slot:
                def __enter__(self_inner):
                    events.append(("enter", timeout))

                def __exit__(self_inner, *exc):
                    events.append(("exit",))

            return Slot()

    rig.gate = RecordingGate()
    real_loader = rig.loader
    monkeypatch.setattr(data, "load_prices", lambda *a, **k: (events.append(("load",)), real_loader(*a, **k))[1])
    rig.load(gate_timeout=7)

    assert events == [("enter", 7), ("load",), ("exit",)]
    assert live.GATE_TIMEOUT_SECONDS == 20  # the default: a 10-ticker x 10-year download can take a while


def test_a_busy_gate_is_a_limit_error_and_uses_no_budget(monkeypatch):
    rig = Rig(monkeypatch, budget=2, concurrent=1)
    with rig.gate.slot(timeout=1):  # somebody else is downloading
        with pytest.raises(limits.LimitError, match="busy"):
            rig.load(gate_timeout=0.05)
    assert rig.loader.calls == []
    assert rig.limiter.remaining("v1") == 2 and rig.limiter.key_count == 0


def test_a_visitor_who_waited_for_the_gate_gets_the_data_the_other_one_loaded(monkeypatch):
    rig = Rig(monkeypatch, budget=2)
    real_slot = rig.gate.slot

    class FillingGate:
        """While we wait for the slot, another visitor finishes the very same download."""

        def slot(self, timeout=5.0):
            rig.cache.put(live.cache_key(request_for()), make_prices(columns=["AAPL", "SPY"]))
            return real_slot(timeout=timeout)

    rig.gate = FillingGate()
    loaded = rig.load()

    assert loaded.from_cache is True
    assert rig.loader.calls == []
    assert rig.limiter.remaining("v1") == 2  # no download, no budget


def test_a_failed_download_still_counts_but_is_not_remembered(monkeypatch):
    rig = Rig(monkeypatch, budget=3, loader_error=data.ProviderUnavailableError("no data came back"))
    with pytest.raises(data.ProviderUnavailableError, match="no data came back"):
        rig.load()
    assert rig.limiter.remaining("v1") == 2  # Yahoo was asked, so it counts
    assert len(rig.cache) == 0
    rig.loader.error = None
    assert rig.load().from_cache is False  # the next try downloads for real
    assert rig.limiter.remaining("v1") == 1


def test_the_slot_is_given_back_when_the_download_fails(monkeypatch):
    rig = Rig(monkeypatch, budget=5, concurrent=1, loader_error=data.PriceDataError("nope"))
    for _ in range(3):
        with pytest.raises(data.PriceDataError):
            rig.load()
    with rig.gate.slot(timeout=0.2):  # the single slot is free again
        pass


# --- the site-wide limit (U1): a reload gives a visitor a fresh personal budget, never a fresh site budget ---

def test_a_real_load_spends_one_unit_of_each_budget_and_a_memory_hit_spends_none(monkeypatch):
    rig = Rig(monkeypatch, budget=5, site_budget=30)
    rig.load()
    assert (rig.limiter.remaining("v1"), rig.site_remaining()) == (4, 29)
    for _ in range(20):
        assert rig.load().from_cache is True
    assert (rig.limiter.remaining("v1"), rig.site_remaining()) == (4, 29)  # free for both
    assert len(rig.loader.calls) == 1


def test_the_31st_real_download_on_the_whole_site_is_refused_with_the_site_message(monkeypatch):
    cap = limits.PUBLIC_LIMITS.max_site_live_loads_per_hour  # 30 on the public site
    rig = Rig(monkeypatch, budget=limits.PUBLIC_LIMITS.max_live_loads_per_hour, site_budget=cap)
    for n in range(cap):  # 30 different visitors, each with a NEW request: all fine
        assert rig.load(new_request(n), visitor=f"visitor-{n}").from_cache is False
    assert len(rig.loader.calls) == cap and rig.site_remaining() == 0

    with pytest.raises(live.SiteRateLimited) as caught:  # the 31st visitor: the site is full
        rig.load(new_request(cap), visitor="visitor-31")
    assert 0 < caught.value.retry_after <= 3600
    assert isinstance(caught.value, live.RateLimited)
    assert len(rig.loader.calls) == cap  # nothing more went to Yahoo
    # The refused visitor lost nothing: their own allowance is untouched, and the site's stays at zero.
    assert rig.limiter.remaining("visitor-31") == limits.PUBLIC_LIMITS.max_live_loads_per_hour
    assert rig.site_remaining() == 0


def test_a_memory_hit_is_still_served_after_the_site_is_full(monkeypatch):
    rig = Rig(monkeypatch, budget=10, site_budget=2)
    rig.load(new_request(1), visitor="a")
    rig.load(new_request(2), visitor="b")
    with pytest.raises(live.SiteRateLimited):
        rig.load(new_request(3), visitor="c")
    served = rig.load(new_request(1), visitor="c")  # somebody else's earlier request: free
    assert served.from_cache is True
    assert len(rig.loader.calls) == 2 and rig.site_remaining() == 0 and rig.limiter.remaining("c") == 10


def test_a_visitor_refusal_does_not_spend_the_site_budget(monkeypatch):
    rig = Rig(monkeypatch, budget=1, site_budget=30)
    rig.load(new_request(1), visitor="alice")
    assert rig.site_remaining() == 29
    for n in range(2, 8):  # alice is out of personal budget: refused again and again
        with pytest.raises(live.RateLimited) as caught:
            rig.load(new_request(n), visitor="alice")
        assert not isinstance(caught.value, live.SiteRateLimited)  # the visitor message, not the site one
    assert rig.site_remaining() == 29  # not a unit spent by the refusals
    assert len(rig.loader.calls) == 1


def test_a_site_refusal_does_not_spend_the_visitor_budget(monkeypatch):
    rig = Rig(monkeypatch, budget=3, site_budget=1)
    rig.load(new_request(1), visitor="alice")
    for n in range(2, 6):
        with pytest.raises(live.SiteRateLimited):
            rig.load(new_request(n), visitor="bob")
    assert rig.limiter.remaining("bob") == 3  # bob never lost a unit
    assert rig.limiter.remaining("alice") == 2


def test_when_both_limits_are_used_up_the_site_message_wins(monkeypatch):
    rig = Rig(monkeypatch, budget=1, site_budget=1)
    rig.load(new_request(1), visitor="alice")
    with pytest.raises(live.SiteRateLimited):  # waiting out alice's own limit would not help: the site is full
        rig.load(new_request(2), visitor="alice")


def test_a_busy_gate_spends_neither_budget(monkeypatch):
    rig = Rig(monkeypatch, budget=2, site_budget=30, concurrent=1)
    with rig.gate.slot(timeout=1):  # somebody else is downloading
        with pytest.raises(limits.LimitError, match="busy"):
            rig.load(gate_timeout=0.05)
    assert rig.loader.calls == []
    assert rig.limiter.remaining("v1") == 2 and rig.limiter.key_count == 0
    assert rig.site_remaining() == 30 and rig.site_limiter.key_count == 0


def test_the_site_budget_comes_back_after_the_hour(monkeypatch):
    rig = Rig(monkeypatch, budget=10, site_budget=1)
    rig.load(new_request(1), visitor="a")
    with pytest.raises(live.SiteRateLimited) as caught:
        rig.load(new_request(2), visitor="b")
    rig.clock.now += caught.value.retry_after + 1
    assert rig.load(new_request(2), visitor="b").from_cache is False


def test_on_your_own_computer_there_is_no_site_cap(monkeypatch):
    local = limits.LOCAL_LIMITS
    assert local.max_site_live_loads_per_hour is None
    rig = Rig(monkeypatch, budget=local.max_live_loads_per_hour, site_budget=local.max_site_live_loads_per_hour)
    for n in range(60):  # twice the public cap, all new requests
        assert rig.load(new_request(n), visitor=f"visitor-{n % 3}").from_cache is False
    assert len(rig.loader.calls) == 60 and rig.site_remaining() is None


class MeetingLimiter(limits.RateLimiter):
    """A limiter whose N-th ``retry_after`` call (per thread) takes its look, then waits for the other thread.

    It makes two threads take their look at the same moment, before either has spent anything, so a race
    that would be rare in real life happens every time. If the code under test lets only one thread in at a time (a lock, a gate),
    the other cannot arrive, the wait times out, and the test goes on: no deadlock either way.
    """

    def __init__(self, *args, meet_on_call: int, parties: int = 2, wait_seconds: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._meet_on_call = meet_on_call
        self._barrier = threading.Barrier(parties, timeout=wait_seconds)
        self._calls = {}

    def retry_after(self, key):
        answer = super().retry_after(key)  # take the look FIRST ...
        me = threading.get_ident()
        self._calls[me] = self._calls.get(me, 0) + 1
        if self._calls[me] == self._meet_on_call:
            try:
                self._barrier.wait()  # ... then wait for the other thread to have taken its look too
            except threading.BrokenBarrierError:
                pass
        return answer  # (an answer that may be out of date by now: that is the race)


def run_visitors(rig, parties: int):
    """``parties`` different visitors press "run" for different requests at the same moment."""
    start_together = threading.Barrier(parties)
    outcomes = []

    def visit(n):
        start_together.wait()
        try:
            rig.load(new_request(n), visitor=f"visitor-{n}")
            outcomes.append("ok")
        except live.SiteRateLimited:
            outcomes.append("site")

    threads = [threading.Thread(target=visit, args=(n,)) for n in range(parties)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return sorted(outcomes)


def test_a_visitor_who_passed_the_first_look_is_still_refused_by_the_look_inside_the_gate(monkeypatch):
    """Two visitors both see room (before the gate). The gate lets them in one by one: the second must be refused."""
    rig = Rig(monkeypatch, budget=10, site_budget=1, concurrent=1)
    rig.site_limiter = MeetingLimiter(1, 3600, clock=rig.clock, meet_on_call=1)  # both pass the FIRST look together
    assert run_visitors(rig, 2) == ["ok", "site"]
    assert len(rig.loader.calls) == 1  # one download for a site budget of one


def test_both_budgets_are_spent_in_one_locked_step_even_when_the_gate_lets_several_downloads_run(monkeypatch):
    """Without one step for "look, then spend both", two visitors could both pass the look and overspend."""
    rig = Rig(monkeypatch, budget=10, site_budget=1, concurrent=4)
    rig.site_limiter = MeetingLimiter(1, 3600, clock=rig.clock, meet_on_call=2)  # both meet INSIDE the spending step
    assert run_visitors(rig, 2) == ["ok", "site"]
    assert len(rig.loader.calls) == 1


# --------------------------------------------------------------------------- #
# charts.py
# --------------------------------------------------------------------------- #

def test_chart_helpers_contain_no_chart_side_math():
    prices = make_prices()
    cols = HOLDINGS + [BENCHMARK]
    compared = pd.DataFrame({"Portfolio": risk.portfolio_value(prices[HOLDINGS], DEMO_WEIGHTS), BENCHMARK: prices[BENCHMARK]})
    colors = charts.series_colors(list(compared.columns), "Portfolio", BENCHMARK)
    built = [
        charts.line_chart(risk.cumulative_returns(compared), "Growth", "Growth of $1", colors=colors),
        charts.line_chart(risk.cumulative_returns(prices[cols]), "Growth", "Growth of $1"),
        charts.line_chart(
            risk.drawdown_series(compared), "Drawdown", "Loss", y_axis_format=".0%", colors=colors, upside_down=True
        ),
        charts.line_chart(
            risk.rolling_volatility(prices[cols], 60), "Volatility", "Annualized volatility", y_axis_format=".0%"
        ),
        charts.correlation_chart(risk.return_correlation(prices[cols])),
    ]
    for chart in built:
        assert_no_chart_math(chart.to_dict())


def test_the_chart_math_guard_catches_the_usual_tricks():
    rows = pd.DataFrame({"Date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "Value": [1.0, 2.0], "Ticker": ["A", "A"]})
    plain = alt.Chart(rows).mark_line().encode(x="Date:T", y="Value:Q")
    assert_no_chart_math(plain.to_dict())  # a plain chart passes

    tricks = {
        "calculate": plain.transform_calculate(Value="datum.Value * 100"),
        "window sum": plain.transform_window(Value="sum(Value)"),
        "filter": plain.transform_filter("datum.Value > 1"),
        "aggregate shorthand": alt.Chart(rows).mark_line().encode(x="Date:T", y="mean(Value):Q"),
        "aggregate argument": alt.Chart(rows).mark_line().encode(x="Date:T", y=alt.Y("Value:Q", aggregate="sum")),
        "other expression": plain.encode(color=alt.condition("datum.Value > 1", alt.value("red"), alt.value("blue"))),
        # M43: imputing plots invented zeros for missing days
        "impute": alt.Chart(rows).mark_line().encode(
            x="Date:T", y=alt.Y("Value:Q", impute=alt.ImputeParams(value=0)), color="Ticker:N"
        ),
        # M44: binning a colour channel groups numbers into ranges
        "bin on colour": alt.Chart(rows).mark_rect().encode(x="Date:T", y="Ticker:N", color=alt.Color("Value:Q", bin=True)),
        "bin on an axis": alt.Chart(rows).mark_bar().encode(x=alt.X("Value:Q", bin=alt.Bin(maxbins=5)), y="count()"),
        "time unit": alt.Chart(rows).mark_line().encode(x=alt.X("Date:T", timeUnit="month"), y="Value:Q"),
        "stack": alt.Chart(rows).mark_area().encode(x="Date:T", y=alt.Y("Value:Q", stack="normalize"), color="Ticker:N"),
        "sort by an operation": alt.Chart(rows).mark_bar().encode(
            x="Ticker:N", y="Value:Q", color=alt.Color("Ticker:N", sort=alt.EncodingSortField(field="Value", op="mean"))
        ),
        "a log scale": plain.encode(y=alt.Y("Value:Q", scale=alt.Scale(type="log"))),
        "an offset channel": plain.encode(xOffset="Ticker:N"),
        "a size channel": plain.encode(size="Value:Q"),
        "a tooltip with an aggregate": plain.encode(tooltip=[alt.Tooltip("sum(Value):Q")]),
    }
    for name, chart in tricks.items():
        with pytest.raises(AssertionError):
            assert_no_chart_math(chart.to_dict())
        assert name  # (the name shows in the failure report)


def test_the_chart_math_guard_accepts_what_the_ui_really_draws():
    from app_helpers import ALLOWED_CHANNELS

    frame = pd.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]}, index=pd.bdate_range("2026-01-05", periods=2, name="Date"))
    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"])
    for chart in (
        charts.line_chart(frame, "Value", "v"),
        charts.line_chart(frame, "Value", "v", y_axis_format=".0%", colors=["#111111", "#222222"], upside_down=True),
        charts.correlation_chart(corr),
    ):
        assert_no_chart_math(chart.to_dict())
    assert ALLOWED_CHANNELS == {"x", "y", "color", "text", "tooltip"}


def test_heatmap_leaves_undefined_correlations_blank():
    corr = pd.DataFrame(
        [[1.0, np.nan, 0.5], [np.nan, np.nan, np.nan], [0.5, np.nan, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    spec = charts.correlation_chart(corr).to_dict()
    rows = next(iter(spec["datasets"].values()))
    assert len(rows) == 4  # only the four defined cells are drawn
    assert all(row["Correlation"] is not None for row in rows)
    # The colour scale is fixed to -1..1 whatever the data looks like.
    scale = spec["layer"][0]["encoding"]["color"]["scale"]
    assert scale["domain"] == [-1, 1]
    assert scale["scheme"] == "redblue" and scale["reverse"] is True


def test_to_long_drops_empty_cells_and_keeps_the_dates():
    frame = pd.DataFrame({"A": [1.0, np.nan], "B": [3.0, 4.0]}, index=pd.bdate_range("2026-01-05", periods=2, name="Date"))
    long = charts.to_long(frame, "Value")
    assert len(long) == 3 and set(long.columns) == {"Date", "Ticker", "Value"}
    assert long["Value"].tolist() == [1.0, 3.0, 4.0] or sorted(long["Value"].tolist()) == [1.0, 3.0, 4.0]


def test_series_colors_gives_the_portfolio_and_the_benchmark_their_own_colours():
    names = ["AAPL", "Portfolio", "SPY", "MSFT"]
    colors = charts.series_colors(names, "Portfolio", "SPY")
    assert colors[1] == charts.PORTFOLIO_COLOR and colors[2] == charts.BENCHMARK_COLOR
    assert colors[0] != colors[3] and len(colors) == 4
    many = charts.series_colors([f"T{n}" for n in range(25)], "Portfolio", "SPY")  # more holdings than colours
    assert len(many) == 25


def test_line_chart_flips_the_axis_only_when_asked():
    frame = pd.DataFrame({"A": [0.0, 0.1]}, index=pd.bdate_range("2026-01-05", periods=2, name="Date"))
    normal = charts.line_chart(frame, "Value", "v").to_dict()["encoding"]["y"]["scale"]
    flipped = charts.line_chart(frame, "Value", "v", upside_down=True).to_dict()["encoding"]["y"]["scale"]
    assert normal.get("reverse") is False and normal["zero"] is False  # growth lines do not start at zero
    assert flipped["reverse"] is True and flipped["zero"] is True
