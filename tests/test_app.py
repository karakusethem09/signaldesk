"""Tests for the dashboard page in DEMO mode (app.py and the ui package). None of these tests use the network.

The page has two modes. These tests cover the default one, the demo portfolio, because the
page looks the same in both modes once the prices are loaded: the key numbers, the six tabs, the
warnings. test_app_custom.py covers what is specific to custom (live) mode: the form, the limits,
the rate limit, the errors.

See app_helpers.py for how the page is run and read. Every expected number below is computed in
the test from ``signaldesk/risk.py`` itself, so the test fails if the page shows something else.
"""

import json
import re
import socket
from datetime import date

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from app_helpers import (
    BENCHMARK,
    DEMO_WEIGHTS,
    HOLDINGS,
    MODE_CUSTOM,
    MODE_DEMO,
    ROOT,
    assert_sortable_table,
    assert_chart_shows,
    assert_no_chart_math,
    captions,
    chart_data,
    chart_spec,
    chart_table,
    chart_tickers,
    charts,
    display_values,
    expected_portfolio,
    expected_summary,
    install_demo,
    make_demo,
    make_prices,
    metrics_table,
    run_app,
    shown_table,
    tab,
    texts,
)

from signaldesk import data, limits, risk
from ui import analysis, sections, state

# The tests call risk.py themselves to know what the page must show; its warnings are not news there.
pytestmark = [
    pytest.mark.usefixtures("app_env"),
    pytest.mark.filterwarnings("ignore::signaldesk.risk.ShortSampleWarning"),
    pytest.mark.filterwarnings("ignore::signaldesk.risk.GapWarning"),
    pytest.mark.filterwarnings("ignore::signaldesk.risk.SkippedTickerWarning"),
]

FOOTER = "Not investment advice. Data: Yahoo Finance, end-of-day."
ALL_TABLE_ROWS = HOLDINGS + ["Portfolio", BENCHMARK]


def pct(value: float) -> str:
    """How a percent must look, written independently of the page's own formatter."""
    return f"{value:,.1%}"


def kpi_texts(at) -> dict:
    """The five key numbers as shown: label -> (big text, small benchmark text)."""
    return {m.label: (m.value, m.delta) for m in at.metric}


# --------------------------------------------------------------------------- #
# The page opens in demo mode
# --------------------------------------------------------------------------- #

def test_the_page_opens_in_demo_mode_and_renders(monkeypatch):
    install_demo(monkeypatch, make_demo())
    at = run_app()

    assert not at.exception
    assert at.sidebar.radio(key="mode").value == MODE_DEMO  # demo is the default
    assert at.title[0].value == "SignalDesk"
    assert [t.label for t in at.tabs] == ["Overview", "Risk", "Correlation", "Holdings", "Data", "About"]
    assert [s.value for s in at.subheader] == [
        "Growth of $1",
        "Holdings and starting weights",
        "Drawdown (loss from the previous peak)",
        "Rolling volatility (60-day window)",
        "Correlation of daily returns",
        "Key metrics",
        "Data quality",
        "Data source",
        "What this is",
        "How it was built",
        "How the numbers are defined",
        "Disclaimer",
    ]
    assert len(charts(at)) == 4  # growth, drawdown, rolling volatility, heatmap
    assert len(at.metric) == 5
    assert not at.error
    assert not at.warning  # clean synthetic data: nothing to warn about
    assert not at.info
    assert FOOTER in captions(at)
    # Charts must plot risk.py's numbers as they are: no Vega-Lite transform or aggregate.
    for chart in charts(at):
        assert_no_chart_math(chart_spec(chart))


def test_the_offline_guard_really_blocks_the_network(app_env):
    """If the guard did nothing, the "works with the network blocked" test below would prove nothing."""
    with pytest.raises(AssertionError, match="Python socket"):
        socket.create_connection(("127.0.0.1", 1), timeout=1)
    assert [kind for kind, _ in app_env] == ["socket"]
    app_env.clear()  # expected: do not let the fixture's final check fail this test


def test_the_real_demo_snapshot_renders_with_the_network_blocked(app_env):
    at = run_app()  # nothing is faked: this reads data/demo/*.csv and *.json

    assert not at.exception
    assert not at.error and not at.warning
    assert at.sidebar.radio(key="mode").value == MODE_DEMO
    snapshot = data.load_demo()
    meta = json.loads((ROOT / "data" / "demo" / "demo_meta.json").read_text())
    header = [m.value for m in at.markdown if m.value.startswith("**Demo data as of")]
    assert len(header) == 1
    assert f"Demo data as of {meta['as_of']}**" in header[0]  # the date in demo_meta.json
    assert meta["as_of"] == f"{snapshot.as_of:%Y-%m-%d}"
    assert header[0] == "**Demo data as of 2026-10-19** · [GitHub](https://github.com/karakusethem09/signaldesk)"

    # The starting weights are the ones in demo_portfolio.json.
    config = json.loads((ROOT / "data" / "demo" / "demo_portfolio.json").read_text())
    weights_table = display_values(tab(at, "Overview").dataframe[0])
    assert list(weights_table.index) == list(config["holdings"])
    assert weights_table["Starting weight"].tolist() == [pct(w) for w in config["holdings"].values()]

    # The key numbers are risk_summary of the Portfolio and of SPY, formatted.
    value, bench = expected_portfolio(snapshot.prices, config["holdings"], config["benchmark"])
    expected = risk.risk_summary(pd.DataFrame({"Portfolio": value, "SPY": bench}), confidence=0.95, risk_free_rate=0.0)
    shown = kpi_texts(at)
    for label, field in [
        ("Annualized return", "annualized_return"),
        ("Annualized volatility", "annualized_volatility"),
        ("Max drawdown (loss)", "max_drawdown"),
        ("1-day VaR 95% (loss)", "var_95"),
    ]:
        assert shown[label] == (pct(expected.loc["Portfolio", field]), f"SPY: {pct(expected.loc['SPY', field])}"), label
    assert shown["Sharpe ratio"] == (
        f"{expected.loc['Portfolio', 'sharpe_ratio']:.2f}",
        f"SPY: {expected.loc['SPY', 'sharpe_ratio']:.2f}",
    )
    assert len(charts(at)) == 4


# --------------------------------------------------------------------------- #
# The key numbers
# --------------------------------------------------------------------------- #

def test_the_key_numbers_are_risk_summary_of_portfolio_and_benchmark_formatted(monkeypatch):
    demo = install_demo(monkeypatch, make_demo())
    shown = kpi_texts(run_app())
    value, bench = expected_portfolio(demo.prices)
    expected = risk.risk_summary(pd.DataFrame({"Portfolio": value, "SPY": bench}), confidence=0.95, risk_free_rate=0.0)

    assert list(shown) == [
        "Annualized return",
        "Annualized volatility",
        "Sharpe ratio",
        "Max drawdown (loss)",
        "1-day VaR 95% (loss)",
    ]
    for label, field in [
        ("Annualized return", "annualized_return"),
        ("Annualized volatility", "annualized_volatility"),
        ("Max drawdown (loss)", "max_drawdown"),
        ("1-day VaR 95% (loss)", "var_95"),
    ]:
        big, small = shown[label]
        assert big == pct(expected.loc["Portfolio", field]), label
        assert small == f"SPY: {pct(expected.loc['SPY', field])}", label  # the benchmark's own number
    assert shown["Sharpe ratio"] == (
        f"{expected.loc['Portfolio', 'sharpe_ratio']:.2f}",
        f"SPY: {expected.loc['SPY', 'sharpe_ratio']:.2f}",
    )


def test_losses_in_the_key_numbers_have_no_minus_sign(monkeypatch):
    install_demo(monkeypatch, make_demo())
    shown = kpi_texts(run_app())
    for label in ("Max drawdown (loss)", "1-day VaR 95% (loss)", "Annualized volatility"):
        for text in shown[label]:
            assert "-" not in text, (label, text)
            assert re.search(r"\d+\.\d%$", text), (label, text)  # one percent sign, one decimal


def test_the_key_numbers_use_the_chosen_risk_free_rate(monkeypatch):
    demo = install_demo(monkeypatch, make_demo())
    at = run_app()
    at.sidebar.number_input(key="risk_free_rate").set_value(0.04).run()

    assert not at.exception
    value, bench = expected_portfolio(demo.prices)
    expected = risk.risk_summary(pd.DataFrame({"Portfolio": value, "SPY": bench}), confidence=0.95, risk_free_rate=0.04)
    assert kpi_texts(at)["Sharpe ratio"] == (
        f"{expected.loc['Portfolio', 'sharpe_ratio']:.2f}",
        f"SPY: {expected.loc['SPY', 'sharpe_ratio']:.2f}",
    )
    # ... and the holdings table's Sharpe column follows too.
    full = expected_summary(demo.prices, rf=0.04)
    assert metrics_table(at)["Sharpe ratio"].tolist() == full["sharpe_ratio"].tolist()
    assert shown_table(at)["Sharpe ratio"].tolist() == [f"{value:.2f}" for value in full["sharpe_ratio"]]


def test_risk_free_rate_has_sensible_bounds_and_the_page_says_which_rate_it_used(monkeypatch):
    install_demo(monkeypatch, make_demo())
    at = run_app()
    box = at.sidebar.number_input(key="risk_free_rate")

    assert (box.min, box.max) == (-0.05, 0.20)  # typing 4.0 instead of 0.04 is not possible
    assert "Sharpe uses a risk-free rate of 0.0%/yr." in captions(at)
    at.sidebar.number_input(key="risk_free_rate").set_value(0.04).run()
    assert "Sharpe uses a risk-free rate of 4.0%/yr." in captions(at)


def test_a_cleared_risk_free_box_asks_for_a_number_instead_of_crashing(monkeypatch):
    install_demo(monkeypatch, make_demo())
    real_number_input = st.sidebar.number_input

    def cleared_box(*args, **kwargs):
        real_number_input(*args, **kwargs)  # the widget is drawn as usual ...
        return None  # ... but the visitor has emptied it, which Streamlit reports as None

    monkeypatch.setattr(st.sidebar, "number_input", cleared_box)
    at = run_app()

    assert not at.exception
    assert any("risk-free rate" in message for message in texts(at.info))
    assert len(at.metric) == 0
    assert FOOTER in captions(at)


# --------------------------------------------------------------------------- #
# Tab: Holdings (the metrics table)
# --------------------------------------------------------------------------- #

def test_metrics_table_has_a_row_per_holding_then_portfolio_and_benchmark(monkeypatch):
    install_demo(monkeypatch, make_demo())
    table = metrics_table(run_app())

    assert list(table.index) == ALL_TABLE_ROWS
    assert list(table.columns) == [
        "Annualized return",
        "Annualized volatility",
        "Sharpe ratio",
        "Max drawdown (loss)",
        "1-day VaR 95% (loss)",
        "Daily returns (n)",
        "Note",
    ]
    assert (table["Daily returns (n)"] == 299).all()  # 300 prices -> 299 returns
    assert not table.drop(columns="Note").isna().any().any()  # nothing empty
    assert (table["Note"] == "").all()


PERCENT_COLUMNS = {
    "Annualized return": "annualized_return",
    "Annualized volatility": "annualized_volatility",
    "Max drawdown (loss)": "max_drawdown",
    "1-day VaR 95% (loss)": "var_95",
}


def test_every_cell_the_user_sees_is_the_formatted_risk_summary_value(monkeypatch):
    demo = install_demo(monkeypatch, make_demo())
    shown = shown_table(run_app())
    expected = expected_summary(demo.prices)

    assert list(shown.index) == ALL_TABLE_ROWS
    for row in ALL_TABLE_ROWS:  # the holdings AND the Portfolio and SPY rows
        for label, column in PERCENT_COLUMNS.items():  # all four percent columns
            assert shown.loc[row, label] == pct(expected.loc[row, column]), (row, label)
        assert shown.loc[row, "Sharpe ratio"] == f"{expected.loc[row, 'sharpe_ratio']:.2f}", row
        assert shown.loc[row, "Daily returns (n)"] == str(int(expected.loc[row, "n_obs"])), row


def test_table_text_has_the_right_shape_whatever_the_formatter_does(monkeypatch):
    """Independent of the page's formatter: percents end in %, losses carry no minus, Sharpe has 2 decimals."""
    install_demo(monkeypatch, make_demo())
    shown = shown_table(run_app())
    percent = re.compile(r"^-?\d{1,3}(,\d{3})*\.\d%$")
    loss = re.compile(r"^\d{1,3}(,\d{3})*\.\d%$")  # no minus sign: losses are positive numbers
    for label in PERCENT_COLUMNS:
        assert shown[label].map(lambda text: bool(percent.match(text))).all(), (label, shown[label].tolist())
    for label in ("Max drawdown (loss)", "1-day VaR 95% (loss)", "Annualized volatility"):
        assert shown[label].map(lambda text: bool(loss.match(text))).all(), (label, shown[label].tolist())
    assert shown["Sharpe ratio"].map(lambda text: bool(re.match(r"^-?\d+\.\d{2}$", text))).all()
    assert shown["Daily returns (n)"].map(lambda text: text.isdigit()).all()


def test_hand_checked_numbers_appear_exactly_as_expected(monkeypatch):
    # AAPL is the whole portfolio. Prices 100, 110, 99, 108.9 give returns +10%, -10%, +10%. By hand:
    #   volatility = 0.11547 * sqrt(252) = 183.3%        Sharpe = 0.03333 / 0.11547 * sqrt(252) = 4.58
    #   max drawdown = 110 -> 99 = 10.0%                  VaR 95% = -(-0.1 + 0.1 * 0.2) = 8.0%
    #   3 returns stretched to a year give an absurd return, so the cap text is shown.
    index = pd.bdate_range("2025-01-01", periods=4, name="Date")
    prices = pd.DataFrame({"AAPL": [100.0, 110.0, 99.0, 108.9], "SPY": [100.0, 101.0, 102.0, 103.0]}, index=index)
    install_demo(monkeypatch, make_demo(frame=prices, weights={"AAPL": 1.0}))
    at = run_app()

    assert not at.exception
    shown = shown_table(at)
    for row in ("AAPL", "Portfolio"):  # a one-holding portfolio IS the holding
        assert shown.loc[row, "Annualized volatility"] == "183.3%", row
        assert shown.loc[row, "Sharpe ratio"] == "4.58", row
        assert shown.loc[row, "Max drawdown (loss)"] == "10.0%", row
        assert shown.loc[row, "1-day VaR 95% (loss)"] == "8.0%", row
        assert shown.loc[row, "Annualized return"] == ">100,000%", row
        assert shown.loc[row, "Daily returns (n)"] == "3", row
    # The same numbers on the key-number row.
    assert kpi_texts(at)["Annualized volatility"][0] == "183.3%"
    assert kpi_texts(at)["Max drawdown (loss)"][0] == "10.0%"
    assert kpi_texts(at)["1-day VaR 95% (loss)"][0] == "8.0%"
    assert kpi_texts(at)["Annualized return"][0] == ">100,000%"


def test_table_numbers_come_straight_from_risk_summary(monkeypatch):
    demo = install_demo(monkeypatch, make_demo())
    at = run_app()
    table = metrics_table(at)  # what is sent to the browser: the REAL numbers (so the columns sort by value)
    expected = expected_summary(demo.prices)

    assert table["Annualized return"].tolist() == expected["annualized_return"].tolist()
    assert table["Annualized volatility"].tolist() == expected["annualized_volatility"].tolist()
    assert table["1-day VaR 95% (loss)"].tolist() == expected["var_95"].tolist()
    assert table["Max drawdown (loss)"].tolist() == expected["max_drawdown"].tolist()
    assert table["Sharpe ratio"].tolist() == expected["sharpe_ratio"].tolist()
    assert table["Daily returns (n)"].tolist() == expected["n_obs"].tolist()
    assert_sortable_table(tab(at, "Holdings").dataframe[0], text_columns=["Note"])


# --------------------------------------------------------------------------- #
# Header, weights and the dates
# --------------------------------------------------------------------------- #

def test_as_of_line_uses_the_snapshots_date_not_todays(monkeypatch):
    demo = install_demo(monkeypatch, make_demo())
    at = run_app()

    header = [m.value for m in at.markdown if m.value.startswith("**Demo data as of")]
    assert len(header) == 1
    assert f"**Demo data as of {demo.as_of:%Y-%m-%d}**" in header[0]
    assert f"{data.market_today():%Y-%m-%d}" not in header[0]
    assert f"{date.today():%Y-%m-%d}" not in header[0]


def test_overview_lists_the_holdings_with_their_starting_weights(monkeypatch):
    install_demo(monkeypatch, make_demo())
    at = run_app()
    weights = display_values(tab(at, "Overview").dataframe[0])

    assert list(weights.index) == HOLDINGS
    assert weights["Starting weight"].tolist() == ["20.0%", "20.0%", "15.0%", "15.0%", "15.0%", "15.0%"]
    assert any(c.startswith("Period: 5Y. The portfolio starts on 2025-01-01") for c in captions(tab(at, "Overview")))


# --------------------------------------------------------------------------- #
# Periods
# --------------------------------------------------------------------------- #

def snapshot_like_demo(monkeypatch) -> data.DemoData:
    """Made-up prices over the same dates as the real snapshot (2021-10-20 to 2026-10-19)."""
    index = pd.bdate_range("2021-10-20", "2026-10-19", name="Date")
    frame = make_prices(len(index))
    frame.index = index
    return install_demo(monkeypatch, make_demo(frame=frame))


@pytest.mark.parametrize(
    "label, first_day",
    [
        ("1Y", "2025-10-20"),  # the last day 2026-10-19 minus one calendar year is a Sunday: the next row is Monday
        ("3Y", "2023-10-19"),
        ("5Y", "2021-10-20"),  # 2021-10-19 is before the snapshot starts: everything is shown
    ],
)
def test_demo_presets_pick_the_start_that_earliest_allowed_start_gives(monkeypatch, label, first_day):
    demo = snapshot_like_demo(monkeypatch)
    at = run_app()
    at.sidebar.radio(key="demo_period").set_value(label).run()

    assert not at.exception
    end_exclusive = demo.as_of + pd.Timedelta(days=1)
    start = limits.earliest_allowed_start(end_exclusive, int(label.removesuffix("Y")))
    rows = demo.prices.index[(demo.prices.index >= start)]
    assert f"{rows[0]:%Y-%m-%d}" == first_day  # the rows the preset must select
    value = chart_table(charts(tab(at, "Overview"))[0], "Growth")
    assert value.index[0] == rows[0]
    assert len(value) == len(rows)
    assert f"The portfolio starts on {first_day} and runs to 2026-10-19" in " ".join(captions(tab(at, "Overview")))


def test_demo_custom_dates_select_exactly_those_days(monkeypatch):
    demo = snapshot_like_demo(monkeypatch)
    at = run_app()
    at.sidebar.radio(key="demo_period").set_value("Custom dates").run()
    at.sidebar.date_input(key="demo_start").set_value(date(2024, 3, 4))
    at.sidebar.date_input(key="demo_end").set_value(date(2024, 6, 28))
    at.run()

    assert not at.exception
    growth = chart_table(charts(tab(at, "Overview"))[0], "Growth")
    inside = demo.prices.index[(demo.prices.index >= "2024-03-04") & (demo.prices.index <= "2024-06-28")]
    assert list(growth.index) == list(inside)  # the end day is INCLUDED
    assert growth.index[-1] == pd.Timestamp("2024-06-28")


def test_demo_dates_with_fewer_than_two_trading_days_get_a_message(monkeypatch):
    snapshot_like_demo(monkeypatch)
    at = run_app()
    at.sidebar.radio(key="demo_period").set_value("Custom dates").run()
    at.sidebar.date_input(key="demo_start").set_value(date(2024, 3, 9))  # a Saturday ...
    at.sidebar.date_input(key="demo_end").set_value(date(2024, 3, 10))  # ... to a Sunday: no trading day
    at.run()

    assert not at.exception
    assert any("fewer than 2 trading days" in message for message in texts(at.warning))
    assert len(at.metric) == 0
    assert FOOTER in captions(at)  # the page is still a page


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def test_growth_chart_plots_portfolio_against_benchmark_from_risk_py(monkeypatch):
    demo = install_demo(monkeypatch, make_demo())
    chart = charts(tab(run_app(), "Overview"))[0]
    value, bench = expected_portfolio(demo.prices)

    assert chart_tickers(chart) == ["Portfolio", "SPY"]
    assert_chart_shows(chart, "Growth", risk.cumulative_returns(pd.DataFrame({"Portfolio": value, "SPY": bench})))


def test_growth_chart_can_also_show_each_holding(monkeypatch):
    demo = install_demo(monkeypatch, make_demo())
    at = run_app()
    at.sidebar.toggle(key="show_holdings").set_value(True).run()

    assert not at.exception
    chart = charts(tab(at, "Overview"))[0]
    value, bench = expected_portfolio(demo.prices)
    frame = pd.concat([demo.prices.loc[value.index, HOLDINGS], pd.DataFrame({"Portfolio": value, "SPY": bench})], axis=1)
    assert chart_tickers(chart) == sorted(HOLDINGS + ["Portfolio", "SPY"])
    assert_chart_shows(chart, "Growth", risk.cumulative_returns(frame))
    assert_no_chart_math(chart_spec(chart))


def test_drawdown_chart_plots_risk_py_drawdowns_as_positive_numbers(monkeypatch):
    demo = install_demo(monkeypatch, make_demo())
    chart = charts(tab(run_app(), "Risk"))[0]
    value, bench = expected_portfolio(demo.prices)
    expected = risk.drawdown_series(pd.DataFrame({"Portfolio": value, "SPY": bench}))

    assert chart_tickers(chart) == ["Portfolio", "SPY"]  # the holdings only on request
    assert_chart_shows(chart, "Drawdown", expected)
    assert (chart_data(chart)["Drawdown"] >= 0).all()  # a loss is a positive number, never "-12%"
    spec = chart_spec(chart)
    assert spec["encoding"]["y"]["scale"]["reverse"] is True  # drawn hanging down from zero ("underwater")
    assert spec["encoding"]["y"]["axis"]["format"] == ".0%"


def test_rolling_volatility_chart_plots_risk_py_numbers_for_the_chosen_window(monkeypatch):
    demo = install_demo(monkeypatch, make_demo())
    at = run_app()
    at.sidebar.slider(key="window").set_value(30).run()

    value, bench = expected_portfolio(demo.prices)
    rolling_chart = charts(tab(at, "Risk"))[1]
    expected = risk.rolling_volatility(pd.DataFrame({"Portfolio": value, "SPY": bench}), 30)
    assert chart_tickers(rolling_chart) == ["Portfolio", "SPY"]  # the holdings only on request
    assert_chart_shows(rolling_chart, "Volatility", expected)
    assert chart_data(rolling_chart)["Date"].nunique() == 299 - 29  # a 30-day window starts after 29 returns


def test_the_volatility_window_slider_is_not_ignored(monkeypatch):
    install_demo(monkeypatch, make_demo())
    at = run_app()
    assert at.sidebar.slider(key="window").value == 60
    assert (at.sidebar.slider(key="window").min, at.sidebar.slider(key="window").max) == (20, 252)
    first = chart_table(charts(tab(at, "Risk"))[1], "Volatility")
    at.sidebar.slider(key="window").set_value(120).run()
    second = chart_table(charts(tab(at, "Risk"))[1], "Volatility")

    assert "Rolling volatility (120-day window)" in [s.value for s in tab(at, "Risk").subheader]
    assert len(second) == 299 - 119 and len(first) == 299 - 59
    assert not first.tail(10).equals(second.tail(10))  # different window, different numbers


def test_rolling_volatility_can_show_each_holding(monkeypatch):
    demo = install_demo(monkeypatch, make_demo())
    at = run_app()
    at.sidebar.toggle(key="show_holdings").set_value(True).run()

    chart = charts(tab(at, "Risk"))[1]
    value, bench = expected_portfolio(demo.prices)
    frame = pd.concat([demo.prices.loc[value.index, HOLDINGS], pd.DataFrame({"Portfolio": value, "SPY": bench})], axis=1)
    assert_chart_shows(chart, "Volatility", risk.rolling_volatility(frame, 60))


def test_heatmap_plots_the_return_correlation_of_holdings_and_benchmark_from_risk_py(monkeypatch):
    demo = install_demo(monkeypatch, make_demo())
    heatmap = charts(tab(run_app(), "Correlation"))[0]
    value, _ = expected_portfolio(demo.prices)
    expected = risk.return_correlation(demo.prices.loc[value.index, HOLDINGS + [BENCHMARK]])

    assert len(chart_data(heatmap)) == 49  # 7 x 7 squares: six holdings and the benchmark (not the Portfolio)
    assert_chart_shows(heatmap, "Correlation", expected, rows="Ticker A", columns="Ticker B")


def test_heatmap_colour_scale_is_red_blue_reversed_and_fixed_to_minus_one_one(monkeypatch):
    install_demo(monkeypatch, make_demo())
    spec = chart_spec(charts(tab(run_app(), "Correlation"))[0])
    squares, numbers = spec["layer"]

    assert squares["mark"]["type"] == "rect"
    scale = squares["encoding"]["color"]["scale"]
    assert scale["scheme"] == "redblue"  # a two-sided scheme: the middle is neutral
    assert scale["reverse"] is True  # positive correlation warm, negative cool
    assert scale["domain"] == [-1, 1]  # fixed, whatever the data looks like
    assert numbers["encoding"]["text"]["format"] == ".2f"  # numbers printed with 2 decimals


def test_portfolio_and_benchmark_lines_keep_their_own_colours(monkeypatch):
    install_demo(monkeypatch, make_demo())
    spec = chart_spec(charts(tab(run_app(), "Overview"))[0])
    scale = spec["encoding"]["color"]["scale"]

    assert scale["domain"] == ["Portfolio", "SPY"]
    assert len(scale["range"]) == 2 and scale["range"][0] != scale["range"][1]


# --------------------------------------------------------------------------- #
# Tab: Data
# --------------------------------------------------------------------------- #

def test_data_tab_lists_gaps_missing_days_and_where_the_data_came_from(monkeypatch):
    demo = install_demo(monkeypatch, make_demo())
    at = run_app()
    data_tab = tab(at, "Data")

    assert len(data_tab.dataframe) == 1  # find_gaps; the made-up days have no missing weekday, so no second table
    gaps = data_tab.dataframe[0].value  # what is sent to the browser
    assert list(gaps.index) == HOLDINGS + [BENCHMARK]
    # Readable column names, not find_gaps' internal ones ...
    assert list(gaps.columns) == ["Missing prices", "First missing", "Last missing"]
    assert gaps["Missing prices"].tolist() == [0] * 7  # counts are numbers (they sort as numbers)
    assert gaps["First missing"].isna().all() and gaps["Last missing"].isna().all()  # no date: empty, shown as a dash
    shown = display_values(data_tab.dataframe[0])
    assert (shown["First missing"] == "—").all() and (shown["Last missing"] == "—").all()  # not None, not NaT
    assert_sortable_table(data_tab.dataframe[0])
    # ... and a dash instead of "None"/"NaT" where there is no date.
    shown = display_values(data_tab.dataframe[0])
    assert (shown["First missing"] == "—").all() and (shown["Last missing"] == "—").all()
    assert (shown["Missing prices"] == "0").all()
    labels = [m.value for m in data_tab.markdown]
    assert any("Possible gaps (most are exchange holidays)" in label for label in labels)
    assert "None." in labels
    assert any(demo.source in text.value for text in data_tab.markdown)
    assert any("Demo snapshot: 300 trading days" in text.value for text in data_tab.markdown)
    assert any("Rows (trading days): **300**" in text.value for text in data_tab.markdown)


def test_missing_prices_show_readable_dates_in_the_data_tab(monkeypatch):
    def tweak(frame):
        frame.loc[frame.index[100:103], "MSFT"] = np.nan

    install_demo(monkeypatch, make_demo(tweak=tweak))
    at = run_app()
    shown = display_values(tab(at, "Data").dataframe[0])

    assert shown.loc["MSFT", "Missing prices"] == "3"
    assert shown.loc["MSFT", "First missing"] == "2025-05-21" and shown.loc["MSFT", "Last missing"] == "2025-05-23"
    assert shown.loc["AAPL", "First missing"] == "—"


def test_find_missing_days_is_asked_about_the_window_that_was_shown(monkeypatch):
    demo = snapshot_like_demo(monkeypatch)
    asked = []
    real_function = data.find_missing_days

    def spy(prices, start=None, end=None):
        asked.append((start, end))
        return real_function(prices, start, end)

    monkeypatch.setattr(data, "find_missing_days", spy)
    at = run_app()  # 5Y: the preset starts a day before the snapshot does
    assert asked == [(demo.start, demo.as_of + pd.Timedelta(days=1))]  # starts where the data starts; end EXCLUDED

    asked.clear()
    at.sidebar.radio(key="demo_period").set_value("1Y").run()
    assert asked == [(pd.Timestamp("2025-10-19"), demo.as_of + pd.Timedelta(days=1))]  # exactly the 1Y window


# --------------------------------------------------------------------------- #
# When the demo cannot be shown
# --------------------------------------------------------------------------- #

def test_a_broken_demo_snapshot_gives_a_friendly_error_and_a_working_page(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("The demo file demo_prices.csv is missing (looked in /somewhere).")

    monkeypatch.setattr(data, "load_demo", broken)
    at = run_app()

    assert not at.exception
    assert len(at.error) == 1
    assert "could not be loaded" in at.error[0].value
    assert "demo_prices.csv is missing" in at.error[0].value
    assert at.title[0].value == "SignalDesk"
    assert FOOTER in captions(at)  # the page ends cleanly
    assert len(at.metric) == 0 and len(charts(at)) == 0
    assert len(at.button) == 0  # no "Back to the demo": the demo is what failed
    # The sidebar still works: Custom mode is one click away.
    at.sidebar.radio(key="mode").set_value(MODE_CUSTOM).run()
    assert not at.exception
    assert any("press Run analysis" in message for message in texts(at.info))


# --------------------------------------------------------------------------- #
# Benchmark alignment (a holding that starts late shortens the portfolio)
# --------------------------------------------------------------------------- #

LATE_START_ROWS = 100  # XOM has no price on the first 100 of 300 days


def late_starter(frame):
    frame.loc[frame.index[:LATE_START_ROWS], "XOM"] = np.nan


def test_the_benchmark_is_measured_over_the_portfolios_dates_only(monkeypatch):
    demo = install_demo(monkeypatch, make_demo(tweak=late_starter))
    at = run_app()

    assert not at.exception
    value, aligned = expected_portfolio(demo.prices)
    assert len(value) == 200  # the portfolio starts when XOM does
    on_portfolio_dates = risk.risk_summary(pd.DataFrame({"Portfolio": value, "SPY": aligned}), 0.95, 0.0)
    over_full_history = risk.risk_summary(demo.prices[["SPY"]], 0.95, 0.0)
    # Sanity check of the test itself: the two periods really give different benchmark numbers.
    assert on_portfolio_dates.loc["SPY", "annualized_return"] != over_full_history.loc["SPY", "annualized_return"]

    shown = kpi_texts(at)
    assert shown["Annualized return"] == (
        pct(on_portfolio_dates.loc["Portfolio", "annualized_return"]),
        f"SPY: {pct(on_portfolio_dates.loc['SPY', 'annualized_return'])}",
    )
    assert shown["Annualized return"][1] != f"SPY: {pct(over_full_history.loc['SPY', 'annualized_return'])}"
    assert shown["Max drawdown (loss)"][1] == f"SPY: {pct(on_portfolio_dates.loc['SPY', 'max_drawdown'])}"
    # The table's benchmark row and its holdings use the same 199 returns as the portfolio.
    table = metrics_table(at)
    assert (table["Daily returns (n)"] == 199).all()
    assert table.loc["SPY", "Annualized return"] == on_portfolio_dates.loc["SPY", "annualized_return"]


def test_a_late_starting_holding_shows_the_shorter_history_warning_and_the_real_start_date(monkeypatch):
    demo = install_demo(monkeypatch, make_demo(tweak=late_starter))
    at = run_app()

    shorter = [m for m in texts(at.warning) if m.startswith("Short sample. portfolio_value:")]
    assert len(shorter) == 1
    assert "the portfolio starts on" in shorter[0] and "'XOM'" in shorter[0]
    start = demo.prices.index[LATE_START_ROWS]
    assert f"the portfolio starts on {start:%Y-%m-%d}" in shorter[0]
    overview = " ".join(captions(tab(at, "Overview")))
    assert f"The portfolio starts on {start:%Y-%m-%d}" in overview
    assert "later than the first date loaded (2025-01-01)" in overview
    # 199 returns is a real sample, so no row carries the per-row "short sample" flag; only the warning above.
    assert (metrics_table(at)["Note"] == "").all()


def test_benchmark_gaps_inside_the_portfolio_period_do_not_break_alignment(monkeypatch):
    def tweak(frame):
        frame.loc[frame.index[150], "SPY"] = np.nan  # one missing benchmark price in the middle

    demo = install_demo(monkeypatch, make_demo(tweak=tweak))
    at = run_app()

    assert not at.exception
    value, aligned = expected_portfolio(demo.prices)
    expected = risk.risk_summary(pd.DataFrame({"Portfolio": value, "SPY": aligned}), 0.95, 0.0)
    assert kpi_texts(at)["Annualized return"][1] == f"SPY: {pct(expected.loc['SPY', 'annualized_return'])}"


# --------------------------------------------------------------------------- #
# Awkward data
# --------------------------------------------------------------------------- #

def test_a_benchmark_with_only_two_prices_is_skipped_with_a_warning(monkeypatch):
    def tweak(frame):
        frame.loc[frame.index[:-2], "SPY"] = np.nan  # SPY has just 2 prices, at the end

    install_demo(monkeypatch, make_demo(tweak=tweak))
    at = run_app()

    assert not at.exception
    skipped = [m for m in texts(at.warning) if m.startswith("Skipped ticker")]
    assert len(skipped) == 1
    assert "SPY" in skipped[0]

    table = metrics_table(at)
    assert list(table.index) == ALL_TABLE_ROWS  # the row is still there ...
    # ... but empty: no number in the data, and a dash on the page (the placeholder; without it the browser
    # would draw "None" for an empty number cell)
    assert table.loc["SPY"].drop(["Daily returns (n)", "Note"]).isna().all()
    assert (shown_table(at).loc["SPY"].drop(["Daily returns (n)", "Note"]) == "—").all()
    assert table.loc["SPY", "Daily returns (n)"] == 1  # the real count: 2 prices give 1 return (too few to use)
    assert table.loc["SPY", "Note"] == "skipped (see warning above)"
    assert not table.drop(index="SPY").isna().any().any()  # the Portfolio and the holdings are fine
    assert_sortable_table(tab(at, "Holdings").dataframe[0], text_columns=["Note"])
    assert kpi_texts(at)["Annualized return"][1] == "SPY: —"  # no benchmark number to show

    overview = tab(at, "Overview")
    assert chart_tickers(charts(overview)[0]) == ["Portfolio"]  # no line for the skipped benchmark
    assert any(c.startswith("No line for SPY") for c in captions(overview))
    assert chart_tickers(charts(tab(at, "Risk"))[0]) == ["Portfolio"]
    assert chart_tickers(charts(tab(at, "Correlation"))[0]) == sorted(HOLDINGS)  # read from the plotted data


def test_a_stale_holding_with_one_price_is_skipped_and_the_rest_still_work(monkeypatch):
    def tweak(frame):
        frame.loc[frame.index[1:], "JPM"] = np.nan  # JPM has a price on the first day only (it stopped trading)

    install_demo(monkeypatch, make_demo(tweak=tweak))
    at = run_app()

    assert not at.exception
    assert any(m.startswith("Skipped ticker") and "JPM" in m for m in texts(at.warning))
    # The portfolio held JPM at its last price for 299 days: risk.py calls that a stale holding, and it is
    # longer than any holiday, so it is a warning and not the soft holiday note.
    stale = [m for m in texts(at.warning) if m.startswith("Possible missing data")]
    assert any("Stale holding: 'JPM' has no price after 2025-01-01" in m for m in stale)
    table = metrics_table(at)
    assert table.loc["JPM", "Note"] == "skipped (see warning above)"
    assert table.loc["JPM"].drop(["Daily returns (n)", "Note"]).isna().all()
    assert (shown_table(at).loc["JPM"].drop(["Daily returns (n)", "Note"]) == "—").all()
    assert not table.drop(index="JPM").isna().any().any()
    assert "JPM" not in chart_tickers(charts(tab(at, "Correlation"))[0])


def test_every_column_skipped_shows_an_error_instead_of_crashing(monkeypatch):
    def tweak(frame):
        frame.iloc[:-2, :] = np.nan  # only the last two rows have prices

    install_demo(monkeypatch, make_demo(tweak=tweak))
    at = run_app()

    assert not at.exception
    assert len(at.error) == 1
    assert "Could not calculate the risk numbers" in at.error[0].value
    assert len(charts(at)) == 0
    assert FOOTER in captions(at)


def test_a_holding_without_any_common_start_date_shows_an_error(monkeypatch):
    def tweak(frame):
        frame.loc[frame.index[:150], "XOM"] = np.nan
        frame.loc[frame.index[150:], "JNJ"] = np.nan  # no date on which XOM and JNJ both have a price

    install_demo(monkeypatch, make_demo(tweak=tweak))
    at = run_app()

    assert not at.exception
    assert len(at.error) == 1
    assert "there is no date on which every holding has a price" in at.error[0].value
    assert len(charts(at)) == 0


SHORT_BENCHMARK_WARNING = "Short sample."


def test_short_sample_is_flagged_in_the_table_and_warned_about(monkeypatch):
    def tweak(frame):
        frame.loc[frame.index[:-31], "SPY"] = np.nan  # SPY has 31 prices = 30 returns

    install_demo(monkeypatch, make_demo(tweak=tweak))
    at = run_app()

    assert not at.exception
    short = [m for m in texts(at.warning) if m.startswith(SHORT_BENCHMARK_WARNING)]
    assert short and all("SPY" in message for message in short)

    table = metrics_table(at)
    assert table.loc["SPY", "Daily returns (n)"] == 30
    assert table.loc["SPY", "Note"].startswith("⚠")
    assert (table.drop(index="SPY")["Note"] == "").all()
    assert any("⚠" in caption for caption in captions(tab(at, "Holdings")))  # the explanation is shown


def test_extreme_short_sample_row_is_flagged_capped_and_does_not_break_the_charts(monkeypatch):
    def tweak(frame):
        # SPY has 4 prices: returns of +18%, +93% and +135% (GME in January 2021). Stretched
        # to a year, that is a CAGR with 60 digits, which is why the flag exists.
        frame["SPY"] = np.nan
        frame.iloc[-4:, frame.columns.get_loc("SPY")] = [100.0, 118.0, 227.74, 535.0]

    demo = install_demo(monkeypatch, make_demo(tweak=tweak))
    at = run_app()

    assert not at.exception
    table = metrics_table(at)
    assert table.loc["SPY", "Daily returns (n)"] == 3
    assert table.loc["SPY", "Annualized return"] > 1000  # absurdly large, as expected
    assert table.loc["SPY", "Note"].startswith("⚠")
    # The shown text is capped, not a 60-digit number ...
    assert shown_table(at).loc["SPY", "Annualized return"] == ">100,000%"
    assert kpi_texts(at)["Annualized return"][1] == "SPY: >100,000%"  # ... on the key numbers too
    # risk_summary calculates this wild ticker, and so do the price-taking chart functions:
    # nothing says "this chart could not be drawn" any more (that was the old A4 problem).
    assert not at.error
    assert not any("could not be drawn" in m for m in texts(at.warning))
    assert chart_tickers(charts(tab(at, "Correlation"))[0]) == sorted(HOLDINGS + [BENCHMARK])
    rolling = charts(tab(at, "Risk"))[1]
    assert chart_tickers(rolling) == ["Portfolio"]  # SPY has 3 returns: fewer than the window, so no line
    assert "No line for SPY: fewer than 60 returns in the range." in captions(tab(at, "Risk"))


def test_a_gme_like_holding_does_not_remove_the_rolling_and_correlation_charts(monkeypatch):
    def tweak(frame):
        # A price that doubles and halves every day: returns of +100% and -50%. Its median absolute
        # return is 0.75, so the old price-checking functions would take its RETURNS for prices.
        frame["GME"] = 100.0 * np.where(np.arange(len(frame)) % 2 == 0, 1.0, 2.0)

    weights = {**DEMO_WEIGHTS, "JNJ": 0.10, "GME": 0.05}
    demo = install_demo(monkeypatch, make_demo(tweak=tweak, weights=weights))
    # Proof that the old route refuses this ticker (and that the page does not use it):
    wild_returns = risk.daily_returns(demo.prices[["GME"]])
    with pytest.raises(ValueError, match="looks like prices"):
        risk.annualized_volatility(wild_returns, window=60)
    with pytest.raises(ValueError, match="looks like prices"):
        risk.correlation_matrix(pd.concat([wild_returns, wild_returns.rename(columns={"GME": "GME2"})], axis=1))

    at = run_app()
    at.sidebar.toggle(key="show_holdings").set_value(True).run()

    assert not at.exception
    assert not any("could not be drawn" in m for m in texts(at.warning))
    rolling = charts(tab(at, "Risk"))[1]
    assert "GME" in chart_tickers(rolling)
    heatmap = charts(tab(at, "Correlation"))[0]
    assert "GME" in chart_tickers(heatmap)
    assert len(chart_data(heatmap)) == 64  # 8 x 8: seven holdings and the benchmark
    value, _ = expected_portfolio(demo.prices, weights)
    assert_chart_shows(
        heatmap, "Correlation", risk.return_correlation(demo.prices.loc[value.index, list(weights) + [BENCHMARK]]),
        rows="Ticker A", columns="Ticker B",
    )


def test_a_gap_is_shown_once_as_an_info_note_not_a_warning(monkeypatch):
    def tweak(frame):
        frame.loc[frame.index[100], "MSFT"] = np.nan  # one missing price in the middle

    install_demo(monkeypatch, make_demo(tweak=tweak))
    at = run_app()

    assert not at.exception
    gap_notes = [m for m in texts(at.info) if m.startswith("Gaps in the data")]
    # risk_summary and daily_returns both warn about the bridged return: we show it once. The
    # portfolio's own note (it valued MSFT at its last price that day) is a different thing.
    returns_notes = [m for m in gap_notes if "bridged" in m]
    portfolio_notes = [m for m in gap_notes if "valued it at its last known price" in m]
    assert len(returns_notes) == 1 and len(portfolio_notes) == 1 and len(gap_notes) == 2
    # These sentences come from risk.py's own messages, not from the page's introductions.
    assert "'MSFT': 1 bridged (longest 1 row)" in returns_notes[0]
    assert "'MSFT': 1 valued at last price (longest 1 row)" in portfolio_notes[0]
    assert not at.warning


def test_correlation_warning_about_short_pairs_reaches_the_page(monkeypatch):
    def tweak(frame):
        for ticker in ("JPM", "XOM"):  # two tickers with 30 returns each
            frame.loc[frame.index[:-31], ticker] = np.nan

    # JPM and XOM are listed with weight 0, so they do not shorten the portfolio (a held late
    # starter would), but they are still in the table and in the heatmap.
    weights = {"AAPL": 0.4, "MSFT": 0.3, "NVDA": 0.3, "JPM": 0.0, "XOM": 0.0}
    install_demo(monkeypatch, make_demo(tweak=tweak, weights=weights))
    at = run_app()

    assert not at.exception
    pair_warnings = [m for m in texts(at.warning) if "correlation_matrix:" in m]
    assert len(pair_warnings) == 1
    assert pair_warnings[0].startswith("Short sample")
    assert "'JPM'/'XOM': 30" in pair_warnings[0]  # the pair that shares only 30 days
    # The summary's own short-sample warning (a different message) is shown as well.
    assert any("risk_summary:" in m and "'JPM': 30" in m for m in texts(at.warning))


@pytest.mark.parametrize(
    "n_prices, n_returns, flagged",
    [(61, 60, False), (60, 59, True)],  # exactly risk.MIN_RELIABLE_OBS returns is long enough
)
def test_short_sample_flag_starts_below_exactly_60_returns(monkeypatch, n_prices, n_returns, flagged):
    def tweak(frame):
        frame.loc[frame.index[:-n_prices], "SPY"] = np.nan

    install_demo(monkeypatch, make_demo(tweak=tweak))
    table = metrics_table(run_app())

    assert table.loc["SPY", "Daily returns (n)"] == n_returns
    assert table.loc["SPY", "Note"].startswith("⚠") is flagged
    assert risk.MIN_RELIABLE_OBS == 60  # if risk.py changes the limit, rethink these numbers


def test_a_line_without_enough_returns_for_the_window_is_named(monkeypatch):
    def tweak(frame):
        frame.loc[frame.index[:-40], "XOM"] = np.nan  # 40 prices = 39 returns, window is 60

    # XOM has weight 0 here, so the portfolio is not shortened.
    weights = {"AAPL": 0.5, "MSFT": 0.5, "XOM": 0.0}
    install_demo(monkeypatch, make_demo(tweak=tweak, weights=weights))
    at = run_app()
    assert not any(c.startswith("No line for") for c in captions(tab(at, "Risk")))  # XOM is not drawn: not named

    at.sidebar.toggle(key="show_holdings").set_value(True).run()
    assert not at.exception
    assert "No line for XOM: fewer than 60 returns in the range." in captions(tab(at, "Risk"))
    assert chart_tickers(charts(tab(at, "Risk"))[1]) == ["AAPL", "MSFT", "Portfolio", "SPY"]  # XOM really has no line


def test_no_note_when_every_line_has_a_rolling_value(monkeypatch):
    install_demo(monkeypatch, make_demo())
    at = run_app()
    assert not any(c.startswith("No line for") for c in captions(at))


def test_a_long_gap_is_a_warning_not_a_holiday_note(monkeypatch):
    def tweak(frame):
        frame.loc[frame.index[100:110], "MSFT"] = np.nan  # a 10-row outage, longer than any holiday

    install_demo(monkeypatch, make_demo(tweak=tweak))
    at = run_app()

    assert not at.exception
    long_gap = [m for m in texts(at.warning) if m.startswith("Possible missing data")]
    # Two notes: one about the returns that bridge the outage, one about the portfolio valuing MSFT at a stale price.
    assert len(long_gap) == 2
    assert any("'MSFT': 1 bridged (longest 10 rows)" in m for m in long_gap)
    assert any("'MSFT': 10 valued at last price (longest 10 rows)" in m for m in long_gap)
    assert all(analysis.LONG_GAP_PHRASE in m for m in long_gap)  # that sentence is risk.py's, not the page's
    assert not any(m.startswith("Gaps in the data") for m in texts(at.info))  # not the soft note
    # Each note has an introduction that describes ITS OWN problem.
    returns_note = next(m for m in long_gap if "bridged" in m)
    valuation_note = next(m for m in long_gap if "valued at last price" in m)
    assert "multi-week moves counted as one day" in returns_note
    assert "multi-week moves" not in valuation_note  # that sentence is about returns, not about valuing a holding
    assert "the portfolio valued it at its last known price all that time" in valuation_note
    assert "probably missing or stale" in valuation_note
    assert "more than 5 rows in a row" in valuation_note


def test_risk_py_still_uses_the_gap_wording_the_page_looks_for():
    prices = make_prices()[["AAPL"]]
    prices.iloc[100:110] = np.nan  # a 10-row outage
    with pytest.warns(risk.GapWarning, match=analysis.LONG_GAP_PHRASE):
        risk.daily_returns(prices)
    prices = make_prices()[["AAPL"]]
    prices.iloc[100:101] = np.nan  # a one-day holiday-style gap: no such sentence
    with pytest.warns(risk.GapWarning) as caught:
        risk.daily_returns(prices)
    assert analysis.LONG_GAP_PHRASE not in str(caught[0].message)

    # portfolio_value says it too, and starts its messages with its own name (the page tells
    # the portfolio's note from the returns note by that prefix).
    frame = make_prices()[["AAPL", "MSFT"]]
    frame.iloc[100:110, 0] = np.nan
    with pytest.warns(risk.GapWarning, match=analysis.LONG_GAP_PHRASE) as caught:
        risk.portfolio_value(frame, {"AAPL": 0.5, "MSFT": 0.5})
    assert str(caught[0].message).startswith(analysis.PORTFOLIO_MESSAGE_PREFIX)
    with pytest.warns(risk.GapWarning) as caught:
        risk.daily_returns(frame)
    assert not str(caught[0].message).startswith(analysis.PORTFOLIO_MESSAGE_PREFIX)


def test_a_one_holding_portfolio_still_gets_a_heatmap_with_the_benchmark(monkeypatch):
    install_demo(monkeypatch, make_demo(weights={"AAPL": 1.0}))
    at = run_app()

    assert not at.exception
    assert chart_tickers(charts(tab(at, "Correlation"))[0]) == ["AAPL", "SPY"]
    assert list(metrics_table(at).index) == ["AAPL", "Portfolio", "SPY"]


def test_fewer_than_two_comparable_tickers_shows_a_message_instead_of_the_heatmap(monkeypatch):
    def tweak(frame):
        frame.loc[frame.index[:-2], "SPY"] = np.nan  # the benchmark cannot be calculated

    install_demo(monkeypatch, make_demo(tweak=tweak, weights={"AAPL": 1.0}))
    at = run_app()

    assert not at.exception
    assert any("at least 2 tickers" in message for message in texts(tab(at, "Correlation").info))
    assert len(charts(tab(at, "Correlation"))) == 0


def test_window_longer_than_the_data_shows_a_message_instead_of_the_chart(monkeypatch):
    install_demo(monkeypatch, make_demo(frame=make_prices(100)))  # 99 returns
    at = run_app()
    at.sidebar.slider(key="window").set_value(252).run()

    assert not at.exception
    assert any("longer than the 99 daily returns" in message for message in texts(tab(at, "Risk").info))
    assert len(charts(tab(at, "Risk"))) == 1  # the drawdown chart; the rolling-volatility chart is replaced
    assert len(charts(tab(at, "Correlation"))) == 1


# --------------------------------------------------------------------------- #
# A chart that risk.py refuses must not take the page down
# --------------------------------------------------------------------------- #

def test_a_refused_chart_gets_a_plain_sentence_before_risk_pys_own_message(monkeypatch):
    install_demo(monkeypatch, make_demo())
    real = risk.return_correlation

    def refuse(prices):
        raise ValueError("returns: 'XOM' looks like prices, not returns; call daily_returns first")

    monkeypatch.setattr(risk, "return_correlation", refuse)
    at = run_app()

    assert not at.exception
    problems = [m for m in texts(at.warning) if m.startswith("This chart could not be drawn")]
    assert len(problems) == 1
    assert "so extreme that risk.py's safety check refused" in problems[0]
    assert problems[0].index("so extreme") < problems[0].index("looks like prices")  # plain words come first
    assert len(charts(at)) == 3  # growth, drawdown, rolling: only the heatmap is missing
    assert len(at.metric) == 5  # the key numbers stay
    assert real is not refuse


def test_another_refusal_is_labelled_generically(monkeypatch):
    install_demo(monkeypatch, make_demo())

    def refuse(prices, window):
        raise ValueError("something else")

    monkeypatch.setattr(risk, "rolling_volatility", refuse)
    at = run_app()

    problems = [m for m in texts(at.warning) if m.startswith("This chart could not be drawn")]
    assert len(problems) == 1 and "risk.py refused to calculate this chart." in problems[0]
    assert "risk.py said: something else" in problems[0]


def test_the_about_tab_links_to_the_catch_log_and_the_repository(monkeypatch):
    install_demo(monkeypatch, make_demo())
    about = " ".join(m.value for m in tab(run_app(), "About").markdown)
    assert "[catch log](https://github.com/karakusethem09/signaldesk/blob/main/docs/catch-log.md)" in about
    assert "[GitHub](https://github.com/karakusethem09/signaldesk)" in about
    assert sections.GITHUB_URL == "https://github.com/karakusethem09/signaldesk"


def skipped_benchmark(frame):
    frame.loc[frame.index[:-2], "SPY"] = np.nan  # SPY cannot be calculated: its whole row is empty


def stale_holding(frame):
    frame.loc[frame.index[1:], "JPM"] = np.nan  # JPM stopped trading on day one


def gap_in_the_middle(frame):
    frame.loc[frame.index[100:103], "MSFT"] = np.nan  # also gives find_gaps real dates, not "no date"


def short_benchmark(frame):
    frame.loc[frame.index[:-31], "SPY"] = np.nan  # 30 returns: a flagged row, not an empty one


def mixed_magnitudes(frame):
    """Single-digit and negative returns next to double-digit ones: the case where text and number order differ."""
    frame["XOM"] = frame["XOM"] * np.linspace(1.0, 0.45, len(frame))
    frame["JNJ"] = frame["JNJ"] * np.linspace(1.0, 0.8, len(frame))


def four_price_benchmark(frame):
    frame["SPY"] = np.nan
    frame.iloc[-4:, frame.columns.get_loc("SPY")] = [100.0, 118.0, 227.74, 535.0]  # a GME-like row


@pytest.mark.parametrize(
    "tweak",
    [None, skipped_benchmark, stale_holding, gap_in_the_middle, short_benchmark, four_price_benchmark],
    ids=["clean", "skipped-benchmark", "stale-holding", "gap", "short-benchmark", "gme-like-benchmark"],
)
def test_every_table_sorts_by_value_and_draws_a_dash_for_an_empty_cell(monkeypatch, tweak):
    """What a visitor does with a table in the browser, as far as AppTest can see it.

    - Numbers and dates are sent as numbers and dates, so a click on a header sorts by value (a column of
      texts would sort alphabetically: "5.1%" above "29.3%").
    - ``placeholder`` is the dash: without it the browser draws "None" for an empty cell and ignores the dash
      the Styler chose (AppTest reads only the Styler's text, so a test of that text alone would pass).
    """
    install_demo(monkeypatch, make_demo(tweak=tweak))
    at = run_app()

    assert not at.exception
    tables = [element for tab_ in at.tabs for element in tab_.dataframe]
    assert len(tables) >= 3  # the weights, the key metrics and the gaps (and the missing days, if there are any)
    for element in tables:
        # The missing-days table is texts on purpose (a date and a weekday name); every other table is numbers/dates.
        text_columns = ["Note"] + ["Date", "Weekday"]
        assert_sortable_table(element, text_columns=text_columns)


@pytest.mark.parametrize("tweak", [skipped_benchmark, mixed_magnitudes], ids=["skipped-benchmark", "mixed-magnitudes"])
def test_every_non_empty_cell_shows_the_formatted_risk_summary_value_and_an_empty_cell_a_dash(monkeypatch, tweak):
    demo = install_demo(monkeypatch, make_demo(tweak=tweak))
    at = run_app()
    shown = shown_table(at)
    expected = expected_summary(demo.prices)

    for row in ALL_TABLE_ROWS:
        for label, column in PERCENT_COLUMNS.items():
            value = expected.loc[row, column]
            assert shown.loc[row, label] == ("—" if pd.isna(value) else pct(value)), (row, label)
        sharpe = expected.loc[row, "sharpe_ratio"]
        assert shown.loc[row, "Sharpe ratio"] == ("—" if pd.isna(sharpe) else f"{sharpe:.2f}"), row
        assert shown.loc[row, "Daily returns (n)"] == f"{int(expected.loc[row, 'n_obs']):,}", row


def test_every_table_is_drawn_with_the_dash_placeholder(monkeypatch):
    install_demo(monkeypatch, make_demo(tweak=skipped_benchmark))
    at = run_app()
    tables = [element for tab_ in at.tabs for element in tab_.dataframe]
    # The weights, the key metrics and the gaps. (The made-up days have no missing weekday, so no fourth table.)
    assert len(tables) == 3 and all(element.proto.placeholder == "—" for element in tables)
    assert len(at.sidebar.dataframe) == 0  # demo mode: no editor in the sidebar


def test_all_four_tables_of_the_real_demo_have_the_placeholder_and_sort_by_value():
    at = run_app()  # the real snapshot has exchange holidays, so the missing-days table is there too
    tables = [element for tab_ in at.tabs for element in tab_.dataframe]
    assert len(tables) == 4
    for element in tables:
        assert_sortable_table(element, text_columns=["Note", "Date", "Weekday"])


def test_clicking_a_header_would_sort_the_numbers_by_value_not_as_text(monkeypatch):
    demo = install_demo(monkeypatch, make_demo(tweak=mixed_magnitudes))
    at = run_app()
    sent = metrics_table(at)
    expected = expected_summary(demo.prices)

    for column, field in [("Annualized return", "annualized_return"), ("Max drawdown (loss)", "max_drawdown")]:
        by_value = sent[column].sort_values(ascending=False).index.tolist()  # what the browser does with numbers
        assert by_value == expected[field].sort_values(ascending=False).index.tolist(), column
    # The scenario really is the awkward one: sorted as TEXT the returns would come out in another order,
    # and a negative number that is closer to zero would rank below a more negative one.
    shown = shown_table(at)
    as_text = shown["Annualized return"].sort_values(ascending=False).index.tolist()
    assert as_text != sent["Annualized return"].sort_values(ascending=False).index.tolist()
    assert (sent["Annualized return"] < 0).any() and (sent["Annualized return"].abs() < 0.1).any()


def test_dates_in_the_gaps_table_sort_chronologically(monkeypatch):
    def two_gaps(frame):
        frame.loc[frame.index[200:202], "MSFT"] = np.nan  # later
        frame.loc[frame.index[10:12], "NVDA"] = np.nan  # earlier, but NVDA comes after MSFT in the table

    install_demo(monkeypatch, make_demo(tweak=two_gaps))
    at = run_app()
    gaps = tab(at, "Data").dataframe[0].value

    assert pd.api.types.is_datetime64_any_dtype(gaps["First missing"])  # real dates, not text
    ordered = gaps["First missing"].sort_values().dropna().index.tolist()
    assert ordered == ["NVDA", "MSFT"]  # the earlier gap first, whatever the row order of the table
    assert gaps["Missing prices"].sort_values(ascending=False).iloc[0] == 2 and gaps["Missing prices"].dtype.kind == "i"


def test_the_note_column_is_wide_enough_to_read_and_the_numbers_are_right_aligned(monkeypatch):
    """The reviewer saw "skipped (se" cut off in a real browser."""
    import json

    install_demo(monkeypatch, make_demo(tweak=skipped_benchmark))
    columns = json.loads(tab(run_app(), "Holdings").dataframe[0].proto.columns)

    assert columns["Note"]["width"] == "large"
    for name in ("Annualized return", "Annualized volatility", "Sharpe ratio", "Max drawdown (loss)", "1-day VaR 95% (loss)", "Daily returns (n)"):
        assert columns[name]["alignment"] == "right", name


def test_a_number_that_does_not_exist_is_empty_in_the_data_and_a_dash_on_the_page(monkeypatch):
    install_demo(monkeypatch, make_demo(tweak=skipped_benchmark))
    at = run_app()
    metric_columns = ["Annualized return", "Annualized volatility", "Sharpe ratio", "Max drawdown (loss)", "1-day VaR 95% (loss)"]
    sent = metrics_table(at)
    assert sent.loc["SPY", metric_columns].isna().all()  # empty in the data (so it sorts apart from real numbers)
    assert shown_table(at).loc["SPY", metric_columns].tolist() == ["—"] * 5  # what the Styler says ...
    assert tab(at, "Holdings").dataframe[0].proto.placeholder == "—"  # ... and what the browser draws for the empty cell


def test_every_table_in_the_ui_goes_through_show_table_so_none_forgets_the_placeholder():
    import re as _re

    from app_helpers import ROOT as _root

    calls = {}
    for path in sorted((_root / "ui").glob("*.py")) + [_root / "app.py"]:
        text = path.read_text()
        for match in _re.finditer(r"st\.(dataframe|data_editor|table)\(", text):
            calls.setdefault(path.name, []).append(match.group(1))
    # The one st.dataframe is inside show_table (sections.py); the one st.data_editor is the sidebar's holdings editor.
    assert calls == {"sections.py": ["dataframe"], "state.py": ["data_editor"]}
    assert 'st.dataframe(table, placeholder=PLACEHOLDER, **options)' in (_root / "ui" / "sections.py").read_text()


# --------------------------------------------------------------------------- #
# Any failure while loading the demo keeps the page (title, message, footer)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "error",
    [KeyError("holdings"), RuntimeError("the demo reader blew up"), OSError(13, "Permission denied: /secret/path")],
    ids=["KeyError", "RuntimeError", "OSError"],
)
def test_an_unexpected_error_while_loading_the_demo_still_gives_a_page(monkeypatch, caplog, error):
    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(data, "load_demo", broken)
    with caplog.at_level("ERROR"):
        at = run_app()

    assert not at.exception  # no generic Streamlit error box
    assert at.title[0].value == "SignalDesk"
    assert len(at.error) == 1
    assert "could not be loaded because of an unexpected problem" in at.error[0].value
    for secret in ("holdings", "blew up", "/secret/path", "Permission", "Traceback"):
        assert secret not in at.error[0].value  # internals stay in the server log
    assert any(secret in str(r.exc_info[1]) for r in caplog.records if r.exc_info for secret in (str(error),))
    assert FOOTER in captions(at)
    assert len(at.metric) == 0 and len(at.button) == 0  # no "Back to the demo": the demo is what failed
    # The sidebar still works: Custom mode is one click away.
    at.sidebar.radio(key="mode").set_value(MODE_CUSTOM).run()
    assert not at.exception and any("press Run analysis" in m for m in texts(at.info))


def test_a_bug_while_reading_the_inputs_gives_a_page_with_a_way_back(monkeypatch, caplog):
    install_demo(monkeypatch, make_demo())
    at = run_app()
    at.sidebar.radio(key="mode").set_value(MODE_CUSTOM)

    def broken_form():
        raise TypeError("a bug in the form code")

    monkeypatch.setattr(state, "custom_form", broken_form)
    with caplog.at_level("ERROR"):
        at.run()

    assert not at.exception
    assert texts(at.error) == ["Something unexpected went wrong while preparing this page. Please reload it, or try again in a moment."]
    assert [b.label for b in at.button if b.label == "Back to the demo"]  # custom mode: offer the demo
    assert FOOTER in captions(at)
    assert any("a bug in the form code" in str(r.exc_info[1]) for r in caplog.records if r.exc_info)


def test_a_bug_in_the_calculation_keeps_the_page(monkeypatch, caplog):
    install_demo(monkeypatch, make_demo())

    def broken(*args, **kwargs):
        raise KeyError("XOM")

    monkeypatch.setattr(risk, "portfolio_value", broken)
    with caplog.at_level("ERROR"):
        at = run_app()

    assert not at.exception
    assert texts(at.error) == ["Something unexpected went wrong while calculating the risk numbers. Please try again."]
    assert at.title[0].value == "SignalDesk" and FOOTER in captions(at)
    assert any("XOM" in str(r.exc_info[1]) for r in caplog.records if r.exc_info)


# --------------------------------------------------------------------------- #
# A bug while DRAWING keeps the rest of the page (title, key numbers, other tabs, footer)
# --------------------------------------------------------------------------- #

def test_a_bug_in_the_metrics_table_leaves_the_rest_of_the_page_alone(monkeypatch, caplog):
    install_demo(monkeypatch, make_demo())

    def broken(summary):
        raise KeyError("Note")

    monkeypatch.setattr(sections, "metrics_for_display", broken)
    with caplog.at_level("ERROR"):
        at = run_app()

    assert not at.exception  # no generic Streamlit error box
    assert at.title[0].value == "SignalDesk"
    assert len(at.metric) == 5  # the key numbers are still there
    assert FOOTER in captions(at)
    assert texts(tab(at, "Holdings").error) == [
        "Something unexpected went wrong while drawing the metrics table. The rest of the page is not affected."
    ]
    assert len(tab(at, "Holdings").dataframe) == 0
    assert len(charts(at)) == 4  # growth, drawdown, rolling volatility and the heatmap are all drawn
    assert len(tab(at, "Data").dataframe) == 1 and tab(at, "About").subheader  # the Data and About tabs too
    assert any("Note" in str(r.exc_info[1]) for r in caplog.records if r.exc_info)  # the cause is in the log


def test_a_bug_in_one_chart_leaves_the_other_tabs_alone(monkeypatch):
    install_demo(monkeypatch, make_demo())

    def broken(*args, **kwargs):
        raise TypeError("a chart bug")

    monkeypatch.setattr(sections, "correlation_chart", broken)
    at = run_app()

    assert not at.exception
    assert len(texts(tab(at, "Correlation").error)) == 1 and "correlation chart" in tab(at, "Correlation").error[0].value
    assert len(charts(tab(at, "Overview"))) == 1 and len(charts(tab(at, "Risk"))) == 2
    assert len(tab(at, "Holdings").dataframe) == 1 and len(at.metric) == 5 and FOOTER in captions(at)


def test_a_bug_in_the_key_numbers_leaves_the_tabs_alone(monkeypatch):
    install_demo(monkeypatch, make_demo())
    monkeypatch.setattr(sections, "render_kpis", lambda *args: (_ for _ in ()).throw(ValueError("kpi bug")))
    at = run_app()

    assert not at.exception
    assert len(at.metric) == 0
    assert any("key numbers" in message for message in texts(at.error))
    assert len(charts(at)) == 4 and FOOTER in captions(at)


def test_anything_else_that_breaks_the_page_still_gets_a_message_and_the_footer(monkeypatch, caplog):
    install_demo(monkeypatch, make_demo())

    def broken(*args, **kwargs):
        raise RuntimeError("the page code blew up")

    monkeypatch.setattr(sections, "render_title", broken)  # drawn by render_page itself, outside the per-part guards
    with caplog.at_level("ERROR"):
        at = run_app()

    assert not at.exception
    assert texts(at.error) == [
        "Something unexpected went wrong while drawing this page. Please reload it, or try again in a moment."
    ]
    assert FOOTER in captions(at)
    assert "blew up" not in at.error[0].value  # internals stay in the log
    assert any("blew up" in str(r.exc_info[1]) for r in caplog.records if r.exc_info)
    assert len(at.button) == 0  # demo mode: nothing to go back to


# --------------------------------------------------------------------------- #
# The entry point
# --------------------------------------------------------------------------- #

def test_importing_app_does_not_draw_anything():
    import app

    assert callable(app.main)
    assert callable(sections.render_page)
