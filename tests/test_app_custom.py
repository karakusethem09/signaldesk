"""Tests for the dashboard's CUSTOM (live data) mode. None of these tests use the network.

Custom mode is where a visitor types holdings, a benchmark and dates, and the page downloads prices.
That is the risky part of a public website, so the tests are strict about WHEN a download may
happen (only after "Run analysis", only for a request that passed the limits), WHAT happens when
it fails (a message, and a page that is still usable), and WHO pays for it (the rate limit).

``data.load_prices`` is replaced by a fake that serves made-up prices and records its calls
(see app_helpers.py); the ``app_env`` fixture additionally refuses every network attempt.

AppTest detail: a widget inside an ``st.form`` only counts when its value is set in the SAME run
that presses the form's button. So the helpers set the values first and press "Run analysis" last.
"""

import re
import threading
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from app_helpers import (
    BENCHMARK,
    DEMO_WEIGHTS,
    HOLDINGS,
    MODE_CUSTOM,
    MODE_DEMO,
    assert_chart_shows,
    captions,
    chart_data,
    chart_tickers,
    charts,
    display_values,
    expected_portfolio,
    expected_summary,
    holdings_state,
    install_loader,
    make_prices,
    metrics_table,
    new_app,
    open_custom,
    press_run,
    run_app,
    run_custom,
    tab,
    texts,
)

from signaldesk import data, limits, risk
from ui import inputs, live, sections, state

pytestmark = [
    pytest.mark.usefixtures("app_env"),
    pytest.mark.filterwarnings("ignore::signaldesk.risk.ShortSampleWarning"),
    pytest.mark.filterwarnings("ignore::signaldesk.risk.GapWarning"),
    pytest.mark.filterwarnings("ignore::signaldesk.risk.SkippedTickerWarning"),
]

TODAY = pd.Timestamp("2027-03-17")  # the pinned "today" (New York): far from the real date, so a stray date.today() shows
FOOTER = "Not investment advice. Data: Yahoo Finance, end-of-day."
LOADED_NOTE = (
    "Loaded just now through the data layer: a fresh Yahoo Finance download, or its one-day cache on this server."
)


@pytest.fixture
def pinned_today(monkeypatch):
    """Freeze data.market_today() at 2027-03-17, so the presets have known dates."""
    monkeypatch.setattr(data, "_today", lambda: TODAY)
    return TODAY.date()


def kpi_texts(at) -> dict:
    return {m.label: (m.value, m.delta) for m in at.metric}


def pct(value: float) -> str:
    return f"{value:,.1%}"


def page_is_usable(at) -> None:
    """After a failure: no crash, a way back to the demo, the sidebar and the footer are all there."""
    assert not at.exception
    assert at.title[0].value == "SignalDesk"
    assert [b.label for b in at.button if b.label == "Back to the demo"] == ["Back to the demo"]
    assert len(at.sidebar.form_submit_button) == 1  # the form is still there to try again
    assert FOOTER in captions(at)
    assert len(at.metric) == 0 and len(charts(at)) == 0


# --------------------------------------------------------------------------- #
# Nothing is downloaded before "Run analysis"
# --------------------------------------------------------------------------- #

def test_custom_mode_starts_with_a_prompt_and_downloads_nothing(monkeypatch):
    loader = install_loader(monkeypatch)
    at = open_custom()

    assert not at.exception
    assert at.sidebar.radio(key="mode").value == MODE_CUSTOM
    assert any("press Run analysis" in m and "Nothing is downloaded until you do" in m for m in texts(at.info))
    assert [b.label for b in at.button if b.label == "Back to the demo"]
    assert loader.calls == []


def test_changing_the_form_and_the_other_settings_never_downloads_anything(monkeypatch):
    loader = install_loader(monkeypatch)
    at = open_custom()
    # Edit everything in the form, run the script after each change ...
    at.session_state["holdings_editor"] = holdings_state([("AAPL", 60), ("MSFT", 40)])
    at.sidebar.text_input(key="benchmark").set_value("QQQ")
    at.sidebar.radio(key="custom_period").set_value("1Y")
    at.run()
    at.sidebar.radio(key="custom_period").set_value("Custom dates").run()
    at.sidebar.slider(key="window").set_value(30).run()
    at.sidebar.number_input(key="risk_free_rate").set_value(0.03).run()
    at.sidebar.toggle(key="show_holdings").set_value(True).run()
    assert not at.exception
    assert loader.calls == []  # ... and still nothing was asked of Yahoo

    press_run(at)
    assert len(loader.calls) == 1


def test_after_a_run_the_window_the_rate_and_the_holdings_switch_do_not_download_again(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom()
    assert len(loader.calls) == 1 and len(at.metric) == 5

    at.sidebar.slider(key="window").set_value(30).run()
    at.sidebar.number_input(key="risk_free_rate").set_value(0.03).run()
    at.sidebar.toggle(key="show_holdings").set_value(True).run()

    assert not at.exception
    assert len(loader.calls) == 1  # the prices stay in the session
    assert len(at.metric) == 5
    assert "Rolling volatility (30-day window)" in [s.value for s in tab(at, "Risk").subheader]
    assert "Sharpe uses a risk-free rate of 3.0%/yr." in captions(at)


# --------------------------------------------------------------------------- #
# A successful run
# --------------------------------------------------------------------------- #

def test_a_run_shows_the_page_for_the_typed_portfolio(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom(rows=[("AAPL", 60), ("msft ", 40)], benchmark=" spy")

    assert not at.exception and not at.error and not at.warning
    assert loader.calls[0][0] == ["AAPL", "MSFT", "SPY"]  # clean: upper case, holdings first, benchmark last
    weights = {"AAPL": 0.6, "MSFT": 0.4}
    value, bench = expected_portfolio(loader.frame, weights)
    expected = risk.risk_summary(pd.DataFrame({"Portfolio": value, "SPY": bench}), confidence=0.95, risk_free_rate=0.0)
    shown = kpi_texts(at)
    assert shown["Annualized return"] == (
        pct(expected.loc["Portfolio", "annualized_return"]),
        f"SPY: {pct(expected.loc['SPY', 'annualized_return'])}",
    )
    assert shown["Max drawdown (loss)"][0] == pct(expected.loc["Portfolio", "max_drawdown"])
    table = display_values(tab(at, "Overview").dataframe[0])
    assert list(table.index) == ["AAPL", "MSFT"] and table["Starting weight"].tolist() == ["60.0%", "40.0%"]
    assert list(metrics_table(at).index) == ["AAPL", "MSFT", "Portfolio", "SPY"]
    assert len(charts(at)) == 4


def test_default_form_is_the_demo_portfolio(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom()

    assert loader.calls[0][0] == HOLDINGS + [BENCHMARK]
    shown = display_values(tab(at, "Overview").dataframe[0])
    assert shown["Starting weight"].tolist() == ["20.0%", "20.0%", "15.0%", "15.0%", "15.0%", "15.0%"]
    expected = expected_summary(loader.frame)
    assert metrics_table(at)["Annualized return"].tolist() == expected["annualized_return"].tolist()


def test_the_as_of_line_uses_the_last_date_in_the_data_not_the_requested_end(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom()

    header = [m.value for m in at.markdown if m.value.startswith("**Live data")]
    assert len(header) == 1
    assert f"Live data, last close {loader.frame.index[-1]:%Y-%m-%d}" in header[0]
    assert f"{data.market_today():%Y-%m-%d}" not in header[0]
    assert "[GitHub](" in header[0]


def test_percent_weights_reach_portfolio_value_normalized_by_risk_py(monkeypatch):
    install_loader(monkeypatch)
    raw_seen, weights_seen = [], []
    real_normalize, real_value = risk.normalize_weights, risk.portfolio_value

    def spy_normalize(raw):
        raw_seen.append(dict(raw))
        return real_normalize(raw)

    def spy_value(prices, weights):
        weights_seen.append(weights)
        return real_value(prices, weights)

    monkeypatch.setattr(risk, "normalize_weights", spy_normalize)
    monkeypatch.setattr(risk, "portfolio_value", spy_value)

    at = open_custom()  # (the first page, the demo, calls portfolio_value too: forget that call)
    raw_seen.clear()
    weights_seen.clear()
    run_custom(rows=[("AAPL", 50), ("MSFT", 25), ("NVDA", 25)], at=at)
    # The page hands the typed PERCENTAGES to risk.normalize_weights (it never divides them itself) ...
    assert raw_seen == [{"AAPL": 50.0, "MSFT": 25.0, "NVDA": 25.0}]
    # ... and portfolio_value receives the fractions that came back.
    assert weights_seen == [{"AAPL": 0.5, "MSFT": 0.25, "NVDA": 0.25}]
    assert all(type(w) is float for w in weights_seen[0].values())

    # The percentages do not have to add up to 100: only the proportions count.
    raw_seen.clear()
    weights_seen.clear()
    run_custom(rows=[("AAPL", 10), ("MSFT", 30)], at=at)
    assert raw_seen == [{"AAPL": 10.0, "MSFT": 30.0}]
    assert weights_seen == [{"AAPL": 0.25, "MSFT": 0.75}]


def test_the_holdings_editor_is_a_text_column_and_a_bounded_number_column(monkeypatch):
    import json

    install_loader(monkeypatch)
    at = open_custom()
    assert len(at.sidebar.dataframe) == 1  # the editor is the only table in the sidebar
    editor = at.sidebar.dataframe[0]
    columns = json.loads(editor.proto.columns)
    ticker, weight = columns["Ticker"], columns["Weight (%)"]

    assert ticker["type_config"]["type"] == "text"
    assert weight["type_config"]["type"] == "number"
    assert weight["type_config"]["min_value"] == 0 and weight["type_config"]["max_value"] == 100
    assert weight["type_config"]["step"] in (0.5, 1)
    assert weight["default"] == 10
    assert editor.proto.editing_mode == 2  # dynamic: rows can be added and deleted
    # Prefilled with the demo portfolio in percent.
    assert editor.value["Ticker"].tolist() == HOLDINGS
    assert editor.value["Weight (%)"].tolist() == [20.0, 20.0, 15.0, 15.0, 15.0, 15.0]


# --------------------------------------------------------------------------- #
# Periods and dates
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "label, first_day",
    [("1Y", date(2026, 3, 17)), ("3Y", date(2024, 3, 17)), ("5Y", date(2022, 3, 17)), ("10Y", date(2017, 3, 17))],
)
def test_presets_start_where_earliest_allowed_start_says_and_end_today(monkeypatch, pinned_today, label, first_day):
    loader = install_loader(monkeypatch)
    at = run_custom(period=label)

    assert not at.exception and not at.warning and not at.error  # even 10Y passes the public site's limit
    _tickers, start, end_exclusive = loader.calls[0]
    assert start == first_day
    assert end_exclusive == date(2027, 3, 18)  # "today" is included in the picker; load_prices excludes its end
    assert pd.Timestamp(start) == limits.earliest_allowed_start(end_exclusive, int(label.removesuffix("Y")))


def test_the_ten_year_preset_passes_check_request_with_the_public_limits(pinned_today):
    end_exclusive = state.inputs.end_exclusive_from(pinned_today)
    start = state.inputs.preset_start(end_exclusive, "10Y")
    assert start == pd.Timestamp("2017-03-17")
    limits.check_request(HOLDINGS + [BENCHMARK], start, end_exclusive, limits.PUBLIC_LIMITS)  # does not raise
    with pytest.raises(limits.LimitError):  # one day earlier would be refused
        limits.check_request(HOLDINGS + [BENCHMARK], start - pd.Timedelta(days=1), end_exclusive, limits.PUBLIC_LIMITS)


def test_the_end_date_sent_to_load_prices_is_one_day_after_the_chosen_end(monkeypatch):
    loader = install_loader(monkeypatch)
    run_custom(period="Custom dates", dates=(date(2025, 3, 3), date(2025, 9, 30)))

    _tickers, start, end_exclusive = loader.calls[0]
    assert start == date(2025, 3, 3)
    assert end_exclusive == date(2025, 10, 1)  # the picker's end is INCLUDED, load_prices' end is not


def test_load_prices_is_called_without_a_cache_dir_so_the_default_folder_is_pruned(monkeypatch):
    loader = install_loader(monkeypatch)
    run_custom()
    assert loader.options == [{"cache_dir": None, "use_cache": True}]


def test_find_missing_days_is_asked_about_the_window_load_prices_loaded(monkeypatch):
    loader = install_loader(monkeypatch)
    asked = []
    real_function = data.find_missing_days

    def spy(prices, start=None, end=None):
        asked.append((start, end))
        return real_function(prices, start, end)

    monkeypatch.setattr(data, "find_missing_days", spy)
    at = open_custom()
    asked.clear()  # (the first page, the demo, asked too)
    run_custom(period="Custom dates", dates=(date(2025, 1, 6), date(2026, 2, 20)), at=at)

    assert not at.exception
    _tickers, start, end_exclusive = loader.calls[0]
    assert asked == [(start, end_exclusive)]  # the very same start and (exclusive) end
    assert end_exclusive == date(2026, 2, 21)


def test_start_after_end_is_refused_without_a_download(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom(period="Custom dates", dates=(date(2025, 6, 2), date(2025, 5, 1)))

    assert any("start date must be before the end date" in m for m in texts(at.warning))
    assert loader.calls == []
    page_is_usable(at)


# --------------------------------------------------------------------------- #
# The public-site limits (a refused request never reaches Yahoo)
# --------------------------------------------------------------------------- #

def test_eleven_tickers_are_refused_with_the_limit_text_and_nothing_is_downloaded(monkeypatch):
    loader = install_loader(monkeypatch)
    symbols = [f"T{n}" for n in range(1, 11)]  # ten holdings; with SPY that is 11 tickers
    at = run_custom(rows=[(symbol, 10) for symbol in symbols])

    with pytest.raises(limits.LimitError) as expected:
        limits.check_request(symbols + ["SPY"], date(2021, 10, 21), date(2026, 10, 22), limits.PUBLIC_LIMITS)
    assert texts(at.warning) == [str(expected.value)]  # shown exactly as the limit module wrote it
    assert "Up to 10 tickers per request" in at.warning[0].value and "(you asked for 11)" in at.warning[0].value
    assert loader.calls == []
    page_is_usable(at)


def test_ten_tickers_in_total_are_fine(monkeypatch):
    loader = install_loader(monkeypatch, frame=make_prices(columns=[f"T{n}" for n in range(1, 10)] + ["SPY"]))
    at = run_custom(rows=[(f"T{n}", 10) for n in range(1, 10)])  # nine holdings and SPY

    assert not at.warning and not at.error
    assert len(loader.calls[0][0]) == 10 and len(at.metric) == 5


def test_an_eleven_year_range_is_refused_with_the_limit_text_and_nothing_is_downloaded(monkeypatch, pinned_today):
    loader = install_loader(monkeypatch)
    start = date(2016, 3, 17)  # 11 years back
    at = run_custom(period="Custom dates", dates=(start, pinned_today))

    with pytest.raises(limits.LimitError) as expected:
        limits.check_request(HOLDINGS + [BENCHMARK], start, pinned_today + timedelta(days=1), limits.PUBLIC_LIMITS)
    assert texts(at.warning) == [str(expected.value)]
    assert "Up to 10 years per request" in at.warning[0].value and "2017-03-17" in at.warning[0].value
    assert loader.calls == []
    page_is_usable(at)


def test_ten_years_to_the_day_is_accepted(monkeypatch, pinned_today):
    loader = install_loader(monkeypatch)
    at = run_custom(period="Custom dates", dates=(date(2017, 3, 17), pinned_today))
    assert not at.warning and len(loader.calls) == 1
    at = run_custom(period="Custom dates", dates=(date(2017, 3, 16), pinned_today))  # one day more is refused
    assert len(at.warning) == 1 and "Up to 10 years" in at.warning[0].value and len(loader.calls) == 1


def test_a_bad_symbol_is_refused_without_a_download(monkeypatch):
    loader = install_loader(monkeypatch)
    for bad in ("AA PL", "AAPL!!", "A" * 40, "<script>"):
        at = run_custom(rows=[(bad, 50), ("MSFT", 50)])
        assert len(at.warning) == 1 and "is not a valid ticker symbol" in at.warning[0].value, bad
        assert len(at.warning[0].value) < 250  # a 40-character symbol is cut short in the message
        page_is_usable(at)
    assert loader.calls == []


def test_a_bad_benchmark_symbol_is_refused_without_a_download(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom(benchmark="S&P 500")
    assert any("is not a valid ticker symbol" in m for m in texts(at.warning))
    assert loader.calls == []


def test_on_your_own_computer_there_are_no_limits_but_symbols_are_still_checked(monkeypatch, local_mode):
    symbols = [f"T{n}" for n in range(1, 13)]  # twelve holdings
    loader = install_loader(monkeypatch, frame=make_prices(columns=symbols + ["SPY"]))
    at = run_custom(rows=[(symbol, 10) for symbol in symbols], period="Custom dates", dates=(date(2005, 1, 3), date(2026, 2, 20)))

    assert not at.warning and not at.error  # 13 tickers and 21 years: fine in local mode
    assert len(loader.calls) == 1 and len(loader.calls[0][0]) == 13

    at = run_custom(rows=[("AAPL!!", 50), ("MSFT", 50)])  # check_request skips this in local mode, we do not
    assert any("is not a valid ticker symbol" in m for m in texts(at.warning))
    assert len(loader.calls) == 1


# --------------------------------------------------------------------------- #
# What was typed in the holdings table
# --------------------------------------------------------------------------- #

def spy_on_normalize_weights(monkeypatch) -> list:
    calls = []
    real = risk.normalize_weights
    monkeypatch.setattr(risk, "normalize_weights", lambda raw: (calls.append(raw), real(raw))[1])
    return calls


def test_an_empty_weight_cell_gets_a_plain_message(monkeypatch):
    loader = install_loader(monkeypatch)
    calls = spy_on_normalize_weights(monkeypatch)
    at = run_custom(rows=[("AAPL", 50), ("MSFT", None)])

    assert texts(at.warning) == ["Fill in a weight for every ticker (or delete the row)."]
    for text in texts(at.warning) + texts(at.error):  # never Python's own words
        assert "nan" not in text.lower() and "float" not in text and "Error" not in text
    assert loader.calls == [] and calls == []  # it never even reached risk.normalize_weights
    page_is_usable(at)


def test_a_weight_cell_holding_nan_gets_the_same_plain_message(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom(rows=[("AAPL", 50), ("MSFT", float("nan"))])
    assert texts(at.warning) == ["Fill in a weight for every ticker (or delete the row)."]
    assert loader.calls == []


def test_an_empty_ticker_cell_gets_a_plain_message(monkeypatch):
    loader = install_loader(monkeypatch)
    for blank in (None, "", "   "):
        at = run_custom(rows=[("AAPL", 50), (blank, 10)])
        assert texts(at.warning) == ["Fill in a ticker for every weight (or delete the row)."], blank
    assert loader.calls == []


def test_untouched_empty_rows_are_ignored(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom(rows=[("AAPL", 60), (None, None), ("MSFT", 40), ("", None)])
    assert not at.warning and not at.error
    assert loader.calls[0][0] == ["AAPL", "MSFT", "SPY"]


@pytest.mark.parametrize("weight", [1e308, float("inf"), -5, 100.5, 1e30])
def test_absurd_weights_never_reach_risk_py(monkeypatch, weight):
    loader = install_loader(monkeypatch)
    calls = spy_on_normalize_weights(monkeypatch)
    at = run_custom(rows=[("AAPL", weight), ("MSFT", 10)])

    assert texts(at.warning) == ["The weight of AAPL must be between 0 and 100 (percent)."]
    assert loader.calls == [] and calls == []  # no OverflowError, no "np.float64(...)" text
    page_is_usable(at)


def test_all_zero_weights_get_risk_pys_plain_message(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom(rows=[("AAPL", 0), ("MSFT", 0)])
    assert texts(at.warning) == ["All the weights are zero. At least one holding needs a weight above 0."]
    assert loader.calls == []


def test_an_empty_table_asks_for_a_holding(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom(rows=[])
    assert texts(at.warning) == ["Add at least one holding: a ticker and a weight."]
    assert loader.calls == []
    page_is_usable(at)


def test_a_repeated_ticker_is_refused_instead_of_silently_overwritten(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom(rows=[("AAPL", 30), (" aapl ", 20), ("MSFT", 50)])
    assert texts(at.warning) == ["AAPL is listed more than once. Keep one row per ticker."]
    assert loader.calls == []


def test_a_benchmark_that_is_also_a_holding_is_refused(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom(rows=[("AAPL", 50), ("SPY", 50)])
    assert texts(at.warning) == ["The benchmark SPY is also one of your holdings. Pick a different benchmark."]
    assert loader.calls == []


@pytest.mark.parametrize("benchmark, message", [("", "Enter a benchmark ticker"), ("SPY, QQQ", "Enter just one benchmark ticker.")])
def test_the_benchmark_must_be_exactly_one_ticker(monkeypatch, benchmark, message):
    loader = install_loader(monkeypatch)
    at = run_custom(benchmark=benchmark)
    assert len(at.warning) == 1 and message in at.warning[0].value
    assert loader.calls == []


def test_the_same_extra_spacing_and_case_gives_the_same_tickers(monkeypatch):
    loader = install_loader(monkeypatch)
    run_custom(rows=[(" tsla ", 50), ("aapl", 50)])
    assert loader.calls[-1][0] == ["TSLA", "AAPL", "SPY"]  # upper case, in the order typed


# --------------------------------------------------------------------------- #
# When the download fails the page stays usable
# --------------------------------------------------------------------------- #

def test_provider_unavailable_shows_its_own_message_as_it_is_and_the_page_stays_usable(monkeypatch):
    message = (
        "No price data came back for AAPL, MSFT between 2025-03-01 and 2025-03-03. The data provider may be "
        "rate-limiting or unavailable; the symbols may not exist; or those dates may hold no trading days."
    )
    loader = install_loader(monkeypatch, error=data.ProviderUnavailableError(message))
    at = run_custom()

    assert texts(at.error) == [message]  # not "Yahoo is down", not prefixed: the message names every cause
    assert len(loader.calls) == 1
    page_is_usable(at)
    # ... and from there the visitor can go back to the demo with one click.
    at.button(key="back_to_demo").click().run()
    assert not at.exception
    assert at.sidebar.radio(key="mode").value == MODE_DEMO
    assert len(at.metric) == 5 and not at.error


def test_a_missing_ticker_shows_the_price_data_message(monkeypatch):
    install_loader(monkeypatch)
    at = run_custom(rows=[("AAPL", 50), ("zzzz", 50)])

    assert len(at.error) == 1
    assert at.error[0].value == "Could not load prices: No price data returned for: ZZZZ."
    assert len(charts(at)) == 0 and len(at.dataframe) == 1  # only the editor: no result tables
    page_is_usable(at)


def test_a_plain_value_error_from_the_data_layer_is_shown(monkeypatch):
    install_loader(monkeypatch, error=ValueError("start 2026-10-20 must be earlier than end 2026-10-20."))
    at = run_custom()
    assert texts(at.error) == ["Could not load prices (bad input): start 2026-10-20 must be earlier than end 2026-10-20."]
    page_is_usable(at)


def test_an_unexpected_error_does_not_blank_the_page(monkeypatch, caplog):
    install_loader(monkeypatch, error=RuntimeError("a bug deep inside a library"))
    with caplog.at_level("ERROR"):
        at = run_custom()
    assert texts(at.error) == ["Something unexpected went wrong while loading prices. Please try again, or use the demo."]
    assert "a bug deep inside a library" not in at.error[0].value  # internals stay in the server log
    assert any("a bug deep inside a library" in str(r.exc_info[1]) for r in caplog.records if r.exc_info)
    page_is_usable(at)


def test_a_failed_run_can_be_followed_by_a_good_one(monkeypatch):
    loader = install_loader(monkeypatch)
    at = run_custom(rows=[("AAPL", 50), ("zzzz", 50)])
    assert len(at.error) == 1
    at.session_state["holdings_editor"] = holdings_state([("AAPL", 50), ("MSFT", 50)])
    press_run(at)

    assert not at.error and not at.warning
    assert len(at.metric) == 5
    assert len(loader.calls) == 2


def test_prices_that_load_but_cannot_be_analysed_show_an_error_and_a_way_back(monkeypatch):
    def tweak(frame):
        frame.loc[frame.index[:150], "XOM"] = np.nan
        frame.loc[frame.index[150:], "JNJ"] = np.nan  # no date on which XOM and JNJ both have a price

    loader = install_loader(monkeypatch, tweak=tweak)
    at = run_custom()

    assert len(loader.calls) == 1  # the download itself worked
    assert len(at.error) == 1 and "there is no date on which every holding has a price" in at.error[0].value
    assert "Could not calculate the risk numbers" in at.error[0].value
    assert len(at.metric) == 0 and len(charts(at)) == 0
    assert [b.label for b in at.button if b.label == "Back to the demo"]
    assert FOOTER in captions(at)
    at.button(key="back_to_demo").click().run()
    assert at.sidebar.radio(key="mode").value == MODE_DEMO and len(at.metric) == 5


def test_a_drawing_bug_after_a_good_run_keeps_the_footer_and_offers_the_way_back(monkeypatch, caplog):
    install_loader(monkeypatch)
    at = run_custom()
    assert len(at.metric) == 5

    def broken(*args, **kwargs):
        raise RuntimeError("the page code blew up")

    monkeypatch.setattr(sections, "render_page", broken)
    with caplog.at_level("ERROR"):
        at.run()

    assert not at.exception
    assert texts(at.error) == [
        "Something unexpected went wrong while drawing this page. Please reload it, or try again in a moment."
    ]
    assert [b.label for b in at.button if b.label == "Back to the demo"]
    assert FOOTER in captions(at)
    assert any("blew up" in str(r.exc_info[1]) for r in caplog.records if r.exc_info)


def test_the_data_tab_says_whether_the_prices_were_downloaded_or_reused(monkeypatch):
    install_loader(monkeypatch)
    first = run_custom()
    assert LOADED_NOTE in [m.value for m in tab(first, "Data").markdown]
    assert "Yahoo Finance" in " ".join(m.value for m in tab(first, "Data").markdown if m.value.startswith("**Source"))

    second = run_custom()  # another visitor, the same request, a moment later
    notes = [m.value for m in tab(second, "Data").markdown]
    assert "Reused from this server's short-term memory (at most 6 hours old): nothing was downloaded." in notes
    assert LOADED_NOTE not in notes


def test_the_result_of_one_run_does_not_outlive_a_visit_to_the_demo(monkeypatch):
    install_loader(monkeypatch)
    at = run_custom()
    assert len(at.metric) == 5
    at.sidebar.radio(key="mode").set_value(MODE_DEMO).run()
    assert "Demo data as of" in at.markdown[0].value
    at.sidebar.radio(key="mode").set_value(MODE_CUSTOM).run()
    assert len(at.metric) == 0  # the form is back to its defaults, so the old result must not reappear
    assert any("press Run analysis" in m for m in texts(at.info))


# --------------------------------------------------------------------------- #
# The same analysis as the demo: alignment, awkward data, GME
# --------------------------------------------------------------------------- #

def test_in_custom_mode_too_the_benchmark_is_measured_over_the_portfolios_dates(monkeypatch):
    def late_starter(frame):
        frame.loc[frame.index[:100], "XOM"] = np.nan

    loader = install_loader(monkeypatch, tweak=late_starter)
    at = run_custom()

    value, aligned = expected_portfolio(loader.frame)
    expected = risk.risk_summary(pd.DataFrame({"Portfolio": value, "SPY": aligned}), 0.95, 0.0)
    assert len(value) == 200
    assert kpi_texts(at)["Annualized return"] == (
        pct(expected.loc["Portfolio", "annualized_return"]),
        f"SPY: {pct(expected.loc['SPY', 'annualized_return'])}",
    )
    assert any(m.startswith("Short sample. portfolio_value: the portfolio starts on") for m in texts(at.warning))
    assert (metrics_table(at)["Daily returns (n)"] == 199).all()


def test_a_gme_like_ticker_still_gets_its_rolling_line_and_heatmap_cell(monkeypatch):
    def gme(frame):
        frame["GME"] = 100.0 * np.where(np.arange(len(frame)) % 2 == 0, 1.0, 2.0)  # +100%, -50%, +100%, ...

    loader = install_loader(monkeypatch, tweak=gme)
    at = run_custom(rows=[("AAPL", 40), ("MSFT", 40), ("GME", 20)])
    at.sidebar.toggle(key="show_holdings").set_value(True).run()

    assert not at.exception and not at.error
    assert not any("could not be drawn" in m for m in texts(at.warning))
    assert "GME" in chart_tickers(charts(tab(at, "Risk"))[1])
    heatmap = charts(tab(at, "Correlation"))[0]
    assert chart_tickers(heatmap) == ["AAPL", "GME", "MSFT", "SPY"]
    value, _ = expected_portfolio(loader.frame, {"AAPL": 0.4, "MSFT": 0.4, "GME": 0.2})
    assert_chart_shows(
        heatmap, "Correlation", risk.return_correlation(loader.frame.loc[value.index, ["AAPL", "MSFT", "GME", "SPY"]]),
        rows="Ticker A", columns="Ticker B",
    )


# --------------------------------------------------------------------------- #
# The rate limit and the short-term memory
# --------------------------------------------------------------------------- #

@pytest.fixture
def two_loads_an_hour(monkeypatch):
    """The public limits, but with only 2 live analyses per hour, to keep the tests short."""
    monkeypatch.setattr(limits, "PUBLIC_LIMITS", limits.RequestLimits(10, 10, 2, 1))


def other_dates(n: int) -> dict:
    """A request that differs from every other one (so the short-term memory cannot answer it)."""
    return {"period": "Custom dates", "dates": (date(2025, 1, 6) + timedelta(days=n), date(2026, 2, 20))}


def test_the_rate_limit_triggers_after_the_budget_and_says_when_to_retry(monkeypatch, two_loads_an_hour):
    loader = install_loader(monkeypatch)
    at = run_custom(**other_dates(1))
    run_custom(**other_dates(2), at=at)
    assert len(loader.calls) == 2 and not at.warning

    at = run_custom(**other_dates(3), at=at)  # a third NEW request within the hour
    assert len(loader.calls) == 2  # refused before any download
    assert len(at.warning) == 1
    message = at.warning[0].value
    assert message.startswith("You have used your 2 live analyses for this hour.")
    assert "Please try again in about 1 hour" in message or "Please try again in about 59 minutes" in message
    assert "demo portfolio" in message
    page_is_usable(at)


def test_requests_answered_from_memory_do_not_use_up_the_budget(monkeypatch, two_loads_an_hour):
    loader = install_loader(monkeypatch)
    at = run_custom(**other_dates(1))
    for _ in range(6):  # the very same request, again and again
        at = run_custom(**other_dates(1), at=at)
        assert not at.warning and len(at.metric) == 5
    assert len(loader.calls) == 1  # one real download

    run_custom(**other_dates(2), at=at)  # the budget (2) still has room for one more new request
    assert len(loader.calls) == 2 and not at.warning
    at = run_custom(**other_dates(1), at=at)  # ... and a repeat is STILL served after the budget is used up
    assert not at.warning and len(at.metric) == 5
    assert len(loader.calls) == 2


def test_each_visitor_has_their_own_budget_and_they_share_the_memory(monkeypatch, two_loads_an_hour):
    loader = install_loader(monkeypatch)
    first = run_custom(**other_dates(1))
    run_custom(**other_dates(2), at=first)
    blocked = run_custom(**other_dates(3), at=first)
    assert len(blocked.warning) == 1

    second = new_app().run()  # another browser session
    second = run_custom(**other_dates(3), at=second)
    assert not second.warning and len(second.metric) == 5  # not blocked by the first visitor's use
    assert len(loader.calls) == 3

    second = run_custom(**other_dates(1), at=second)  # the first visitor's data comes out of the shared memory
    assert not second.warning and len(loader.calls) == 3


@pytest.fixture
def site_cap_of_two(monkeypatch):
    """The public limits, but only 2 live analyses per hour for the whole site (5 per visitor)."""
    monkeypatch.setattr(limits, "PUBLIC_LIMITS", limits.RequestLimits(10, 10, 5, 1, 2))


def test_the_whole_site_has_an_hourly_cap_that_a_new_session_does_not_reset(monkeypatch, site_cap_of_two):
    loader = install_loader(monkeypatch)
    run_custom(**other_dates(1))  # visitor 1
    run_custom(**other_dates(2))  # visitor 2: a brand-new browser session with a fresh personal budget
    third = run_custom(**other_dates(3))  # visitor 3: a fresh session too, but the SITE is full

    assert len(loader.calls) == 2  # nothing went to Yahoo for visitor 3
    assert len(third.warning) == 1
    message = third.warning[0].value
    assert message.startswith("The site has reached its hourly limit of live analyses. Please try again in ")
    assert re.search(r"Please try again in (about 1 hour|about \d+ minutes); the demo still works\.$", message)
    assert "You have used your" not in message  # not the personal-limit message
    page_is_usable(third)
    # The demo still works from there.
    third.button(key="back_to_demo").click().run()
    assert third.sidebar.radio(key="mode").value == MODE_DEMO and len(third.metric) == 5


def test_a_request_somebody_else_already_loaded_is_still_served_when_the_site_is_full(monkeypatch, site_cap_of_two):
    loader = install_loader(monkeypatch)
    run_custom(**other_dates(1))
    run_custom(**other_dates(2))
    blocked = run_custom(**other_dates(3))
    assert len(blocked.warning) == 1

    again = run_custom(**other_dates(1))  # the same request as visitor 1: from the shared memory, free
    assert not again.warning and len(again.metric) == 5
    assert len(loader.calls) == 2


def test_a_busy_server_does_not_hide_the_site_limit_message(monkeypatch, site_cap_of_two):
    install_loader(monkeypatch)
    monkeypatch.setattr(live, "GATE_TIMEOUT_SECONDS", 0.05)
    run_custom(**other_dates(1))
    run_custom(**other_dates(2))  # the site's two loads are spent
    gate = state.shared_download_gate(limits.PUBLIC_LIMITS.max_concurrent_downloads)
    with gate.slot(timeout=1):  # another download is running right now
        third = run_custom(**other_dates(3))

    assert len(third.warning) == 1
    assert third.warning[0].value.startswith("The site has reached its hourly limit of live analyses.")
    assert limits.BUSY_MESSAGE not in texts(third.warning)  # the limit, not "busy"


def test_a_busy_server_does_not_hide_the_personal_limit_message(monkeypatch):
    monkeypatch.setattr(limits, "PUBLIC_LIMITS", limits.RequestLimits(10, 10, 1, 1, 30))
    install_loader(monkeypatch)
    monkeypatch.setattr(live, "GATE_TIMEOUT_SECONDS", 0.05)
    at = run_custom(**other_dates(1))
    gate = state.shared_download_gate(1)
    with gate.slot(timeout=1):
        at = run_custom(**other_dates(2), at=at)

    assert len(at.warning) == 1 and at.warning[0].value.startswith("You have used your 1 live analyses")
    assert limits.BUSY_MESSAGE not in texts(at.warning)


def test_the_personal_limit_message_is_not_the_site_message(monkeypatch):
    monkeypatch.setattr(limits, "PUBLIC_LIMITS", limits.RequestLimits(10, 10, 1, 1, 30))
    install_loader(monkeypatch)
    at = run_custom(**other_dates(1))
    at = run_custom(**other_dates(2), at=at)
    assert at.warning[0].value.startswith("You have used your 1 live analyses for this hour.")
    assert "The site has reached" not in at.warning[0].value


def test_on_your_own_computer_the_site_limiter_has_no_cap(local_mode):
    request_limits = limits.active_limits()
    assert request_limits.max_site_live_loads_per_hour is None
    site = state.shared_site_rate_limiter(request_limits.max_site_live_loads_per_hour)
    assert site.remaining(live.SITE_KEY) is None and site.retry_after(live.SITE_KEY) == 0.0
    assert state.shared_site_rate_limiter(None) is site


def test_the_site_limiter_is_one_shared_object_and_not_the_visitor_limiter():
    site = state.shared_site_rate_limiter(30)
    assert site is state.shared_site_rate_limiter(30)
    assert site is not state.shared_rate_limiter(30)  # same size, still a separate counter
    assert live.SITE_KEY == "__site__"


def test_the_download_is_wrapped_in_a_gate_slot_with_a_20_second_timeout(monkeypatch):
    from contextlib import contextmanager

    events = []
    real_slot = limits.DownloadGate.slot

    @contextmanager
    def recording_slot(self, timeout=5.0):
        events.append(("enter", timeout))
        with real_slot(self, timeout=timeout):
            yield
        events.append(("exit",))

    loader = install_loader(monkeypatch)
    monkeypatch.setattr(limits.DownloadGate, "slot", recording_slot)
    monkeypatch.setattr(data, "load_prices", lambda *a, **k: (events.append(("load",)), loader(*a, **k))[1])
    run_custom()

    assert events == [("enter", 20), ("load",), ("exit",)]  # the download happens INSIDE the slot


def test_a_busy_server_gives_a_friendly_message_and_uses_no_budget(monkeypatch):
    loader = install_loader(monkeypatch)
    monkeypatch.setattr(live, "GATE_TIMEOUT_SECONDS", 0.05)
    gate = state.shared_download_gate(limits.PUBLIC_LIMITS.max_concurrent_downloads)
    limiter = state.shared_rate_limiter(limits.PUBLIC_LIMITS.max_live_loads_per_hour)
    with gate.slot(timeout=1):  # another visitor's download is running
        at = run_custom()

    assert texts(at.warning) == [limits.BUSY_MESSAGE]
    assert loader.calls == []
    assert limiter.key_count == 0  # nothing was counted against anybody
    page_is_usable(at)
    at = press_run(at)  # the slot is free again
    assert len(at.metric) == 5 and len(loader.calls) == 1


def test_the_gate_and_the_limiter_are_shared_by_every_visitor():
    assert state.shared_download_gate(1) is state.shared_download_gate(1)
    assert state.shared_rate_limiter(10) is state.shared_rate_limiter(10)
    assert state.shared_price_cache() is state.shared_price_cache()


def test_two_visitors_one_after_the_other_each_get_their_own_page(monkeypatch):
    """Two browser sessions share the server's gate, limiters and memory, but never each other's page.

    (Two AppTest runs in two threads at the same time are not safe inside AppTest itself, so the
    visits are made one after the other. The threads are tested on the shared objects directly, below.)
    """
    loader = install_loader(monkeypatch)
    alice = run_custom(rows=[("AAPL", 100)])
    bob = run_custom(rows=[("MSFT", 100)])

    assert list(metrics_table(alice).index) == ["AAPL", "Portfolio", "SPY"]
    assert list(metrics_table(bob).index) == ["MSFT", "Portfolio", "SPY"]
    alice.run()  # alice's session is untouched by what bob did
    assert list(metrics_table(alice).index) == ["AAPL", "Portfolio", "SPY"]
    assert not alice.exception and not bob.exception
    assert len(loader.calls) == 2


def test_the_shared_gate_serializes_downloads_from_many_threads_and_each_gets_its_own_data(monkeypatch):
    """Visitors are threads of one process. Use the very objects the page shares, without AppTest.

    The fake download waits (up to a moment) for a second download to start. Through the gate no
    second download can start, so the wait simply times out; without the gate the downloads overlap.
    """
    frame = make_prices(columns=["AAPL", "MSFT", "NVDA", "JPM", "SPY"])
    state_lock = threading.Lock()
    counts = {"active": 0, "most_at_once": 0}
    overlapped = threading.Event()

    def slow_download(tickers, start, end, cache_dir=None, use_cache=True):
        with state_lock:
            counts["active"] += 1
            counts["most_at_once"] = max(counts["most_at_once"], counts["active"])
            if counts["active"] >= 2:
                overlapped.set()
        overlapped.wait(timeout=0.1)
        with state_lock:
            counts["active"] -= 1
        return frame[list(tickers)].copy()

    monkeypatch.setattr(data, "load_prices", slow_download)
    request_limits = limits.PUBLIC_LIMITS
    gate = state.shared_download_gate(request_limits.max_concurrent_downloads)
    limiter = state.shared_rate_limiter(request_limits.max_live_loads_per_hour)
    site_limiter = state.shared_site_rate_limiter(request_limits.max_site_live_loads_per_hour)
    cache = state.shared_price_cache()
    assert gate is state.shared_download_gate(1) and cache is state.shared_price_cache()  # the same objects everywhere

    wanted = {"a": "AAPL", "b": "MSFT", "c": "NVDA", "d": "JPM"}
    got, errors = {}, []
    start_together = threading.Barrier(len(wanted))

    def visit(name, ticker):
        request = inputs.LiveRequest(
            weights={ticker: 1.0}, benchmark="SPY", tickers=[ticker, "SPY"], start=date(2025, 1, 6), end_exclusive=date(2025, 12, 31)
        )
        start_together.wait()
        try:
            got[name] = live.load_live_prices(
                request, visitor=f"visitor-{name}", cache=cache, limiter=limiter, site_limiter=site_limiter, gate=gate
            )
        except Exception as error:  # pragma: no cover - would fail the asserts below
            errors.append(error)

    threads = [threading.Thread(target=visit, args=(name, ticker)) for name, ticker in wanted.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors
    assert counts["most_at_once"] == 1  # the gate let one download run at a time
    for name, ticker in wanted.items():
        assert list(got[name].prices.columns) == [ticker, "SPY"]  # each visitor got their own tickers, nothing else
        assert got[name].from_cache is False
    assert site_limiter.remaining(live.SITE_KEY) == request_limits.max_site_live_loads_per_hour - 4


def test_without_the_gate_the_same_downloads_would_overlap(monkeypatch):
    """The check above only means something if overlapping really happens when nothing stops it."""
    frame = make_prices(columns=["AAPL", "MSFT", "SPY"])
    overlapped = threading.Event()
    active = []
    state_lock = threading.Lock()

    def slow_download(tickers, start, end, cache_dir=None, use_cache=True):
        with state_lock:
            active.append(1)
            if len(active) >= 2:
                overlapped.set()
        overlapped.wait(timeout=2)
        return frame[list(tickers)].copy()

    monkeypatch.setattr(data, "load_prices", slow_download)
    no_gate = limits.DownloadGate(None)
    cache = live.PriceCache()
    free = limits.RateLimiter(None, 3600)
    start_together = threading.Barrier(2)

    def visit(ticker):
        request = inputs.LiveRequest(
            weights={ticker: 1.0}, benchmark="SPY", tickers=[ticker, "SPY"], start=date(2025, 1, 6), end_exclusive=date(2025, 12, 31)
        )
        start_together.wait()
        live.load_live_prices(request, visitor=ticker, cache=cache, limiter=free, site_limiter=free, gate=no_gate)

    threads = [threading.Thread(target=visit, args=(ticker,)) for ticker in ("AAPL", "MSFT")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert overlapped.is_set()
