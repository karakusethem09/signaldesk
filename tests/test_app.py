"""Tests for app.py (the Streamlit dashboard). None of these tests use the network.

How it works:
    - ``streamlit.testing.v1.AppTest`` runs app.py inside this process, with no browser and
      no server, and lets us read the elements it drew (title, tables, charts, messages)
      and change the sidebar widgets.
    - app.py calls ``data.load_prices(...)``. Here we replace that ONE function (with
      ``monkeypatch``) by a fake that returns made-up prices, so nothing is downloaded.
      The fake never reads or writes the cache; we also point the default cache folder at
      a temporary folder, in case something touches it by mistake.

The risk numbers themselves are tested in test_risk.py. Here we check what the USER SEES:
the text in every table cell, the data behind every chart, the warnings and notes, and that
the page survives awkward data without crashing. (A test that only says "something was
drawn" would pass even if a number were shown with the wrong sign or unit.)
"""

import ast
import json
import re
import warnings
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pyarrow as pa  # installed with Streamlit; used to read the tables Streamlit sends out
import pytest
from streamlit.testing.v1 import AppTest

import app
from signaldesk import data, risk

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
KNOWN_TICKERS = DEFAULT_TICKERS + ["TSLA"]  # TSLA is there so a typed extra ticker works


# --------------------------------------------------------------------------- #
# Helpers: fake prices and a way to run the app
# --------------------------------------------------------------------------- #

def make_prices(n_days: int = 300) -> pd.DataFrame:
    """Made-up prices: business days, one column per known ticker, fixed random seed."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2025-01-01", periods=n_days, name="Date")
    daily = rng.normal(loc=0.0004, scale=0.01, size=(n_days, len(KNOWN_TICKERS)))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + daily, axis=0), index=dates, columns=KNOWN_TICKERS)


class FakeLoader:
    """Stands in for data.load_prices. Records every call so tests can look at the arguments."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls: list[tuple] = []

    def __call__(self, tickers, start, end, cache_dir=None, use_cache=True):
        self.calls.append((list(tickers), start, end))
        unknown = [t for t in tickers if t not in self.frame.columns]
        if unknown:  # the real function raises PriceDataError for an unknown ticker too
            raise data.PriceDataError(f"No price data returned for: {', '.join(unknown)}.")
        return self.frame[list(tickers)].copy()


@pytest.fixture
def install_fake(monkeypatch, tmp_path):
    """Return a function that installs a fake load_prices (optionally tweaked) and returns it."""
    monkeypatch.setattr(data, "_default_cache_dir", lambda: tmp_path / "cache")

    def install(n_days: int = 300, tweak=None, frame: pd.DataFrame | None = None) -> FakeLoader:
        frame = make_prices(n_days) if frame is None else frame
        if tweak is not None:
            tweak(frame)  # changes the table in place
        loader = FakeLoader(frame)
        monkeypatch.setattr(data, "load_prices", loader)
        return loader

    return install


def run_app() -> AppTest:
    """Run app.py once with the default sidebar settings."""
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def run_app_with(install_fake, **kwargs) -> AppTest:
    """Install the fake prices, then run the app once."""
    install_fake(**kwargs)
    return run_app()


def charts(at: AppTest) -> list:
    """The Altair charts on the page (Streamlit draws them as 'vega_lite_chart' elements)."""
    return at.get("vega_lite_chart")


def texts(elements) -> list[str]:
    """The text of a list of message elements (warnings, infos, errors)."""
    return [element.value for element in elements]


def metrics_table(at: AppTest) -> pd.DataFrame:
    """The first table on the page is the metrics table (the raw numbers behind it)."""
    return at.dataframe[0].value


def shown_table(at: AppTest) -> pd.DataFrame:
    """The metrics table as the USER SEES it: the formatted text of every cell ("25.6%")."""
    arrow = at.dataframe[0].proto.arrow_data
    return pa.ipc.open_stream(arrow.styler.display_values).read_all().to_pandas()


def chart_spec(chart) -> dict:
    """The Vega-Lite description of a rendered chart (colours, axes, ...) as a dict."""
    return json.loads(chart.proto.spec)


def chart_data(chart) -> pd.DataFrame:
    """The rows a rendered chart plots. The spec only names a dataset; the rows sit beside it."""
    name = chart_spec(chart)["data"]["name"]
    for dataset in chart.proto.datasets:
        if dataset.name == name:
            return pa.ipc.open_stream(dataset.data.data).read_all().to_pandas()
    raise AssertionError(f"the chart has no dataset called {name!r}")


def chart_tickers(chart) -> list[str]:
    """Which tickers a rendered chart plots, read from its data (not from its sort list)."""
    rows = chart_data(chart)
    column = "Ticker" if "Ticker" in rows.columns else "Ticker A"
    return sorted(rows[column].unique())


def chart_table(chart, value: str, rows: str = "Date", columns: str = "Ticker") -> pd.DataFrame:
    """A chart's plotted rows as a wide table (rows x tickers), ready to compare with risk.py."""
    wide = chart_data(chart).pivot(index=rows, columns=columns, values=value)
    wide.columns = wide.columns.astype(object)
    if rows == "Date":
        wide.index = pd.DatetimeIndex(wide.index).astype("datetime64[ns]")
    else:
        wide.index = wide.index.astype(object)
    return wide.astype("float64")


# The only Vega expression the charts may contain: it picks the label colour in the heatmap
# (white text on dark squares). It sits in a colour condition, not in a data transform.
ALLOWED_EXPRESSIONS = {"abs(datum.Correlation) > 0.6"}


def assert_no_chart_math(spec: dict) -> None:
    """A chart spec must not calculate anything: no transform, no aggregate, no expression."""
    text = json.dumps({key: value for key, value in spec.items() if key != "datasets"})
    assert '"transform"' not in text, "the chart transforms its data (that is math: put it in risk.py)"
    assert '"aggregate"' not in text, "the chart aggregates its data (that is math: put it in risk.py)"
    assert '"expr"' not in text
    expressions = set(re.findall(r'"([^"]*datum\.[^"]*)"', text))
    assert expressions <= ALLOWED_EXPRESSIONS, f"unexpected chart expression(s): {expressions}"


def assert_chart_shows(chart, value: str, expected: pd.DataFrame, rows: str = "Date", columns: str = "Ticker"):
    """The chart plots exactly the numbers risk.py gave (same cells, no rescaling)."""
    got = chart_table(chart, value, rows, columns).reindex(index=expected.index, columns=expected.columns)
    pd.testing.assert_frame_equal(
        got, expected.astype("float64"), check_names=False, check_freq=False, rtol=1e-12, atol=0.0
    )


# --------------------------------------------------------------------------- #
# The page with normal data
# --------------------------------------------------------------------------- #

def test_page_renders_with_default_settings(install_fake):
    install_fake()
    at = run_app()

    assert not at.exception
    assert at.title[0].value == "SignalDesk"
    assert [s.value for s in at.subheader] == [
        "Key metrics",
        "Growth of $1",
        "Rolling volatility (60-day window)",
        "Correlation of daily returns",
    ]
    assert len(charts(at)) == 3  # growth, rolling volatility, heatmap
    assert not at.error
    assert not at.warning  # clean synthetic data: nothing to warn about
    assert not at.info
    # Charts must plot risk.py's numbers as they are: no Vega-Lite transform or aggregate.
    for chart in charts(at):
        assert_no_chart_math(chart_spec(chart))


def test_metrics_table_has_one_row_per_ticker_and_the_right_columns(install_fake):
    install_fake()
    table = metrics_table(run_app())

    assert list(table.index) == DEFAULT_TICKERS
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
    assert not table.drop(columns="Note").isna().any().any()
    assert (table["Note"] == "").all()


PERCENT_COLUMNS = {
    "Annualized return": "annualized_return",
    "Annualized volatility": "annualized_volatility",
    "Max drawdown (loss)": "max_drawdown",
    "1-day VaR 95% (loss)": "var_95",
}


def test_every_cell_the_user_sees_is_the_formatted_risk_summary_value(install_fake):
    loader = install_fake()
    shown = shown_table(run_app())
    expected = risk.risk_summary(loader.frame[DEFAULT_TICKERS], confidence=0.95, risk_free_rate=0.0)

    assert list(shown.index) == DEFAULT_TICKERS
    for ticker in DEFAULT_TICKERS:
        for label, column in PERCENT_COLUMNS.items():  # all four percent columns
            assert shown.loc[ticker, label] == app.format_percent(expected.loc[ticker, column]), (ticker, label)
        assert shown.loc[ticker, "Sharpe ratio"] == f"{expected.loc[ticker, 'sharpe_ratio']:.2f}", ticker
        assert shown.loc[ticker, "Daily returns (n)"] == str(int(expected.loc[ticker, "n_obs"])), ticker


def test_table_text_has_the_right_shape_whatever_format_percent_does(install_fake):
    """Independent of app.format_percent: percents end in %, losses carry no minus, Sharpe has 2 decimals."""
    shown = shown_table(run_app_with(install_fake))
    percent = re.compile(r"^-?\d{1,3}(,\d{3})*\.\d%$")
    loss = re.compile(r"^\d{1,3}(,\d{3})*\.\d%$")  # no minus sign: losses are positive numbers
    for label in PERCENT_COLUMNS:
        assert shown[label].map(lambda text: bool(percent.match(text))).all(), (label, shown[label].tolist())
    for label in ("Max drawdown (loss)", "1-day VaR 95% (loss)", "Annualized volatility"):
        assert shown[label].map(lambda text: bool(loss.match(text))).all(), (label, shown[label].tolist())
    assert shown["Sharpe ratio"].map(lambda text: bool(re.match(r"^-?\d+\.\d{2}$", text))).all()
    assert shown["Daily returns (n)"].map(lambda text: text.isdigit()).all()


def test_hand_checked_numbers_appear_exactly_as_expected(install_fake):
    # Prices 100, 110, 99, 108.9 give returns +10%, -10%, +10%. By hand:
    #   volatility = 0.11547 * sqrt(252) = 183.3%        Sharpe = 0.03333 / 0.11547 * sqrt(252) = 4.58
    #   max drawdown = 110 -> 99 = 10.0%                  VaR 95% = -(-0.1 + 0.1 * 0.2) = 8.0%
    #   3 returns stretched to a year give an absurd return, so the cap text is shown.
    prices = pd.DataFrame(
        {"AAPL": [100.0, 110.0, 99.0, 108.9]},
        index=pd.bdate_range("2025-01-01", periods=4, name="Date"),
    )
    install_fake(frame=prices)
    at = run_app()
    at.sidebar.multiselect(key="tickers").set_value(["AAPL"]).run()

    assert not at.exception
    row = shown_table(at).loc["AAPL"]
    assert row["Annualized volatility"] == "183.3%"
    assert row["Sharpe ratio"] == "4.58"
    assert row["Max drawdown (loss)"] == "10.0%"
    assert row["1-day VaR 95% (loss)"] == "8.0%"
    assert row["Annualized return"] == ">100,000%"
    assert row["Daily returns (n)"] == "3"


def test_table_numbers_come_straight_from_risk_summary(install_fake):
    loader = install_fake()
    table = metrics_table(run_app())

    expected = risk.risk_summary(loader.frame[DEFAULT_TICKERS], confidence=0.95, risk_free_rate=0.0)
    assert table["Annualized return"].tolist() == expected["annualized_return"].tolist()
    assert table["1-day VaR 95% (loss)"].tolist() == expected["var_95"].tolist()


def test_as_of_line_uses_the_last_date_in_the_data_not_the_requested_end(install_fake):
    loader = install_fake()
    at = run_app()

    last_in_data = f"{loader.frame.index[-1]:%Y-%m-%d}"
    as_of = [m.value for m in at.markdown if m.value.startswith("**Data as of")]
    assert len(as_of) == 1
    assert last_in_data in as_of[0]
    assert f"{date.today():%Y-%m-%d}" not in as_of[0]


def test_end_date_sent_to_load_prices_is_one_day_after_the_sidebar_end(install_fake):
    loader = install_fake()
    at = run_app()

    tickers, start, end = loader.calls[-1]
    assert tickers == DEFAULT_TICKERS
    assert start == at.sidebar.date_input(key="start_date").value
    assert end == at.sidebar.date_input(key="end_date").value + timedelta(days=1)


def test_data_quality_expander_lists_gaps_and_missing_days(install_fake):
    install_fake()
    at = run_app()

    assert len(at.expander) == 1
    assert at.expander[0].label == "Data quality"
    # Inside the expander: the find_gaps table and the missing-days table.
    assert len(at.dataframe) == 3
    labels = [m.value for m in at.expander[0].markdown]
    assert any("Possible gaps (most are exchange holidays)" in label for label in labels)


def test_changing_the_window_changes_the_rolling_volatility_title(install_fake):
    install_fake()
    at = run_app()
    at.sidebar.slider(key="window").set_value(30).run()

    assert not at.exception
    assert "Rolling volatility (30-day window)" in [s.value for s in at.subheader]


def test_risk_free_rate_has_sensible_bounds_and_the_page_says_which_rate_it_used(install_fake):
    install_fake()
    at = run_app()
    box = at.sidebar.number_input(key="risk_free_rate")

    assert (box.min, box.max) == (-0.05, 0.20)  # typing 4.0 instead of 0.04 is not possible
    assert "Sharpe uses a risk-free rate of 0.0%/yr." in [c.value for c in at.caption]
    at.sidebar.number_input(key="risk_free_rate").set_value(0.04).run()
    assert "Sharpe uses a risk-free rate of 4.0%/yr." in [c.value for c in at.caption]


def test_risk_free_rate_goes_to_risk_summary(install_fake):
    loader = install_fake()
    at = run_app()
    at.sidebar.number_input(key="risk_free_rate").set_value(0.04).run()

    assert not at.exception
    expected = risk.risk_summary(loader.frame[DEFAULT_TICKERS], confidence=0.95, risk_free_rate=0.04)
    assert metrics_table(at)["Sharpe ratio"].tolist() == expected["sharpe_ratio"].tolist()


def test_growth_chart_plots_the_cumulative_returns_from_risk_py(install_fake):
    loader = install_fake()
    growth_chart = charts(run_app())[0]
    assert_chart_shows(growth_chart, "Growth", risk.cumulative_returns(loader.frame[DEFAULT_TICKERS]))


def test_rolling_volatility_chart_plots_risk_py_numbers_for_the_chosen_window(install_fake):
    loader = install_fake()
    at = run_app()
    at.sidebar.slider(key="window").set_value(30).run()

    returns = risk.daily_returns(loader.frame[DEFAULT_TICKERS])
    rolling_chart = charts(at)[1]
    assert_chart_shows(rolling_chart, "Volatility", risk.annualized_volatility(returns, window=30))
    assert chart_data(rolling_chart)["Date"].nunique() == 299 - 29  # a 30-day window starts after 29 returns


def test_heatmap_plots_the_correlation_matrix_from_risk_py(install_fake):
    loader = install_fake()
    heatmap = charts(run_app())[2]
    expected = risk.correlation_matrix(risk.daily_returns(loader.frame[DEFAULT_TICKERS]))

    assert len(chart_data(heatmap)) == 25  # 5 x 5 squares
    assert_chart_shows(heatmap, "Correlation", expected, rows="Ticker A", columns="Ticker B")


def test_heatmap_colour_scale_is_red_blue_reversed_and_fixed_to_minus_one_one(install_fake):
    spec = chart_spec(charts(run_app_with(install_fake))[2])
    squares, numbers = spec["layer"]

    assert squares["mark"]["type"] == "rect"
    scale = squares["encoding"]["color"]["scale"]
    assert scale["scheme"] == "redblue"  # a two-sided scheme: the middle is neutral
    assert scale["reverse"] is True  # positive correlation warm, negative cool
    assert scale["domain"] == [-1, 1]  # fixed, whatever the data looks like
    assert numbers["encoding"]["text"]["format"] == ".2f"  # numbers printed with 2 decimals


def test_chart_helpers_contain_no_chart_side_math():
    prices = make_prices()[DEFAULT_TICKERS]
    returns = risk.daily_returns(prices)
    charts_built = [
        app.line_chart(risk.cumulative_returns(prices), "Growth", "Growth of $1"),
        app.line_chart(
            risk.annualized_volatility(returns, window=60), "Volatility", "Annualized volatility", y_axis_format=".0%"
        ),
        app.correlation_chart(risk.correlation_matrix(returns)),
    ]
    for chart in charts_built:
        assert_no_chart_math(chart.to_dict())


def test_the_chart_math_guard_catches_the_usual_tricks():
    rows = pd.DataFrame({"Date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "Value": [1.0, 2.0]})
    plain = alt.Chart(rows).mark_line().encode(x="Date:T", y="Value:Q")
    assert_no_chart_math(plain.to_dict())  # a plain chart passes

    tricks = {
        "calculate": plain.transform_calculate(Value="datum.Value * 100"),
        "window sum": plain.transform_window(Value="sum(Value)"),
        "aggregate shorthand": alt.Chart(rows).mark_line().encode(x="Date:T", y="mean(Value):Q"),
        "other expression": plain.encode(color=alt.condition("datum.Value > 1", alt.value("red"), alt.value("blue"))),
    }
    for name, chart in tricks.items():
        with pytest.raises(AssertionError):
            assert_no_chart_math(chart.to_dict())


def test_find_missing_days_is_asked_about_the_window_load_prices_loaded(install_fake, monkeypatch):
    loader = install_fake()
    asked = []
    real_function = data.find_missing_days

    def spy(prices, start=None, end=None):
        asked.append((start, end))
        return real_function(prices, start, end)

    monkeypatch.setattr(data, "find_missing_days", spy)
    at = run_app()

    _tickers, start, end_exclusive = loader.calls[-1]
    assert asked == [(start, end_exclusive)]  # the very same start and (exclusive) end
    assert end_exclusive == at.sidebar.date_input(key="end_date").value + timedelta(days=1)


# --------------------------------------------------------------------------- #
# Sidebar: tickers and dates
# --------------------------------------------------------------------------- #

def test_extra_tickers_are_upper_cased_and_repeats_are_removed(install_fake):
    loader = install_fake()
    at = run_app()
    at.sidebar.text_input(key="extra_tickers").set_value(" tsla , aapl,, ").run()

    assert not at.exception
    assert loader.calls[-1][0] == DEFAULT_TICKERS + ["TSLA"]
    assert list(metrics_table(at).index) == DEFAULT_TICKERS + ["TSLA"]


def test_unknown_ticker_shows_an_error_and_no_charts(install_fake):
    install_fake()
    at = run_app()
    at.sidebar.text_input(key="extra_tickers").set_value("zzzz").run()

    assert not at.exception
    assert len(at.error) == 1
    assert "Could not load prices" in at.error[0].value
    assert "ZZZZ" in at.error[0].value
    assert len(charts(at)) == 0
    assert len(at.dataframe) == 0


def test_start_after_end_shows_an_error(install_fake):
    loader = install_fake()
    at = run_app()
    today = date.today()
    at.sidebar.date_input(key="start_date").set_value(today)
    at.sidebar.date_input(key="end_date").set_value(today - timedelta(days=10))
    at.run()

    assert not at.exception
    assert any("start date must be before the end date" in message for message in texts(at.error))
    assert len(loader.calls) == 1  # only the first run loaded prices
    assert len(charts(at)) == 0


def test_no_tickers_selected_asks_for_one(install_fake):
    loader = install_fake()
    at = run_app()
    at.sidebar.multiselect(key="tickers").set_value([]).run()

    assert not at.exception
    assert any("at least one ticker" in message for message in texts(at.info))
    assert len(loader.calls) == 1


# --------------------------------------------------------------------------- #
# Awkward data
# --------------------------------------------------------------------------- #

def test_ticker_with_only_two_prices_is_skipped_with_a_warning(install_fake):
    def tweak(frame):
        frame.loc[frame.index[:-2], "JPM"] = np.nan  # JPM has just 2 prices, at the end

    install_fake(tweak=tweak)
    at = run_app()

    assert not at.exception
    skipped = [m for m in texts(at.warning) if m.startswith("Skipped ticker")]
    assert len(skipped) == 1
    assert "JPM" in skipped[0]

    table = metrics_table(at)
    assert list(table.index) == DEFAULT_TICKERS  # the row is still there ...
    assert table.loc["JPM"].drop(["Daily returns (n)", "Note"]).isna().all()  # ... but empty
    assert table.loc["JPM", "Note"] == "skipped (see warning above)"
    assert not table.drop(index="JPM").isna().any().any()  # the other four are fine

    assert len(charts(at)) == 3  # growth, volatility and heatmap use the 4 good tickers
    for chart in charts(at):  # read from the plotted data, not from the axis sort lists
        assert chart_tickers(chart) == ["AAPL", "MSFT", "NVDA", "XOM"]


def test_every_ticker_skipped_shows_an_error_instead_of_crashing(install_fake):
    def tweak(frame):
        frame.iloc[:-2, :] = np.nan  # only the last two rows have prices

    install_fake(tweak=tweak)
    at = run_app()

    assert not at.exception
    assert len(at.error) == 1
    assert "Could not calculate the risk numbers" in at.error[0].value
    assert len(charts(at)) == 0


def test_short_sample_is_flagged_in_the_table_and_warned_about(install_fake):
    def tweak(frame):
        frame.loc[frame.index[:-31], "XOM"] = np.nan  # XOM has 31 prices = 30 returns

    install_fake(tweak=tweak)
    at = run_app()

    assert not at.exception
    short = [m for m in texts(at.warning) if m.startswith("Short sample")]
    assert short and all("XOM" in message for message in short)

    table = metrics_table(at)
    assert table.loc["XOM", "Daily returns (n)"] == 30
    assert table.loc["XOM", "Note"].startswith("⚠")
    assert (table.drop(index="XOM")["Note"] == "").all()
    assert any("⚠" in caption.value for caption in at.caption)  # the explanation is shown


def test_extreme_short_sample_row_is_flagged_and_does_not_break_the_charts(install_fake):
    def tweak(frame):
        # XOM has 4 prices: returns of +18%, +93% and +135% (GME in January 2021). Stretched
        # to a year, that is a CAGR with 60 digits, which is why the flag exists.
        frame["XOM"] = np.nan
        frame.iloc[-4:, frame.columns.get_loc("XOM")] = [100.0, 118.0, 227.74, 535.0]

    loader = install_fake(tweak=tweak)
    at = run_app()

    assert not at.exception
    table = metrics_table(at)
    assert table.loc["XOM", "Daily returns (n)"] == 3
    assert table.loc["XOM", "Annualized return"] > 1000  # absurdly large, as expected
    assert table.loc["XOM", "Note"].startswith("⚠")
    # The shown text is capped, not a 60-digit number.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", risk.ShortSampleWarning)  # expected: only 3 returns
        xom_summary = risk.risk_summary(loader.frame[["XOM"]].dropna())
    assert "100,000%" in app.style_metrics(app.metrics_for_display(xom_summary)).to_html()

    # risk_summary calculates this wild ticker, but annualized_volatility(window=...) and
    # correlation_matrix refuse it ("looks like prices": its median daily move is 93%).
    # The page must survive that: the table and growth chart stay, and each of the two
    # charts that cannot be drawn says why instead of crashing the page.
    assert not at.error
    problems = [m for m in texts(at.warning) if m.startswith("This chart could not be drawn")]
    assert len(problems) == 2
    assert all("XOM" in m and "looks like prices" in m for m in problems)
    # A plain sentence comes before risk.py's programmer-style message.
    assert all("so extreme that risk.py's safety check refused" in m for m in problems)
    assert all(m.index("so extreme") < m.index("looks like prices") for m in problems)
    assert len(charts(at)) == 1  # only the growth chart


def test_a_gap_is_shown_once_as_an_info_note_not_a_warning(install_fake):
    def tweak(frame):
        frame.loc[frame.index[100], "MSFT"] = np.nan  # one missing price in the middle

    install_fake(tweak=tweak)
    at = run_app()

    assert not at.exception
    gap_notes = [m for m in texts(at.info) if m.startswith("Gaps in the data")]
    assert len(gap_notes) == 1  # risk_summary and daily_returns both warn; we show one
    # This sentence comes from risk.py's own message, not from the page's introduction.
    assert "'MSFT': 1 bridged (longest 1 row)" in gap_notes[0]
    assert not at.warning


def test_correlation_warning_about_short_pairs_reaches_the_page(install_fake):
    def tweak(frame):
        for ticker in ("JPM", "XOM"):  # two tickers with 30 returns each
            frame.loc[frame.index[:-31], ticker] = np.nan

    install_fake(tweak=tweak)
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
def test_short_sample_flag_starts_below_exactly_60_returns(install_fake, n_prices, n_returns, flagged):
    def tweak(frame):
        frame.loc[frame.index[:-n_prices], "XOM"] = np.nan

    install_fake(tweak=tweak)
    table = metrics_table(run_app())

    assert table.loc["XOM", "Daily returns (n)"] == n_returns
    assert table.loc["XOM", "Note"].startswith("⚠") is flagged
    assert risk.MIN_RELIABLE_OBS == 60  # if risk.py changes the limit, rethink these numbers


def test_late_starter_without_a_rolling_line_is_named(install_fake):
    def tweak(frame):
        frame.loc[frame.index[:-40], "XOM"] = np.nan  # 40 prices = 39 returns, window is 60

    install_fake(tweak=tweak)
    at = run_app()

    assert not at.exception
    assert "No line for XOM: fewer than 60 returns in the range." in [c.value for c in at.caption]
    rolling_chart = charts(at)[1]
    assert chart_tickers(rolling_chart) == ["AAPL", "JPM", "MSFT", "NVDA"]  # XOM really has no line
    assert len(charts(at)) == 3


def test_no_note_when_every_ticker_has_a_rolling_line(install_fake):
    at = run_app_with(install_fake)
    assert not any(c.value.startswith("No line for") for c in at.caption)


def test_a_long_gap_is_a_warning_not_a_holiday_note(install_fake):
    def tweak(frame):
        frame.loc[frame.index[100:110], "MSFT"] = np.nan  # a 10-row outage, longer than any holiday

    install_fake(tweak=tweak)
    at = run_app()

    assert not at.exception
    long_gap = [m for m in texts(at.warning) if m.startswith("Possible missing data")]
    assert len(long_gap) == 1
    assert "'MSFT': 1 bridged (longest 10 rows)" in long_gap[0]
    assert app.LONG_GAP_PHRASE in long_gap[0]  # that sentence is risk.py's, not the page's
    assert not any(m.startswith("Gaps in the data") for m in texts(at.info))  # not the soft note


def test_risk_py_still_uses_the_long_gap_wording_the_page_looks_for():
    prices = make_prices()[["AAPL"]]
    prices.iloc[100:110] = np.nan  # a 10-row outage
    with pytest.warns(risk.GapWarning, match=app.LONG_GAP_PHRASE):
        risk.daily_returns(prices)
    prices = make_prices()[["AAPL"]]
    prices.iloc[100:101] = np.nan  # a one-day holiday-style gap: no such sentence
    with pytest.warns(risk.GapWarning) as caught:
        risk.daily_returns(prices)
    assert app.LONG_GAP_PHRASE not in str(caught[0].message)


def test_one_ticker_shows_the_pick_two_message_instead_of_the_heatmap(install_fake):
    install_fake()
    at = run_app()
    at.sidebar.multiselect(key="tickers").set_value(["AAPL"]).run()

    assert not at.exception
    assert any("at least 2 tickers" in message for message in texts(at.info))
    assert len(charts(at)) == 2  # growth and rolling volatility, no heatmap
    assert list(metrics_table(at).index) == ["AAPL"]


def test_window_longer_than_the_data_shows_a_message_instead_of_the_chart(install_fake):
    install_fake(n_days=100)  # 99 returns
    at = run_app()
    at.sidebar.slider(key="window").set_value(252).run()

    assert not at.exception
    assert any("longer than the 99 daily returns" in message for message in texts(at.info))
    assert len(charts(at)) == 2  # growth and heatmap; the rolling-volatility chart is replaced


# --------------------------------------------------------------------------- #
# Small helper functions
# --------------------------------------------------------------------------- #

def test_importing_app_does_not_draw_anything():
    # `import app` at the top of this file already worked without a Streamlit server.
    assert callable(app.main)


def test_parse_extra_tickers():
    assert app.parse_extra_tickers("") == []
    assert app.parse_extra_tickers(" tsla , amd,, ") == ["TSLA", "AMD"]


def test_format_percent():
    assert app.format_percent(0.256) == "25.6%"
    assert app.format_percent(-0.03) == "-3.0%"
    assert app.format_percent(27.081) == "2,708.1%"  # thousands separators
    assert app.format_percent(999.9) == "99,990.0%"
    assert app.format_percent(1000) == ">100,000%"
    assert app.format_percent(float("nan")) == "—"
    # A 3-day sample stretched to a year (GME-like): never a 60-digit number.
    assert app.format_percent(1.6e61) == ">100,000%"


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
    assert app.skipped_rows(summary).tolist() == [False, True, True]


def test_metrics_for_display_finds_the_var_column_by_its_prefix():
    summary = pd.DataFrame(
        {
            "annualized_return": [0.1, 0.2],
            "annualized_volatility": [0.2, 0.3],
            "sharpe_ratio": [0.5, 0.6],
            "max_drawdown": [0.1, 0.2],
            "var_99": [0.02, 0.03],  # not var_95
            "n_obs": [100, risk.MIN_RELIABLE_OBS - 1],
        },
        index=pd.Index(["A", "B"], name="Ticker"),
    )
    table = app.metrics_for_display(summary)
    assert "1-day VaR 99% (loss)" in table.columns
    assert table["1-day VaR 99% (loss)"].tolist() == [0.02, 0.03]  # values are not changed
    assert table["Note"].tolist() == ["", f"⚠ short sample (under {risk.MIN_RELIABLE_OBS} returns)"]


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
    notes = app.metrics_for_display(summary)["Note"].tolist()
    assert [bool(note) for note in notes] == [True, False, False]


def test_capture_risk_warnings_keeps_ours_and_passes_all_others_on():
    bucket = []
    with pytest.warns(FutureWarning, match="old pandas thing") as passed_on:
        with app.capture_risk_warnings(bucket):
            warnings.warn("old pandas thing", FutureWarning)
            warnings.warn("too short", risk.ShortSampleWarning)

    assert [str(w.message) for w in bucket] == ["too short"]
    assert not any(issubclass(w.category, risk.ShortSampleWarning) for w in passed_on)  # ours stay off the terminal


def test_capture_risk_warnings_still_delivers_when_the_calculation_raises():
    bucket = []
    with pytest.warns(DeprecationWarning, match="going away"):
        with pytest.raises(ValueError, match="boom"):
            with app.capture_risk_warnings(bucket):
                warnings.warn("going away", DeprecationWarning)
                warnings.warn("too short", risk.ShortSampleWarning)
                raise ValueError("boom")
    assert [str(w.message) for w in bucket] == ["too short"]


def test_dedupe_warnings_keeps_one_gap_warning_and_drops_exact_repeats():
    def record(category, message):
        return warnings.WarningMessage(message, category, "x.py", 1)

    caught = [
        record(risk.GapWarning, "risk_summary: gaps in 'A'"),
        record(risk.GapWarning, "daily_returns: gaps in 'A'"),  # same gaps, other wording
        record(risk.ShortSampleWarning, "short B"),
        record(risk.ShortSampleWarning, "short B"),  # exact repeat
        record(risk.ShortSampleWarning, "short C"),
    ]
    assert app.dedupe_warnings(caught) == [
        (risk.GapWarning, "risk_summary: gaps in 'A'"),
        (risk.ShortSampleWarning, "short B"),
        (risk.ShortSampleWarning, "short C"),
    ]


def test_heatmap_leaves_undefined_correlations_blank():
    corr = pd.DataFrame(
        [[1.0, np.nan, 0.5], [np.nan, np.nan, np.nan], [0.5, np.nan, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    spec = app.correlation_chart(corr).to_dict()
    rows = next(iter(spec["datasets"].values()))
    assert len(rows) == 4  # only the four defined cells are drawn
    assert all(row["Correlation"] is not None for row in rows)
    # The colour scale is fixed to -1..1 whatever the data looks like.
    scale = spec["layer"][0]["encoding"]["color"]["scale"]
    assert scale["domain"] == [-1, 1]


# --------------------------------------------------------------------------- #
# The "no math in app.py" rule
# --------------------------------------------------------------------------- #

FORBIDDEN_METHODS = {
    # pandas / numpy calculations
    "pct_change", "std", "var", "mean", "median", "cumprod", "cumsum", "cummax", "cummin", "sqrt",
    "quantile", "corr", "cov", "rolling", "expanding", "ewm", "resample", "log", "log1p", "exp",
    "expm1", "prod", "sum", "diff", "shift", "pow", "mul", "div", "sub", "truediv", "floordiv",
    "abs", "rank",
}  # (.apply is allowed: the Styler uses it to set CSS colours, not to calculate)
FORBIDDEN_NAMES = {"sum", "pow", "abs", "divmod"}  # built-in functions that calculate
AGGREGATE_SHORTHAND = re.compile(
    r"^\s*(mean|sum|average|median|min|max|count|distinct|stdev|stdevp|variance|variancep|q1|q3)\(", re.I
)


def find_math(source: str) -> list[str]:
    """Look at the CODE of a Python source (not its comments) for calculations. Returns the findings."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_METHODS:
                found.append(f"line {node.lineno}: .{node.attr}")
            if node.attr.startswith("transform"):  # Altair: chart.transform_calculate(...), ...
                found.append(f"line {node.lineno}: .{node.attr} (a chart transform)")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in FORBIDDEN_NAMES:
            found.append(f"line {node.lineno}: {node.id}(...)")
        if isinstance(node, ast.keyword) and node.arg in {"aggregate", "calculate", "transform"}:
            found.append(f"line {node.value.lineno}: {node.arg}= (chart-side math)")
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Pow, ast.MatMult)):
            found.append(f"line {node.lineno}: power / matrix product")
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
            for side in (node.left, node.right):
                if isinstance(side, ast.Constant) and side.value in (100, 100.0):
                    found.append(f"line {node.lineno}: multiplying or dividing by 100")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and AGGREGATE_SHORTHAND.match(node.value):
            found.append(f"line {node.lineno}: {node.value!r} (Altair aggregate shorthand)")
    return found


def test_app_py_contains_no_financial_math():
    """All math lives in signaldesk/risk.py, so app.py's code must not calculate anything."""
    found = find_math(APP_PATH.read_text())
    assert not found, f"app.py must not calculate numbers itself: {found}"


def test_the_math_scan_catches_the_usual_tricks():
    """Run the scan on snippets full of tricks, to prove it is not asleep."""
    tricks = {
        "pct_change": "x = df.pct_change()",
        ".cummin": "y = df.cummin()",
        ".sum": "y = df.sum()",
        ".diff": "y = df.diff()",
        ".mul": "z = df.mul(100)",
        ".resample": "z = df.resample('W')",
        ".pow": "z = df.pow(2)",
        "transform_calculate": "c = chart.transform_calculate(a='datum.a * 100')",
        "transform_window": "c = chart.transform_window(a='sum(a)')",
        "aggregate shorthand": "d = alt.Y('mean(Value):Q')",
        "aggregate keyword": "e = alt.Y(field='v', aggregate='mean')",
        "times 100": "f = df * 100",
        "power": "g = df ** 2",
        "pow()": "g = pow(2, 3)",
    }
    for name, code in tricks.items():
        assert find_math(code), f"the scan missed: {name}"
    # ... and it leaves harmless code alone (including .apply, which the Styler needs for CSS).
    assert find_math("a = df.dropna().apply(f, axis=1)\nb = df.melt(id_vars='Date')\nc = 64 * len(x)") == []
