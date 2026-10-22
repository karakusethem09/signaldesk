"""Helpers for the dashboard tests: made-up prices, fakes for the data layer, and readers for what a page shows.

How the dashboard tests work:
    - ``streamlit.testing.v1.AppTest`` runs app.py inside this process, with no browser and no
      server, and lets us read the elements it drew (key numbers, tables, charts, messages) and
      change the sidebar widgets.
    - The prices are made up. In DEMO mode we replace ``data.load_demo`` by a fake that returns
      made-up prices; in CUSTOM mode we replace ``data.load_prices`` (and count its calls).
      The ``app_env`` fixture (conftest.py) additionally refuses every network attempt.

The risk numbers themselves are tested in test_risk.py. Here we check what the USER SEES: the text
in every table cell, the data behind every chart, the warnings and notes, and that the page
survives awkward data without crashing. (A test that only says "something was drawn" would pass
even if a number were shown with the wrong sign or unit.)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa  # installed with Streamlit; used to read the tables Streamlit sends out
from streamlit.testing.v1 import AppTest

from signaldesk import data, risk

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "app.py"

MODE_DEMO = "Demo portfolio"
MODE_CUSTOM = "Custom (live data)"

# The made-up universe has the same shape as the real demo: six holdings and SPY as the benchmark.
HOLDINGS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ"]
DEMO_WEIGHTS = {"AAPL": 0.20, "MSFT": 0.20, "NVDA": 0.15, "JPM": 0.15, "XOM": 0.15, "JNJ": 0.15}
BENCHMARK = "SPY"
UNIVERSE = HOLDINGS + [BENCHMARK, "TSLA", "GME"]  # TSLA and GME are there for tests that type extra tickers


# --------------------------------------------------------------------------- #
# Made-up data and fakes for the data layer
# --------------------------------------------------------------------------- #

def make_prices(n_days: int = 300, columns=UNIVERSE) -> pd.DataFrame:
    """Made-up prices: business days, one column per ticker, fixed random seed."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2025-01-01", periods=n_days, name="Date")
    daily = rng.normal(loc=0.0004, scale=0.01, size=(n_days, len(columns)))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + daily, axis=0), index=dates, columns=list(columns))


def make_demo(frame: pd.DataFrame | None = None, tweak=None, weights=None, benchmark: str = BENCHMARK) -> data.DemoData:
    """A fake demo snapshot with made-up prices. ``tweak(frame)`` may change the table in place."""
    frame = make_prices() if frame is None else frame.copy()
    if tweak is not None:
        tweak(frame)
    weights = dict(DEMO_WEIGHTS) if weights is None else dict(weights)
    columns = list(weights) + [benchmark]
    return data.DemoData(
        prices=frame[columns].copy(),
        holdings=weights,
        benchmark=benchmark,
        name="Test portfolio",
        as_of=frame.index[-1],
        start=frame.index[0],
        source="Made-up test prices",
    )


def install_demo(monkeypatch, demo: data.DemoData) -> data.DemoData:
    """Make ``data.load_demo`` return ``demo`` (no files are read)."""
    monkeypatch.setattr(data, "load_demo", lambda *args, **kwargs: demo)
    return demo


class FakeLoader:
    """Stands in for data.load_prices. Records every call so tests can look at the arguments."""

    def __init__(self, frame: pd.DataFrame, error: Exception | None = None):
        self.frame = frame
        self.error = error
        self.calls: list[tuple] = []  # (tickers, start, end) of every call
        self.options: list[dict] = []  # the cache_dir / use_cache each call was given

    def __call__(self, tickers, start, end, cache_dir=None, use_cache=True):
        self.calls.append((list(tickers), start, end))
        self.options.append({"cache_dir": cache_dir, "use_cache": use_cache})
        if self.error is not None:
            raise self.error
        unknown = [t for t in tickers if t not in self.frame.columns]
        if unknown:  # the real function raises PriceDataError for an unknown ticker too
            raise data.PriceDataError(f"No price data returned for: {', '.join(unknown)}.")
        return self.frame[list(tickers)].copy()


def install_loader(monkeypatch, frame: pd.DataFrame | None = None, tweak=None, error: Exception | None = None) -> FakeLoader:
    """Make ``data.load_prices`` a fake that serves ``frame`` (made-up prices by default)."""
    frame = make_prices() if frame is None else frame.copy()
    if tweak is not None:
        tweak(frame)
    loader = FakeLoader(frame, error)
    monkeypatch.setattr(data, "load_prices", loader)
    return loader


# --------------------------------------------------------------------------- #
# Running the app
# --------------------------------------------------------------------------- #

def new_app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=60)


def run_app() -> AppTest:
    """Run app.py once with the default settings (demo mode)."""
    return new_app().run()


def holdings_state(rows: list[tuple], n_existing: int = len(HOLDINGS)) -> dict:
    """What the holdings editor reports when the visitor replaced the table with ``rows``.

    ``rows`` are (ticker, weight in percent) pairs. The editor starts with the six demo rows,
    so they are all marked deleted and the new rows are added. Use None for an empty cell.
    """
    return {
        "edited_rows": {},
        "added_rows": [{"Ticker": ticker, "Weight (%)": weight} for ticker, weight in rows],
        "deleted_rows": list(range(n_existing)),
    }


def open_custom(at: AppTest | None = None) -> AppTest:
    """Run the app and switch the sidebar to Custom mode."""
    at = run_app() if at is None else at
    if at.sidebar.radio(key="mode").value != MODE_CUSTOM:
        at.sidebar.radio(key="mode").set_value(MODE_CUSTOM).run()
    return at


def press_run(at: AppTest) -> AppTest:
    """Press "Run analysis" (the button inside the custom form)."""
    at.sidebar.form_submit_button(key="run_analysis").click().run()
    return at


def run_custom(rows=None, benchmark: str | None = None, period: str | None = None, dates=None, at=None) -> AppTest:
    """Switch to Custom mode, optionally fill the form, and press "Run analysis".

    ``rows``: (ticker, percent) pairs replacing the default table. ``benchmark``: the benchmark text.
    ``period``: "1Y", "3Y", "5Y", "10Y" or "Custom dates". ``dates``: (start, end) for "Custom dates".
    """
    at = open_custom(at)
    if rows is not None:
        at.session_state["holdings_editor"] = holdings_state(rows)
    if benchmark is not None:
        at.sidebar.text_input(key="benchmark").set_value(benchmark)
    if period is not None:
        at.sidebar.radio(key="custom_period").set_value(period)
    if dates is not None:
        at.sidebar.date_input(key="custom_start").set_value(dates[0])
        at.sidebar.date_input(key="custom_end").set_value(dates[1])
    return press_run(at)


# --------------------------------------------------------------------------- #
# Reading what the page shows
# --------------------------------------------------------------------------- #

def tab(at: AppTest, name: str):
    """One tab of the page (a block with its own tables, charts and texts)."""
    return next(t for t in at.tabs if t.label == name)


def charts(block) -> list:
    """The Altair charts in a tab or on the page (Streamlit draws them as 'vega_lite_chart' elements)."""
    return block.get("vega_lite_chart")


def texts(elements) -> list[str]:
    """The text of a list of message elements (warnings, infos, errors)."""
    return [element.value for element in elements]


def captions(block) -> list[str]:
    return [caption.value for caption in block.caption]


def metrics_table(at: AppTest) -> pd.DataFrame:
    """The Holdings tab's table EXACTLY AS STREAMLIT SENDS IT to the browser: real numbers, NaN where empty.

    Use ``shown_table`` for the text a visitor reads, and ``assert_sortable_table`` for what the browser does with it.
    """
    return tab(at, "Holdings").dataframe[0].value


def shown_table(at: AppTest) -> pd.DataFrame:
    """The Holdings tab's table as the USER SEES it: the formatted text of every cell ("25.6%")."""
    return display_values(tab(at, "Holdings").dataframe[0])


def assert_sortable_table(element, text_columns=()) -> None:
    """A table is sent so that clicking a header SORTS BY VALUE, and an empty cell is drawn as a dash.

    - ``placeholder`` is "—": without it the browser draws its own "None" for an empty cell, whatever text a
      Styler chose for it (AppTest reads the Styler's text, so a test of that text alone would pass anyway).
    - Every column except ``text_columns`` is numeric or a date: a column of strings would sort alphabetically
      ("5.1%" above "29.3%", "-17.4%" above "-17.1%").
    - The Styler's text for an empty cell is the dash, and for any other cell never "None", "nan" or "NaT".
    """
    assert element.proto.placeholder == "—", repr(element.proto.placeholder)
    sent = element.value
    for column in sent.columns:
        if column in text_columns:
            assert pd.api.types.is_string_dtype(sent[column]) or sent[column].dtype == object, column
        else:
            kind = sent[column].dtype
            assert pd.api.types.is_numeric_dtype(kind) or pd.api.types.is_datetime64_any_dtype(kind), (
                f"column {column!r} is sent as {kind}: it would sort as text"
            )
    if element.proto.arrow_data.styler.display_values:
        shown = display_values(element)  # a Styler chose the text
    else:  # a plain table: the browser draws the data itself, and the placeholder for an empty cell
        shown = sent.astype(object).where(sent.notna(), "—").astype(str)
    assert list(sent.columns) == list(shown.columns) and list(sent.index.astype(str)) == list(shown.index.astype(str))
    empty = sent.isna().to_numpy()
    shown_values = shown.to_numpy()
    assert (shown_values[empty] == "—").all(), "an empty cell must show the dash"
    assert not pd.Series(shown_values[~empty].ravel()).isin(["None", "nan", "NaN", "NaT", "<NA>"]).any()


def display_values(element) -> pd.DataFrame:
    arrow = element.proto.arrow_data
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

# A chart must only DRAW the numbers it was given. Vega-Lite can also compute: a transform, an
# aggregate, a binning, an imputation (it invents values), a stack, a time unit, a sort by an operation.
# These keys are refused anywhere in a specification ...
FORBIDDEN_SPEC_KEYS = {
    "transform", "aggregate", "calculate", "impute", "bin", "timeUnit", "op", "expr", "joinaggregate",
    "window", "fold", "flatten", "pivot", "density", "loess", "regression", "quantile", "lookup", "filter",
    "sample", "extent", "stack",
}
# ... and what is left is an ALLOW-LIST: inside "encoding" only these keys may appear, so a computing
# feature that nobody thought of yet is refused as well.
ALLOWED_CHANNELS = {"x", "y", "color", "text", "tooltip"}
ALLOWED_CHANNEL_KEYS = {"field", "type", "title", "scale", "axis", "legend", "sort", "format", "condition", "value"}
ALLOWED_SCALE_KEYS = {"domain", "range", "scheme", "reverse", "zero"}
ALLOWED_AXIS_KEYS = {"format", "labelAngle"}
ALLOWED_LEGEND_KEYS = {"orient", "title"}
ALLOWED_CONDITION_KEYS = {"test", "value"}
ALLOWED_MARK_KEYS = {"type", "point"}


def _all_keys(node, path=""):
    """Every (path, key) in a nested dict/list, except inside the data."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "datasets":
                continue
            yield path, key, value
            yield from _all_keys(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _all_keys(value, f"{path}[{index}]")


def _encodings(spec: dict):
    """Every "encoding" block of a (possibly layered) specification."""
    for path, key, value in _all_keys(spec):
        if key == "encoding" and isinstance(value, dict):
            yield f"{path}/encoding", value


def _check_keys(where: str, block, allowed: set) -> None:
    if isinstance(block, dict):
        unexpected = set(block) - allowed
        assert not unexpected, f"{where} uses {sorted(unexpected)}: not an allowed way to draw (it may compute)"


def assert_no_chart_math(spec: dict) -> None:
    """A chart spec must not calculate anything: no transform, aggregate, bin, impute, stack, expression ..."""
    # 1. Forbidden keys anywhere (a "stack": null, which Altair writes for "no stacking", is fine).
    for path, key, value in _all_keys(spec):
        if key in FORBIDDEN_SPEC_KEYS and not (key == "stack" and value is None):
            raise AssertionError(f"the chart uses {key!r} at {path or '/'}: that is math, put it in risk.py")
    # 2. Expressions: only the heatmap's text-colour test.
    text = json.dumps({key: value for key, value in spec.items() if key != "datasets"})
    expressions = set(re.findall(r'"([^"]*datum\.[^"]*)"', text))
    assert expressions <= ALLOWED_EXPRESSIONS, f"unexpected chart expression(s): {expressions}"
    # 3. The allow-list for what an encoding may say.
    for where, encoding in _encodings(spec):
        _check_keys(f"{where}", encoding, ALLOWED_CHANNELS)
        for channel, definition in encoding.items():
            for part in definition if isinstance(definition, list) else [definition]:  # tooltip is a list
                _check_keys(f"{where}/{channel}", part, ALLOWED_CHANNEL_KEYS)
                if not isinstance(part, dict):
                    continue
                _check_keys(f"{where}/{channel}/scale", part.get("scale"), ALLOWED_SCALE_KEYS)
                _check_keys(f"{where}/{channel}/axis", part.get("axis"), ALLOWED_AXIS_KEYS)
                _check_keys(f"{where}/{channel}/legend", part.get("legend"), ALLOWED_LEGEND_KEYS)
                condition = part.get("condition")
                for one in condition if isinstance(condition, list) else [condition]:
                    _check_keys(f"{where}/{channel}/condition", one, ALLOWED_CONDITION_KEYS)
                assert not isinstance(part.get("sort"), dict), f"{where}/{channel} sorts by a computed value"
    # 4. The mark only says what shape to draw.
    for path, key, value in _all_keys(spec):
        if key == "mark":
            _check_keys(f"{path}/mark", value, ALLOWED_MARK_KEYS)


def assert_chart_shows(chart, value: str, expected: pd.DataFrame, rows: str = "Date", columns: str = "Ticker"):
    """The chart plots exactly the numbers risk.py gave (same cells, no rescaling)."""
    got = chart_table(chart, value, rows, columns).reindex(index=expected.index, columns=expected.columns)
    pd.testing.assert_frame_equal(
        got, expected.astype("float64"), check_names=False, check_freq=False, rtol=1e-12, atol=0.0
    )


# --------------------------------------------------------------------------- #
# What risk.py says the numbers are (computed here, independently of the page)
# --------------------------------------------------------------------------- #

def expected_portfolio(frame: pd.DataFrame, weights=None, benchmark: str = BENCHMARK):
    """(portfolio value, benchmark on the portfolio's dates) straight from risk.py."""
    weights = DEMO_WEIGHTS if weights is None else weights
    value = risk.portfolio_value(frame[list(weights)], weights)
    return value, frame[benchmark].loc[value.index]


def expected_summary(frame: pd.DataFrame, weights=None, benchmark: str = BENCHMARK, rf: float = 0.0) -> pd.DataFrame:
    """risk_summary of the holdings, the Portfolio and the benchmark, all on the portfolio's dates."""
    weights = DEMO_WEIGHTS if weights is None else weights
    value, bench = expected_portfolio(frame, weights, benchmark)
    table = pd.concat([frame.loc[value.index, list(weights)], pd.DataFrame({"Portfolio": value, benchmark: bench})], axis=1)
    return risk.risk_summary(table, confidence=0.95, risk_free_rate=rf)
