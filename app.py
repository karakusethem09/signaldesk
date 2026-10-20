"""SignalDesk dashboard (Phase 1): a small risk and performance page for a few stocks.

This file is ONLY the user interface. It does no financial math at all:
    - every price comes from ``signaldesk/data.py``
    - every number (returns, volatility, correlation, drawdown, VaR, Sharpe) comes from
      ``signaldesk/risk.py``
If a number is missing, the right fix is a new function in ``risk.py`` (with a test), not
a calculation here. What this file does do: pick which columns/rows to show, reshape tables
for the charts, and format numbers as text (for example 0.25 -> "25.0%").

How to run it:
    .venv/bin/streamlit run app.py

How the page is built: ``main()`` runs top to bottom on every click. It reads the sidebar,
loads prices, runs all the risk calculations ONCE (``analyze``), and then draws each
section with a small ``render_...`` helper, in the order the page shows them.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from signaldesk import data, risk

# --------------------------------------------------------------------------- #
# Settings and constants
# --------------------------------------------------------------------------- #

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
# Extra ready-made choices in the picker. Any other symbol can be typed in the text box.
MORE_TICKERS = ["GOOGL", "AMZN", "META", "TSLA", "JNJ", "WMT", "SPY", "QQQ"]

CONFIDENCE = 0.95  # VaR confidence level (95%)
DEFAULT_WINDOW = 60  # default rolling-volatility window, in trading days
MIN_WINDOW, MAX_WINDOW = 20, 252  # 20 is the smallest window risk.py does not warn about
MIN_RISK_FREE_RATE, MAX_RISK_FREE_RATE = -0.05, 0.20  # a year's rate as a fraction: -5% to 20%

# Part of the sentence risk.py's GapWarning adds when a bridged gap is longer than any holiday.
# test_app.py checks that risk.py still says it, so a reworded message cannot go unnoticed.
LONG_GAP_PHRASE = "longer than a holiday"

SCOPE_NOTE = (
    "Built for stocks and ETFs only (a year = 252 trading days). Crypto trades 7 days a week, "
    "so its annualized numbers come out wrong. Mixing exchanges (say US and Japan) is fine, "
    "but expect a gap notice because the holidays differ."
)

# The kinds of warning that risk.py raises. Only these are shown on the page.
RISK_WARNINGS = (risk.SkippedTickerWarning, risk.ShortSampleWarning, risk.GapWarning)


@dataclass(frozen=True)
class Settings:
    """Everything the user picked in the sidebar."""

    tickers: list[str]
    start: date
    end: date  # the last day the user wants INCLUDED (what the sidebar shows)
    end_exclusive: date  # end + 1 day: what load_prices wants, because its end is excluded
    window: int  # rolling-volatility window in trading days
    risk_free_rate: float  # annual FRACTION, 0.04 = 4%


@dataclass(frozen=True)
class Analysis:
    """The results of every risk.py call the page needs, computed once per run."""

    summary: pd.DataFrame  # one row per ticker (NaN row = ticker skipped)
    skipped: list[str]  # tickers that risk_summary could not calculate
    usable: list[str]  # the other tickers (these feed the charts)
    growth: pd.DataFrame | None  # growth of 1 dollar, usable tickers only
    rolling_vol: pd.DataFrame | None  # None when the window is longer than the data
    n_returns: int  # number of daily-return rows (used in the "window too long" message)
    correlation: pd.DataFrame | None  # None when fewer than 2 usable tickers
    problems: dict[str, str]  # chart name -> why risk.py refused to calculate it
    warnings: list[tuple[type, str]]  # (category, message), without repeats


# --------------------------------------------------------------------------- #
# Sidebar: what the user can change
# --------------------------------------------------------------------------- #

def parse_extra_tickers(text: str) -> list[str]:
    """Turn 'tsla, amd ,spy' into ['TSLA', 'AMD', 'SPY'] (upper case, no blanks)."""
    return [piece.strip().upper() for piece in text.split(",") if piece.strip()]


def render_sidebar() -> Settings:
    """Draw the sidebar controls and return what the user chose."""
    st.sidebar.header("Settings")

    chosen = st.sidebar.multiselect(
        "Tickers",
        options=DEFAULT_TICKERS + MORE_TICKERS,
        default=DEFAULT_TICKERS,
        key="tickers",
    )
    extra = st.sidebar.text_input(
        "Other tickers (comma-separated)",
        placeholder="e.g. AMD, KO",
        key="extra_tickers",
    )
    # dict.fromkeys removes repeats but keeps the order the user picked.
    tickers = list(dict.fromkeys(chosen + parse_extra_tickers(extra)))

    today = date.today()
    earliest = date(2000, 1, 1)
    start = st.sidebar.date_input(
        "Start date",
        value=today - timedelta(days=2 * 365),  # default: about two years of history
        min_value=earliest,
        max_value=today,
        key="start_date",
    )
    end = st.sidebar.date_input(
        "End date (included)",
        value=today,
        min_value=earliest,
        max_value=today,
        key="end_date",
    )

    window = st.sidebar.slider(
        "Rolling volatility window (trading days)",
        min_value=MIN_WINDOW,
        max_value=MAX_WINDOW,
        value=DEFAULT_WINDOW,
        key="window",
    )
    risk_free_rate = st.sidebar.number_input(
        "Risk-free rate (per year, as a fraction)",
        # Bounds stop a slip like 4.0 (meaning 4%), which would give a Sharpe of about -25.
        min_value=MIN_RISK_FREE_RATE,
        max_value=MAX_RISK_FREE_RATE,
        value=0.0,
        step=0.005,
        format="%.3f",
        help="0.04 = 4%. Used only for the Sharpe ratio. Allowed: -0.05 to 0.20.",
        key="risk_free_rate",
    )

    st.sidebar.caption(SCOPE_NOTE)

    # A cleared date or number box gives None. Stop with a hint instead of crashing.
    if start is None or end is None or risk_free_rate is None:
        st.info("Fill in the start date, end date and risk-free rate in the sidebar.")
        st.stop()

    return Settings(
        tickers=tickers,
        start=start,
        end=end,
        # data.load_prices treats its end date as EXCLUDED, but the sidebar says
        # "included", so we hand over one day later. This is date handling, not math.
        end_exclusive=end + timedelta(days=1),
        window=int(window),
        risk_free_rate=float(risk_free_rate),
    )


# --------------------------------------------------------------------------- #
# Loading data
# --------------------------------------------------------------------------- #

def load_data(settings: Settings) -> pd.DataFrame:
    """Get the price table, or show a friendly error and stop the page.

    We deliberately do NOT use st.cache_data here: data.load_prices already keeps its own
    CSV cache on disk and refreshes it daily, and Streamlit's cache would freeze old data.
    """
    if not settings.tickers:
        st.info("Pick at least one ticker in the sidebar.")
        st.stop()
    if settings.start >= settings.end:
        st.error("The start date must be before the end date.")
        st.stop()

    try:
        with st.spinner("Loading prices..."):
            return data.load_prices(settings.tickers, settings.start, settings.end_exclusive)
    except data.PriceDataError as error:  # unknown ticker or network problem (check first)
        st.error(f"Could not load prices: {error}")
        st.stop()
    except ValueError as error:  # bad input, for example an invalid date
        st.error(f"Could not load prices (bad input): {error}")
        st.stop()


# --------------------------------------------------------------------------- #
# Running the risk calculations (no math here: only calls into risk.py)
# --------------------------------------------------------------------------- #

@contextmanager
def capture_risk_warnings(bucket: list):
    """Collect the warnings risk.py raises, so we can show them on the page.

    Python prints warnings to the server's terminal, which the person using the dashboard
    never sees. Recording them lets us put them on the page. Warnings of other kinds
    (for example from pandas or numpy) are not for the page, but recording would hide them
    from the terminal too, so those are passed on to ``warnings.showwarning`` afterwards.
    """
    caught: list = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")  # record every one, even repeats
            yield
    finally:
        # Runs after catch_warnings has finished, also when the calculation raised.
        for item in caught:
            if issubclass(item.category, RISK_WARNINGS):
                bucket.append(item)
            else:
                warnings.showwarning(
                    item.message, item.category, item.filename, item.lineno, item.file, item.line
                )


def dedupe_warnings(caught: list) -> list[tuple[type, str]]:
    """Drop repeated warnings.

    risk_summary and daily_returns both report gaps, in nearly the same words. We keep the
    first GapWarning (the summary's) and skip later ones; for the other kinds we remove
    exact repeats of (category, message).
    """
    result: list[tuple[type, str]] = []
    seen: set[tuple[type, str]] = set()
    gap_shown = False
    for item in caught:
        category, message = item.category, str(item.message)
        if issubclass(category, risk.GapWarning):
            if gap_shown:
                continue
            gap_shown = True
        if (category, message) in seen:
            continue
        seen.add((category, message))
        result.append((category, message))
    return result


def skipped_rows(summary: pd.DataFrame) -> pd.Series:
    """True for each ticker whose summary row is all NaN (risk_summary skipped it).

    We look at the NaN metrics and not at n_obs, because n_obs can be 0 or more than 0 for
    a skipped ticker (it is the real count of returns when the problem was too few of them
    or a flat price, and 0 when a price was bad).
    """
    return summary.drop(columns="n_obs").isna().all(axis=1)


def analyze(prices: pd.DataFrame, settings: Settings) -> Analysis:
    """Run every risk.py function the page needs, once, and collect their warnings.

    Raises ValueError if no ticker at all could be calculated (the caller shows it).
    """
    caught: list = []
    problems: dict[str, str] = {}

    def attempt(chart: str, function, *args, **kwargs):
        """Call a risk.py function for a chart. If it refuses (ValueError), remember why.

        One chart that cannot be drawn must not take the metrics table down with it.
        """
        try:
            return function(*args, **kwargs)
        except ValueError as error:
            problems[chart] = str(error)
            return None

    # Metrics table. risk_summary skips tickers it cannot calculate (NaN row + warning).
    with capture_risk_warnings(caught):
        summary = risk.risk_summary(
            prices, confidence=CONFIDENCE, risk_free_rate=settings.risk_free_rate
        )

    # The charts only get tickers that were calculated: the single-ticker functions
    # (growth, returns, correlation) refuse tickers with too little data.
    is_skipped = skipped_rows(summary)
    skipped = list(summary.index[is_skipped.to_numpy()])
    usable = list(summary.index[~is_skipped.to_numpy()])
    usable_prices = prices[usable]

    with capture_risk_warnings(caught):
        growth = attempt("growth", risk.cumulative_returns, usable_prices)
        returns = risk.daily_returns(usable_prices)
        n_returns = len(returns)

        # A rolling window longer than the data raises ValueError, so check first.
        rolling_vol = None
        if settings.window <= n_returns:
            rolling_vol = attempt("rolling", risk.annualized_volatility, returns, window=settings.window)

        # A correlation needs at least two tickers.
        correlation = None
        if len(usable) >= 2:
            correlation = attempt("correlation", risk.correlation_matrix, returns)

    return Analysis(
        summary=summary,
        skipped=skipped,
        usable=usable,
        growth=growth,
        rolling_vol=rolling_vol,
        n_returns=n_returns,
        correlation=correlation,
        problems=problems,
        warnings=dedupe_warnings(caught),
    )


# --------------------------------------------------------------------------- #
# Section 1: the "as of" line
# --------------------------------------------------------------------------- #

def render_as_of(prices: pd.DataFrame) -> None:
    """Say how fresh the data is. We use the last date IN the table, not the requested end."""
    first, last = prices.index[0], prices.index[-1]
    st.markdown(
        f"**Data as of {last:%Y-%m-%d}** (the last trading day in the data). "
        f"{len(prices)} trading days from {first:%Y-%m-%d}."
    )


# --------------------------------------------------------------------------- #
# Section 2: warnings
# --------------------------------------------------------------------------- #

def render_warnings(items: list[tuple[type, str]]) -> None:
    """Show the warnings risk.py raised: problems as warnings, gaps as softer notes."""
    for category, message in items:
        if issubclass(category, risk.GapWarning) and LONG_GAP_PHRASE in message:
            # risk.py found a gap longer than any exchange holiday: this is probably missing
            # data, so it must not look like the harmless holiday note below.
            st.warning(
                "Possible missing data: at least one gap is longer than "
                f"{risk.MAX_HOLIDAY_GAP_ROWS} rows, which is more than any exchange holiday. "
                f"Returns across it are multi-week moves counted as one day.\n\n{message}"
            )
        elif issubclass(category, risk.GapWarning):
            st.info(
                "Gaps in the data: some returns span missing prices (bridged returns). "
                "These are usually exchange holidays; gaps longer than "
                f"{risk.MAX_HOLIDAY_GAP_ROWS} rows are likely missing data.\n\n{message}"
            )
        elif issubclass(category, risk.SkippedTickerWarning):
            st.warning(f"Skipped ticker. {message}")
        else:  # ShortSampleWarning
            st.warning(f"Short sample. {message}")


# --------------------------------------------------------------------------- #
# Section 3: metrics table
# --------------------------------------------------------------------------- #

def format_percent(value: float) -> str:
    """Show a fraction as a percent, e.g. 0.256 -> '25.6%' and 27.08 -> '2,708.1%'. Display only, nothing is changed.

    A huge value (more than 100,000%) comes from stretching a few days of data to a full
    year; we show '>100,000%' instead of a 60-digit number.
    """
    if pd.isna(value):
        return "—"
    if value >= 1000:
        return ">100,000%"
    return f"{value:,.1%}"


def metrics_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Rename the summary's columns for people and add a 'Note' column. Values are untouched."""
    # The VaR column is named after the confidence (var_95, var_99, ...): find it by prefix.
    var_column = next(name for name in summary.columns if name.startswith("var_"))
    var_label = f"1-day VaR {var_column.removeprefix('var_')}% (loss)"

    table = summary.rename(
        columns={
            "annualized_return": "Annualized return",
            "annualized_volatility": "Annualized volatility",
            "sharpe_ratio": "Sharpe ratio",
            "max_drawdown": "Max drawdown (loss)",
            var_column: var_label,
            "n_obs": "Daily returns (n)",
        }
    )

    skipped = skipped_rows(summary)
    short = (summary["n_obs"] < risk.MIN_RELIABLE_OBS) & ~skipped
    table["Note"] = ""
    table.loc[skipped, "Note"] = "skipped (see warning above)"
    table.loc[short, "Note"] = f"⚠ short sample (under {risk.MIN_RELIABLE_OBS} returns)"
    return table


def style_metrics(table: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Format the numbers as text and grey out the rows that carry a note."""
    percent_columns = [
        name
        for name in table.columns
        if name in ("Annualized return", "Annualized volatility")
        or name.startswith(("Max drawdown", "1-day VaR"))
    ]

    def grey_out(row: pd.Series) -> list[str]:
        return ["color: gray" if row["Note"] else ""] * len(row)

    return (
        table.style.format(format_percent, subset=percent_columns)
        .format("{:.2f}", subset=["Sharpe ratio"], na_rep="—")
        .format("{:,d}", subset=["Daily returns (n)"])
        .apply(grey_out, axis=1)
    )


def render_metrics(summary: pd.DataFrame, risk_free_rate: float) -> None:
    """The table of headline risk numbers, one row per ticker."""
    st.subheader("Key metrics")
    table = metrics_for_display(summary)
    st.dataframe(
        style_metrics(table),
        width="stretch",
        column_config={
            "Annualized return": st.column_config.Column(
                help="Steady yearly growth rate (geometric, a year = 252 trading days)."
            ),
            "Annualized volatility": st.column_config.Column(
                help="How much daily returns jump around, scaled to one year."
            ),
            "Sharpe ratio": st.column_config.Column(
                help="Return above the risk-free rate per unit of volatility."
            ),
            "Max drawdown (loss)": st.column_config.Column(
                help="Biggest fall from an earlier peak. A loss, shown as a positive number."
            ),
            next(c for c in table.columns if c.startswith("1-day VaR")): st.column_config.Column(
                help=(
                    "Historical Value at Risk: a one-day loss that was not exceeded on "
                    f"{CONFIDENCE:.0%} of past days. A loss, shown as a positive number. "
                    "Negative means there was no loss at this level."
                )
            ),
            "Daily returns (n)": st.column_config.Column(
                help="How many daily returns the numbers are based on."
            ),
        },
    )
    caption = "Max drawdown and VaR are losses, so 12.0% means a 12% loss (no minus sign)."
    if (table["Note"].str.startswith("⚠")).any():
        caption += (
            f" Greyed rows marked ⚠ have fewer than {risk.MIN_RELIABLE_OBS} daily returns "
            "(about 3 months). Stretching so few days to a year can give extreme numbers, "
            "so do not trust them."
        )
    st.caption(caption)
    # Say which risk-free rate the Sharpe column used (formatting only, no math).
    st.caption(f"Sharpe uses a risk-free rate of {risk_free_rate:.1%}/yr.")


# --------------------------------------------------------------------------- #
# Charts (Altair, which comes with Streamlit)
# --------------------------------------------------------------------------- #

def to_long(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Reshape a Date x Ticker table into rows of (Date, Ticker, value) for Altair.

    Empty (NaN) cells are dropped, so each line simply skips holidays and the start-up
    period of a rolling window instead of showing a hole.
    """
    long = (
        frame.rename_axis(index="Date", columns="Ticker")
        .reset_index()
        .melt(id_vars="Date", var_name="Ticker", value_name=value_name)
    )
    return long.dropna(subset=[value_name])


def line_chart(
    frame: pd.DataFrame, value_name: str, y_title: str, y_axis_format: str | None = None
) -> alt.Chart:
    """One coloured line per ticker. ``y_axis_format`` is a display format, e.g. '%'."""
    long = to_long(frame, value_name)
    tickers = list(frame.columns)  # keeps the colours and legend in the sidebar's order
    return (
        alt.Chart(long)
        # Invisible dots on every point make the hover tooltip work along the whole line.
        .mark_line(point=alt.OverlayMarkDef(opacity=0))
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(
                f"{value_name}:Q",
                title=y_title,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format=y_axis_format) if y_axis_format else alt.Axis(),
            ),
            color=alt.Color("Ticker:N", sort=tickers, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[
                alt.Tooltip("Date:T"),
                alt.Tooltip("Ticker:N"),
                alt.Tooltip(f"{value_name}:Q", format=y_axis_format or ".3f"),
            ],
        )
        .properties(height=320)
    )


def correlation_chart(correlation: pd.DataFrame) -> alt.LayerChart:
    """A heatmap: one coloured square per pair of tickers, with the number printed on it."""
    tickers = list(correlation.columns)
    long = (
        correlation.rename_axis(index="Ticker A", columns="Ticker B")
        .reset_index()
        .melt(id_vars="Ticker A", var_name="Ticker B", value_name="Correlation")
        .dropna(subset=["Correlation"])  # an undefined correlation is left blank
    )
    base = alt.Chart(long).encode(
        x=alt.X("Ticker B:N", sort=tickers, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Ticker A:N", sort=tickers, title=None),
    )
    squares = base.mark_rect().encode(
        # Fixed domain -1..1 with a two-sided colour scheme: 0 is always the neutral middle.
        # reverse=True makes positive (move together) warm and negative (move opposite) cool.
        color=alt.Color(
            "Correlation:Q",
            scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
            legend=alt.Legend(title="Correlation"),
        ),
        tooltip=[
            alt.Tooltip("Ticker A:N"),
            alt.Tooltip("Ticker B:N"),
            alt.Tooltip("Correlation:Q", format=".2f"),
        ],
    )
    numbers = base.mark_text().encode(
        text=alt.Text("Correlation:Q", format=".2f"),
        # White text on dark squares, black text on light ones.
        color=alt.condition("abs(datum.Correlation) > 0.6", alt.value("white"), alt.value("black")),
    )
    return alt.layer(squares, numbers).properties(height=max(240, 64 * len(tickers)))


# --------------------------------------------------------------------------- #
# Sections 4-6: growth chart, rolling volatility chart, correlation heatmap
# --------------------------------------------------------------------------- #

def render_chart_problem(analysis: Analysis, chart: str) -> None:
    """Say why a chart is missing when risk.py refused to calculate it.

    A plain sentence comes first, then risk.py's own message (written for programmers).
    """
    reason = analysis.problems[chart]
    if "looks like prices" in reason:
        plain = (
            "One ticker's few returns are so extreme that risk.py's safety check refused "
            "them (it takes them for prices). Remove that ticker or widen the date range."
        )
    else:
        plain = "risk.py refused to calculate this chart."
    st.warning(f"This chart could not be drawn. {plain}\n\nrisk.py said: {reason}")


def render_growth(analysis: Analysis) -> None:
    """Growth of $1: what 1 dollar invested on a ticker's first day would be worth."""
    st.subheader("Growth of $1")
    if analysis.growth is None:
        render_chart_problem(analysis, "growth")
        return
    st.caption("1.25 means +25% since the ticker's first day in the chart. Every line starts at 1.")
    st.altair_chart(line_chart(analysis.growth, "Growth", "Growth of $1"), width="stretch")


def render_rolling_volatility(analysis: Analysis, window: int) -> None:
    """Annualized volatility over a trailing window, to see how risk changes over time."""
    st.subheader(f"Rolling volatility ({window}-day window)")
    if "rolling" in analysis.problems:
        render_chart_problem(analysis, "rolling")
        return
    if analysis.rolling_vol is None:
        st.info(
            f"The window ({window} days) is longer than the {analysis.n_returns} daily "
            "returns available. Pick a shorter window or a longer date range."
        )
        return
    # A ticker with fewer returns than the window has no value at all (every cell is empty).
    # This only looks for empty columns; it calculates nothing.
    no_line = [name for name in analysis.rolling_vol.columns if analysis.rolling_vol[name].isna().all()]
    if to_long(analysis.rolling_vol, "Volatility").empty:
        st.info(
            f"No ticker has enough returns to fill one {window}-day window yet. "
            "Try a shorter window or a longer date range."
        )
        return
    st.caption("Annualized. Each line starts once a full window of returns exists.")
    if no_line:
        st.caption(f"No line for {', '.join(no_line)}: fewer than {window} returns in the range.")
    st.altair_chart(
        line_chart(analysis.rolling_vol, "Volatility", "Annualized volatility", y_axis_format=".0%"),
        width="stretch",
    )


def render_correlation(analysis: Analysis, n_selected: int) -> None:
    """Heatmap of how closely the tickers' daily returns move together."""
    st.subheader("Correlation of daily returns")
    if "correlation" in analysis.problems:
        render_chart_problem(analysis, "correlation")
        return
    if analysis.correlation is None:
        if n_selected < 2:
            st.info("Pick at least 2 tickers to see correlations.")
        else:
            st.info("Fewer than 2 tickers could be calculated (see the warnings above), so there is nothing to compare.")
        return
    st.caption("+1 = move together, 0 = unrelated, -1 = move in opposite directions.")
    st.altair_chart(correlation_chart(analysis.correlation), width="stretch")


# --------------------------------------------------------------------------- #
# Section 7: data quality
# --------------------------------------------------------------------------- #

def render_data_quality(prices: pd.DataFrame, settings: Settings) -> None:
    """A collapsed box with the raw facts about gaps, for anyone who wants to check the data."""
    with st.expander("Data quality"):
        st.write(
            f"First date in the data: **{prices.index[0]:%Y-%m-%d}**. "
            f"Last date: **{prices.index[-1]:%Y-%m-%d}**. "
            f"Rows (trading days): **{len(prices)}**."
        )

        st.markdown("**Missing prices per ticker** (empty cells, not whole missing days)")
        st.dataframe(data.find_gaps(prices), width="stretch")

        st.markdown("**Possible gaps (most are exchange holidays)**: weekdays with no row at all")
        try:
            # Same start and end we gave load_prices, so the check covers the loaded window.
            missing = data.find_missing_days(prices, settings.start, settings.end_exclusive)
        except ValueError as error:
            st.warning(f"Could not check for missing days: {error}")
            return
        if len(missing) == 0:
            st.write("None.")
        else:
            st.dataframe(
                pd.DataFrame({"Date": missing.strftime("%Y-%m-%d"), "Weekday": missing.strftime("%A")}),
                hide_index=True,
                width="stretch",
            )


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #

def main() -> None:
    """Build the whole page, top to bottom."""
    st.set_page_config(page_title="SignalDesk", page_icon="📈", layout="wide")
    st.title("SignalDesk")
    st.caption("A small portfolio risk and research dashboard: performance, volatility, correlation and risk.")

    settings = render_sidebar()
    prices = load_data(settings)
    try:
        analysis = analyze(prices, settings)
    except ValueError as error:  # no ticker could be calculated at all
        st.error(f"Could not calculate the risk numbers: {error}")
        st.stop()

    render_as_of(prices)
    render_warnings(analysis.warnings)
    render_metrics(analysis.summary, settings.risk_free_rate)
    render_growth(analysis)
    render_rolling_volatility(analysis, settings.window)
    render_correlation(analysis, n_selected=len(settings.tickers))
    render_data_quality(prices, settings)


# Streamlit runs this file as "__main__"; a plain `import app` does not, so importing is harmless.
if __name__ == "__main__":
    main()
