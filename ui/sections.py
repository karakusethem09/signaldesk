"""The page itself: header, key numbers, warnings, the six tabs and the footer.

Each ``render_...`` function draws one part of the page from an ``Analysis`` (the numbers
risk.py calculated, see ui/analysis.py) and a ``Dataset`` (what was analysed). They only choose
which rows and columns to show, reshape tables for the charts and format numbers as text.

The page, top to bottom:
    title and byline    "Demo data as of ..." and the GitHub link
    key numbers         five numbers for the portfolio, the benchmark in small text under each
    warnings            what risk.py warned about (short samples, gaps, skipped tickers)
    tabs                Overview | Risk | Correlation | Holdings | Data | About
    footer              the disclaimer
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from signaldesk import data, risk
from ui import inputs
from ui.analysis import CONFIDENCE, LONG_GAP_PHRASE, PORTFOLIO, PORTFOLIO_MESSAGE_PREFIX, Analysis, analyze
from ui.charts import correlation_chart, line_chart, series_colors, to_long
from ui.formatting import (
    PLACEHOLDER,
    format_percent,
    format_ratio,
    metrics_for_display,
    style_gaps,
    style_metrics,
    style_weights,
    var_column,
    var_label,
)
from ui.state import Controls, Dataset, Unavailable, go_to_demo

log = logging.getLogger(__name__)

GITHUB_URL = "https://github.com/karakusethem09/signaldesk"
CATCH_LOG_URL = f"{GITHUB_URL}/blob/main/docs/catch-log.md"
FOOTER_TEXT = "Not investment advice. Data: Yahoo Finance, end-of-day."
TAB_NAMES = ["Overview", "Risk", "Correlation", "Holdings", "Data", "About"]


# --------------------------------------------------------------------------- #
# Title, byline and footer
# --------------------------------------------------------------------------- #

def render_title(as_of_text: str | None = None) -> None:
    """The page title, and under it the data's date (when there is data) and the GitHub link."""
    st.title("SignalDesk")
    lead = f"**{as_of_text}**" if as_of_text else "A portfolio risk dashboard"
    st.markdown(f"{lead} · [GitHub]({GITHUB_URL})")


def render_footer() -> None:
    """The disclaimer, at the bottom of every page."""
    st.divider()
    st.caption(FOOTER_TEXT)


def offer_demo() -> None:
    """A way back to the demo, for when custom mode has nothing (or nothing good) to show."""
    st.button("Back to the demo", on_click=go_to_demo, key="back_to_demo")


def render_unavailable(problem: Unavailable) -> None:
    """The page when there is nothing to analyse: the reason, a way back, and the footer."""
    render_title()
    {"info": st.info, "warning": st.warning}.get(problem.kind, st.error)(problem.message)
    if problem.offer_demo:
        offer_demo()
    render_footer()


def show_table(table, **options) -> None:
    """Draw a table. EVERY table goes through here, so none can forget the placeholder.

    ``placeholder`` is the text the browser draws for an empty cell. Without it Streamlit writes "None"
    (and ignores the dash a Styler chose for that cell). The cells themselves stay real numbers or dates,
    so clicking a column header sorts by value.
    """
    st.dataframe(table, placeholder=PLACEHOLDER, **options)


# --------------------------------------------------------------------------- #
# Key numbers
# --------------------------------------------------------------------------- #

def render_kpis(analysis: Analysis, risk_free_rate: float) -> None:
    """Five key numbers for the portfolio, with the benchmark's value as small text under each.

    The values are read from ``risk.risk_summary`` of the Portfolio and the benchmark (two columns on
    the same dates). The small text is just the benchmark's own number: no difference is calculated.
    """
    kpi = analysis.kpi
    cells = [
        ("Annualized return", "annualized_return", format_percent),
        ("Annualized volatility", "annualized_volatility", format_percent),
        ("Sharpe ratio", "sharpe_ratio", format_ratio),
        ("Max drawdown (loss)", "max_drawdown", format_percent),
        (var_label(kpi), var_column(kpi), format_percent),
    ]
    for column, (label, field, as_text) in zip(st.columns(len(cells)), cells, strict=True):
        with column:
            st.metric(
                label,
                as_text(kpi.loc[PORTFOLIO, field]),
                delta=f"{analysis.benchmark}: {as_text(kpi.loc[analysis.benchmark, field])}",
                delta_color="off",  # grey: the benchmark is a reference, not a gain or a loss
                delta_arrow="off",
            )
    st.caption(
        f"The small grey text under each number is {analysis.benchmark} over the same dates. "
        "Max drawdown and VaR are losses, so 12.0% means a 12% loss (no minus sign)."
    )
    # Say which risk-free rate the Sharpe number used (formatting only, no math).
    st.caption(f"Sharpe uses a risk-free rate of {risk_free_rate:.1%}/yr.")


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #

def render_warnings(items: list[tuple[type, str]]) -> None:
    """Show the warnings risk.py raised: problems as warnings, gaps as softer notes."""
    for category, message in items:
        if (
            issubclass(category, risk.GapWarning)
            and LONG_GAP_PHRASE in message
            and message.startswith(PORTFOLIO_MESSAGE_PREFIX)
        ):
            # The portfolio held a holding at its last known price for longer than any holiday:
            # the holding's data is probably missing or stale.
            st.warning(
                "Possible missing data: a holding had no price for more than "
                f"{risk.MAX_HOLIDAY_GAP_ROWS} rows in a row, so the portfolio valued it at its last known "
                "price all that time. That is longer than any exchange holiday, so the data is probably "
                "missing or stale, and the portfolio may have ignored what the holding really did.\n\n"
                f"{message}"
            )
        elif issubclass(category, risk.GapWarning) and LONG_GAP_PHRASE in message:
            # risk.py found a returns gap longer than any exchange holiday: this is probably missing
            # data, so it must not look like the harmless holiday note below.
            st.warning(
                "Possible missing data: at least one gap is longer than "
                f"{risk.MAX_HOLIDAY_GAP_ROWS} rows, which is more than any exchange holiday. "
                f"Returns across it are multi-week moves counted as one day.\n\n{message}"
            )
        elif issubclass(category, risk.GapWarning) and message.startswith(PORTFOLIO_MESSAGE_PREFIX):
            st.info(
                "Gaps in the data: on some days a holding had no price, so the portfolio valued it "
                f"at its last known price. This is usual around exchange holidays.\n\n{message}"
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


# --------------------------------------------------------------------------- #
# Tab 1: Overview
# --------------------------------------------------------------------------- #

def chart_columns(analysis: Analysis, show_holdings: bool) -> list[str]:
    """Which lines a chart draws: the portfolio and the benchmark, plus the holdings on request.

    Columns that risk.py could not calculate are left out (there is nothing to draw for them).
    """
    wanted = [PORTFOLIO, analysis.benchmark] + (analysis.holdings if show_holdings else [])
    return [name for name in wanted if name in analysis.usable]


def colors_for(analysis: Analysis, names: list[str]) -> list[str]:
    return series_colors(names, PORTFOLIO, analysis.benchmark)


def render_overview(dataset: Dataset, analysis: Analysis, controls: Controls) -> None:
    """Growth of $1 for the portfolio against the benchmark, then the holdings and their weights."""
    first, last = analysis.value.index[0], analysis.value.index[-1]
    st.subheader("Growth of $1")
    st.caption(
        f"Period: {dataset.period}. The portfolio starts on {first:%Y-%m-%d} and runs to {last:%Y-%m-%d} "
        f"({len(analysis.value):,} trading days)."
    )
    if first > dataset.prices.index[0]:
        st.caption(
            f"That is later than the first date loaded ({dataset.prices.index[0]:%Y-%m-%d}): "
            "a holding had no price before the start date, and every number on this page covers the "
            "portfolio's dates only."
        )
    if analysis.growth is None:
        render_chart_problem(analysis, "growth")
    else:
        names = chart_columns(analysis, controls.show_holdings)
        st.caption("1.25 means +25% since the first day. Every line starts at 1. Buy-and-hold: never rebalanced.")
        if analysis.benchmark not in names:
            st.caption(f"No line for {analysis.benchmark}: it could not be calculated (see the warnings above).")
        st.altair_chart(
            line_chart(analysis.growth[names], "Growth", "Growth of $1", colors=colors_for(analysis, names)),
            width="stretch",
        )

    st.subheader("Holdings and starting weights")
    show_table(style_weights(dataset.weights), width="content")
    st.caption(
        f"Benchmark: {analysis.benchmark}. The weights apply on the first day only. "
        "A winner grows into a bigger share of the portfolio, because nothing is ever sold or bought again."
    )


# --------------------------------------------------------------------------- #
# Tab 2: Risk
# --------------------------------------------------------------------------- #

def render_drawdown(analysis: Analysis, controls: Controls) -> None:
    """How far below its previous peak the portfolio (and the benchmark) is on each day."""
    st.subheader("Drawdown (loss from the previous peak)")
    if analysis.drawdown is None:
        render_chart_problem(analysis, "drawdown")
        return
    names = chart_columns(analysis, controls.show_holdings)
    st.caption("A loss, shown as a positive number: 20% means 20% below the earlier high. Deeper falls hang lower.")
    st.altair_chart(
        line_chart(
            analysis.drawdown[names],
            "Drawdown",
            "Loss from the previous peak",
            y_axis_format=".0%",
            colors=colors_for(analysis, names),
            upside_down=True,
        ),
        width="stretch",
    )


def render_rolling_volatility(analysis: Analysis, controls: Controls) -> None:
    """Annualized volatility over a trailing window, to see how risk changes over time."""
    window = controls.window
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
    names = chart_columns(analysis, controls.show_holdings)
    shown = analysis.rolling_vol[names]
    # A ticker with fewer returns than the window has no value at all (every cell is empty).
    # This only looks for empty columns; it calculates nothing.
    no_line = [name for name in shown.columns if shown[name].isna().all()]
    if to_long(shown, "Volatility").empty:
        st.info(
            f"No ticker has enough returns to fill one {window}-day window yet. "
            "Try a shorter window or a longer date range."
        )
        return
    st.caption("Annualized. Each line starts once a full window of returns exists.")
    if no_line:
        st.caption(f"No line for {', '.join(no_line)}: fewer than {window} returns in the range.")
    st.altair_chart(
        line_chart(
            shown, "Volatility", "Annualized volatility", y_axis_format=".0%", colors=colors_for(analysis, names)
        ),
        width="stretch",
    )


def render_risk(analysis: Analysis, controls: Controls) -> None:
    render_drawdown(analysis, controls)
    render_rolling_volatility(analysis, controls)


# --------------------------------------------------------------------------- #
# Tab 3: Correlation
# --------------------------------------------------------------------------- #

def render_correlation(analysis: Analysis) -> None:
    """Heatmap of how closely the holdings' (and the benchmark's) daily returns move together."""
    st.subheader("Correlation of daily returns")
    if "correlation" in analysis.problems:
        render_chart_problem(analysis, "correlation")
        return
    if analysis.correlation is None:
        st.info(
            "A correlation needs at least 2 tickers (the holdings and the benchmark count). "
            "Fewer than 2 could be calculated, so there is nothing to compare."
        )
        return
    st.caption("+1 = move together, 0 = unrelated, -1 = move in opposite directions.")
    st.altair_chart(correlation_chart(analysis.correlation), width="stretch")


# --------------------------------------------------------------------------- #
# Tab 4: Holdings (the metrics table)
# --------------------------------------------------------------------------- #

def render_metrics(analysis: Analysis) -> None:
    """The table of headline risk numbers: one row per holding, then the portfolio and the benchmark."""
    st.subheader("Key metrics")
    table = metrics_for_display(analysis.summary)
    right = "right"  # numbers read better right-aligned
    show_table(
        style_metrics(table),
        width="stretch",
        column_config={
            "Annualized return": st.column_config.Column(
                help="Steady yearly growth rate (geometric, a year = 252 trading days).", alignment=right
            ),
            "Annualized volatility": st.column_config.Column(
                help="How much daily returns jump around, scaled to one year.", alignment=right
            ),
            "Sharpe ratio": st.column_config.Column(
                help="Return above the risk-free rate per unit of volatility.", alignment=right
            ),
            "Max drawdown (loss)": st.column_config.Column(
                help="Biggest fall from an earlier peak. A loss, shown as a positive number.", alignment=right
            ),
            next(c for c in table.columns if c.startswith("1-day VaR")): st.column_config.Column(
                help=(
                    "Historical Value at Risk: a one-day loss that was not exceeded on "
                    f"{CONFIDENCE:.0%} of past days. A loss, shown as a positive number. "
                    "Negative means there was no loss at this level."
                ),
                alignment=right,
            ),
            "Daily returns (n)": st.column_config.Column(
                help="How many daily returns the numbers are based on.", alignment=right
            ),
            "Note": st.column_config.TextColumn(
                width="large", help="Why a row's numbers should be read with care (a short sample, or a skipped ticker)."
            ),
        },
    )
    caption = (
        f"Every row covers the same dates as the portfolio. Portfolio is the buy-and-hold value of the "
        f"holdings; {analysis.benchmark} is the benchmark. Max drawdown and VaR are losses, so 12.0% means a "
        "12% loss (no minus sign). Sharpe uses the risk-free rate set in the sidebar."
    )
    if (table["Note"].str.startswith("⚠")).any():
        caption += (
            f" Greyed rows marked ⚠ have fewer than {risk.MIN_RELIABLE_OBS} daily returns "
            "(about 3 months). Stretching so few days to a year can give extreme numbers, "
            "so do not trust them."
        )
    st.caption(caption)


# --------------------------------------------------------------------------- #
# Tab 5: Data
# --------------------------------------------------------------------------- #

def render_data(dataset: Dataset) -> None:
    """The raw facts about the data: gaps, missing days, where it came from."""
    prices = dataset.prices
    st.subheader("Data quality")
    st.write(
        f"First date in the data: **{prices.index[0]:%Y-%m-%d}**. "
        f"Last date: **{prices.index[-1]:%Y-%m-%d}**. "
        f"Rows (trading days): **{len(prices):,}**."
    )

    st.markdown("**Missing prices per ticker** (empty cells, not whole missing days)")
    show_table(style_gaps(data.find_gaps(prices)), width="stretch")

    st.markdown("**Possible gaps (most are exchange holidays)**: weekdays with no row at all")
    try:
        # The same start and end that were asked for, so the check covers the whole requested window.
        missing = data.find_missing_days(prices, dataset.window_start, dataset.window_end_exclusive)
    except ValueError as error:
        st.warning(f"Could not check for missing days: {error}")
    else:
        if len(missing) == 0:
            st.write("None.")
        else:
            show_table(
                pd.DataFrame({"Date": missing.strftime("%Y-%m-%d"), "Weekday": missing.strftime("%A")}),
                hide_index=True,
                width="stretch",
            )

    st.subheader("Data source")
    st.write(f"**Source:** {dataset.source}.")
    st.write("Prices are adjusted closes (corrected for splits and dividends), one per trading day.")
    for note in dataset.notes:
        st.write(note)


# --------------------------------------------------------------------------- #
# Tab 6: About
# --------------------------------------------------------------------------- #

def render_about() -> None:
    """How the project was built, how the numbers are defined, and the disclaimer."""
    st.subheader("What this is")
    st.write(
        "SignalDesk is a small portfolio risk dashboard in the spirit of a very simple Bloomberg Terminal. "
        "Every number you see is calculated by tested Python functions in `signaldesk/risk.py`. "
        "This page only draws them: it does no math of its own, and tests check that."
    )

    st.subheader("How it was built")
    st.write(
        "I did not write the code by hand. I designed a small team of AI agents and gave each one a role: "
        "a supervisor that plans and delegates, builders for the data layer, the risk math and this page, "
        "and an independent reviewer that can only read and report. The reviewer runs on a different model "
        "from the builders, re-checks numbers by hand, tries the code on real data, and plants bugs on "
        "purpose to see whether the tests notice. Nothing is committed before the reviewer has gone through it."
    )
    st.markdown(
        f"Every mistake the agents caught is written up, with the evidence and the fix, in the "
        f"[catch log]({CATCH_LOG_URL}). The code and the project rules are on [GitHub]({GITHUB_URL})."
    )

    st.subheader("How the numbers are defined")
    st.markdown(
        "- Returns are simple daily returns. A year has 252 trading days.\n"
        "- The portfolio is buy-and-hold: the weights apply on the first day and nothing is rebalanced. "
        "It starts on the first day every holding has a price.\n"
        "- Volatility is the standard deviation of daily returns, scaled to one year. "
        "The Sharpe ratio uses the risk-free rate from the sidebar.\n"
        "- Max drawdown and Value at Risk are losses, shown as positive numbers. "
        "VaR is historical: the one-day loss not exceeded on 95% of past days.\n"
        "- Statistics from fewer than 60 daily returns are flagged: stretching a few days to a year gives "
        "numbers that look precise and are not."
    )

    st.subheader("Disclaimer")
    st.write(
        "This is a learning project. It is not investment advice, and past performance does not predict "
        "future results. Prices come from Yahoo Finance, are end-of-day, and can contain errors."
    )


# --------------------------------------------------------------------------- #
# The whole page
# --------------------------------------------------------------------------- #

def draw_safely(what: str, function, *args) -> None:
    """Draw one part of the page. If it fails, log it and say so in its place; the rest of the page goes on."""
    try:
        function(*args)
    except Exception:  # a bug while drawing must not take the other parts (or the footer) down with it
        log.exception("Unexpected error while drawing the %s", what)
        st.error(f"Something unexpected went wrong while drawing the {what}. The rest of the page is not affected.")


def render_page(dataset: Dataset, controls: Controls) -> None:
    """Analyse the data set once, then draw the page top to bottom."""
    render_title(dataset.as_of_text)
    try:
        analysis = analyze(
            dataset.prices, dataset.weights, dataset.benchmark, controls.window, controls.risk_free_rate
        )
    except ValueError as error:  # for example no date on which every holding has a price
        st.error(f"Could not calculate the risk numbers: {error}")
        if dataset.mode == inputs.MODE_CUSTOM:
            offer_demo()
        render_footer()
        return
    except Exception:  # a bug in the calculation: log it, and keep the page (title, way back, footer)
        log.exception("Unexpected error while calculating the risk numbers")
        st.error("Something unexpected went wrong while calculating the risk numbers. Please try again.")
        if dataset.mode == inputs.MODE_CUSTOM:
            offer_demo()
        render_footer()
        return

    # Each part is drawn on its own: a bug in one table or chart is reported in its own place, and the
    # other tabs, the key numbers and the footer are still there.
    draw_safely("key numbers", render_kpis, analysis, controls.risk_free_rate)
    draw_safely("warnings", render_warnings, analysis.warnings)

    overview, risk_tab, correlation, holdings, data_tab, about = st.tabs(TAB_NAMES)
    with overview:
        draw_safely("overview", render_overview, dataset, analysis, controls)
    with risk_tab:
        draw_safely("risk charts", render_risk, analysis, controls)
    with correlation:
        draw_safely("correlation chart", render_correlation, analysis)
    with holdings:
        draw_safely("metrics table", render_metrics, analysis)
    with data_tab:
        draw_safely("data tab", render_data, dataset)
    with about:
        draw_safely("about tab", render_about)
    render_footer()
