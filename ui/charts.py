"""Altair charts (Altair comes with Streamlit). They only DRAW numbers that risk.py calculated.

A chart must not calculate anything: no Vega-Lite transform, no aggregate, no expression that
changes a value. tests/test_ui_guards.py reads every chart's specification to make sure.
The one allowed expression picks a text colour in the heatmap (white on dark squares).
"""

from __future__ import annotations

import itertools

import altair as alt
import pandas as pd

# Colours: the portfolio is the deep blue of the page theme, the benchmark a warm grey-orange,
# and the holdings (shown only on request) get softer colours.
PORTFOLIO_COLOR = "#1F4E8C"
BENCHMARK_COLOR = "#E08A1E"
HOLDING_COLORS = [
    "#6BA3D6", "#8DB580", "#B58DB6", "#D6A66B", "#7FBFBF", "#C98A8A", "#9AA5B1", "#B5B56B", "#6F7FBF", "#BF8F6F",
]


def series_colors(names: list[str], portfolio: str, benchmark: str) -> list[str]:
    """One colour per line, in the order of ``names``: portfolio blue, benchmark orange, others soft."""
    palette = itertools.cycle(HOLDING_COLORS)  # repeats if there are more holdings than colours
    colors = []
    for name in names:
        if name == portfolio:
            colors.append(PORTFOLIO_COLOR)
        elif name == benchmark:
            colors.append(BENCHMARK_COLOR)
        else:
            colors.append(next(palette))
    return colors


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
    frame: pd.DataFrame,
    value_name: str,
    y_title: str,
    y_axis_format: str | None = None,
    colors: list[str] | None = None,
    upside_down: bool = False,
) -> alt.Chart:
    """One coloured line per column. ``y_axis_format`` is a display format, e.g. '%'.

    ``colors`` (one per column, in column order) fixes the colour of each line; without it
    Altair picks. ``upside_down`` flips the y axis for the drawdown chart, so a deeper loss hangs
    lower on the chart. That is a drawing choice: the numbers stay positive.
    """
    long = to_long(frame, value_name)
    tickers = list(frame.columns)  # keeps the colours and legend in the order of the columns
    if colors is None:
        color_scale = alt.Scale()
    else:
        color_scale = alt.Scale(domain=tickers, range=colors)
    return (
        alt.Chart(long)
        # Invisible dots on every point make the hover tooltip work along the whole line.
        .mark_line(point=alt.OverlayMarkDef(opacity=0))
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(
                f"{value_name}:Q",
                title=y_title,
                scale=alt.Scale(zero=upside_down, reverse=upside_down),
                axis=alt.Axis(format=y_axis_format) if y_axis_format else alt.Axis(),
            ),
            color=alt.Color(
                "Ticker:N", sort=tickers, scale=color_scale, legend=alt.Legend(orient="bottom", title=None)
            ),
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
    height = 64 * len(tickers)  # not math: layout (a taller chart for more tickers)
    if height < 240:
        height = 240
    return alt.layer(squares, numbers).properties(height=height)
