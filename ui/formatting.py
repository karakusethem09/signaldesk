"""Numbers and tables as text for people. Display only: no value is ever changed.

For example 0.256 becomes "25.6%" and a Sharpe ratio gets two decimals. The percent sign, the
thousands separators and the ">100,000%" cap are the only "calculations" here, and they are
formatting: Python's format language does them, the values underneath stay exactly what
risk.py gave.
"""

from __future__ import annotations

import pandas as pd

from signaldesk import risk
from ui.analysis import CONFIDENCE, skipped_rows

NO_VALUE = "—"  # shown for a number that does not exist (for example a skipped ticker)
PLACEHOLDER = NO_VALUE  # what st.dataframe draws for an empty cell (otherwise the browser writes "None")


def format_percent(value: float) -> str:
    """Show a fraction as a percent, e.g. 0.256 -> '25.6%' and 27.08 -> '2,708.1%'. Display only, nothing is changed.

    A huge value (more than 100,000%) comes from stretching a few days of data to a full
    year; we show '>100,000%' instead of a 60-digit number.
    """
    if pd.isna(value):
        return NO_VALUE
    if value >= 1000:
        return ">100,000%"
    return f"{value:,.1%}"


def format_ratio(value: float) -> str:
    """The Sharpe ratio is a plain number, not a percent: two decimals."""
    if pd.isna(value):
        return NO_VALUE
    return f"{value:.2f}"


def var_column(summary: pd.DataFrame) -> str:
    """The VaR column is named after the confidence (var_95, var_99, ...): find it by its prefix."""
    return next(name for name in summary.columns if name.startswith("var_"))


def var_label(summary: pd.DataFrame) -> str:
    """'1-day VaR 95% (loss)' for the column ``var_95``."""
    return f"1-day VaR {var_column(summary).removeprefix('var_')}% (loss)"


def describe_wait(seconds: float) -> str:
    """Seconds in plain words: 'less than a minute', 'about 12 minutes', 'about 1 hour'.

    The rounding up to whole minutes is done by pandas (``Timedelta.ceil``), not here.
    """
    if not seconds > 0:
        return "a moment"
    wait = pd.Timedelta(seconds=seconds).ceil("min")
    parts = wait.components
    if parts.days:
        return "more than a day"
    if parts.hours:
        unit = "hour" if parts.hours == 1 else "hours"
        extra = f" and {parts.minutes} minute{'s' if parts.minutes != 1 else ''}" if parts.minutes else ""
        return f"about {parts.hours} {unit}{extra}"
    if parts.minutes <= 1:
        return "less than a minute" if seconds < 60 else "about a minute"
    return f"about {parts.minutes} minutes"


def metrics_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Rename the summary's columns for people and add a 'Note' column. Values are untouched."""
    table = summary.rename(
        columns={
            "annualized_return": "Annualized return",
            "annualized_volatility": "Annualized volatility",
            "sharpe_ratio": "Sharpe ratio",
            "max_drawdown": "Max drawdown (loss)",
            var_column(summary): var_label(summary),
            "n_obs": "Daily returns (n)",
        }
    )

    skipped = skipped_rows(summary)
    short = (summary["n_obs"] < risk.MIN_RELIABLE_OBS) & ~skipped
    table["Note"] = ""
    table.loc[skipped, "Note"] = "skipped (see warning above)"
    table.loc[short, "Note"] = f"⚠ short sample (under {risk.MIN_RELIABLE_OBS} returns)"
    return table


def percent_columns(table: pd.DataFrame) -> list[str]:
    """The metric columns that are shown as percents (everything but Sharpe, the count and the note)."""
    return [
        name
        for name in table.columns
        if name in ("Annualized return", "Annualized volatility")
        or name.startswith(("Max drawdown", "1-day VaR"))
    ]


def style_metrics(table: pd.DataFrame) -> pd.io.formats.style.Styler:
    """The metrics table: the NUMBERS stay numbers, a Styler formats how they are shown.

    Why numbers and not text: a column of real numbers sorts by value when a visitor clicks its header
    (a column of texts sorts alphabetically, which puts "5.1%" above "29.3%"). An empty cell stays empty
    in the data, and ``PLACEHOLDER`` (passed to ``st.dataframe``) makes the browser draw a dash for it,
    instead of its own "None". Rows that carry a note are greyed out.
    """

    def grey_out(row: pd.Series) -> list[str]:
        return ["color: gray" if row["Note"] else ""] * len(row)

    return (
        table.style.format(format_percent, subset=percent_columns(table))
        .format(format_ratio, subset=["Sharpe ratio"])
        .format(format_count, subset=["Daily returns (n)"])
        .apply(grey_out, axis=1)
    )


def format_count(value: float) -> str:
    """A whole number with a thousands separator ("1,253"), or a dash when there is none."""
    if pd.isna(value):
        return NO_VALUE
    return f"{int(value):,}"


def format_date(value) -> str:
    """A date as 2026-01-31, or a dash when there is no date (NaT)."""
    if pd.isna(value):
        return NO_VALUE
    return f"{value:%Y-%m-%d}"


def style_weights(weights: dict[str, float]) -> pd.io.formats.style.Styler:
    """The holdings and their starting weights: numbers, shown as percentages (formatting only)."""
    table = pd.DataFrame({"Starting weight": list(weights.values())}, index=pd.Index(list(weights), name="Ticker"))
    return table.style.format(format_percent)


def style_gaps(gaps: pd.DataFrame) -> pd.io.formats.style.Styler:
    """``data.find_gaps`` for people: readable column names, counts and real dates (they sort by value).

    A day with no date stays empty (NaT) in the data and is drawn as a dash through ``PLACEHOLDER``.
    """
    table = gaps.rename(
        columns={"missing_count": "Missing prices", "first_missing": "First missing", "last_missing": "Last missing"}
    )
    return table.style.format(format_count, subset=["Missing prices"]).format(
        format_date, subset=["First missing", "Last missing"]
    )
