"""Turn what a visitor typed into a checked request. Plain Python: no Streamlit, no network.

The dashboard's custom mode lets a visitor type a holdings table (ticker + weight in percent),
a benchmark and a date range. Everything that can be wrong with that is checked HERE, before
anything is downloaded, and every problem becomes a short message in plain words (never a
Python error text such as ``np.float64(nan)``).

No math in this file: weights are rescaled by ``risk.normalize_weights`` and tickers are
cleaned by ``limits.normalize_tickers``. What happens here is presence and range checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from signaldesk import limits, risk

# --------------------------------------------------------------------------- #
# Names and constants
# --------------------------------------------------------------------------- #

MODE_DEMO = "Demo portfolio"
MODE_CUSTOM = "Custom (live data)"

CUSTOM_DATES = "Custom dates"
PRESET_YEARS = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}
DEMO_PRESETS = ("1Y", "3Y", "5Y")  # the demo snapshot holds 5 years
LIVE_PRESETS = ("1Y", "3Y", "5Y", "10Y")  # 10Y is allowed on the public site (its limit is 10 years)

TICKER_COLUMN = "Ticker"
WEIGHT_COLUMN = "Weight (%)"
MIN_WEIGHT_PERCENT = 0  # the editor's NumberColumn bounds; read_holdings checks them again
MAX_WEIGHT_PERCENT = 100
NEW_ROW_WEIGHT_PERCENT = 10  # what a freshly added row starts with
MAX_SYMBOL_CHARS = 15  # the longest ticker limits.normalize_tickers accepts

# The holdings editor starts with the demo portfolio, written out as percentages. It is a
# literal table on purpose (computing "weight * 100" here would be math). A test checks that it
# equals data/demo/demo_portfolio.json after risk.normalize_weights.
DEFAULT_HOLDINGS = [("AAPL", 20), ("MSFT", 20), ("NVDA", 15), ("JPM", 15), ("XOM", 15), ("JNJ", 15)]
DEFAULT_BENCHMARK = "SPY"


class InputProblem(ValueError):
    """Something the visitor typed cannot be used. The message is plain words, safe to show."""


@dataclass(frozen=True)
class LiveRequest:
    """A checked request for live prices."""

    weights: dict[str, float]  # ticker -> fraction, adds up to 1 (from risk.normalize_weights)
    benchmark: str
    tickers: list[str]  # the holdings, then the benchmark (clean, upper case, no repeats)
    start: date
    end_exclusive: date  # what load_prices wants: the day AFTER the last day shown


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

def end_exclusive_from(end: date) -> date:
    """The sidebar's end date is INCLUDED; ``load_prices`` excludes its end, so add one day.

    This is date handling, not math.
    """
    return end + timedelta(days=1)  # not math: date


def resolve_dates(label: str, custom_start, custom_end, today: date) -> tuple[date | None, date | None]:
    """The (first day, last day) a custom request covers. The last day is INCLUDED.

    A preset ("5Y") ends today (New York) and starts where ``limits.earliest_allowed_start`` says;
    "Custom dates" uses the two dates the visitor picked (either can be None if the box was cleared).
    """
    if label == CUSTOM_DATES:
        return custom_start, custom_end
    return preset_start(end_exclusive_from(today), label).date(), today


def preset_start(end_exclusive, label: str) -> pd.Timestamp:
    """The first day of a preset such as "5Y", counted by ``limits.earliest_allowed_start``.

    The same function the public-site limit uses, so a "10Y" preset is exactly as long as
    the longest range ``limits.check_request`` accepts.
    """
    return limits.earliest_allowed_start(end_exclusive, PRESET_YEARS[label])


# --------------------------------------------------------------------------- #
# The holdings table
# --------------------------------------------------------------------------- #

def _is_blank(cell) -> bool:
    """True for an empty cell: None, NaN or only spaces."""
    if cell is None:
        return True
    if isinstance(cell, str):
        return not cell.strip()
    return bool(pd.isna(cell))


def read_holdings(table: pd.DataFrame) -> dict[str, float]:
    """Read the editor's table into ``{TICKER: weight in percent}``, or raise InputProblem.

    Untouched empty rows (no ticker and no weight) are ignored. Everything else must be
    complete: a ticker AND a weight. Tickers are cleaned by ``limits.normalize_tickers`` (a bad
    one raises its LimitError, a message written for visitors). A repeated ticker is refused
    instead of silently overwritten. Weights must lie between 0 and 100, which together with
    the presence check keeps NaN, infinity and absurd numbers away from ``risk.normalize_weights``.
    """
    percent_by_symbol: dict[str, float] = {}
    for ticker_cell, weight_cell in zip(table[TICKER_COLUMN], table[WEIGHT_COLUMN], strict=True):
        no_ticker, no_weight = _is_blank(ticker_cell), _is_blank(weight_cell)
        if no_ticker and no_weight:
            continue  # an empty row the visitor added and never filled in
        if no_ticker:
            raise InputProblem("Fill in a ticker for every weight (or delete the row).")
        if no_weight:
            raise InputProblem("Fill in a weight for every ticker (or delete the row).")

        symbol = limits.normalize_tickers([str(ticker_cell)])[0]  # LimitError for a bad symbol
        try:
            weight = float(weight_cell)
        except (TypeError, ValueError):
            raise InputProblem(f"The weight of {symbol} must be a number.") from None
        if not MIN_WEIGHT_PERCENT <= weight <= MAX_WEIGHT_PERCENT:  # also False for NaN and infinity
            raise InputProblem(
                f"The weight of {symbol} must be between {MIN_WEIGHT_PERCENT} and {MAX_WEIGHT_PERCENT} (percent)."
            )
        if symbol in percent_by_symbol:
            raise InputProblem(f"{symbol} is listed more than once. Keep one row per ticker.")
        percent_by_symbol[symbol] = weight

    if not percent_by_symbol:
        raise InputProblem("Add at least one holding: a ticker and a weight.")
    return percent_by_symbol


def to_fractions(percent_by_symbol: dict[str, float]) -> dict[str, float]:
    """Percent weights -> weights that add up to 1, done by ``risk.normalize_weights`` (never here).

    The percentages do not have to add up to 100: only their proportions count.
    After the checks in ``read_holdings`` the only message left is "All the weights are zero".
    """
    try:
        return risk.normalize_weights(percent_by_symbol)
    except ValueError as error:
        raise InputProblem(str(error)) from None


def read_benchmark(text, holdings) -> str:
    """The benchmark ticker: exactly one valid symbol that is not also a holding."""
    if text is None or not str(text).strip():
        raise InputProblem("Enter a benchmark ticker, for example SPY.")
    symbols = limits.normalize_tickers(str(text))  # LimitError for a bad symbol
    if len(symbols) != 1:
        raise InputProblem("Enter just one benchmark ticker.")
    if symbols[0] in holdings:
        raise InputProblem(f"The benchmark {symbols[0]} is also one of your holdings. Pick a different benchmark.")
    return symbols[0]


def build_live_request(
    table: pd.DataFrame,
    benchmark_text,
    start: date | None,
    end: date | None,
    request_limits: limits.RequestLimits,
) -> LiveRequest:
    """Check everything about a custom request BEFORE any download. ``end`` is INCLUDED.

    Raises InputProblem (plain-words messages) or limits.LimitError (the public-site limits).
    ``limits.normalize_tickers`` is called here in every mode, because in local mode
    ``check_request`` skips the symbol checks.
    """
    if start is None or end is None:
        raise InputProblem("Fill in the start date and the end date.")
    if start >= end:
        raise InputProblem("The start date must be before the end date.")

    percent_by_symbol = read_holdings(table)
    benchmark = read_benchmark(benchmark_text, list(percent_by_symbol))
    weights = to_fractions(percent_by_symbol)

    tickers = limits.normalize_tickers(list(weights) + [benchmark])
    end_exclusive = end_exclusive_from(end)
    limits.check_request(tickers, start, end_exclusive, request_limits)
    return LiveRequest(weights=weights, benchmark=benchmark, tickers=tickers, start=start, end_exclusive=end_exclusive)
