"""The sidebar, the data sets behind the page, and Streamlit's shared (per-process) objects.

``collect_inputs`` is the one function the page calls. It draws the sidebar and returns
    - a ``Dataset``: the prices, the weights and the benchmark to analyse, and
    - ``Controls``: the volatility window and the risk-free rate.
If there is nothing to show (the demo files are broken, a custom request was refused, or the
visitor has not pressed "Run analysis" yet) it raises ``Unavailable`` with a message in plain
words, and the page shows that message instead.

No math here: dates are picked with ``limits.earliest_allowed_start``, weights are rescaled
by ``risk.normalize_weights`` (in ``ui/inputs.py``), and all numbers come from risk.py later.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from signaldesk import data, limits
from ui import inputs, live
from ui.formatting import describe_wait

log = logging.getLogger(__name__)

DEFAULT_WINDOW = 60  # default rolling-volatility window, in trading days
MIN_WINDOW, MAX_WINDOW = 20, 252  # 20 is the smallest window risk.py does not warn about
MIN_RISK_FREE_RATE, MAX_RISK_FREE_RATE = -0.05, 0.20  # a year's rate as a fraction: -5% to 20%
EARLIEST_DATE = date(1990, 1, 1)  # the oldest date the pickers offer

SCOPE_NOTE = (
    "Built for stocks and ETFs only (a year = 252 trading days). Crypto trades 7 days a week, "
    "so its annualized numbers come out wrong. Mixing exchanges (say US and Japan) is fine, "
    "but expect a gap notice because the holidays differ."
)

# Widget and session keys (tests find the widgets by these names).
KEY_MODE = "mode"
KEY_DEMO_PERIOD, KEY_DEMO_START, KEY_DEMO_END = "demo_period", "demo_start", "demo_end"
KEY_FORM = "custom_form"
KEY_CUSTOM_PERIOD, KEY_CUSTOM_START, KEY_CUSTOM_END = "custom_period", "custom_start", "custom_end"
KEY_EDITOR, KEY_BENCHMARK, KEY_RUN = "holdings_editor", "benchmark", "run_analysis"
KEY_WINDOW, KEY_RISK_FREE, KEY_SHOW_HOLDINGS = "window", "risk_free_rate", "show_holdings"
KEY_LIVE_DATASET, KEY_LIVE_NOTICE, KEY_VISITOR = "live_dataset", "live_notice", "visitor_id"


# --------------------------------------------------------------------------- #
# What the sidebar hands to the page
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Controls:
    """The sidebar settings that do not change which prices are loaded."""

    window: int  # rolling-volatility window in trading days
    risk_free_rate: float  # annual FRACTION, 0.04 = 4%
    show_holdings: bool  # also draw one line per holding on the charts


@dataclass(frozen=True)
class Dataset:
    """Everything the page needs to analyse and show one portfolio (demo or custom)."""

    mode: str
    title: str
    prices: pd.DataFrame  # holding columns, then the benchmark; only the chosen period
    weights: dict[str, float]  # ticker -> fraction (adds up to 1)
    benchmark: str
    period: str  # "5Y", "Custom dates", ...
    window_start: object  # the start we asked for; handed to data.find_missing_days
    window_end_exclusive: object  # the (excluded) end we asked for
    as_of_text: str  # the header line, e.g. "Demo data as of 2026-10-19"
    source: str  # where the prices came from
    notes: tuple[str, ...]  # extra lines for the Data tab (snapshot info, cache note)


class Unavailable(Exception):
    """There is nothing to show right now. ``kind`` is "info", "warning" or "error"."""

    def __init__(self, message: str, kind: str = "error", offer_demo: bool = False):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.offer_demo = offer_demo  # show a "Back to the demo" button


# --------------------------------------------------------------------------- #
# Shared objects: ONE of each per server process, whatever the number of visitors
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def cached_demo() -> data.DemoData:
    """The committed demo snapshot, read once and kept in memory (no network involved)."""
    return data.load_demo()


@st.cache_resource
def shared_price_cache() -> live.PriceCache:
    """The 6-hour memory of live price tables, shared by all visitors."""
    return live.PriceCache()


@st.cache_resource
def shared_rate_limiter(max_events: int | None) -> limits.RateLimiter:
    """The live-analyses-per-hour limit, counted for each visitor. One limiter for the whole process."""
    return limits.RateLimiter(max_events, live.RATE_WINDOW_SECONDS)


@st.cache_resource
def shared_site_rate_limiter(max_events: int | None) -> limits.RateLimiter:
    """The live-analyses-per-hour limit for the WHOLE SITE (all visitors together), one per server process.

    A visitor who reloads the page gets a fresh personal allowance, but not a fresh site allowance.
    """
    return limits.RateLimiter(max_events, live.RATE_WINDOW_SECONDS)


@st.cache_resource
def shared_download_gate(max_concurrent: int | None) -> limits.DownloadGate:
    """The limit on downloads running at the same moment, counted over all visitors."""
    return limits.DownloadGate(max_concurrent)


def visitor_id() -> str:
    """A random id for this browser session, used as the rate-limit key.

    A page reload starts a new session and so a new id: the limit is per session, not per person.
    """
    if KEY_VISITOR not in st.session_state:
        st.session_state[KEY_VISITOR] = uuid.uuid4().hex
    return st.session_state[KEY_VISITOR]


# --------------------------------------------------------------------------- #
# Sidebar: mode
# --------------------------------------------------------------------------- #

def go_to_demo() -> None:
    """Button callback for "Back to the demo": switch the mode radio back."""
    st.session_state[KEY_MODE] = inputs.MODE_DEMO


def in_custom_mode() -> bool:
    """True when the sidebar's mode switch is on Custom (used to offer "Back to the demo")."""
    return st.session_state.get(KEY_MODE) == inputs.MODE_CUSTOM


def forget_live_data() -> None:
    """Drop the last custom result, so it cannot reappear under a form that was reset."""
    st.session_state.pop(KEY_LIVE_DATASET, None)
    st.session_state.pop(KEY_LIVE_NOTICE, None)


def choose_mode() -> str:
    """The Demo / Custom switch. Demo is the default."""
    st.sidebar.header("Settings")
    return st.sidebar.radio(
        "Mode",
        [inputs.MODE_DEMO, inputs.MODE_CUSTOM],
        key=KEY_MODE,
        help="The demo portfolio works offline from a saved snapshot. Custom mode downloads live prices.",
    )


# --------------------------------------------------------------------------- #
# Demo mode
# --------------------------------------------------------------------------- #

def load_demo_or_raise() -> data.DemoData:
    """The demo snapshot, or ``Unavailable`` with a friendly message if its files are broken."""
    try:
        return cached_demo()
    except ValueError as error:  # load_demo's own message says which file is wrong: safe to show
        raise Unavailable(
            "The demo portfolio could not be loaded, so there is nothing to show in this mode. "
            f"{error} The Custom (live data) mode in the sidebar may still work.",
            kind="error",
        ) from error
    except Exception as error:  # anything else (a permission problem, a bug): log it, show a plain message
        log.exception("Unexpected error while loading the demo snapshot")
        raise Unavailable(
            "The demo portfolio could not be loaded because of an unexpected problem, so there is nothing "
            "to show in this mode. The Custom (live data) mode in the sidebar may still work.",
            kind="error",
        ) from error


def demo_period(demo: data.DemoData) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    """Draw the period picker; return (label, first day, day AFTER the last day) for the snapshot.

    The last day is the snapshot's last date. The presets count back from it with
    ``limits.earliest_allowed_start``; "Custom dates" lets the visitor pick days inside the snapshot.
    """
    end_exclusive = inputs.end_exclusive_from(demo.as_of)
    label = st.sidebar.radio(
        "Period",
        [*inputs.DEMO_PRESETS, inputs.CUSTOM_DATES],
        index=inputs.DEMO_PRESETS.index("5Y"),
        horizontal=True,
        key=KEY_DEMO_PERIOD,
    )
    if label != inputs.CUSTOM_DATES:
        return label, inputs.preset_start(end_exclusive, label), end_exclusive

    first, last = demo.start.date(), demo.as_of.date()
    default_start = inputs.preset_start(end_exclusive, "3Y").date()
    if default_start < first:  # the 3-year start may precede the snapshot: begin at the snapshot's first day
        default_start = first
    start = st.sidebar.date_input(
        "Start date", value=default_start, min_value=first, max_value=last, key=KEY_DEMO_START
    )
    end = st.sidebar.date_input(
        "End date (included)", value=last, min_value=first, max_value=last, key=KEY_DEMO_END
    )
    if start is None or end is None:
        raise Unavailable("Fill in the start date and the end date in the sidebar.", kind="info")
    return label, pd.Timestamp(start), pd.Timestamp(inputs.end_exclusive_from(end))


def demo_dataset(demo: data.DemoData, period: tuple[str, pd.Timestamp, pd.Timestamp]) -> Dataset:
    """The demo prices for the chosen period. Choosing rows by date is not math."""
    label, start, end_exclusive = period
    rows = demo.prices.loc[(demo.prices.index >= start) & (demo.prices.index < end_exclusive)]
    if len(rows) < 2:
        raise Unavailable(
            "The chosen dates hold fewer than 2 trading days of the demo snapshot. Pick a wider period.",
            kind="warning",
        )
    # A preset can reach back before the snapshot's first day (5Y starts a day earlier). The data
    # cannot hold those days, so the missing-days check starts at the snapshot's first day.
    window_start = start if start >= demo.start else demo.start
    return Dataset(
        mode=inputs.MODE_DEMO,
        title=demo.name,
        prices=rows,
        weights=dict(demo.holdings),
        benchmark=demo.benchmark,
        period=label,
        window_start=window_start,
        window_end_exclusive=end_exclusive,
        as_of_text=f"Demo data as of {demo.as_of:%Y-%m-%d}",
        source=demo.source,
        notes=(
            f"Demo snapshot: {len(demo.prices):,} trading days, {demo.start:%Y-%m-%d} to {demo.as_of:%Y-%m-%d}. "
            "It is saved in the repository, so this view works even when Yahoo Finance is down.",
        ),
    )


# --------------------------------------------------------------------------- #
# Custom mode (live data)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Submission:
    """What the visitor sent with "Run analysis"."""

    table: pd.DataFrame  # the holdings editor's content
    benchmark_text: str
    period: str
    start: date | None
    end: date | None  # INCLUDED


def default_holdings_table() -> pd.DataFrame:
    """The editor's starting content: the demo portfolio, in percent."""
    return pd.DataFrame(
        {
            inputs.TICKER_COLUMN: [ticker for ticker, _ in inputs.DEFAULT_HOLDINGS],
            inputs.WEIGHT_COLUMN: [float(percent) for _, percent in inputs.DEFAULT_HOLDINGS],
        }
    )


def custom_form() -> Submission | None:
    """Draw the custom form. Typing inside it does nothing until "Run analysis" is pressed."""
    today = data.market_today()
    request_limits = limits.active_limits()
    with st.sidebar.form(KEY_FORM):
        label = st.radio(
            "Period",
            [*inputs.LIVE_PRESETS, inputs.CUSTOM_DATES],
            index=inputs.LIVE_PRESETS.index("5Y"),
            horizontal=True,
            key=KEY_CUSTOM_PERIOD,
        )
        start = st.date_input(
            "Start date (for Custom dates)",
            value=inputs.preset_start(inputs.end_exclusive_from(today), "3Y").date(),
            min_value=EARLIEST_DATE,
            max_value=today,
            key=KEY_CUSTOM_START,
        )
        end = st.date_input(
            "End date, included (for Custom dates)",
            value=today,
            min_value=EARLIEST_DATE,
            max_value=today,
            key=KEY_CUSTOM_END,
        )
        st.markdown("**Holdings**")
        table = st.data_editor(
            default_holdings_table(),
            key=KEY_EDITOR,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                inputs.TICKER_COLUMN: st.column_config.TextColumn(
                    inputs.TICKER_COLUMN,
                    max_chars=inputs.MAX_SYMBOL_CHARS,
                    required=True,
                    help="A Yahoo Finance symbol, for example AAPL, BRK-B or 7203.T.",
                ),
                inputs.WEIGHT_COLUMN: st.column_config.NumberColumn(
                    inputs.WEIGHT_COLUMN,
                    min_value=inputs.MIN_WEIGHT_PERCENT,
                    max_value=inputs.MAX_WEIGHT_PERCENT,
                    step=0.5,
                    default=inputs.NEW_ROW_WEIGHT_PERCENT,
                    format="%.1f",
                    required=True,
                    help="Percent of the money put into this ticker on the first day.",
                ),
            },
        )
        if request_limits.max_tickers is None:
            st.caption("Weights are rescaled to add up to 100%. There is no ticker limit on your own computer.")
        else:
            st.caption(
                "Weights are rescaled to add up to 100%. Up to "
                f"{request_limits.max_tickers} tickers in total, benchmark included."
            )
        benchmark = st.text_input("Benchmark", value=inputs.DEFAULT_BENCHMARK, max_chars=inputs.MAX_SYMBOL_CHARS, key=KEY_BENCHMARK)
        pressed = st.form_submit_button("Run analysis", key=KEY_RUN, type="primary")
    st.sidebar.caption(SCOPE_NOTE)

    if not pressed:
        return None
    first, last = inputs.resolve_dates(label, start, end, today)
    return Submission(table=table, benchmark_text=benchmark, period=label, start=first, end=last)


def remember_notice(kind: str, message: str) -> None:
    """Keep a message for the main area; it stays until the next "Run analysis"."""
    st.session_state[KEY_LIVE_NOTICE] = (kind, message)


def run_custom_analysis(submission: Submission) -> None:
    """Check the request, load the prices, and keep the result (or the reason it failed) in the session."""
    forget_live_data()
    request_limits = limits.active_limits()
    try:
        request = inputs.build_live_request(
            submission.table, submission.benchmark_text, submission.start, submission.end, request_limits
        )
        with st.spinner("Loading prices from Yahoo Finance..."):
            loaded = live.load_live_prices(
                request,
                visitor=visitor_id(),
                cache=shared_price_cache(),
                limiter=shared_rate_limiter(request_limits.max_live_loads_per_hour),
                site_limiter=shared_site_rate_limiter(request_limits.max_site_live_loads_per_hour),
                gate=shared_download_gate(request_limits.max_concurrent_downloads),
                gate_timeout=live.GATE_TIMEOUT_SECONDS,
            )
    except (limits.LimitError, inputs.InputProblem) as error:  # a refused request: the text is for visitors
        remember_notice("warning", str(error))
    except live.SiteRateLimited as error:  # (before RateLimited: it is a kind of RateLimited)
        remember_notice(
            "warning",
            "The site has reached its hourly limit of live analyses. "
            f"Please try again in {describe_wait(error.retry_after)}; the demo still works.",
        )
    except live.RateLimited as error:
        remember_notice(
            "warning",
            f"You have used your {request_limits.max_live_loads_per_hour} live analyses for this hour. "
            f"Please try again in {describe_wait(error.retry_after)}, or look at the demo portfolio meanwhile.",
        )
    except data.ProviderUnavailableError as error:  # the message names every possible cause: show it as it is
        remember_notice("error", str(error))
    except data.PriceDataError as error:  # for example one ticker without data
        remember_notice("error", f"Could not load prices: {error}")
    except ValueError as error:  # bad input, for example an invalid date
        remember_notice("error", f"Could not load prices (bad input): {error}")
    except Exception:  # anything else must not blank the page: log it for the server's owner
        log.exception("Unexpected error while loading live prices")
        remember_notice("error", "Something unexpected went wrong while loading prices. Please try again, or use the demo.")
    else:
        prices = loaded.prices
        st.session_state[KEY_LIVE_DATASET] = Dataset(
            mode=inputs.MODE_CUSTOM,
            title="Custom portfolio",
            prices=prices,
            weights=request.weights,
            benchmark=request.benchmark,
            period=submission.period,
            window_start=request.start,
            window_end_exclusive=request.end_exclusive,
            as_of_text=f"Live data, last close {prices.index[-1]:%Y-%m-%d}",
            source="Yahoo Finance via yfinance, adjusted close, end-of-day",
            notes=(
                "Reused from this server's short-term memory (at most 6 hours old): nothing was downloaded."
                if loaded.from_cache
                else "Loaded just now through the data layer: a fresh Yahoo Finance download, or its one-day cache on this server.",
            ),
        )


def current_live_dataset() -> Dataset:
    """The last custom result, or ``Unavailable`` saying why there is none."""
    dataset = st.session_state.get(KEY_LIVE_DATASET)
    if dataset is not None:
        return dataset
    notice = st.session_state.get(KEY_LIVE_NOTICE)
    if notice is not None:
        kind, message = notice
        raise Unavailable(message, kind=kind, offer_demo=True)
    raise Unavailable(
        "Set your holdings in the sidebar and press Run analysis. Nothing is downloaded until you do.",
        kind="info",
        offer_demo=True,
    )


# --------------------------------------------------------------------------- #
# Settings that never need a download
# --------------------------------------------------------------------------- #

def common_controls() -> Controls:
    """Window, risk-free rate and the "show each holding" switch."""
    window = st.sidebar.slider(
        "Rolling volatility window (trading days)",
        min_value=MIN_WINDOW,
        max_value=MAX_WINDOW,
        value=DEFAULT_WINDOW,
        key=KEY_WINDOW,
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
        key=KEY_RISK_FREE,
    )
    show_holdings = st.sidebar.toggle(
        "Show each holding on the charts", value=False, key=KEY_SHOW_HOLDINGS
    )
    # A cleared number box gives None. Say what to do instead of crashing.
    if risk_free_rate is None:
        raise Unavailable("Fill in the risk-free rate in the sidebar.", kind="info")
    return Controls(window=int(window), risk_free_rate=float(risk_free_rate), show_holdings=bool(show_holdings))


# --------------------------------------------------------------------------- #
# The one function the page calls
# --------------------------------------------------------------------------- #

def collect_inputs() -> tuple[Dataset, Controls]:
    """Draw the sidebar and return what to analyse. Raises ``Unavailable`` if there is nothing to show."""
    mode = choose_mode()
    if mode == inputs.MODE_DEMO:
        forget_live_data()
        demo = load_demo_or_raise()
        period = demo_period(demo)
        controls = common_controls()
        return demo_dataset(demo, period), controls

    submission = custom_form()
    if submission is not None:
        run_custom_analysis(submission)
    controls = common_controls()
    return current_live_dataset(), controls
