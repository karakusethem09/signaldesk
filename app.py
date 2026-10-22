"""SignalDesk dashboard (Phase 1.5): a portfolio risk page, for recruiters and for my own analysis.

This file is only the entry point. The page itself lives in the ``ui`` package, and it does no
financial math at all: every price comes from ``signaldesk/data.py``, every number from
``signaldesk/risk.py`` and every request limit from ``signaldesk/limits.py``.

How to run it:
    .venv/bin/streamlit run app.py                       # public behaviour: strict request limits
    SIGNALDESK_MODE=local .venv/bin/streamlit run app.py # your own computer: no limits

Two modes (the radio at the top of the sidebar):
    Demo portfolio       the default; reads a committed snapshot (data/demo/), works offline
    Custom (live data)   your own holdings and benchmark; downloads prices from Yahoo Finance
                         only when you press "Run analysis"

How the page is built: ``main()`` runs top to bottom on every click. ``state.collect_inputs`` draws
the sidebar and says what to analyse, then ``sections.render_page`` runs the risk calculations
ONCE and draws each part of the page.
"""

from __future__ import annotations

import logging

import streamlit as st

from ui import sections, state

log = logging.getLogger(__name__)


def main() -> None:
    """Build the whole page, top to bottom."""
    st.set_page_config(page_title="SignalDesk", page_icon=":material/monitoring:", layout="wide")
    try:
        dataset, controls = state.collect_inputs()
    except state.Unavailable as problem:  # nothing to show: say why, and offer a way back
        sections.render_unavailable(problem)
        return
    except Exception:  # a bug or an odd failure while reading inputs: the page must not go blank
        log.exception("Unexpected error while preparing the page")
        sections.render_unavailable(
            state.Unavailable(
                "Something unexpected went wrong while preparing this page. Please reload it, "
                "or try again in a moment.",
                kind="error",
                offer_demo=state.in_custom_mode(),
            )
        )
        return
    try:
        sections.render_page(dataset, controls)
    except Exception:  # whatever else goes wrong while drawing: say so, offer a way back, keep the footer
        log.exception("Unexpected error while drawing the page")
        st.error("Something unexpected went wrong while drawing this page. Please reload it, or try again in a moment.")
        if state.in_custom_mode():
            sections.offer_demo()
        sections.render_footer()


# Streamlit runs this file as "__main__"; a plain `import app` does not, so importing is harmless.
if __name__ == "__main__":
    main()
