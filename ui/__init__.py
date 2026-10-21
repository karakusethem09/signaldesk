"""SignalDesk dashboard pieces (Phase 1.5). ``app.py`` at the project root only wires them together.

This package is ONLY the user interface. It does no financial math at all:
    - every price comes from ``signaldesk/data.py``
    - every number (returns, volatility, correlation, drawdown, VaR, Sharpe, portfolio value,
      weights) comes from ``signaldesk/risk.py``
    - every request limit comes from ``signaldesk/limits.py``
If a number is missing, the right fix is a new function in ``risk.py`` (with a test), not a
calculation here. tests/test_ui_guards.py scans EVERY file in this folder (and app.py) for math.

The files, from the visitor's input inward:
    inputs.py     turn what a visitor typed (holdings table, benchmark, dates) into a checked request
    live.py       the live-data pipeline: short-term memory, rate limit, download gate (no Streamlit)
    state.py      the sidebar widgets, the demo/live data sets, Streamlit's shared caches
    analysis.py   run every risk.py call once (one lock keeps analyses apart) and collect the warnings
    formatting.py numbers and tables as text for people (for example 0.256 -> "25.6%")
    charts.py     Altair chart builders (no chart-side math)
    sections.py   the page itself: header, key numbers, tabs, footer
"""
