# SignalDesk

A portfolio risk and research platform, like a simple Bloomberg Terminal.
This is a learning project. The main goal is to understand how AI subagents work
together, while also building a strong portfolio piece.

## About me

- I'm learning, so explain what you're doing and why in simple terms.
- Before big changes, propose a plan and wait for my approval.
- Keep changes small and step by step. Don't build ahead of the current phase.

## Stack

- Python 3, with a virtual environment in `.venv/` inside this folder
- `pandas` and `numpy` for calculations
- `yfinance` for stock prices (later: FRED for macro data, SEC EDGAR for filings)
- Streamlit for the dashboard
- Anthropic Python SDK for the agents (phase two)
- `pytest` for tests

## Phases

### Phase 1: Dashboard (done)
A dashboard with a few stocks showing performance, volatility, correlations and
risk, using our own calculations.

Rules:
- All math lives in plain Python functions (for example `signaldesk/risk.py`),
  not inside the dashboard code. These functions will later be the agents' tools.
- Every calculation function gets a test with a known expected result.
- Functions should be deterministic: same input, same output.

### Phase 1.5: Public website (current phase)
Deploy the dashboard on Streamlit Community Cloud so recruiters can open it, while
keeping it useful for my own analysis. Work happens on the `phase-1.5` branch and
reaches `main` (the live site) only through a reviewed pull request.

Rules:
- Phase 1 rules still apply. Existing functions in `data.py` and `risk.py` keep their
  names, inputs and behaviour; new work is added next to them.
- The default view is a demo portfolio loaded from a committed snapshot, so it works
  even when Yahoo Finance is down or blocks the server.
- Live data requests are limited on the public site (ticker count, date span, rate).
- Stay on Streamlit (no React/Next.js). No new runtime dependencies unless approved.
- All tests stay offline.

### Phase 2: Agents (later, one at a time)
A supervisor agent that delegates to specialist agents. Build them in this order:
1. Quant agent: calls the phase-one functions. It never computes numbers in its
   own text; numbers must come from tool calls.
2. Macro agent: inflation, interest rates (FRED).
3. Filings agent: company filings (SEC EDGAR). Long documents, so needs chunking.
4. Risk agent: challenges the other agents' claims.

Agent design rules:
- Write the agent loop by hand first with the Anthropic SDK, before using any framework.
- Start the supervisor with simple hard-coded routing, then upgrade to LLM-driven delegation.
- Every agent returns claims with their source, e.g.
  `volatility = 28%, source: calc_volatility(AAPL, 60d)`.
- The risk agent checks: can each number be traced to a tool output, is the sample
  period long enough, is there look-ahead bias, do agents contradict each other.
- Log every agent call: which agent, input, output, token usage.
- Keep a small eval set of questions with known answers to measure whether changes help.

## Project goal for the portfolio
Show a case where the risk agent catches another agent's mistake (for example,
a deliberately planted too-short sample period).

## Documentation
- Whenever an agent catches a mistake (another agent's, the supervisor's, or its own),
  add an entry to `docs/catch-log.md` using the template at the top of that file.
- Record design decisions in `docs/decisions.md`, and add a short journal file to
  `docs/sessions/` after each work session. Keep `README.md` up to date as the front page.
