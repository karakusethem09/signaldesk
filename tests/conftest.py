"""Shared fixtures for the dashboard tests (test_app*.py, test_ui_*.py).

Nothing here is automatic: a test only gets these fixtures if it asks for them, so the data,
risk and limits tests are not affected.

``offline`` refuses every network attempt (Python sockets, libcurl and its async engine) and fails the
test if there was one. It IS the data tests' ``real_yf`` guard (see test_data.py), imported, not copied.

``app_env`` is the one the page tests use. It is ``offline`` plus
    - removing SIGNALDESK_MODE, so the dashboard runs with the strict PUBLIC limits,
    - emptying Streamlit's in-process caches before and after the test (they would otherwise leak
      the demo data, the rate limiters and the price memory from one test into the next), and
    - pointing the data layer's default cache folder at a temporary folder.
"""

from __future__ import annotations

import pytest
import streamlit as st

from signaldesk import data

# The network guard is written ONCE, in test_data.py (the data tests use it too): it refuses Python sockets,
# libcurl and curl_cffi's async engine, and sets NO_PROXY so that not even a request to the loopback address
# can be handed to a proxy. It is imported here, not copied, so the UI tests can never have a weaker one.
from test_data import network_guard, no_proxy, real_yf  # noqa: F401  (pytest finds fixtures by these names)


@pytest.fixture
def offline(real_yf):  # noqa: F811
    """Refuse and record any network attempt; the test fails if there was one. Yields the list of attempts."""
    return real_yf


@pytest.fixture
def app_env(monkeypatch, tmp_path, offline):
    """Public limits, empty caches, a temporary data folder and no network. Yields the list of network attempts."""
    monkeypatch.delenv("SIGNALDESK_MODE", raising=False)
    monkeypatch.setattr(data, "_default_cache_dir", lambda: tmp_path / "cache")
    st.cache_data.clear()
    st.cache_resource.clear()
    yield offline
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def local_mode(monkeypatch, app_env):
    """Like ``app_env``, but as on your own computer: SIGNALDESK_MODE=local, no limits."""
    monkeypatch.setenv("SIGNALDESK_MODE", "local")
    st.cache_resource.clear()
    return app_env
