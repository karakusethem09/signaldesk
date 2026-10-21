"""Shared fixtures for the dashboard tests (test_app*.py, test_ui_*.py).

Nothing here is automatic: a test only gets these fixtures if it asks for them, so the data,
risk and limits tests are not affected.

``offline`` refuses every network attempt (Python sockets and libcurl, the two ways yfinance can
reach the internet) and fails the test if there was one. See ``real_yf`` in test_data.py.

``app_env`` is the one the page tests use. It is ``offline`` plus
    - removing SIGNALDESK_MODE, so the dashboard runs with the strict PUBLIC limits,
    - emptying Streamlit's in-process caches before and after the test (they would otherwise leak
      the demo data, the rate limiters and the price memory from one test into the next), and
    - pointing the data layer's default cache folder at a temporary folder.
"""

from __future__ import annotations

import socket

import pytest
import streamlit as st

from signaldesk import data


@pytest.fixture
def offline(monkeypatch):
    """Refuse and record any network attempt; the test fails if there was one. Yields the list of attempts."""
    from curl_cffi import curl as curl_module

    attempts: list = []

    def refuse_socket(*args, **kwargs):
        attempts.append(("socket", args))
        raise AssertionError("a test tried to use the network (through a Python socket)")

    def refuse_curl(self, *args, **kwargs):
        attempts.append(("curl", None))
        raise AssertionError("a test tried to use the network (through libcurl)")

    monkeypatch.setattr(socket.socket, "connect", refuse_socket)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_socket)
    monkeypatch.setattr(socket, "create_connection", refuse_socket)
    monkeypatch.setattr(curl_module.Curl, "perform", refuse_curl)
    yield attempts
    assert attempts == []  # nothing even tried to reach the network


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
