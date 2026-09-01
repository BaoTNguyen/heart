"""Suite-wide defaults.

Retrieval is off by default here. `_context_packet` shells out to `art packet`
and `cap find` — a database round trip and a cross-encoder — once per role per
episode, and letting the toy episodes do that took the suite from 32s to 230s
without testing anything the retrieval tests do not already cover.

One place rather than each fixture: every episode-running test class needs it,
including ones not written yet. A test that wants real retrieval overrides it.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _no_live_retrieval(monkeypatch):
    if "ARTERIES_RETRIEVAL" not in os.environ:
        monkeypatch.setenv("ARTERIES_RETRIEVAL", "off")
