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


@pytest.fixture(autouse=True)
def _journal_to_a_tmpdir(tmp_path_factory, monkeypatch):
    """Test episodes write their events somewhere disposable.

    Unset, EVENT_JOURNAL_DIR is ~/.local/share/heart/events -- the real one. So
    the suite's synthetic episodes wrote into the real journal and the real
    per-run inboxes, and a sandboxed run's inbox came back holding 18 events
    belonging to two fixture episodes, task_id "scope", indistinguishable from
    work. Measured on episode 20260912-210719-abce2c5d.
    """
    if "EVENT_JOURNAL_DIR" not in os.environ:
        monkeypatch.setenv("EVENT_JOURNAL_DIR", str(tmp_path_factory.mktemp("events")))
