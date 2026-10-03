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


@pytest.fixture(scope="session")
def _isolated_state_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("isolated")
    events, vascular = root / "events", root / "vascular"
    events.mkdir()
    vascular.mkdir()
    return events, vascular


@pytest.fixture(autouse=True)
def _journal_to_a_tmpdir(_isolated_state_dirs, monkeypatch):
    """Test episodes write their events and state somewhere disposable.

    The rule is override, not setdefault: EVENT_JOURNAL_DIR and VASCULAR_HOME
    are set on every test whatever the environment already holds. The old
    fixture backed off when EVENT_JOURNAL_DIR was set, and inside the sandbox
    the verifier inherits EVENT_JOURNAL_DIR=/journal -- the real inbox. On
    2026-10-01 ~/.vascular/state/heart/events/incoming/ held 1,053 fake events
    written by this suite.

    Function-scoped on purpose: test_route_orchestrate's tearDownModule pops
    EVENT_JOURNAL_DIR, so a value set once per session would be gone for every
    later test. The directories are made once; the setenv repeats per test. A
    test that sets either variable itself still wins, since its setenv runs
    after this one.
    """
    events, vascular = _isolated_state_dirs
    monkeypatch.setenv("EVENT_JOURNAL_DIR", str(events))
    monkeypatch.setenv("VASCULAR_HOME", str(vascular))
