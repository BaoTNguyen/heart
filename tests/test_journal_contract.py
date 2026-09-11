"""The journal contract, asserted independently in each repo that writes to it.

arteries, heart and capillaries each resolve the journal path themselves --
none of them depends on the others, so none can import a shared constant. The
duplication is deliberate; drift is what is dangerous. A rename this session
proved it: with plexus reading the old variable while arteries and heart wrote
the new one, events went to two directories and the only symptom was a test
failing with "spine events not written".

These constants ARE the contract. Change one repo and its own test fails, at the
point of the change rather than in production. Change all of them together and
the rename is complete by construction.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heart.events import journal_dir

JOURNAL_ENV = "EVENT_JOURNAL_DIR"
JOURNAL_DEFAULT = Path.home() / ".local" / "share" / "heart" / "events"


def test_the_default_path_matches_the_contract():
    env = {k: v for k, v in os.environ.items() if k != JOURNAL_ENV}
    with patch.dict(os.environ, env, clear=True):
        assert journal_dir() == JOURNAL_DEFAULT


def test_the_environment_variable_overrides_it():
    with patch.dict(os.environ, {JOURNAL_ENV: "/tmp/elsewhere"}):
        assert journal_dir() == Path("/tmp/elsewhere")


def test_the_sandbox_asks_arteries_rather_than_guessing():
    """heart used to compute the inbox path itself when arteries was absent.
    A container writing where the drain does not read loses the run's memory
    silently, so a missing arteries must raise."""
    from heart.sandbox import inbox_for

    with patch.dict(os.environ, {"PATH": "/nonexistent"}):
        try:
            inbox_for("ep-contract")
        except RuntimeError as exc:
            assert "arteries" in str(exc)
        else:
            raise AssertionError("inbox_for must not guess the path")
