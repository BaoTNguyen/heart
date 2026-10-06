"""Tests for egress-proxy.py LOG_FILE / open_log behaviour.

Loads the proxy module via importlib so it can be tested in isolation.
All writes go to tmp_path — never into the read-only checkout.
"""

import importlib.util
import os
import sys
from io import StringIO
from pathlib import Path

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

PROXY_PATH = Path(__file__).resolve().parent.parent / "contrib" / "egress-proxy.py"


def _load_proxy():
    """Load the proxy module fresh each time so globals don't leak."""
    spec = importlib.util.spec_from_file_location("egress_proxy", PROXY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# pytest restores sys.stdout after each test, but the proxy module
# replaces it with a _Tee that holds the original.  We need to make
# sure the tee is undone after each test so pytest's capture works
# on the next one.
@pytest.fixture(autouse=True)
def _undo_stdout():
    """Remember the original sys.stdout before each test, restore after."""
    _orig = sys.stdout  # pytest's captured stream, or real stdout
    yield
    # If the proxy module replaced sys.stdout with a _Tee, restore it
    # so the next test sees a clean slate.
    if hasattr(_orig, '_tee_out'):
        # _Tee was set on sys.stdout; restore the original
        sys.stdout = _orig._tee_out
    else:
        # Make sure the proxy module's sys.stdout is the current one
        sys.stdout = _orig


class _Tee:
    """Minimal tee matching contrib/egress-proxy.py's _Tee for testing."""
    def __init__(self, out, log, path):
        self.out = out
        self.log = log
        self.path = path
        self.notice = ""
        self.bol = True

    def write(self, s: str) -> int:
        n = self.out.write(s)
        if s:
            self.bol = s.endswith("\n")
        if self.log:
            try:
                self.log.write(s)
            except (OSError, ValueError) as exc:
                self._drop(exc)
        self._announce()
        return n

    def flush(self) -> None:
        self.out.flush()
        if self.log:
            try:
                self.log.flush()
            except (OSError, ValueError) as exc:
                self._drop(exc)
        self._announce()

    def _drop(self, exc):
        try:
            self.log.close()
        except (OSError, ValueError):
            pass
        self.log = None
        self.notice = f"log file {self.path} unavailable ({exc}): logging to stdout only\n"

    def _announce(self):
        if self.notice and self.bol:
            self.out.write(self.notice)
            self.out.flush()
            self.notice = ""

    def __getattr__(self, name):
        return getattr(self.out, name)


# ── tests ────────────────────────────────────────────────────────────────────

def test_log_file_appears_in_both(capsys, tmp_path):
    """When LOG_FILE is set, print() reaches both stdout and the file."""
    log_path = tmp_path / "test.log"
    mod = _load_proxy()

    # Replace sys.stdout with a StringIO so we can capture
    captured = StringIO()
    sys.stdout = captured

    mod.open_log(str(log_path))
    print("hello", flush=True)

    # Restore stdout so capsys can still work
    sys.stdout = captured

    out_text = captured.getvalue()
    assert "hello" in out_text

    file_text = log_path.read_text()
    assert "hello" in file_text


def test_rotation_at_start(capsys, tmp_path):
    """A pre-existing file > 10 MiB is rotated to .1 at start."""
    log_path = tmp_path / "test.log"
    old_one_path = tmp_path / "test.log.1"

    # Create the oversized log file (10 MiB + 1 byte)
    big_content = b"A" * (10 * 1024 * 1024 + 1)
    log_path.write_bytes(big_content)

    # Also create a pre-existing .1 file with different content
    old_one_path.write_bytes(b"OLD ONE")

    mod = _load_proxy()

    # Replace sys.stdout with a StringIO so we can capture
    captured = StringIO()
    sys.stdout = captured

    mod.open_log(str(log_path))
    print("new line", flush=True)

    sys.stdout = captured
    out_text = captured.getvalue()

    # The old .1 should have been replaced with the oversized content
    assert old_one_path.read_bytes() == big_content

    # The new log file should have the fresh line
    file_text = log_path.read_text()
    assert "new line" in file_text

    # The old .1 should NOT still exist
    assert old_one_path.read_bytes() == big_content


def test_no_rotation_under_10MiB(capsys, tmp_path):
    """A file at or under 10 MiB is NOT rotated."""
    log_path = tmp_path / "test.log"
    log_path.write_bytes(b"small content\n")

    mod = _load_proxy()

    captured = StringIO()
    sys.stdout = captured

    mod.open_log(str(log_path))
    print("appended", flush=True)

    sys.stdout = captured

    # The original file should still exist with its original content
    file_text = log_path.read_text()
    assert "small content" in file_text
    assert "appended" in file_text

    # No .1 file should exist
    assert not (tmp_path / "test.log.1").exists()


def test_unwritable_log_file_does_not_break(capsys, tmp_path):
    """An unwritable LOG_FILE leaves the proxy working: one fallback line."""
    # Use a path whose parent doesn't exist — this is the most reliable
    # way to make the file unwritable (even as root, the parent dir must exist)
    log_path = str(tmp_path / "nonexistent_parent_dir" / "test.log")

    mod = _load_proxy()

    # Replace sys.stdout with a StringIO so we can capture
    captured = StringIO()
    sys.stdout = captured

    mod.open_log(log_path)

    # Now print something — it should still reach stdout
    print("still working", flush=True)

    sys.stdout = captured
    out_text = captured.getvalue()

    # The fallback line should appear
    assert "unavailable" in out_text
    assert "logging to stdout only" in out_text
    assert "still working" in out_text


def test_mid_run_write_failure(capsys, tmp_path):
    """Closing the log file mid-run drops teeing but does not raise."""
    log_path = tmp_path / "midrun.log"

    mod = _load_proxy()

    captured = StringIO()
    sys.stdout = captured

    mod.open_log(str(log_path))

    # First print should work fine
    print("before break", flush=True)

    # Get the _Tee and close the underlying log file to simulate failure
    tee = sys.stdout
    tee.log.close()

    # This print should NOT raise, stdout should still get the line,
    # and the fallback notice should appear once
    print("after break", flush=True)

    sys.stdout = captured
    out_text = captured.getvalue()

    assert "before break" in out_text
    assert "after break" in out_text
    # The fallback notice should appear (once)
    assert "unavailable" in out_text
    assert "logging to stdout only" in out_text
