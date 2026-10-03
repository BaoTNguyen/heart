"""Isolation of EVENT_JOURNAL_DIR / VASCULAR_HOME across subprocesses.

The outer test spawns a child pytest process that sets EVENT_JOURNAL_DIR and
VASCULAR_HOME to a sentinel directory and runs an inner probe.  The probe
emits an event and the outer test asserts nothing under the sentinel.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_journal_isolation(tmp_path_factory):
    """Child pytest sets EVENT_JOURNAL_DIR / VASCULAR_HOME to sentinel and
    emits an event; the outer test asserts no files appear under sentinel."""
    sentinel = tmp_path_factory.mktemp("journal_isolation")
    journal_dir = sentinel / "journal"
    home_dir = sentinel / "home"

    child_env = os.environ.copy()
    child_env["EVENT_JOURNAL_DIR"] = str(journal_dir)
    child_env["VASCULAR_HOME"] = str(home_dir)
    child_env["HEART_JOURNAL_PROBE"] = "1"
    child_env["HEART_JOURNAL_SENTINEL"] = str(sentinel)
    child_env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")

    this_file = Path(__file__).resolve()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"{this_file}::test_probe_emits_into_tmp"],
        env=child_env,
        cwd=str(tmp_path_factory.mktemp("cwd")),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"child pytest failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    # No files may appear anywhere under the sentinel.
    assert list(sentinel.rglob("*")) == []


def test_probe_emits_into_tmp(tmp_path_factory):
    """Inner probe: runs only when HEART_JOURNAL_PROBE=1."""
    if os.environ.get("HEART_JOURNAL_PROBE") != "1":
        pytest.skip("not the probe child")

    sentinel = Path(os.environ["HEART_JOURNAL_SENTINEL"])

    # After the autouse fixture sets setenv, the values should no longer point
    # under the sentinel (the autouse fixture overrides them).
    ejd = os.environ.get("EVENT_JOURNAL_DIR", "")
    vh = os.environ.get("VASCULAR_HOME", "")
    assert not ejd.startswith(str(sentinel)), f"EVENT_JOURNAL_DIR {ejd} is under sentinel"
    assert not vh.startswith(str(sentinel)), f"VASCULAR_HOME {vh} is under sentinel"

    from heart.events import emit, journal_dir

    # Emit a probe event
    emit("test", "isolation.probe")

    # The journal_dir must not be under the sentinel (it comes from the
    # autouse fixture, not the child env).
    assert not str(journal_dir()).startswith(str(sentinel)), (
        f"journal_dir {journal_dir()} is under sentinel"
    )

    # An .ndjson file containing 'isolation.probe' must exist under journal_dir().
    found = False
    for ndjson_file in journal_dir().rglob("*.ndjson"):
        text = ndjson_file.read_text()
        for line in text.strip().splitlines():
            obj = json.loads(line)
            if obj.get("kind") == "isolation.probe":
                found = True
                break
        if found:
            break
    assert found, "No 'isolation.probe' event found under journal_dir()"
