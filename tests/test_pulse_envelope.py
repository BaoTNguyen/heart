"""heart.events.emit writes pulse's envelope (tests/pulse_envelope.json)."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from heart import pulse_emit
from heart.events import emit

CONTRACT = json.loads((Path(__file__).parent / "pulse_envelope.json").read_text())
FIELDS = {**CONTRACT["required"], **CONTRACT["optional"]}


@pytest.fixture
def journal(tmp_path, monkeypatch):
    monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EVENT_JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pulse_emit, "_announced", set())
    for var in ("ARTERIES_EPISODE_ID", "ARTERIES_TASK_ID", "PLEXUS_GOAL_ID", "PLEXUS_FEATURE_ID"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path / "journal"


def _lines(journal: Path) -> list[dict]:
    return [json.loads(line) for p in sorted(journal.glob("*.ndjson"))
            for line in p.read_text().splitlines()]


def _events(journal: Path) -> list[dict]:
    return [e for e in _lines(journal) if e["kind"] != "context.started"]


def _type_ok(kind: str, value) -> bool:
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
    if kind == "object":
        return isinstance(value, dict)
    return isinstance(value, str)  # string, uuid, timestamp, name, kind, origin


def test_every_line_matches_the_envelope(journal):
    emit("heart", "episode.started", episode_id="ep1", task_id="t1", role="solo",
         duration_ms=5, agent="shell")
    emit("heart", "episode.finished", episode_id="ep1", outcome="pass")
    lines = _lines(journal)
    assert len(lines) == 3
    for ev in lines:
        assert isinstance(ev, dict)
        for key, value in ev.items():
            assert key in FIELDS, key
            assert _type_ok(FIELDS[key], value), (key, value)
        assert set(CONTRACT["required"]) <= ev.keys()
        for key in ("id", "schema_version", "emitter_seq", "context_id"):
            assert key in ev, key
        assert ev["schema_version"] == 1
    started = [e for e in lines if e["kind"] == "context.started"]
    assert len(started) == 1
    assert "episode_id" not in started[0]


def test_fields_set_as_before(journal):
    emit("heart", "role.finished", episode_id="ep1", task_id="t1", role="solo",
         duration_ms=7, exit_code=0)
    [ev] = _events(journal)
    assert datetime.datetime.fromisoformat(ev["ts"]).utcoffset() == datetime.timedelta(0)
    assert ev["ts"].endswith("+00:00")
    assert (ev["source"], ev["kind"]) == ("heart", "role.finished")
    assert (ev["episode_id"], ev["task_id"], ev["role"], ev["duration_ms"]) == ("ep1", "t1", "solo", 7)
    assert ev["payload"] == {"exit_code": 0}


def test_env_fallbacks_and_goal_stamp(journal, monkeypatch):
    monkeypatch.setenv("ARTERIES_EPISODE_ID", "ep-env")
    monkeypatch.setenv("ARTERIES_TASK_ID", "t-env")
    monkeypatch.setenv("PLEXUS_GOAL_ID", "g-env")
    monkeypatch.setenv("PLEXUS_FEATURE_ID", "f-env")
    emit("heart", "verify.round", attempt=0)
    emit("heart", "verify.round", goal_id="explicit-g")
    first, second = _events(journal)
    assert (first["episode_id"], first["task_id"]) == ("ep-env", "t-env")
    assert first["payload"] == {"attempt": 0, "goal_id": "g-env", "feature_id": "f-env"}
    assert (first["goal_id"], first["feature_id"]) == ("g-env", "f-env")
    # an explicit kwarg wins over the env stamp, in the payload and the envelope
    assert second["payload"]["goal_id"] == "explicit-g"
    assert (second["goal_id"], second["feature_id"]) == ("explicit-g", "f-env")


def test_no_payload_without_kwargs_or_env(journal):
    emit("heart", "episode.started", episode_id="ep1")
    [ev] = _events(journal)
    assert "payload" not in ev
    assert "goal_id" not in ev and "feature_id" not in ev


def test_unwritable_journal_does_not_raise(journal, monkeypatch, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setenv("EVENT_JOURNAL_DIR", str(blocker))
    emit("heart", "episode.started", episode_id="ep1", agent="shell")
