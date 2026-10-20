"""Tests for heart.events.artifact and the origin parameter on emit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from heart.events import artifact, emit, journal_dir
from heart import pulse_emit


def _lines(journal_dir: Path) -> list[dict]:
    """Read every JSON line from *.ndjson files under *journal_dir*."""
    return [
        json.loads(line)
        for p in sorted(journal_dir.glob("*.ndjson"))
        for line in p.read_text().splitlines()
        if line.strip()
    ]


def _events(journal_dir: Path) -> list[dict]:
    """Non-context lines."""
    return [e for e in _lines(journal_dir) if e["kind"] != "context.started"]


# ---------------------------------------------------------------------------
# (a) artifact(bytes) – round-trip through journal_dir/artifacts
# ---------------------------------------------------------------------------
class TestArtifactBytes:

    def test_returns_sha256_hexdigest(self, monkeypatch, tmp_path):
        events_dir = tmp_path / "events"
        events_dir.mkdir()
        monkeypatch.setenv("EVENT_JOURNAL_DIR", str(events_dir))
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "vascular"))
        monkeypatch.setattr(pulse_emit, "_announced", set())

        data = b"hello"
        expected = hashlib.sha256(data).hexdigest()
        digest = artifact(data)
        assert digest == expected

    def test_stores_bytes_at_digest_path(self, monkeypatch, tmp_path):
        events_dir = tmp_path / "events"
        events_dir.mkdir()
        monkeypatch.setenv("EVENT_JOURNAL_DIR", str(events_dir))
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "vascular"))

        data = b"hello"
        digest = artifact(data)
        artifact_path = events_dir / "artifacts" / digest
        assert artifact_path.exists()
        assert artifact_path.read_bytes() == data


# ---------------------------------------------------------------------------
# (b) artifact(Path) – streams file contents
# ---------------------------------------------------------------------------
class TestArtifactPath:

    def test_returns_file_sha256_and_file_exists(self, monkeypatch, tmp_path):
        events_dir = tmp_path / "events"
        events_dir.mkdir()
        monkeypatch.setenv("EVENT_JOURNAL_DIR", str(events_dir))
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "vascular"))
        monkeypatch.setattr(pulse_emit, "_announced", set())

        src = tmp_path / "source.bin"
        payload = b"file contents for artifact"
        src.write_bytes(payload)

        digest = artifact(src)
        expected = hashlib.sha256(payload).hexdigest()
        assert digest == expected

        artifact_path = events_dir / "artifacts" / digest
        assert artifact_path.exists()
        assert artifact_path.read_bytes() == payload


# ---------------------------------------------------------------------------
# (c) emit with origin – origin appears or is absent
# ---------------------------------------------------------------------------
class TestEmitOrigin:

    def test_origin_external_appears(self, monkeypatch, tmp_path):
        events_dir = tmp_path / "events"
        events_dir.mkdir()
        monkeypatch.setenv("EVENT_JOURNAL_DIR", str(events_dir))
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "vascular"))
        monkeypatch.setattr(pulse_emit, "_announced", set())

        emit("heart", "test.kind", origin="external")
        events = _events(events_dir)
        assert len(events) == 1
        assert events[0]["origin"] == "external"

    def test_origin_agent_appears(self, monkeypatch, tmp_path):
        events_dir = tmp_path / "events"
        events_dir.mkdir()
        monkeypatch.setenv("EVENT_JOURNAL_DIR", str(events_dir))
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "vascular"))
        monkeypatch.setattr(pulse_emit, "_announced", set())

        emit("heart", "test.kind", origin="agent")
        events = _events(events_dir)
        assert len(events) == 1
        assert events[0]["origin"] == "agent"

    def test_origin_operator_appears(self, monkeypatch, tmp_path):
        events_dir = tmp_path / "events"
        events_dir.mkdir()
        monkeypatch.setenv("EVENT_JOURNAL_DIR", str(events_dir))
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "vascular"))
        monkeypatch.setattr(pulse_emit, "_announced", set())

        emit("heart", "test.kind", origin="operator")
        events = _events(events_dir)
        assert len(events) == 1
        assert events[0]["origin"] == "operator"

    def test_no_origin_means_no_key(self, monkeypatch, tmp_path):
        events_dir = tmp_path / "events"
        events_dir.mkdir()
        monkeypatch.setenv("EVENT_JOURNAL_DIR", str(events_dir))
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "vascular"))
        monkeypatch.setattr(pulse_emit, "_announced", set())

        emit("heart", "test.kind")
        events = _events(events_dir)
        assert len(events) == 1
        assert "origin" not in events[0]


# ---------------------------------------------------------------------------
# (d) Unwritable journal – EVENT_JOURNAL_DIR points to a regular file
# ---------------------------------------------------------------------------
class TestUnwritableJournal:

    def test_artifact_returns_none_when_journal_is_file(self, monkeypatch, tmp_path):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        monkeypatch.setenv("EVENT_JOURNAL_DIR", str(blocker))
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "vascular"))

        result = artifact(b"x")
        assert result is None

    def test_emit_does_not_raise_when_journal_is_file(self, monkeypatch, tmp_path):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        monkeypatch.setenv("EVENT_JOURNAL_DIR", str(blocker))
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "vascular"))

        # Should not raise at all
        emit("heart", "episode.started", episode_id="ep1", origin="external")
        # nothing else to assert — the point is no exception
