"""pulse.follow: the journal tail behind `heart pulse tail` and the dashboard's
/stream. It used to remember the size it measured before reading, so an
append landing mid-read was printed twice, a half-written line was skipped
for good, and a file born mid-follow (UTC midnight) lost its first events."""

import json

from heart import pulse


def _line(n: int) -> bytes:
    return (json.dumps({"kind": "x", "n": n}) + "\n").encode()


def _drain(gen) -> list[int]:
    """Events up to the next idle marker."""
    out = []
    while (e := next(gen)) is not None:
        out.append(e["n"])
    return out


def test_follow_reads_each_line_once_and_waits_for_a_whole_line(tmp_path, monkeypatch):
    monkeypatch.setenv("EVENT_JOURNAL_DIR", str(tmp_path))
    day1 = tmp_path / "20260101.ndjson"
    day1.write_bytes(_line(0))  # history: printed by the caller, not by follow
    gen = pulse.follow(interval=0)
    assert _drain(gen) == []

    with open(day1, "ab") as f:
        f.write(_line(1) + _line(2)[:7])  # a writer caught mid-line
    assert _drain(gen) == [1]
    with open(day1, "ab") as f:
        f.write(_line(2)[7:])
    assert _drain(gen) == [2]
    assert _drain(gen) == [], "a line was yielded twice"

    # midnight: a new day's file appears with events already in it
    (tmp_path / "20260102.ndjson").write_bytes(_line(3) + _line(4))
    assert _drain(gen) == [3, 4]


def test_follow_survives_non_ascii_and_junk(tmp_path, monkeypatch):
    monkeypatch.setenv("EVENT_JOURNAL_DIR", str(tmp_path))
    day = tmp_path / "20260101.ndjson"
    day.write_bytes(b"")
    gen = pulse.follow(interval=0)
    assert _drain(gen) == []
    with open(day, "ab") as f:
        f.write((json.dumps({"n": 1, "s": "café"}, ensure_ascii=False) + "\n").encode())
        f.write(b"not json\n\xff\xfe\n")
        f.write(_line(2))
    assert _drain(gen) == [1, 2]
    assert _drain(gen) == []
