"""heart pulse compare: one day of socket events vs journal events."""
import http.server
import json
import shutil
import socketserver
import tempfile
import threading
from pathlib import Path

import pytest

from heart.cli import main

DAY = [{"ts": "2026-01-01T00:00:01", "source": "heart", "kind": "a"},
       {"ts": "2026-01-01T00:00:02", "source": "heart", "kind": "b", "payload": {"x": 1}}]
OTHER_DAY = {"ts": "2026-01-02T00:00:01", "source": "heart", "kind": "later"}


@pytest.fixture
def home(tmp_path, monkeypatch, request):
    h = Path(tempfile.mkdtemp(dir="/tmp"))  # AF_UNIX paths cap near 108 bytes
    request.addfinalizer(lambda: shutil.rmtree(h, ignore_errors=True))
    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / "2026-01-01.ndjson").write_text("".join(json.dumps(e) + "\n" for e in DAY))
    (journal / "2026-01-02.ndjson").write_text(json.dumps(OTHER_DAY) + "\n")
    monkeypatch.setenv("VASCULAR_HOME", str(h))
    monkeypatch.setenv("EVENT_JOURNAL_DIR", str(journal))
    monkeypatch.delenv("HEART_PULSE", raising=False)
    return h


@pytest.fixture
def collector(home, request):
    def serve(events):
        pages = [[dict(e, seq=i) for i, e in enumerate(events, 1)], []]
        calls = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append(self.path)
                body = json.dumps(pages[min(len(calls), len(pages)) - 1]).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        sock = home / "state" / "pulse" / "pulse.sock"
        sock.parent.mkdir(parents=True)
        srv = socketserver.ThreadingUnixStreamServer(str(sock), Handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        request.addfinalizer(lambda: (srv.shutdown(), srv.server_close()))
    return serve


def test_match_ignores_other_days(collector, capsys):
    collector(list(reversed(DAY)))
    assert main(["pulse", "compare", "--day", "2026-01-01"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["day 2026-01-01: socket 2 events, file 2 events", "match"]


def test_socket_missing_event(collector, capsys):
    collector(DAY[:1])
    assert main(["pulse", "compare", "--day", "2026-01-01"]) == 1
    out = capsys.readouterr().out.splitlines()
    assert "mismatch: 0 only in socket, 1 only in file" in out
    assert [l for l in out if l.startswith("  file only:")] == ["  file only: 00:00:02  heart      b  x=1"]
    assert not any(l.startswith("  socket only:") for l in out)


def test_no_socket(home, capsys):
    assert main(["pulse", "compare", "--day", "2026-01-01"]) == 2
    assert capsys.readouterr().out.splitlines() == [
        "pulse socket unavailable; nothing to compare", "file: 2 events"]


def test_bad_and_missing_day(home, capsys):
    assert main(["pulse", "compare", "--day", "nope"]) == 2
    assert capsys.readouterr().out.splitlines() == ["bad --day: nope"]
    assert main(["pulse", "compare"]) == 2
    assert "usage: heart pulse compare --day YYYY-MM-DD" in capsys.readouterr().err
