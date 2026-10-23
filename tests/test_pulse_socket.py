"""load_events reads the pulse collector's socket and falls back to the journal."""
import http.server
import json
import shutil
import socketserver
import tempfile
import threading
import urllib.parse
from pathlib import Path

import pytest

from heart import observe as pulse

PAGES = [
    {"events": [{"seq": 1, "ts": "2026-10-20T00:00:03", "source": "heart", "kind": "b"},
                {"seq": 2, "ts": "2026-10-20T00:00:01", "source": "heart", "kind": "a"}],
     "next": "c1"},
    [{"seq": 7, "ts": "2026-10-20T00:00:02", "source": "heart", "kind": "c"}],
    [],
]
JOURNAL = [{"ts": "2026-10-20T00:00:09", "source": "heart", "kind": "file"},
           {"ts": "2026-10-20T00:00:08", "source": "plexus", "kind": "other"}]


@pytest.fixture
def home(tmp_path, monkeypatch, request):
    h = Path(tempfile.mkdtemp(dir="/tmp"))  # AF_UNIX paths cap near 108 bytes
    request.addfinalizer(lambda: shutil.rmtree(h, ignore_errors=True))
    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / "2026-10-20.ndjson").write_text(
        "".join(json.dumps(e) + "\n" for e in JOURNAL) + "not json\n")
    monkeypatch.setenv("VASCULAR_HOME", str(h))
    monkeypatch.setenv("EVENT_JOURNAL_DIR", str(journal))
    monkeypatch.delenv("HEART_PULSE", raising=False)
    return h


@pytest.fixture
def collector(home, request):
    def serve(status=200):
        queries = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                queries.append(urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query))
                body = json.dumps(PAGES[min(len(queries), len(PAGES)) - 1]).encode()
                self.send_response(status)
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
        return queries
    return serve


def test_socket_served(collector):
    queries = collector()
    events = pulse.load_events(source="heart")
    assert [e["kind"] for e in events] == ["a", "c", "b"]
    assert all("seq" not in e for e in events)
    assert len(queries) == 3
    assert all(q["source"] == ["heart"] and q["limit"] == ["5000"] for q in queries)
    assert "after" not in queries[0]
    assert queries[1]["after"] == ["c1"]
    assert queries[2]["after"] == ["7"]


def test_no_socket_falls_back_silently(home, capsys):
    events = pulse.load_events()
    assert events == pulse.file_events()
    assert [e["kind"] for e in events] == ["other", "file"]
    assert capsys.readouterr() == ("", "")


def test_heart_pulse_file_skips_socket(collector, monkeypatch):
    queries = collector()
    monkeypatch.setenv("HEART_PULSE", "file")
    assert pulse.load_events(source="heart") == [JOURNAL[0]]
    assert queries == []


def test_server_error_falls_back(collector):
    queries = collector(status=500)
    assert pulse.load_events(source="heart") == [JOURNAL[0]]
    assert len(queries) == 1
