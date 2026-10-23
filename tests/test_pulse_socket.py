"""load_events reads the pulse collector's socket, with no silent fallback to
the journal files (B11); HEART_PULSE=file reads them on purpose."""
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


def test_no_socket_says_so_rather_than_scanning_files(home):
    with pytest.raises(pulse.PulseUnavailable, match="pulse is not answering"):
        pulse.load_events()


def test_heart_pulse_file_skips_socket(collector, monkeypatch):
    queries = collector()
    monkeypatch.setenv("HEART_PULSE", "file")
    assert pulse.load_events(source="heart") == [JOURNAL[0]]
    assert queries == []


def test_server_error_says_so(collector):
    queries = collector(status=500)
    with pytest.raises(pulse.PulseUnavailable):
        pulse.load_events(source="heart")
    assert len(queries) == 1


def test_since_goes_to_pulse(collector):
    queries = collector()
    pulse.load_events(since="2026-10-20T00:00:02+00:00")
    assert queries[0]["since"] == ["2026-10-20T00:00:02+00:00"]


def test_follow_resumes_from_the_cursor_and_skips_history(home, request):
    """cursor() before history, follow(after=) from it: an event that lands
    between the two is in both and printed once; nothing falls between."""
    rows = [{"seq": n, "id": f"e{n}", "ts": f"2026-10-20T00:00:0{n}Z", "kind": "k"}
            for n in (1, 2, 3)]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            if q.get("order") == ["desc"]:
                page = rows[-1:]
            else:
                after = int(q.get("after", ["0"])[0])
                page = [dict(r) for r in rows if r["seq"] > after]
            body = json.dumps({"events": page,
                               "next": page[-1]["seq"] if page else 0}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    sock = home / "state" / "pulse" / "pulse.sock"
    sock.parent.mkdir(parents=True)
    srv = socketserver.ThreadingUnixStreamServer(str(sock), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    request.addfinalizer(lambda: (srv.shutdown(), srv.server_close()))

    after = pulse.cursor()                       # 3
    rows.append({"seq": 4, "id": "e4", "ts": "2026-10-20T00:00:04Z", "kind": "k"})
    history = pulse.load_events()                # e1..e4: e4 landed after the cursor
    rows.append({"seq": 5, "id": "e5", "ts": "2026-10-20T00:00:05Z", "kind": "k"})
    live = pulse.follow(interval=0, seen=pulse._ids(history), after=after)
    got = [next(live), next(live)]
    assert after == 3
    assert [e["id"] for e in history] == ["e1", "e2", "e3", "e4"]
    assert got[0]["id"] == "e5" and got[1] is None   # e4 skipped as seen, then idle
    assert got[0]["ts"].endswith("+00:00") and "seq" not in got[0]
