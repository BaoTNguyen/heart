"""`heart pulse serve` — the factory floor, as a local web page.

One stdlib HTTP server on localhost: serves a single HTML file, streams the
event spool over Server-Sent Events, and exposes insights/health as JSON.
The browser builds the episode board client-side from the same events the
terminal `pulse tail` prints — no database, no framework, no build step.

ponytail: localhost-only, no auth — this never leaves 127.0.0.1. Add a bearer
token before ever binding beyond loopback.
"""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import pulse
from .cli import WORK_RUNS_DIR
from .events import spool_dir

PAGE = Path(__file__).with_name("pulse.html")

# episode dirs for steering (§6.4 item 1) and drill-down (§6.4 item 2) live
# here; cli.py imports serve lazily inside cmd_pulse, so importing WORK_RUNS_DIR
# here at module scope creates no import cycle
RUNS_DIR = WORK_RUNS_DIR


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the terminal quiet
        pass

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        url = urlparse(self.path)
        if url.path == "/":
            return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/insights":
            hours = float(parse_qs(url.query).get("hours", ["24"])[0])
            h_lines, h_code = pulse.health(hours=hours)
            body = json.dumps({
                "insights": pulse.insights(hours=hours),
                "health": {"lines": h_lines, "code": h_code},
            }).encode()
            return self._send(body, "application/json")
        if url.path == "/api/episode":
            episode = parse_qs(url.query).get("id", [None])[0]
            if not episode:
                return self.send_error(400)
            ep_dir = RUNS_DIR / episode
            diff_path = ep_dir / "diff.patch"
            diff = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else None
            logs = {
                log_path.stem: log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                for log_path in sorted(ep_dir.glob("*.log"))
            } if ep_dir.is_dir() else {}
            body = json.dumps({
                "timeline": pulse.episode_timeline(episode), "diff": diff, "logs": logs,
            }).encode()
            return self._send(body, "application/json")
        if url.path == "/stream":
            return self._stream(float(parse_qs(url.query).get("hours", ["24"])[0]))
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        url = urlparse(self.path)
        if url.path == "/api/steer":
            episode = parse_qs(url.query).get("episode", [None])[0]
            if not episode:
                return self.send_error(400)
            ep_dir = RUNS_DIR / episode
            if not ep_dir.is_dir():
                return self.send_error(404)
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            (ep_dir / "steer.txt").write_bytes(body)
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def _stream(self, hours: float) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        cutoff = pulse._cutoff_iso(hours)
        try:
            for e in pulse.load_events():
                if e.get("ts", "") >= cutoff:
                    self._event(e)
            # follow the spool exactly the way `pulse tail` does
            offsets: dict[Path, int] = {}
            while True:
                for path in sorted(spool_dir().glob("*.ndjson"))[-2:]:
                    if path not in offsets:
                        offsets[path] = path.stat().st_size
                        continue
                    size = path.stat().st_size
                    if size > offsets[path]:
                        with open(path, encoding="utf-8", errors="replace") as f:
                            f.seek(offsets[path])
                            for line in f:
                                try:
                                    self._event(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                        offsets[path] = size
                self.wfile.write(b": ping\n\n")  # keepalive + disconnect probe
                self.wfile.flush()
                time.sleep(1)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser tab closed

    def _event(self, e: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(e).encode() + b"\n\n")
        self.wfile.flush()


def serve(port: int = 7717, runs_dir: str | Path | None = None) -> None:
    global RUNS_DIR
    if runs_dir is not None:
        RUNS_DIR = Path(runs_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"heart pulse: http://127.0.0.1:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
