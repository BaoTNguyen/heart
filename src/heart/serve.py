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
from .events import spool_dir

PAGE = Path(__file__).with_name("pulse.html")


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
        if url.path == "/stream":
            return self._stream(float(parse_qs(url.query).get("hours", ["24"])[0]))
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


def serve(port: int = 7717) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"heart pulse: http://127.0.0.1:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
