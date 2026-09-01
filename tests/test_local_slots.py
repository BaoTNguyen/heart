"""Self-check for the local-endpoint concurrency gate.

    python3 tests/test_local_slots.py

Covers the two pieces the fleet cap rests on: locality detection (which
endpoints count as "the local box") and the cross-process counting semaphore
(N agents at a time, no more, even across separate processes). Stdlib only,
no network.
"""
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heart.agents_api import endpoint_for, is_local_endpoint  # noqa: E402
from heart.runner import _flock_pool, _price  # noqa: E402


def test_locality():
    for ep in ("http://127.0.0.1:8000/v1", "http://localhost:1234",
               "http://192.168.1.5:8000/v1", "http://10.0.0.9/v1", "http://[::1]:8000"):
        assert is_local_endpoint(ep), ep
    for ep in ("https://api.openai.com/v1", "https://api.deepseek.com/v1",
               "http://8.8.8.8:8000"):
        assert not is_local_endpoint(ep), ep
    # tolerant resolver: no profile -> the local default, never an exception
    assert is_local_endpoint(endpoint_for(""))


def _hold(d, n, hold_s, q):
    with _flock_pool(Path(d), n):
        q.put(("enter", time.monotonic()))
        time.sleep(hold_s)
        q.put(("exit", time.monotonic()))


def test_pool_serializes():
    """With one slot, two processes must not overlap: one's whole [enter,exit]
    window sits entirely before the other's enter."""
    with tempfile.TemporaryDirectory() as d:
        q = mp.Queue()
        procs = [mp.Process(target=_hold, args=(d, 1, 0.4, q)) for _ in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(10)
        events = sorted((q.get() for _ in range(4)), key=lambda e: e[1])
        kinds = [k for k, _ in events]
        # a serialized pair reads enter,exit,enter,exit — never enter,enter
        assert kinds == ["enter", "exit", "enter", "exit"], kinds


def test_pool_two_slots_overlap():
    """Two slots let two processes run at once: the enters happen before either
    exit."""
    with tempfile.TemporaryDirectory() as d:
        q = mp.Queue()
        procs = [mp.Process(target=_hold, args=(d, 2, 0.4, q)) for _ in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(10)
        events = sorted((q.get() for _ in range(4)), key=lambda e: e[1])
        assert [k for k, _ in events[:2]] == ["enter", "enter"], events


def test_pricing_local_free():
    """Local endpoints are free even under a broad "api" pricing entry; metered
    APIs and subscription seats both price at the map's API rates."""
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "heart"
        cfg.mkdir()
        (cfg / "models.json").write_text(json.dumps({
            "profiles": {
                "local7b": {"endpoint": "http://127.0.0.1:8000/v1", "model": "x"},
                "gpt": {"endpoint": "https://api.openai.com/v1", "model": "gpt"},
            },
            "pricing": {
                "api": {"in_per_mtok": 1.0, "out_per_mtok": 2.0},
                "claude": {"in_per_mtok": 3.0, "out_per_mtok": 15.0},
            },
        }))
        old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = d
        try:
            M = 1_000_000
            assert _price("api:local7b", M, M) == 0.0   # local: free despite "api" entry
            assert _price("api:gpt", M, M) == 3.0        # metered: 1 + 2
            assert _price("claude", M, M) == 18.0        # subscription seat, API-equiv
            assert _price("api:gpt", None, M) is None    # no tokens -> no price
        finally:
            if old is None:
                os.environ.pop("XDG_CONFIG_HOME", None)
            else:
                os.environ["XDG_CONFIG_HOME"] = old


def test_reasoning_body():
    """The per-request reasoning toggle: off by default (no thinking key, so a
    server that ignores it is unaffected), on adds enable_thinking and floors
    the token budget so a reasoning trace can't silently eat the answer."""
    from heart.agents_api import _build_body, resolve_config

    msgs = [{"role": "user", "content": "hi"}]
    off = _build_body({"model": "m", "max_tokens": 4096}, msgs)
    assert "chat_template_kwargs" not in off, off
    on = _build_body({"model": "m", "reasoning": True, "max_tokens": 8000}, msgs)
    assert on["chat_template_kwargs"] == {"enable_thinking": True}
    assert on["max_tokens"] == 8000

    old = os.environ.get("HEART_API_REASONING")
    os.environ["HEART_API_REASONING"] = "1"
    try:
        cfg = resolve_config()
        assert cfg["reasoning"] is True and cfg["max_tokens"] >= 4000, cfg
    finally:
        if old is None:
            os.environ.pop("HEART_API_REASONING", None)
        else:
            os.environ["HEART_API_REASONING"] = old


if __name__ == "__main__":
    test_locality()
    test_pool_serializes()
    test_pool_two_slots_overlap()
    test_pricing_local_free()
    test_reasoning_body()
    print("ok")
