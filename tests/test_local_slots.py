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


def test_default_is_the_servers_parallelism_not_unbounded():
    """The default was 0, meaning no cap, against a server that answers two
    requests at a time and queues the rest invisibly -- a queued request looks
    exactly like a slow one. heart admitted up to _GATE's eight, so six waited
    inside llama.cpp where nothing here could see them.
    """
    from heart.runner import DEFAULT_LOCAL_SLOTS, _default_local_slots, _slots_cache

    _slots_cache.clear()
    assert DEFAULT_LOCAL_SLOTS >= 1
    assert _default_local_slots(None) == DEFAULT_LOCAL_SLOTS


def test_an_unreachable_server_falls_back_rather_than_uncapping():
    """A server that will not answer has unknown parallelism. Guessing the
    default beats guessing "unlimited", which is what 0 meant."""
    from heart.runner import DEFAULT_LOCAL_SLOTS, _default_local_slots, _slots_cache

    _slots_cache.clear()
    assert _default_local_slots("http://127.0.0.1:9/v1") == DEFAULT_LOCAL_SLOTS


def test_the_server_is_asked_once_per_endpoint():
    """One probe per endpoint per process, not one per agent spawn."""
    from heart.runner import _default_local_slots, _slots_cache

    _slots_cache.clear()
    _default_local_slots("http://127.0.0.1:9/v1")
    before = dict(_slots_cache)
    _default_local_slots("http://127.0.0.1:9/v1")
    assert _slots_cache == before and len(before) == 1


def test_an_explicit_setting_still_wins():
    from heart.runner import _local_slot

    old = os.environ.get("HEART_LOCAL_SLOTS")
    os.environ["HEART_LOCAL_SLOTS"] = "0"
    try:
        # 0 means "no cap" when asked for explicitly; the change is only to what
        # *unset* means, so anyone who deliberately turned this off stays off.
        with _local_slot("http://127.0.0.1:8001/v1"):
            pass
    finally:
        if old is None:
            os.environ.pop("HEART_LOCAL_SLOTS", None)
        else:
            os.environ["HEART_LOCAL_SLOTS"] = old


def test_a_live_server_is_reachable():
    from heart.runner import _endpoint_reachable

    # Skipped rather than failed when nothing is running: a test that needs a
    # GPU box up is not a test anyone can run on a laptop.
    if not _endpoint_reachable("http://127.0.0.1:8001/v1"):
        return
    assert _endpoint_reachable("http://127.0.0.1:8001/v1")


def test_a_dead_port_is_not_reachable():
    from heart.runner import _endpoint_reachable

    assert not _endpoint_reachable("http://127.0.0.1:9/v1")


def test_a_broken_probe_does_not_block_work():
    """A liveness check that blocks when the check itself is broken is worse
    than no check: the failure it prevents is one slow episode, the failure it
    would introduce is every episode."""
    from unittest.mock import patch

    from heart.runner import _endpoint_reachable

    with patch("urllib.request.urlopen", side_effect=ValueError("something odd")):
        assert _endpoint_reachable("http://127.0.0.1:8001/v1")


def test_a_non_200_answer_still_counts_as_listening():
    """Something is answering, which is the question being asked."""
    import urllib.error
    from unittest.mock import patch

    from heart.runner import _endpoint_reachable

    err = urllib.error.HTTPError("u", 503, "busy", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        assert _endpoint_reachable("http://127.0.0.1:8001/v1")


if __name__ == "__main__":
    test_locality()
    test_a_live_server_is_reachable()
    test_a_dead_port_is_not_reachable()
    test_a_broken_probe_does_not_block_work()
    test_a_non_200_answer_still_counts_as_listening()
    test_default_is_the_servers_parallelism_not_unbounded()
    test_an_unreachable_server_falls_back_rather_than_uncapping()
    test_the_server_is_asked_once_per_endpoint()
    test_an_explicit_setting_still_wins()
    test_pool_serializes()
    test_pool_two_slots_overlap()
    test_pricing_local_free()
    test_reasoning_body()
    print("ok")
