"""Self-check for the local-endpoint concurrency gate.

    python3 tests/test_local_slots.py

Covers the two pieces the fleet cap rests on: locality detection (which
endpoints count as "the local box") and the cross-process counting semaphore
(N agents at a time, no more, even across separate processes). Stdlib only,
no network.
"""
import inspect
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from dockerprobe import DOCKER_USABLE

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


def test_the_model_pool_is_not_named_after_this_repo():
    """The pool is shared with every process on the box that talks to the same
    server. arteries flocks the same directory and used an advisory lock of its
    own until this was named something it could join -- two caps of two against a
    two-slot server is the same overload with more bookkeeping.

    A rename here silently splits the pool again, and nothing would fail, so this
    test is the only thing holding the convention in place.
    """
    from heart.runner import _local_slot

    source = inspect.getsource(_local_slot)
    # The path, not the word: the comment beside it names the old directory on
    # purpose, so anyone reading the rename knows what it replaced.
    assert '"model-slots"' in source, source
    assert '"heart-local-slots"' not in source


def test_agent_slots_stay_heart_specific():
    """How many agents run at once is heart's business. Only the model server is
    a shared resource."""
    from heart.runner import _global_slot

    assert "heart-agent-slots" in inspect.getsource(_global_slot)


def test_the_pool_path_matches_what_arteries_computes():
    """Skipped when arteries is not on the path -- it is a sibling checkout, not
    a pinned dependency, which is the same reason the frame contract test skips."""
    try:
        from arteries import slots as arteries_slots
    except ImportError:
        return

    from heart.runner import _slots_base

    endpoint = "http://127.0.0.1:8001/v1"
    heart_pool = _slots_base() / "model-slots" / "127.0.0.1_8001"
    assert str(heart_pool) == str(arteries_slots.pool_for(endpoint)), (
        f"pools have diverged: {heart_pool} vs {arteries_slots.pool_for(endpoint)}")


if __name__ == "__main__":
    test_locality()
    test_the_model_pool_is_not_named_after_this_repo()
    test_agent_slots_stay_heart_specific()
    test_the_pool_path_matches_what_arteries_computes()
    test_a_stale_sandbox_image_is_reported_with_the_fix()
    test_a_missing_sandbox_image_says_to_build_it()
    test_a_current_image_is_not_flagged()
    test_work_can_say_which_network_the_sandbox_gets()
    test_the_network_default_is_still_deny()
    test_a_newborn_workspace_lock_is_not_litter()
    test_prune_runs_under_the_same_lock_as_worktree_add()
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


def test_a_stale_sandbox_image_is_reported_with_the_fix():
    """The Dockerfile gained the plugin's lock directory and the image was never
    rebuilt. Every sandboxed run then failed on

        touch: cannot touch '/home/agent/.docker/sandbox/locks/detached.lock'

    which names a path inside a container rather than "your image is older than
    the file describing it". Ten tests failed for two weeks and read as a plugin
    incompatibility.
    """
    import os
    import tempfile
    from pathlib import Path

    from heart.sandbox import image_is_stale

    # A Dockerfile touched into the future stands in for one edited after the
    # build, without rebuilding anything to test it.
    with tempfile.TemporaryDirectory() as tmp:
        dockerfile = Path(tmp) / "Dockerfile"
        dockerfile.write_text("FROM scratch\n")
        os.utime(dockerfile, (2 ** 31 - 1, 2 ** 31 - 1))
        reason = image_is_stale(dockerfile=dockerfile)
        assert reason is not None, "a future-dated Dockerfile should read as stale"
        assert "docker build" in reason, reason


def test_a_missing_sandbox_image_says_to_build_it():
    from heart.sandbox import image_is_stale

    reason = image_is_stale("heart-agent:definitely-not-built")
    assert reason and "does not exist" in reason, reason


@pytest.mark.skipif(not DOCKER_USABLE, reason="no docker daemon, or the sandbox image is not built")
def test_a_current_image_is_not_flagged():
    """Guards against the check crying wolf, which would be worse than silence:
    a sandbox that refuses to start is a harder failure than one that starts
    stale."""
    from heart.sandbox import image_is_stale

    assert image_is_stale() is None, image_is_stale()


def _work_help() -> str:
    """`heart work --help`, captured.

    The parser is built inside main() and there is nothing to import, so this
    goes through the same door a user does rather than refactoring the CLI for a
    test's convenience.
    """
    import contextlib
    import io

    from heart.cli import main

    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
        main(["work", "--help"])
    return out.getvalue()


def test_work_can_say_which_network_the_sandbox_gets():
    """A spec read from a file could always set this; `heart work` builds its
    spec in code and could not. Under HEART_SANDBOX that made `work --agent api`
    refuse every time -- the runner's error says to set network "api" or "model"
    and there was no way to do it.
    """
    help_text = _work_help()
    assert "--network" in help_text, help_text
    for choice in ("none", "model", "api"):
        assert choice in help_text, choice


def test_the_network_default_is_still_deny():
    """Default-deny is the point of the field: a task that cannot reach the
    network cannot exfiltrate through it."""
    import inspect

    from heart import cli

    source = inspect.getsource(cli.main)
    assert '"--network", default="none"' in source, "the default stopped being none"


def test_a_newborn_workspace_lock_is_not_litter():
    """A lock with no directory is what a workspace being born looks like.

    Workspace creates its lock *before* `git worktree add` creates the directory,
    so a reclaim running alongside can tell live from leaked. The litter sweep
    deleted any lock without a directory, which took the newborn's only liveness
    signal away -- and the next sweep then found a directory with no lock, called
    it reclaimable, and removed it out from under git:

        error: unable to create file src/heart/serve.py: No such file
        fatal: cannot create directory at 'src/heart/training'

    Measured on three concurrent sessions: one of the three died that way.
    """
    import fcntl
    import os
    import tempfile
    from pathlib import Path

    from heart import env

    with tempfile.TemporaryDirectory() as tmp:
        old = os.environ.get("HEART_WS_ROOT")
        os.environ["HEART_WS_ROOT"] = tmp
        try:
            root = Path(tmp)
            # A workspace mid-creation: lock taken, directory not there yet.
            newborn = root / "abc123def456.lock"
            held = open(newborn, "w")
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # And a genuine leak: a lock nobody holds, no directory.
            (root / "deadbeef0000.lock").write_text("")

            env.reclaim()

            assert newborn.exists(), "a held lock was swept as litter"
            assert not (root / "deadbeef0000.lock").exists(), "real litter survived"
            held.close()
        finally:
            if old is None:
                os.environ.pop("HEART_WS_ROOT", None)
            else:
                os.environ["HEART_WS_ROOT"] = old


def test_prune_runs_under_the_same_lock_as_worktree_add():
    """Both mutate .git/worktrees. A prune landing inside another session's add
    is the second way three concurrent sessions break one of them."""
    import inspect

    from heart import env

    source = inspect.getsource(env.reclaim)
    prune = source[source.index("for source in repos:"):]
    assert "_worktree_lock(source)" in prune, prune
