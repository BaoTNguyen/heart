"""Verifier execution and task determinism checks."""
from __future__ import annotations

import os
import subprocess
import tempfile
import time

from .env import Workspace
from .runner import SANDBOX_MODE, sandbox_start_failure, sandbox_wrap
from .taskspec import TaskSpec, Verifier


def run_verifiers(verifiers: list[Verifier], cwd: str, timeout: int,
                  profile=None) -> dict[str, dict]:
    # No bytecode cache: a same-second same-size source edit (common in fast
    # agent fix loops) passes the pyc header's mtime+size check and Python
    # silently runs stale code — verifier results must never depend on that.
    # HEART_TIER_* scrubbed for the same reason as in runner.run_agent
    env = {k: v for k, v in os.environ.items() if not k.startswith("HEART_TIER_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Policy: agents may get network, verifiers never do. `profile` carries
    # that — verifier_profile_for pins network="none" and a read-only worktree.
    # No profile in docker mode means the caller did not build one, and a
    # verifier that quietly runs unsandboxed while the agent is contained is
    # the asymmetry this whole feature exists to remove: fail instead.
    mode = os.environ.get("HEART_SANDBOX", "off")
    if mode == SANDBOX_MODE and profile is None:
        raise RuntimeError(f"HEART_SANDBOX={mode} but no verifier sandbox profile was supplied")
    # The read-only worktree survives -- the plugin enforces :ro -- so a verifier
    # still cannot edit the tree it is judging, which is the reward-integrity
    # half. The network half does not: docker-sbx has no network flag, so a
    # verifier can reach the network and exfiltration-via-test is open again.
    sandboxed = mode == SANDBOX_MODE
    # Optional suite-wide ceiling. Each verifier still gets its own `timeout`, but
    # the whole suite can't exceed HEART_VERIFY_SUITE_TIMEOUT — without it, N
    # verifiers run serially at `timeout` each, so a 4-linter repo can spend 4×
    # the budget in the verify phase. Off by default (unset) to avoid silently
    # starving a legitimately slow second verifier and flipping its reward.
    suite_budget = float(os.environ.get("HEART_VERIFY_SUITE_TIMEOUT", "0") or 0)
    suite_start = time.monotonic()
    results: dict[str, dict] = {}
    for v in verifiers:
        t0 = time.monotonic()
        this_timeout = timeout
        if suite_budget > 0:
            remaining = suite_budget - (t0 - suite_start)
            if remaining <= 0:
                results[v.name] = {
                    "passed": False, "exit_code": -1, "duration_s": 0.0,
                    "output_tail": f"skipped: suite budget {suite_budget:g}s exhausted",
                }
                continue
            this_timeout = min(timeout, remaining)
        cmd, shell = v.command, True
        if sandboxed:
            cmd, shell = sandbox_wrap(v.command, True, cwd, {}, mode=SANDBOX_MODE,
                                      profile=profile)
        try:
            proc = subprocess.run(
                cmd, shell=shell, cwd=cwd, capture_output=True, text=True,
                timeout=this_timeout, env=env,
            )
            passed, code = proc.returncode == 0, proc.returncode
            output = (proc.stdout + proc.stderr)[-4000:]
            if failure := sandbox_start_failure(code, output):
                # a container that never started is not a verifier that failed
                raise RuntimeError(f"sandbox failed to start for verifier "
                                   f"{v.name!r}: {failure}")
        except subprocess.TimeoutExpired:
            passed, code, output = False, -1, f"timeout after {this_timeout:g}s"
        results[v.name] = {
            "passed": passed,
            "exit_code": code,
            "duration_s": round(time.monotonic() - t0, 2),
            "output_tail": output,
        }
    return results


def _check_profile(task: TaskSpec, ws_path, journal: str):
    """The verifier container for a determinism check.

    check-task is not an episode: it has no episode_id and nothing to report to,
    so the journal mount is a throwaway directory rather than an arteries inbox.
    A profile is still required -- run_verifiers refuses to run a verifier
    unsandboxed while HEART_SANDBOX says otherwise, and it is right to. Deciding
    a task is deterministic under conditions the episodes will not reproduce is
    exactly the measurement check-task exists to prevent.
    """
    if os.environ.get("HEART_SANDBOX") != SANDBOX_MODE:
        return None
    from .sandbox import verifier_profile_for

    return verifier_profile_for(task, ws_path, journal)


def check_task(task: TaskSpec, n: int = 3) -> dict:
    """Run public verifiers n times at base_commit (must be bit-stable) and once
    at fix_commit if present (must pass). Flaky verifiers poison reward signal."""
    runs = []
    # under heart's workspace root, not /tmp: Docker Desktop shares only paths
    # it has been told about, and a bind it will not share fails the container
    # rather than the check -- which then reads as "every verifier failed"
    from .env import _ws_root

    _ws_root().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="heart-check-", dir=_ws_root()) as journal:
        for _ in range(n):
            ws = Workspace(task.repo_path, task.base_commit, overlay=task.overlay_files)
            try:
                res = run_verifiers(task.public_verifiers, str(ws.path),
                                    task.timeout_seconds,
                                    profile=_check_profile(task, ws.path, journal))
                runs.append({name: r["passed"] for name, r in res.items()})
            finally:
                ws.destroy()
        deterministic = all(r == runs[0] for r in runs)

        fix_passes = None
        if task.fix_commit:
            ws = Workspace(task.repo_path, task.fix_commit, overlay=task.overlay_files)
            try:
                res = run_verifiers(task.public_verifiers, str(ws.path),
                                    task.timeout_seconds,
                                    profile=_check_profile(task, ws.path, journal))
                fix_passes = all(r["passed"] for r in res.values())
            finally:
                ws.destroy()

    # verifiers that already pass at base make the task worthless as signal:
    # a no-op diff earns full reward
    base_all_pass = bool(runs and runs[0] and all(runs[0].values()))
    ok = deterministic and (fix_passes is not False) and not base_all_pass
    return {
        "task_id": task.task_id,
        "deterministic": deterministic,
        "base_results": runs[0] if runs else {},
        "base_fails": not base_all_pass,
        "fix_passes": fix_passes,
        "ok": ok,
    }
