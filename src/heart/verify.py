"""Verifier execution and task determinism checks."""
from __future__ import annotations

import os
import subprocess
import time

from .env import Workspace
from .runner import sandbox_wrap
from .taskspec import TaskSpec, Verifier


def run_verifiers(verifiers: list[Verifier], cwd: str, timeout: int,
                  profile=None) -> dict[str, dict]:
    # No bytecode cache: a same-second same-size source edit (common in fast
    # agent fix loops) passes the pyc header's mtime+size check and Python
    # silently runs stale code — verifier results must never depend on that.
    # HEART_TIER_* scrubbed for the same reason as in runner.run_agent
    env = {k: v for k, v in os.environ.items() if not k.startswith("HEART_TIER_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Policy: agents get network, verifiers never do — when sandboxing is on
    # at all, verifier subprocesses are always forced into the no-network
    # variant of whichever mode is in force. `profile` carries that for docker
    # (verifier_profile_for pins network="none" and a read-only worktree); for
    # bwrap it is the -nonet mode below.
    mode = os.environ.get("HEART_SANDBOX", "off")
    sandboxed = mode in ("bwrap", "bwrap-nonet") or (mode == "docker" and profile is not None)
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
            cmd, shell = sandbox_wrap(
                v.command, True, cwd, {},
                mode="docker" if profile is not None else "bwrap-nonet",
                profile=profile,
            )
        try:
            proc = subprocess.run(
                cmd, shell=shell, cwd=cwd, capture_output=True, text=True,
                timeout=this_timeout, env=env,
            )
            passed, code = proc.returncode == 0, proc.returncode
            output = (proc.stdout + proc.stderr)[-4000:]
        except subprocess.TimeoutExpired:
            passed, code, output = False, -1, f"timeout after {this_timeout:g}s"
        results[v.name] = {
            "passed": passed,
            "exit_code": code,
            "duration_s": round(time.monotonic() - t0, 2),
            "output_tail": output,
        }
    return results


def check_task(task: TaskSpec, n: int = 3) -> dict:
    """Run public verifiers n times at base_commit (must be bit-stable) and once
    at fix_commit if present (must pass). Flaky verifiers poison reward signal."""
    runs = []
    for _ in range(n):
        ws = Workspace(task.repo_path, task.base_commit, overlay=task.overlay_files)
        try:
            res = run_verifiers(task.public_verifiers, str(ws.path), task.timeout_seconds)
            runs.append({name: r["passed"] for name, r in res.items()})
        finally:
            ws.destroy()
    deterministic = all(r == runs[0] for r in runs)

    fix_passes = None
    if task.fix_commit:
        ws = Workspace(task.repo_path, task.fix_commit, overlay=task.overlay_files)
        try:
            res = run_verifiers(task.public_verifiers, str(ws.path), task.timeout_seconds)
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
