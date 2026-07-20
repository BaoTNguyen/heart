"""Verifier execution and task determinism checks."""
from __future__ import annotations

import os
import subprocess
import time

from .env import Workspace
from .runner import sandbox_wrap
from .taskspec import TaskSpec, Verifier


def run_verifiers(verifiers: list[Verifier], cwd: str, timeout: int) -> dict[str, dict]:
    # No bytecode cache: a same-second same-size source edit (common in fast
    # agent fix loops) passes the pyc header's mtime+size check and Python
    # silently runs stale code — verifier results must never depend on that.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    # Policy: agents get network, verifiers never do — when sandboxing is on
    # at all, verifier subprocesses are always forced into bwrap-nonet.
    sandboxed = os.environ.get("HEART_SANDBOX") in ("bwrap", "bwrap-nonet")
    results: dict[str, dict] = {}
    for v in verifiers:
        t0 = time.monotonic()
        cmd, shell = v.command, True
        if sandboxed:
            cmd, shell = sandbox_wrap(v.command, True, cwd, {}, mode="bwrap-nonet")
        try:
            proc = subprocess.run(
                cmd, shell=shell, cwd=cwd, capture_output=True, text=True,
                timeout=timeout, env=env,
            )
            passed, code = proc.returncode == 0, proc.returncode
            output = (proc.stdout + proc.stderr)[-4000:]
        except subprocess.TimeoutExpired:
            passed, code, output = False, -1, f"timeout after {timeout}s"
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
