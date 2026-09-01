"""Verifier execution and task determinism checks."""
from __future__ import annotations

import os
import re
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
    # a no-op diff earns full reward. Baseline verifiers are exempt and must be:
    # a measurement command is *meant* to succeed at base -- that run is the
    # number the head run is compared against -- so judging it by the same rule
    # would condemn every task whose criteria are performance rather than
    # correctness.
    gating = {v.name for v in task.public_verifiers if not v.baseline}
    at_base = {n: r for n, r in (runs[0] if runs else {}).items() if n in gating}
    base_all_pass = bool(at_base and all(at_base.values()))
    ok = deterministic and (fix_passes is not False) and not base_all_pass
    return {
        "task_id": task.task_id,
        "deterministic": deterministic,
        "base_results": runs[0] if runs else {},
        "base_fails": not base_all_pass,
        "fix_passes": fix_passes,
        "ok": ok,
    }


def _last_float(text: str) -> float | None:
    """The last number in a verifier's output.

    Last, not first: a measurement command prints its working and then its
    answer, and the answer is what the criterion is about.
    """
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(m[-1]) if m else None


def compare_baseline(mode: str, base: str, head: str) -> tuple[bool, str]:
    """Judge one verifier's output against the same command at base_commit.

    Returns (passed, explanation). An unreadable comparison fails: a criterion
    that cannot be evaluated has not been met, and defaulting it to pass is how
    a performance gate silently stops gating.
    """
    if mode == "identical":
        if base.strip() == head.strip():
            return True, "identical to base"
        return False, f"differs from base\n--- base\n{base[-1500:]}\n--- head\n{head[-1500:]}"
    # Two directions, because half of the criteria worth gating on get better by
    # going down. Without "no_more" the only way to gate latency is to print it
    # negative, and a criterion nobody can read is a criterion nobody writes.
    if mode in ("no_worse", "no_more"):
        b, h = _last_float(base), _last_float(head)
        if b is None or h is None:
            return False, f"no number to compare (base={base[-200:]!r} head={head[-200:]!r})"
        op, ok = (">=", h >= b) if mode == "no_worse" else ("<=", h <= b)
        if ok:
            return True, f"{h:g} {op} base {b:g}"
        return False, f"regressed: {h:g} not {op} base {b:g}"
    return False, f"unknown baseline mode {mode!r}"


def run_probes(probes: list[Verifier], cwd: str, timeout: int,
               profile=None) -> tuple[str | None, str]:
    """Measure the environment before anything is planned.

    Returns (blocking_failure, facts_note). The note goes into the agent's
    prompt: the same command that asserts a fact is the one that reports it, so
    the plan is written against what the machine actually has rather than
    against what the prompt's author assumed in the morning.

    Probes run under the *agent's* profile, not the verifier's. A probe asks
    what the agent will be able to reach, and answering that from inside a
    stricter box answers a different question.
    """
    if not probes:
        return None, ""
    results = run_verifiers(probes, cwd, timeout, profile=profile)
    failed = [n for n, r in results.items() if not r["passed"]]
    lines = ["", "", "## Measured environment (probed at base commit, before planning)"]
    for name, r in results.items():
        out = " ".join(r["output_tail"].split())[:300] or "(no output)"
        lines.append(f"- {name}: {'ok' if r['passed'] else 'FAILED'} — {out}")
    lines.append("These are measurements, not assumptions. Plan against them.")
    if not failed:
        return None, "\n".join(lines)
    detail = "; ".join(
        f"{n}: exit {results[n]['exit_code']} {' '.join(results[n]['output_tail'].split())[:200]}"
        for n in failed)
    return f"precondition not met — {detail}", "\n".join(lines)
