"""Headless agent execution inside a workspace."""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

# Worktrees are disposable, so agent permission prompts are disabled.
# "api" is the universal OpenAI-compatible tool-loop agent (agents_api.py) —
#   any paid or local model; select a profile with "api:<name>".
# "shell" runs the prompt as a bash script — tests and scripted baselines.
AGENT_COMMANDS: dict[str, list[str]] = {
    "claude": ["claude", "-p", "{prompt}", "--dangerously-skip-permissions"],
    "codex": ["codex", "exec", "--full-auto", "{prompt}"],
    "gemini": ["gemini", "--yolo", "-p", "{prompt}"],
    "opencode": ["opencode", "run", "{prompt}"],
    "api": ["python3", "-m", "heart.agents_api", "{prompt}"],
    "shell": ["bash", "-c", "{prompt}"],
}

# global cap on concurrent agents: batch --parallel times --candidates can
# otherwise oversubscribe API rate limits or a single local vLLM
_GATE = threading.BoundedSemaphore(int(os.environ.get("HEART_MAX_AGENTS", "8")))

# dirs the agent CLIs legitimately write to; everything else in $HOME stays
# read-only under the sandbox
_SANDBOX_WRITABLE = (
    ".cache", ".config", ".local/share", ".local/state",
    ".claude", ".claude.json", ".codex", ".gemini", ".bun", ".npm",
)
_SANDBOX_HIDDEN = (".ssh", ".aws", ".gnupg")  # tmpfs'd: not even readable


def sandbox_wrap(
    cmd: list[str] | str, shell: bool, cwd: str, extra_env: dict[str, str],
) -> tuple[list[str] | str, bool]:
    """HEART_SANDBOX=bwrap wraps the agent in bubblewrap: filesystem read-only
    except the worktree, /tmp, and agent config/cache dirs; ~/.ssh and friends
    hidden. Network stays shared (agents call APIs). Off by default — turn it
    on for unattended batches.

    ponytail: containment for accidents and reward hacking, not a security
    boundary against a hostile model — network is open, $HOME is readable.
    Upgrade path: --unshare-net + a proxy allowlist."""
    mode = os.environ.get("HEART_SANDBOX", "off")
    if mode in ("off", ""):
        return cmd, shell
    if mode != "bwrap":
        raise ValueError(f"HEART_SANDBOX={mode!r}: only 'bwrap' or 'off' supported")
    if not shutil.which("bwrap"):
        # a requested sandbox must never silently degrade to no sandbox
        raise RuntimeError("HEART_SANDBOX=bwrap but bubblewrap is not installed")
    home = Path.home()
    args = [
        "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
        "--bind", "/tmp", "/tmp", "--bind", cwd, cwd,
        "--unshare-pid", "--die-with-parent",
    ]
    for rel in _SANDBOX_WRITABLE:
        p = home / rel
        if p.exists():
            args += ["--bind", str(p), str(p)]
    for rel in _SANDBOX_HIDDEN:
        if (home / rel).exists():
            args += ["--tmpfs", str(home / rel)]
    # arteries JSONL fallback anchors at the source repo; keep that one dir writable
    repo = extra_env.get("ARTERIES_REPO")
    if repo and (Path(repo) / ".arteries").exists():
        p = str(Path(repo) / ".arteries")
        args += ["--bind", p, p]
    if shell:
        return [*args, "sh", "-c", str(cmd)], False
    return [*args, *cmd], False


def run_agent(
    agent: str,
    prompt: str,
    cwd: str,
    extra_env: dict[str, str],
    timeout: int,
    log_path: str | Path,
    agent_cmd: str | None = None,
) -> dict:
    base, _, profile = agent.partition(":")
    if profile:
        extra_env = {**extra_env, "HEART_MODEL_PROFILE": profile}

    if agent_cmd:
        # custom template runs under sh; prompt is provided as $HEART_PROMPT to
        # avoid shell-quoting the prompt into the command line
        cmd: list[str] | str = agent_cmd
        extra_env = {**extra_env, "HEART_PROMPT": prompt}
        shell = True
    elif base in AGENT_COMMANDS:
        cmd = [part.replace("{prompt}", prompt) for part in AGENT_COMMANDS[base]]
        shell = False
    else:
        raise ValueError(f"unknown agent {agent!r}; known: {sorted(AGENT_COMMANDS)}")

    cmd, shell = sandbox_wrap(cmd, shell, cwd, extra_env)
    env = {**os.environ, **extra_env}
    t0 = time.monotonic()
    timed_out = False
    with _GATE, open(log_path, "w") as log:
        try:
            proc = subprocess.run(
                cmd, shell=shell, cwd=cwd, env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            exit_code, timed_out = -1, True
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_s": round(time.monotonic() - t0, 2),
    }
