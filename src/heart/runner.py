"""Headless agent execution inside a workspace."""
from __future__ import annotations

import os
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
ALIASES = {"local": "api"}

# global cap on concurrent agents: batch --parallel times --candidates can
# otherwise oversubscribe API rate limits or a single local vLLM
_GATE = threading.BoundedSemaphore(int(os.environ.get("HEART_MAX_AGENTS", "8")))


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
    base = ALIASES.get(base, base)
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
