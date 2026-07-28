"""Headless agent execution inside a workspace."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

# Worktrees are disposable, so agent permission prompts are disabled.
# "api" is the universal OpenAI-compatible tool-loop agent (agents_api.py) —
#   any paid or local model; select a profile with "api:<name>".
# "shell" runs the prompt as a bash script — tests and scripted baselines.
AGENT_COMMANDS: dict[str, list[str]] = {
    "claude": ["claude", "-p", "{prompt}", "--dangerously-skip-permissions",
               "--output-format", "json"],
    "codex": ["codex", "exec", "--full-auto", "{prompt}"],
    "gemini": ["gemini", "--yolo", "-p", "{prompt}"],
    "opencode": ["opencode", "run", "{prompt}"],
    "api": ["python3", "-m", "heart.agents_api", "{prompt}"],
    "shell": ["bash", "-c", "{prompt}"],
}

# CLI agents that take a `--model <id>` flag, so a task can pin a specific model
# on a subscription seat (Claude Pro/Max, ChatGPT/codex) instead of the metered
# API. `claude:sonnet` -> `claude … --model claude-sonnet-5`. The `api` agent is
# absent on purpose: it selects its model from the profile inside agents_api.py.
_MODEL_FLAG = {"claude": "--model", "codex": "--model", "gemini": "--model", "opencode": "--model"}

# per-process cap on concurrent agents: batch --parallel times --candidates can
# otherwise oversubscribe API rate limits or a single local vLLM
_GATE = threading.BoundedSemaphore(int(os.environ.get("HEART_MAX_AGENTS", "8")))


@contextlib.contextmanager
def _global_slot():
    """Cross-process cap on concurrent agents, opt-in via HEART_MAX_AGENTS_GLOBAL.

    _GATE bounds one process, but several `plexus run` processes (or `heart batch`
    runs) sharing one API key or one local vLLM sum to N*_GATE and blow the rate
    limit. A pool of N flock'd slot files caps the total across every heart
    process on the machine. Unset or <=0 -> no-op, so default behavior is
    unchanged.
    # ponytail: 0.2s poll when all slots are busy — fine at agent timescales
    # (seconds to minutes); a proper wait needs an IPC semaphore dependency.
    """
    n = int(os.environ.get("HEART_MAX_AGENTS_GLOBAL", "0") or 0)
    if n <= 0:
        yield
        return
    d = Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()) / "heart-agent-slots"
    d.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(n):
            f = open(d / f"slot{i}", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            return
        time.sleep(0.2)

# dirs the agent CLIs legitimately write to; everything else in $HOME stays
# read-only under the sandbox
_SANDBOX_WRITABLE = (
    ".cache", ".config", ".local/share", ".local/state",
    ".claude", ".claude.json", ".codex", ".gemini", ".bun", ".npm",
)
_SANDBOX_HIDDEN = (".ssh", ".aws", ".gnupg")  # tmpfs'd: not even readable


def sandbox_wrap(
    cmd: list[str] | str, shell: bool, cwd: str, extra_env: dict[str, str],
    *, mode: str | None = None,
) -> tuple[list[str] | str, bool]:
    """HEART_SANDBOX=bwrap wraps the agent in bubblewrap: filesystem read-only
    except the worktree, /tmp, and agent config/cache dirs; ~/.ssh and friends
    hidden. Network stays shared (agents call APIs). Off by default — turn it
    on for unattended batches.

    HEART_SANDBOX=bwrap-nonet additionally unshares the network — for verifier
    runs and local-model agents that need no egress at all.

    ponytail: containment for accidents and reward hacking, not a security
    boundary against a hostile model — in plain bwrap mode network is open and
    $HOME is readable. Upgrade path for API agents: a proxy allowlist."""
    if mode is None:
        mode = os.environ.get("HEART_SANDBOX", "off")
    if mode in ("off", ""):
        return cmd, shell
    if mode not in ("bwrap", "bwrap-nonet"):
        raise ValueError(f"HEART_SANDBOX={mode!r}: only 'bwrap', 'bwrap-nonet' or 'off' supported")
    if not shutil.which("bwrap"):
        # a requested sandbox must never silently degrade to no sandbox
        raise RuntimeError("HEART_SANDBOX=bwrap but bubblewrap is not installed")
    home = Path.home()
    args = [
        "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
        "--bind", "/tmp", "/tmp", "--bind", cwd, cwd,
        "--unshare-pid", "--die-with-parent",
    ]
    if mode == "bwrap-nonet":
        args.append("--unshare-net")
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


def _claude_envelope(text: str) -> dict | None:
    """The result envelope from a Claude CLI log.

    The log is not necessarily pure JSON: the CLI writes advisories to stdout
    ahead of the envelope ("Warning: no stdin data received in 3s..."), and a
    whole-text json.loads then fails. That failure used to be silent and
    expensive — usage came back as None *and* the log was left as raw JSON, so
    every downstream consumer that greps it for plain text (review verdicts,
    failure tails, plexus's planner) got an envelope it could not read.
    Scanning from the end also picks the final envelope if the CLI emitted
    more than one object.
    """
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("result"), str):
            return obj
    return None


def _extract_usage(log_path: str | Path, base_agent: str) -> dict:
    """Pull tokens_in/tokens_out out of a role log, per agent family. Never
    raises: a log that doesn't parse (older CLI, crash, empty timeout log)
    just means honest Nones, not a broken episode."""
    none = {"tokens_in": None, "tokens_out": None}
    log_path = Path(log_path)
    if base_agent == "claude":
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return none
        envelope = _claude_envelope(text)
        if envelope is None:
            return none
        result = envelope["result"]
        usage = envelope.get("usage") or {}
        # total_cost_usd is deliberately ignored here: cost comes only from
        # our own pricing map (_price) so subscription seats never report
        # fake dollars — Claude CLI's cost field assumes metered pricing that
        # a Pro/Max seat doesn't actually pay.
        tokens_in = usage.get("input_tokens")
        tokens_out = usage.get("output_tokens")
        # downstream code greps role logs for verdicts/failure tails and expects
        # plain text, so rewrite the log to just the result — but keep the full
        # envelope alongside so `pulse episode <id>` drill-down isn't reduced to
        # the final message (it's a *.log so the board still surfaces it).
        try:
            log_path.with_name(log_path.stem + ".raw.log").write_text(text, encoding="utf-8")
        except OSError:
            pass
        log_path.write_text(result, encoding="utf-8")
        return {"tokens_in": tokens_in, "tokens_out": tokens_out}
    if base_agent == "api":
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return none
        for line in reversed(lines):
            if line.startswith("HEART_USAGE="):
                try:
                    payload = json.loads(line[len("HEART_USAGE="):])
                except json.JSONDecodeError:
                    return none
                return {"tokens_in": payload.get("tokens_in"), "tokens_out": payload.get("tokens_out")}
        return none
    return none


def _price(agent: str, tokens_in: int | None, tokens_out: int | None) -> float | None:
    if tokens_in is None or tokens_out is None:
        return None
    path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "heart" / "models.json"
    try:
        pricing = json.loads(path.read_text()).get("pricing", {})
    except (OSError, json.JSONDecodeError):
        return None
    base = agent.partition(":")[0]
    entry = pricing.get(agent) or pricing.get(base)
    if not entry:
        return None
    try:
        in_rate = entry["in_per_mtok"]
        out_rate = entry["out_per_mtok"]
    except KeyError:
        return None
    return round(tokens_in * in_rate / 1e6 + tokens_out * out_rate / 1e6, 6)


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the timed-out agent's whole process group, then SIGKILL any
    stragglers. proc led its own session (start_new_session), so its pgid is the
    tree — grandchildren die with it instead of orphaning to init."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def _resolve_model(profile: str) -> str:
    """A CLI agent's profile token -> a concrete model id. `claude:sonnet` looks
    up models.json profiles[sonnet].model; an unknown token is used verbatim, so
    `claude:claude-opus-4-8` also works without a profile entry."""
    path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "heart" / "models.json"
    try:
        prof = json.loads(path.read_text()).get("profiles", {}).get(profile, {})
    except (OSError, json.JSONDecodeError):
        prof = {}
    return prof.get("model") or profile


def _agent_command(agent: str, prompt: str, agent_cmd: str | None = None) -> tuple[list[str] | str, bool]:
    """Build the (command, shell) for an agent string, before sandbox wrapping.
    Pins `--model` for CLI agents carrying a profile (claude:<x>, codex:<x>) —
    placed immediately before the prompt argument so it lands between the
    subcommand and a trailing positional prompt (`codex exec --model X <prompt>`,
    `claude -p --model X <prompt>`), which parses cleanly for both CLIs."""
    if agent_cmd:
        return agent_cmd, True  # caller supplies the prompt via $HEART_PROMPT
    base, _, profile = agent.partition(":")
    if base not in AGENT_COMMANDS:
        raise ValueError(f"unknown agent {agent!r}; known: {sorted(AGENT_COMMANDS)}")
    model_args = ([_MODEL_FLAG[base], _resolve_model(profile)]
                  if profile and base in _MODEL_FLAG else [])
    cmd: list[str] = []
    for part in AGENT_COMMANDS[base]:
        if part == "{prompt}":
            cmd.extend(model_args)
            cmd.append(prompt)
        else:
            cmd.append(part)
    return cmd, False


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
        extra_env = {**extra_env, "HEART_PROMPT": prompt}
    cmd, shell = _agent_command(agent, prompt, agent_cmd)
    cmd, shell = sandbox_wrap(cmd, shell, cwd, extra_env)
    # HEART_TIER_* is this process's routing config, never the child's: a
    # nested heart invocation (agents working on heart itself) must not
    # inherit ambient tier overrides — that leak broke real episodes once
    env = {k: v for k, v in os.environ.items() if not k.startswith("HEART_TIER_")}
    env.update(extra_env)
    t0 = time.monotonic()
    timed_out = False
    with _GATE, _global_slot(), open(log_path, "w") as log:
        # start_new_session so the agent is its own process-group leader: agent
        # CLIs spawn grandchildren (node, MCP servers, model procs) that a plain
        # subprocess timeout would orphan to init. On timeout we kill the whole
        # group, so nothing leaks. (bwrap's --die-with-parent covers this only in
        # sandbox mode, which is off by default.)
        proc = subprocess.Popen(
            cmd, shell=shell, cwd=cwd, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            exit_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = -1
            _kill_group(proc)
    u = _extract_usage(log_path, base)
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_s": round(time.monotonic() - t0, 2),
        "tokens_in": u["tokens_in"],
        "tokens_out": u["tokens_out"],
        "cost_usd": _price(agent, u["tokens_in"], u["tokens_out"]),
    }
