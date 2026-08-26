"""Headless agent execution inside a workspace."""
from __future__ import annotations

import contextlib
import datetime
import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from . import agents_api

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


def _slots_base() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir())


@contextlib.contextmanager
def _flock_pool(d: Path, n: int):
    """Counting semaphore over n flock'd files in dir d. The kernel drops a
    flock the instant its holder dies, so a crashed or killed agent never
    wedges a slot — the same property serve.py's liveness probe relies on.
    # ponytail: 0.2s poll when all slots are busy — fine at agent timescales
    # (seconds to minutes); a proper wait needs an IPC semaphore dependency.
    """
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


@contextlib.contextmanager
def _global_slot():
    """Cross-process cap on concurrent agents, opt-in via HEART_MAX_AGENTS_GLOBAL.

    _GATE bounds one process, but several `plexus run` processes (or `heart batch`
    runs) sharing one API key or one local vLLM sum to N*_GATE and blow the rate
    limit. A pool of N flock'd slot files caps the total across every heart
    process on the machine. Unset or <=0 -> no-op, so default behavior is
    unchanged.
    """
    n = int(os.environ.get("HEART_MAX_AGENTS_GLOBAL", "0") or 0)
    if n <= 0:
        yield
        return
    with _flock_pool(_slots_base() / "heart-agent-slots", n):
        yield


@contextlib.contextmanager
def _local_slot(endpoint: str | None):
    """Cross-process cap on concurrent agents hitting one local model server,
    opt-in via HEART_LOCAL_SLOTS. This is the fleet knob: many `plexus run`
    goals routing cheap-tier work to the same local vLLM oversubscribe the one
    GPU, while their paid-API roles (endpoint None here) must stay unthrottled.
    Keyed by host:port, so profiles that point at the *same* server share one
    pool while two servers (say one llama.cpp per GPU on :8000 and :8001) get
    independent pools — each GPU is bounded on its own, and HEART_LOCAL_SLOTS is
    the per-server ceiling. One server spanning both GPUs (tensor-parallel vLLM
    on a single port) is a single pool, which is also correct: it's one queue.
    Non-local or unset -> no-op.
    """
    n = int(os.environ.get("HEART_LOCAL_SLOTS", "0") or 0)
    if not endpoint or n <= 0:
        yield
        return
    key = (urlsplit(endpoint).netloc or "local").replace(":", "_")
    with _flock_pool(_slots_base() / "heart-local-slots" / key, n):
        yield

# dirs the agent CLIs legitimately write to; everything else in $HOME stays
# read-only under the sandbox
_SANDBOX_WRITABLE = (
    ".cache", ".config", ".local/share", ".local/state",
    ".claude", ".claude.json", ".codex", ".gemini", ".bun", ".npm",
)
_SANDBOX_HIDDEN = (".ssh", ".aws", ".gnupg")  # tmpfs'd: not even readable


def sandbox_wrap(
    cmd: list[str] | str, shell: bool, cwd: str, extra_env: dict[str, str],
    *, mode: str | None = None, profile=None,
) -> tuple[list[str] | str, bool]:
    """HEART_SANDBOX=bwrap wraps the agent in bubblewrap: filesystem read-only
    except the worktree, /tmp, and agent config/cache dirs; ~/.ssh and friends
    hidden. Network stays shared (agents call APIs). Off by default — turn it
    on for unattended batches.

    HEART_SANDBOX=bwrap-nonet additionally unshares the network — for verifier
    runs and local-model agents that need no egress at all.

    HEART_SANDBOX=docker runs the command in a container instead, described by
    the `profile` a caller built from the task spec (see sandbox.profile_for and
    sandbox.verifier_profile_for). The three modes are one axis of increasing
    strength, so callers keep passing commands through here and never learn
    which one is in force.

    ponytail: containment for accidents and reward hacking, not a security
    boundary against a hostile model — in plain bwrap mode network is open and
    $HOME is readable. Upgrade path for API agents: a proxy allowlist."""
    if mode is None:
        mode = os.environ.get("HEART_SANDBOX", "off")
    if mode in ("off", ""):
        return cmd, shell
    if mode == "docker":
        if profile is None:
            # a requested sandbox must never silently degrade to no sandbox --
            # the same rule the missing-bwrap check below enforces
            raise RuntimeError("HEART_SANDBOX=docker but no sandbox profile was supplied")
        if not shutil.which("docker"):
            raise RuntimeError("HEART_SANDBOX=docker but docker is not installed")
        inner = str(cmd) if shell else " ".join(shlex.quote(c) for c in cmd)
        # The timeout goes inside the container, not just on the docker client.
        # subprocess's timeout kills the client; the container it started keeps
        # running, holding the worktree and its share of the model's slots. A
        # self-terminating container needs nobody to remember to clean up.
        if profile.timeout_seconds:
            inner = f"timeout -s KILL {int(profile.timeout_seconds)}s sh -c {shlex.quote(inner)}"
        return ["docker", "run", *profile.docker_args(), "sh", "-c", inner], False
    if mode not in ("bwrap", "bwrap-nonet"):
        raise ValueError(
            f"HEART_SANDBOX={mode!r}: only 'bwrap', 'bwrap-nonet', 'docker' or 'off' supported")
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


# Cache tokens are billed against the base input rate at these multipliers.
# Both vendors price the same shape: a read is a tenth of fresh input, a write
# carries a premium for the privilege of storing it.
#
# Public because plexus prices subscription turns from its own per-provider rate
# card and needs the same multipliers. Two copies of a vendor constant in two
# repos is a drift waiting to happen, and the drift would be silent — a wrong
# cost is still a number.
CACHE_MULTIPLIERS = {"cache_read": 0.1, "cache_write_5m": 1.25, "cache_write_1h": 2.0}

# Fast mode bills Opus at 2x base ($10/$50 against $5/$25) across the whole
# context window. Modelled as a multiplier rather than a second rate row per
# model because that is how the vendor documents it, and because caching
# multipliers stack on top of it — a duplicate row would have to restate the
# whole cache table to say the same thing.
# Verified 2026-08-05 against
# platform.claude.com/docs/en/docs/about-claude/pricing#fast-mode-pricing.
# An unrecognised speed is 1.0: a new tier must not silently discount a turn.
SPEED_MULTIPLIERS = {"fast": 2.0, "standard": 1.0}


def speed_multiplier(speed: str | None) -> float:
    return SPEED_MULTIPLIERS.get(str(speed or "standard").lower(), 1.0)
_USAGE_KEYS = ("tokens_in", "tokens_out", *CACHE_MULTIPLIERS)


def _extract_usage(log_path: str | Path, base_agent: str) -> dict:
    """Pull token counts out of a role log, per agent family. Never raises: a
    log that doesn't parse (older CLI, crash, empty timeout log) just means
    honest Nones, not a broken episode.

    `tokens_in` is *uncached* input only — that is what the vendors report
    under that name, and conflating it with cache traffic would misprice both.
    An agentic loop re-reads a growing cached prefix every turn, so cache reads
    are usually the bulk of what a turn actually sends; counting only
    `tokens_in` is why a real turn could log 15 input tokens against 2,692 out.
    Cache counts are 0 when the vendor reports none, never None: absent means
    the turn had no cache traffic, which is a number, unlike a log we failed to
    parse at all."""
    none = dict.fromkeys(_USAGE_KEYS, None)
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
        # `cache_creation` breaks the write down by TTL, and the two are priced
        # differently (1.25x vs 2x), so use it when present rather than lumping
        # the total into one bucket and averaging the error.
        creation = usage.get("cache_creation") or {}
        write_5m = creation.get("ephemeral_5m_input_tokens")
        write_1h = creation.get("ephemeral_1h_input_tokens")
        if write_5m is None and write_1h is None:
            # older CLI: only the flat total, which is the 5m case in practice
            write_5m, write_1h = usage.get("cache_creation_input_tokens") or 0, 0
        cache = {
            "cache_read": usage.get("cache_read_input_tokens") or 0,
            "cache_write_5m": write_5m or 0,
            "cache_write_1h": write_1h or 0,
        }
        # downstream code greps role logs for verdicts/failure tails and expects
        # plain text, so rewrite the log to just the result — but keep the full
        # envelope alongside so `pulse episode <id>` drill-down isn't reduced to
        # the final message (it's a *.log so the board still surfaces it).
        try:
            log_path.with_name(log_path.stem + ".raw.log").write_text(text, encoding="utf-8")
        except OSError:
            pass
        log_path.write_text(result, encoding="utf-8")
        return {"tokens_in": tokens_in, "tokens_out": tokens_out, **cache}
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
                # OpenAI-shaped usage reports cached input under
                # prompt_tokens_details.cached_tokens; a local server reports
                # nothing, which is 0 traffic rather than an unparsed log
                return {"tokens_in": payload.get("tokens_in"),
                        "tokens_out": payload.get("tokens_out"),
                        "cache_read": payload.get("cache_read") or 0,
                        "cache_write_5m": payload.get("cache_write_5m") or 0,
                        "cache_write_1h": payload.get("cache_write_1h") or 0}
        return none
    return none


def models_json_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "heart" / "models.json"


def _load_models_json() -> dict:
    try:
        return json.loads(models_json_path().read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _effective_row(rows: object, on: str) -> dict | None:
    """The rate row in force on `on` (YYYY-MM-DD).

    A model may carry several rows, each with an `effective_from`: vendors
    announce price changes ahead of the date (Claude Sonnet 5 moved from $2/$10
    to $3/$15 on 2026-09-01, published weeks earlier). Recording the future row
    now means the change lands on the day instead of whenever someone
    remembers. A row with no `effective_from` has always been in force.
    """
    candidates = rows if isinstance(rows, list) else [rows]
    best = None
    for row in candidates:
        if not isinstance(row, dict):
            continue
        starts = str(row.get("effective_from") or "")
        if starts and starts > on:
            continue  # not yet in force
        if best is None or starts >= str(best.get("effective_from") or ""):
            best = row
    return best


def model_pricing(on: str | None = None) -> dict[str, dict[str, float]]:
    """Concrete model id -> {"input", "output"} USD per million tokens.

    Two sources, in order of precedence:

      model_pricing  keyed by the model id a CLI transcript actually reports
                     (`claude-opus-5`). This is what `heart models set-price`
                     writes, with the vendor URL and date it was verified.
      pricing        the legacy `provider:profile` map, joined through
                     `profiles` to recover a model id. Kept so an existing card
                     keeps working untouched.

    A model absent from both is absent here rather than zero, so the caller can
    fall back to a provider-wide rate instead of billing it as free.
    """
    config = _load_models_json()
    on = on or datetime.date.today().isoformat()
    out: dict[str, dict[str, float]] = {}

    def put(model: str, row: dict) -> None:
        try:
            out[str(model)] = {"input": float(row["in_per_mtok"]),
                               "output": float(row["out_per_mtok"])}
        except (KeyError, TypeError, ValueError):
            pass

    pricing = config.get("pricing") or {}
    for profile, spec in (config.get("profiles") or {}).items():
        model = (spec or {}).get("model")
        if not model:
            continue
        for key, entry in pricing.items():
            _, _, key_profile = str(key).partition(":")
            if key_profile == profile:
                row = _effective_row(entry, on)
                if row:
                    put(model, row)

    # model-id keyed entries win: they name the exact model the vendor prices,
    # where the profile join only ever infers it
    for model, rows in (config.get("model_pricing") or {}).items():
        row = _effective_row(rows, on)
        if row:
            put(model, row)
    return out


def pricing_provenance() -> dict[str, dict]:
    """Per model: where its rate came from and when that was checked.

    Cost telemetry that cannot say how old it is invites being trusted longer
    than it deserves. `heart models check` reads this to age the card.
    """
    out = {}
    for model, rows in (_load_models_json().get("model_pricing") or {}).items():
        candidates = rows if isinstance(rows, list) else [rows]
        for row in candidates:
            if not isinstance(row, dict):
                continue
            prior = out.get(model) or {}
            verified = str(row.get("verified") or "")
            if not prior or verified >= str(prior.get("verified") or ""):
                out[model] = {"verified": verified or None,
                              "source": row.get("source"),
                              "effective_from": row.get("effective_from")}
    return out


def set_model_price(model: str, in_per_mtok: float, out_per_mtok: float, *,
                    source: str, verified: str | None = None,
                    effective_from: str | None = None) -> dict:
    """Record a rate for one model, with where it came from.

    `source` is required and is meant to be the vendor's own pricing page. A
    rate with no provenance cannot be re-checked later, and a cost nobody can
    re-check is one people stop questioning — which is how a stale number
    survives a price change.

    An `effective_from` row is added alongside the existing ones rather than
    replacing them, so a scheduled change can be entered the day it is
    announced and takes effect on its own.
    """
    if not source:
        raise ValueError("source is required: a rate with no provenance cannot be re-verified")
    for value in (in_per_mtok, out_per_mtok):
        if float(value) < 0:
            raise ValueError("rates cannot be negative")
    path = models_json_path()
    config = _load_models_json()
    table = config.setdefault("model_pricing", {})
    row = {"in_per_mtok": float(in_per_mtok), "out_per_mtok": float(out_per_mtok),
           "source": source, "verified": verified or datetime.date.today().isoformat()}
    if effective_from:
        row["effective_from"] = effective_from
        rows = table.get(model)
        rows = list(rows) if isinstance(rows, list) else ([rows] if rows else [])
        # one row per start date: re-running the same correction updates it
        rows = [r for r in rows
                if not (isinstance(r, dict) and r.get("effective_from") == effective_from)]
        table[model] = rows + [row]
    else:
        table[model] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    return row


def _price(agent: str, tokens_in: int | None, tokens_out: int | None,
           cache_read: int | None = 0, cache_write_5m: int | None = 0,
           cache_write_1h: int | None = 0) -> float | None:
    """Dollars for one turn. Cache buckets default to 0 so every existing
    two-argument caller keeps working and keeps its old answer; passing them is
    what makes the answer right for a cached prompt."""
    if tokens_in is None or tokens_out is None:
        return None
    base, _, profile = agent.partition(":")
    # A local model server costs nothing to run, so it's free regardless of the
    # pricing map — this guard is what lets a broad "api" pricing entry bill
    # every metered profile without also billing the local one. Everything else
    # (metered APIs and subscription seats alike) prices at the map's API rates:
    # a Pro/Max seat's marginal cost is zero, but the fleet accounts for the
    # spend it *would* cost, so routing decisions see a real dollar signal.
    if base == "api":
        try:
            if agents_api.is_local_endpoint(agents_api.endpoint_for(profile)):
                return 0.0
        except Exception:
            pass
    path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "heart" / "models.json"
    try:
        pricing = json.loads(path.read_text()).get("pricing", {})
    except (OSError, json.JSONDecodeError):
        return None
    entry = pricing.get(agent) or pricing.get(base)
    if not entry:
        return None
    try:
        in_rate = entry["in_per_mtok"]
        out_rate = entry["out_per_mtok"]
    except KeyError:
        return None
    # every cache bucket is priced off the base input rate, per its multiplier
    cached = {"cache_read": cache_read, "cache_write_5m": cache_write_5m,
              "cache_write_1h": cache_write_1h}
    cache_cost = sum((cached[k] or 0) * in_rate * mult for k, mult in CACHE_MULTIPLIERS.items())
    return round((tokens_in * in_rate + tokens_out * out_rate + cache_cost) / 1e6, 6)


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
    profile=None,
) -> dict:
    base, _, profile = agent.partition(":")
    if profile:
        extra_env = {**extra_env, "HEART_MODEL_PROFILE": profile}
    if agent_cmd:
        # custom template runs under sh; prompt is provided as $HEART_PROMPT to
        # avoid shell-quoting the prompt into the command line
        extra_env = {**extra_env, "HEART_PROMPT": prompt}
    cmd, shell = _agent_command(agent, prompt, agent_cmd)
    cmd, shell = sandbox_wrap(cmd, shell, cwd, extra_env, profile=profile)
    # HEART_TIER_* is this process's routing config, never the child's: a
    # nested heart invocation (agents working on heart itself) must not
    # inherit ambient tier overrides — that leak broke real episodes once
    env = {k: v for k, v in os.environ.items() if not k.startswith("HEART_TIER_")}
    env.update(extra_env)
    # Only api agents pointing at a local server take a local slot; paid APIs
    # and CLI agents pass endpoint=None and skip that gate. Probe never raises
    # (endpoint_for is tolerant) — an uncertain probe just means "not local".
    local_endpoint = None
    if base == "api":
        try:
            ep = agents_api.endpoint_for(profile)
            if agents_api.is_local_endpoint(ep):
                local_endpoint = ep
        except Exception:
            local_endpoint = None
    t0 = time.monotonic()
    timed_out = False
    with _GATE, _global_slot(), _local_slot(local_endpoint), open(log_path, "w") as log:
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
        **{k: u[k] for k in _USAGE_KEYS},
        "cost_usd": _price(agent, u["tokens_in"], u["tokens_out"],
                           u["cache_read"], u["cache_write_5m"], u["cache_write_1h"]),
    }
