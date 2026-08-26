"""One episode: reset repo -> orchestrate agents -> capture diff -> verify on a
clean checkout -> score -> persist. This is the vertical slice everything else
feeds.

Orchestration logic (coding-specific, borrowed from what works):
- verify-fix loop: run verifiers in the workspace after implementation; on
  failure, hand the failing output to a fix agent for up to fix_rounds attempts
  (evaluator-optimizer with a ground-truth evaluator). Optionally escalate to a
  stronger model on the final attempt.
- role pipeline: implement/test/review subagents with per-role arteries memory
  modes (implementer normal, test-writer clean, reviewer readonly).
- candidates: N independent episodes in parallel worktrees, best one wins
  (orchestrator-workers / best-of-N; doubles as the RL data engine).
"""
from __future__ import annotations

import concurrent.futures
import datetime
import json
import os
import re
import uuid
from pathlib import Path

from . import reward as reward_mod
from . import router
from .env import Workspace
from .events import emit
from .guard import scan_secrets
from .runner import run_agent
from .taskspec import TaskSpec
from .verify import run_verifiers

# Memory modes follow the handoff doc's subagent pattern: implementer sees
# project memory, test-writer runs clean so tests aren't biased by the
# implementer's assumptions, reviewer reads memory but leaves no trace.
# Outcomes carrying no correctness signal, so reward is None rather than 0.0.
# 0.0 asserts the episode did badly; None says the axes do not apply. Getting
# this wrong is worse than it looks -- compute() renormalizes over surviving
# components, and the survivors are diff_quality and efficiency, so a do-nothing
# episode would score ~0.97.
#
# scope_denied is here for the mirror-image reason: the sandbox refused the
# writes, so the model never got to produce a signal. Scoring it 0.0 would blame
# the model for a misconfiguration and teach it the task is impossible.
UNSCOREABLE = ("blocked", "unverified", "scope_denied")

DEFAULT_ROLES: list[dict] = [
    {"name": "implement", "memory": "normal", "verify_after": True, "prompt": "{prompt}"},
    {
        "name": "test",
        "memory": "clean",
        "tier": "cheap",  # routine work when routing (--agent auto) is on
        "prompt": (
            "Run `git diff` to see changes made for the task below. Add or strengthen "
            "tests covering those changes, then run the test suite.\nTask: {prompt}"
        ),
    },
    {
        "name": "review",
        "memory": "readonly",
        "prompt": (
            "Run `git diff` and review all changes for the task below: correctness, "
            "unintended edits, missing tests. Tests added by the pipeline's test "
            "role are expected and in scope. Your final line must be exactly "
            "APPROVE or REJECT followed by a one-line reason.\nTask: {prompt}"
        ),
    },
]


def _review_verdict(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    hits = re.findall(r"\b(APPROVE|REJECT)\b", log_path.read_text(errors="replace"))
    return hits[-1].lower() if hits else None


def _review_notes(log_path: Path, limit: int = 1500) -> str:
    """The reviewer's reasoning, whatever the verdict was.

    Only a rejection currently does anything with this: `review-fix` gets the
    log tail. An APPROVE that also says "this works but the retry logic will
    break under concurrency" is read by nobody and deleted with the run
    directory -- the most considered thing an episode produced, thrown away for
    passing.

    Emitting it puts the reasoning in the journal, which is the path into
    arteries' memory. Insight from a review is worth remembering precisely
    because it survives the episode that generated it.
    """
    if not log_path.exists():
        return ""
    text = log_path.read_text(errors="replace").strip()
    return text[-limit:]


def _consume_steer(out: Path) -> str | None:
    """Read+clear a mid-run steer note the dashboard dropped into the
    episode's own out dir (`serve.py`'s POST /api/steer writes steer.txt
    there). Returns None when absent or whitespace-only."""
    path = out / "steer.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return None
    path.write_text("")
    return text


def _failure_tail(results: dict[str, dict]) -> str:
    return "\n".join(
        f"[{name}] FAILED (exit {r['exit_code']}):\n{r['output_tail'][-1500:]}"
        for name, r in results.items() if not r["passed"]
    )


def _diff_paths(diff_text: str) -> set[str]:
    # ponytail: parses ---/+++ headers only; git renames also emit these, so
    # rename-only tricks still surface here
    paths = set()
    for line in diff_text.splitlines():
        for prefix in ("--- a/", "+++ b/"):
            if line.startswith(prefix):
                paths.add(line[len(prefix):])
    return paths


def _within(path: str, prefix: str) -> bool:
    """True if `path` is `prefix` (a file) or lives under it (a dir), matched at
    path boundaries — so `src` covers `src/a.py` but NOT `src_gen/a.py`, which a
    bare startswith would wrongly allow/deny."""
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def path_violations(diff_text: str, allowed: list[str], denied: list[str]) -> list[str]:
    bad = []
    for p in _diff_paths(diff_text):
        if any(_within(p, d) for d in denied):
            bad.append(p)
        elif allowed and not any(_within(p, a) for a in allowed):
            bad.append(p)
    return sorted(bad)


def _blocked_reason(diff_text: str, marker: str | None) -> str | None:
    """The decision the agent stopped for, if it stopped for one. Scans the
    captured diff rather than the agent log: callers tell agents to write the
    marker to a file (heart captures untracked files in the diff), which
    survives an agent that streams nothing useful to stdout."""
    if not marker:
        return None
    for line in diff_text.splitlines():
        stripped = (line[1:] if line.startswith("+") else line).strip()
        if stripped.startswith(marker):
            return stripped[len(marker):].strip() or "(no reason given)"
    return None


# What a refused write looks like coming back from the kernel, a shell, or git.
# Collected from the agent's own log because that is where the failure surfaces:
# the diff shows only what the agent managed to write, which is precisely the
# evidence a denial removes.
_DENIAL_SIGNS = (
    "read-only file system",
    "permission denied",
    "operation not permitted",
    "insufficient permission for adding an object",
    "erofs",
    "eacces",
)


# Paths named in a refusal, so a denial can be attributed. Quoted first, since
# both Python and git quote the offending path; a bare token with a separator
# is the fallback for shells that do not.
_DENIED_PATH_RE = re.compile(r"""['"]([^'"\n]{1,200})['"]|(?<![\w/])([\w.-]+/[\w./-]+)""")


def _denial_paths(line: str) -> list[str]:
    return [(a or b).lstrip("/") for a, b in _DENIED_PATH_RE.findall(line) if (a or b)]


def _probed_forbidden(denials: list[str], denied_paths: list[str]) -> bool:
    """True when a refusal names a path the spec explicitly forbids.

    The distinction decides whether an episode is unscoreable or a violation,
    and without it `scope_denied` is a reward hack: an agent heading for a bad
    score writes one byte into a denied path, produces an empty diff plus a
    refusal, and escapes being scored at all. A refusal on forbidden ground is
    the agent's doing; a refusal on ground the spec permitted is ours.
    """
    if not denied_paths:
        return False
    return any(
        any(_within(path, d) for d in denied_paths)
        for line in denials for path in _denial_paths(line)
    )


def _scope_denials(out: Path, limit: int = 12) -> list[str]:
    """Lines from this episode's agent logs that look like the sandbox refusing.

    A profile derived from a task spec is a prediction of what the work needs,
    and predictions are wrong. When one is too tight the agent writes nothing
    and the diff comes back empty -- which the outcome ladder below would read
    as `no_change`, i.e. "the agent did nothing". That is a lie about what
    happened, and the expensive kind: the task looks unproductive rather than
    misconfigured, and the fix (widen allowed_paths) is invisible.

    Scanning for the refusal turns a silent misattribution into a named
    outcome. It over-reports by design -- a test asserting on EACCES will match
    -- so `scope_denied` is only ever reached when the diff is otherwise empty.
    """
    hits: list[str] = []
    for log in sorted(out.glob("*.log")):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            low = line.lower()
            if any(sign in low for sign in _DENIAL_SIGNS):
                hits.append(f"{log.stem}: {line.strip()[:200]}")
                if len(hits) >= limit:
                    return hits
    return hits


def _sandbox_profiles(task: TaskSpec, ws_path, episode_id: str, out: Path):
    """(agent, verifier) container profiles, or (None, None) when not in docker mode.

    Built once per episode because the container is per-episode: the agent and
    the verifiers share one worktree, one image and one journal inbox, and
    differ only in what they are allowed to do with them.

    Returning None outside docker mode is what keeps bwrap and off untouched --
    sandbox_wrap falls through to its existing branches when no profile arrives.
    """
    if os.environ.get("HEART_SANDBOX") != "docker":
        return None, None
    from . import sandbox

    # Read-only, built on the host: whatever the agent is given to work from.
    # The container never fetches it, so it needs no credential to read memory.
    context = out / "context"
    context.mkdir(parents=True, exist_ok=True)
    inbox = sandbox.inbox_for(episode_id)
    env = {k: v for k, v in os.environ.items() if k.startswith("ARTERIES_")}
    return (
        sandbox.profile_for(task, ws_path, context, inbox, env=env),
        sandbox.verifier_profile_for(task, ws_path, inbox, env=env),
    )


def _agent_turn(
    role: str, agent: str, prompt: str, ws: Workspace, env: dict,
    task: TaskSpec, out: Path, agent_cmd: str | None, runs_log: list[dict],
    memory: str | None = None, profile=None,
) -> dict:
    """One agent invocation: run, record in runs_log, emit role.finished."""
    r = run_agent(
        agent, prompt, str(ws.path), {**env, "HEART_ROLE": role},
        task.timeout_seconds, out / f"{role}.log", agent_cmd=agent_cmd,
        profile=profile,
    )
    runs_log.append(
        {"role": role, "agent": agent, **({"memory": memory} if memory else {}), **r}
    )
    emit("heart", "role.finished", episode_id=env.get("ARTERIES_EPISODE_ID"),
         task_id=task.task_id, role=role, duration_ms=int(r["duration_s"] * 1000),
         agent=agent, exit_code=r["exit_code"], timed_out=r["timed_out"],
         tokens_in=r.get("tokens_in"), tokens_out=r.get("tokens_out"),
         cache_read=r.get("cache_read"), cache_write_5m=r.get("cache_write_5m"),
         cache_write_1h=r.get("cache_write_1h"),
         cost_usd=r.get("cost_usd"))
    return r


def _fix_loop(
    task: TaskSpec, ws: Workspace, out: Path, agent: str, env: dict,
    fix_rounds: int, escalate: str | None, agent_cmd: str | None,
    runs_log: list[dict], agent_profile=None, verifier_profile=None,
) -> list[dict]:
    """In-workspace verify; on failure, feed the failing output to a fix agent."""
    rounds: list[dict] = []
    episode_id = env.get("ARTERIES_EPISODE_ID")
    for attempt in range(fix_rounds + 1):
        results = run_verifiers(task.public_verifiers, str(ws.path), task.timeout_seconds,
                                profile=verifier_profile)
        passed = all(r["passed"] for r in results.values())
        rounds.append({"attempt": attempt, "passed": passed})
        emit("heart", "verify.round", episode_id=episode_id, task_id=task.task_id,
             attempt=attempt, passed=passed)
        if passed or attempt == fix_rounds:
            break
        fix_agent = escalate if (escalate and attempt == fix_rounds - 1) else agent
        prompt = (
            f"These verifier commands failed:\n{_failure_tail(results)}\n"
            f"Fix the code so they pass. Do not weaken or delete tests.\n"
            f"Original task: {task.prompt}"
        )
        steer = _consume_steer(out)
        if steer:
            prompt += f"\n\nOperator note (mid-run steer): {steer}"
            emit("heart", "steer.received", episode_id=episode_id, task_id=task.task_id,
                 chars=len(steer))
        _agent_turn(f"fix{attempt + 1}", fix_agent, prompt, ws, env, task, out,
                    agent_cmd, runs_log, profile=agent_profile)
    return rounds


def run_episode(
    task: TaskSpec,
    agent: str = "claude",
    memory_mode: str = "normal",
    retrieval: bool = True,
    runs_dir: str | Path = "runs",
    agent_cmd: str | None = None,
    roles: list[dict] | None = None,
    fix_rounds: int = 0,
    escalate: str | None = None,
    isolated: bool = False,
    parent_agent_id: str | None = None,
) -> dict:
    episode_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    out = Path(runs_dir) / episode_id
    out.mkdir(parents=True, exist_ok=True)
    routed = agent == "auto"
    if routed:
        tier, signals = router.classify(task)
        agent = router.resolve(tier)
        if escalate is None:
            escalate = router.resolve("strong", default=agent)
        emit("heart", "route.decided", episode_id=episode_id, task_id=task.task_id,
             tier=tier, agent=agent, **signals)
    try:
        return _run_episode(
            task, agent, memory_mode, retrieval, agent_cmd, roles,
            fix_rounds, escalate, episode_id, out, routed, isolated,
            parent_agent_id,
        )
    except Exception as exc:
        # a crash must be a visible error signal, not a silent gap in the journal
        emit("heart", "episode.failed", episode_id=episode_id, task_id=task.task_id,
             error=f"{type(exc).__name__}: {exc}")
        raise


def _subagent_env(parent_agent_id: str) -> dict[str, str]:
    """Mark this episode as a subagent of `parent_agent_id`, using arteries'
    identity contract as the source of truth. Falls back to an inline copy so
    heart still runs where arteries isn't installed — heart imports arteries,
    never the reverse."""
    try:
        from arteries.subagent import subagent_env
        return subagent_env(parent_agent_id)
    except Exception:
        return {"ARTERIES_PARENT_AGENT_ID": parent_agent_id,
                "ARTERIES_AGENT_ID": f"{parent_agent_id}-sub-{uuid.uuid4().hex[:8]}",
                "ARTERIES_AGENT_ROLE": "subagent"}


def _memory_env(memory_mode: str, retrieval: bool, isolated: bool,
                parent_agent_id: str | None) -> dict[str, str]:
    """The arteries memory policy for one episode's child process. The one branch
    that matters: an orchestration subagent (parent_agent_id set) inherits the
    parent lineage and MUST NOT get ARTERIES_EPHEMERAL=discard — its ephemeral has
    to survive to compile up under the parent. Only a redundant best-of-N
    candidate (isolated, no parent) discards, so its arm can't feed the others."""
    env: dict[str, str] = {}
    if parent_agent_id:
        env.update(_subagent_env(parent_agent_id))
    if memory_mode != "normal":
        env["ARTERIES_MEMORY"] = memory_mode
    if not retrieval:
        env["ARTERIES_RETRIEVAL"] = "off"
    if isolated and not parent_agent_id:
        env["ARTERIES_EPHEMERAL"] = "discard"
    return env


def _run_episode(
    task: TaskSpec, agent: str, memory_mode: str, retrieval: bool,
    agent_cmd: str | None, roles: list[dict] | None,
    fix_rounds: int, escalate: str | None, episode_id: str, out: Path,
    routed: bool = False, isolated: bool = False, parent_agent_id: str | None = None,
) -> dict:
    repo = Path(task.repo_path).resolve()
    env = {
        "ARTERIES_EPISODE_ID": episode_id, "ARTERIES_TASK_ID": task.task_id,
        # attribute events to the source repo, not the random worktree dir;
        # ARTERIES_REPO also anchors JSONL fallbacks somewhere that survives destroy
        "ARTERIES_PROJECT": repo.name, "ARTERIES_REPO": str(repo),
    }
    env.update(_memory_env(memory_mode, retrieval, isolated, parent_agent_id))

    emit("heart", "episode.started", episode_id=episode_id, task_id=task.task_id,
         agent=agent, memory_mode=memory_mode, retrieval=retrieval,
         base_commit=task.base_commit[:12], fix_rounds=fix_rounds,
         pipeline=[r["name"] for r in roles] if roles else "solo")
    ws = Workspace(task.repo_path, task.base_commit, overlay=task.overlay_files)
    agent_profile, verifier_profile = _sandbox_profiles(task, ws.path, episode_id, out)
    clean = None
    runs_log: list[dict] = []
    verify_rounds: list[dict] = []
    review_verdict: str | None = None
    verifier_results: dict[str, dict] = {}
    hidden_results: dict[str, dict] = {}
    blocked_reason: str | None = None
    diff = ""
    can_fix = fix_rounds > 0 and bool(task.public_verifiers)
    try:
        for role in roles or [{"name": "solo", "memory": memory_mode,
                               "verify_after": can_fix, "prompt": "{prompt}"}]:
            role_env = dict(env)
            mem = role.get("memory", memory_mode)
            role_env.pop("ARTERIES_MEMORY", None)
            if mem != "normal":
                role_env["ARTERIES_MEMORY"] = mem
            role_agent = role.get("agent") or (
                router.resolve(role["tier"], default=agent)
                if routed and role.get("tier") else agent
            )
            prompt = role["prompt"].format(prompt=task.prompt)
            steer = _consume_steer(out)
            if steer:
                prompt += f"\n\nOperator note (mid-run steer): {steer}"
                emit("heart", "steer.received", episode_id=episode_id, task_id=task.task_id,
                     chars=len(steer))
            emit("heart", "role.started", episode_id=episode_id, task_id=task.task_id,
                 role=role["name"], agent=role_agent, memory=mem)
            _agent_turn(role["name"], role_agent, prompt,
                        ws, role_env, task, out, agent_cmd, runs_log, memory=mem,
                        profile=agent_profile)
            if role.get("verify_after") and can_fix:
                verify_rounds = _fix_loop(
                    task, ws, out, agent, env, fix_rounds, escalate, agent_cmd, runs_log,
                    agent_profile, verifier_profile
                )
        if any(r["role"] == "review" for r in runs_log):
            review_verdict = _review_verdict(out / "review.log")
            notes = _review_notes(out / "review.log")
            if notes:
                emit("heart", "review.notes", episode_id=episode_id, task_id=task.task_id,
                     verdict=review_verdict, notes=notes)
            if review_verdict == "reject" and can_fix:
                # a rejection must act, not just be recorded: one fix turn on
                # the reviewer's feedback, then a fresh verify round
                tail = (out / "review.log").read_text(errors="replace")[-1500:]
                prompt = (
                    f"A code reviewer rejected the current changes:\n{tail}\n"
                    f"Address the review feedback. Do not weaken or delete tests.\n"
                    f"Original task: {task.prompt}"
                )
                _agent_turn("review-fix", agent, prompt, ws, env, task, out,
                            agent_cmd, runs_log, profile=agent_profile)
                verify_rounds += _fix_loop(
                    task, ws, out, agent, env, 0, None, agent_cmd, runs_log,
                    agent_profile, verifier_profile
                )
                # the gate must reflect the post-fix state: without re-review,
                # a resolved rejection still blocks --apply forever
                review_role = next(r for r in roles if r["name"] == "review")
                _agent_turn("review2", agent,
                            review_role["prompt"].format(prompt=task.prompt),
                            ws, {**env, "ARTERIES_MEMORY": "readonly"}, task, out,
                            agent_cmd, runs_log, profile=agent_profile)
                review_verdict = _review_verdict(out / "review2.log") or review_verdict

        diff = ws.diff()
        (out / "diff.patch").write_text(diff)
        # The commit is the artifact; the patch stays because reward, serve, cli
        # and orchestrate all still read diff.patch, and because a text diff is
        # what the path/secret scanners below know how to inspect.
        commit_sha = ws.commit(f"heart episode {episode_id}: {task.prompt[:64]}",
                               ref=f"episodes/{episode_id}")
        if commit_sha:
            (out / "commit").write_text(commit_sha + "\n")
        emit("heart", "diff.captured", episode_id=episode_id, task_id=task.task_id,
             diff_lines=reward_mod.diff_changed_lines(diff), commit=commit_sha)

        violations = path_violations(diff, task.allowed_paths, task.denied_paths)
        secret_hits = scan_secrets(diff)
        blocked_reason = _blocked_reason(diff, task.blocked_marker)
        if violations:
            outcome = "path_violation"
        elif secret_hits:
            # mirrors path_violation handling: a secret in the diff zeroes
            # reward exactly like an out-of-bounds edit, no verify run
            outcome = "guardrail_violation"
            emit("heart", "guardrail.hit", episode_id=episode_id, task_id=task.task_id,
                 rules=sorted({h.split(":", 1)[0] for h in secret_hits}))
        elif blocked_reason:
            # the agent declined to act rather than guess. Checked before
            # no_change because a block is usually an (almost) empty diff, and
            # "wrote nothing" and "asked instead of writing" are not the same event.
            outcome = "blocked"
        elif not diff.strip() and (denials := _scope_denials(out)):
            # An empty diff plus the kernel refusing writes is a sandbox that
            # was drawn too tight, not an agent that had nothing to say -- unless
            # the refusal names ground the spec forbade, in which case the agent
            # went where it was told not to and that is a violation like any
            # other. Without this branch, scope_denied is an escape hatch from
            # being scored.
            if _probed_forbidden(denials, task.denied_paths):
                outcome = "path_violation"
                violations = sorted({p for line in denials for p in _denial_paths(line)
                                     if any(_within(p, d) for d in task.denied_paths)})
                emit("heart", "guardrail.hit", episode_id=episode_id, task_id=task.task_id,
                     rules=["denied_path_probe"], paths=violations)
            else:
                outcome = "scope_denied"
                emit("heart", "sandbox.denied", episode_id=episode_id, task_id=task.task_id,
                     allowed_paths=task.allowed_paths, denied_paths=task.denied_paths,
                     network=getattr(task, "network", "none"), evidence=denials[:5])
        elif not diff.strip():
            outcome = "no_change"
        else:
            # verify on a clean worktree with only the agent's diff applied —
            # leftover workspace state (edited tests, caches) can't game the verifier
            clean = Workspace(task.repo_path, task.base_commit, overlay=task.overlay_files)
            try:
                clean.apply(diff)
            except RuntimeError:
                outcome = "apply_failed"
            else:
                # a fresh workspace means a fresh mount table: the profile
                # above points at ws, and verifying the applied diff must read
                # the tree it was applied to
                _, clean_profile = _sandbox_profiles(task, clean.path, episode_id, out)
                verifier_results = run_verifiers(
                    task.public_verifiers, str(clean.path), task.timeout_seconds,
                    profile=clean_profile
                )
                if task.hidden_verifiers:
                    hidden_results = run_verifiers(
                        task.hidden_verifiers, str(clean.path), task.timeout_seconds,
                        profile=clean_profile
                    )
                if not verifier_results:
                    # all([]) is True, so "pass" here would be a vacuous claim:
                    # nothing was checked. check_task already refuses tasks whose
                    # verifiers pass at base for the same reason — absence of
                    # evidence must not score like evidence of correctness.
                    outcome = "unverified"
                else:
                    outcome = ("pass" if all(r["passed"] for r in verifier_results.values())
                               else "fail")
    finally:
        ws.destroy()
        if clean is not None:
            clean.destroy()

    agent_result = {
        "exit_code": 0 if all(r["exit_code"] == 0 for r in runs_log) else 1,
        "timed_out": any(r["timed_out"] for r in runs_log),
        "duration_s": round(sum(r["duration_s"] for r in runs_log), 2),
    }
    if outcome in ("pass", "fail"):
        budget = task.timeout_seconds * max(1, len(runs_log))
        score = reward_mod.compute(
            verifier_results, diff, agent_result["duration_s"], budget,
            hidden_results=hidden_results,
        )
    elif outcome in UNSCOREABLE:
        # No correctness signal exists, so there is nothing to score. 0.0 would
        # assert the episode did badly; None says the axes don't apply. Scoring
        # these is actively harmful: compute() renormalizes over the surviving
        # components, and the survivors are diff_quality and efficiency — small
        # diff, finished fast — which is exactly what stopping early looks like.
        # A blocked episode would score ~0.97 for doing nothing.
        #
        # scope_denied belongs here for the opposite reason: 0.0 would blame the
        # model for a sandbox drawn too tight. The agent never got the chance to
        # produce a correctness signal, so there is nothing to score either way,
        # and training on it would teach the model that this task is impossible.
        score = {"total": None, "components": {}}
    else:
        score = {"total": 0.0, "components": {}}

    def _sum(key: str) -> float | int | None:
        vals = [r[key] for r in runs_log if r.get(key) is not None]
        return sum(vals) if vals else None

    cost_total = _sum("cost_usd")
    usage = {
        "tokens_in": _sum("tokens_in"), "tokens_out": _sum("tokens_out"),
        # cache traffic rolls up alongside, never folded into tokens_in: they
        # are billed at different rates, so a consumer that adds them would be
        # overcharging reads tenfold
        "cache_read": _sum("cache_read"),
        "cache_write_5m": _sum("cache_write_5m"),
        "cache_write_1h": _sum("cache_write_1h"),
        "cost_usd": round(cost_total, 6) if cost_total is not None else None,
    }

    episode = {
        "episode_id": episode_id,
        "task_id": task.task_id,
        "prompt": task.prompt,
        "repo_path": task.repo_path,
        "base_commit": task.base_commit,
        "agent": agent,
        "memory_mode": memory_mode,
        "retrieval": retrieval,
        "outcome": outcome,
        "blocked_reason": blocked_reason,
        "violations": violations if outcome == "path_violation"
        else secret_hits if outcome == "guardrail_violation" else [],
        "agent_result": agent_result,
        "roles": runs_log,
        "verify_rounds": verify_rounds,
        "review_verdict": review_verdict,
        "env_snapshot": {k: v for k, v in env.items() if k.startswith("ARTERIES_")},
        "verifier_results": verifier_results,
        "hidden_verifier_results": hidden_results,
        "diff_lines": reward_mod.diff_changed_lines(diff),
        "reward": score,
        "usage": usage,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (out / "episode.json").write_text(json.dumps(episode, indent=2))
    usage_payload = {k: v for k, v in usage.items() if v is not None}
    # skills/difficulty ride the event so route.aggregate can build per-(model,
    # skill,difficulty) reward stats — the measured signal that corrects declared
    # manifest scores. Empty skills (unrouted task) simply don't feed the loop.
    route_payload = {}
    if getattr(task, "skills", None):
        route_payload = {"skills": task.skills, "difficulty": task.difficulty}
    emit("heart", "episode.finished", episode_id=episode_id, task_id=task.task_id,
         duration_ms=int(agent_result["duration_s"] * 1000), outcome=outcome,
         reward=score["total"], review_verdict=review_verdict, agent=agent,
         blocked_reason=blocked_reason, **usage_payload, **route_payload)
    return episode


def best_episode(episodes: list[dict]) -> dict:
    # unscored (blocked/unverified) sorts below every scored episode: with no
    # correctness signal there is no case for preferring it over one that has one.
    return max(episodes, key=lambda e: (e["outcome"] == "pass",
                                        _score(e) if _score(e) is not None else -1.0))


def _score(episode: dict) -> float | None:
    return episode["reward"]["total"]


def run_candidates(task: TaskSpec, n: int, parallel: int | None = None, **kwargs) -> list[dict]:
    """N independent episodes in parallel worktrees. Threads suffice: episodes
    are subprocess/IO bound."""
    if n <= 1:
        return [run_episode(task, **kwargs)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel or n) as pool:
        futures = [pool.submit(run_episode, task, isolated=True, **kwargs) for _ in range(n)]
        return [f.result() for f in futures]


