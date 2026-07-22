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
import re
import tempfile
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


def path_violations(diff_text: str, allowed: list[str], denied: list[str]) -> list[str]:
    bad = []
    for p in _diff_paths(diff_text):
        if any(p.startswith(d) for d in denied):
            bad.append(p)
        elif allowed and not any(p.startswith(a) for a in allowed):
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


def _agent_turn(
    role: str, agent: str, prompt: str, ws: Workspace, env: dict,
    task: TaskSpec, out: Path, agent_cmd: str | None, runs_log: list[dict],
    memory: str | None = None,
) -> dict:
    """One agent invocation: run, record in runs_log, emit role.finished."""
    r = run_agent(
        agent, prompt, str(ws.path), {**env, "HEART_ROLE": role},
        task.timeout_seconds, out / f"{role}.log", agent_cmd=agent_cmd,
    )
    runs_log.append(
        {"role": role, "agent": agent, **({"memory": memory} if memory else {}), **r}
    )
    emit("heart", "role.finished", episode_id=env.get("ARTERIES_EPISODE_ID"),
         task_id=task.task_id, role=role, duration_ms=int(r["duration_s"] * 1000),
         agent=agent, exit_code=r["exit_code"], timed_out=r["timed_out"],
         tokens_in=r.get("tokens_in"), tokens_out=r.get("tokens_out"),
         cost_usd=r.get("cost_usd"))
    return r


def _fix_loop(
    task: TaskSpec, ws: Workspace, out: Path, agent: str, env: dict,
    fix_rounds: int, escalate: str | None, agent_cmd: str | None,
    runs_log: list[dict],
) -> list[dict]:
    """In-workspace verify; on failure, feed the failing output to a fix agent."""
    rounds: list[dict] = []
    episode_id = env.get("ARTERIES_EPISODE_ID")
    for attempt in range(fix_rounds + 1):
        results = run_verifiers(task.public_verifiers, str(ws.path), task.timeout_seconds)
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
                    agent_cmd, runs_log)
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
        )
    except Exception as exc:
        # a crash must be a visible error signal, not a silent gap in the spool
        emit("heart", "episode.failed", episode_id=episode_id, task_id=task.task_id,
             error=f"{type(exc).__name__}: {exc}")
        raise


def _run_episode(
    task: TaskSpec, agent: str, memory_mode: str, retrieval: bool,
    agent_cmd: str | None, roles: list[dict] | None,
    fix_rounds: int, escalate: str | None, episode_id: str, out: Path,
    routed: bool = False, isolated: bool = False,
) -> dict:
    repo = Path(task.repo_path).resolve()
    env = {
        "ARTERIES_EPISODE_ID": episode_id, "ARTERIES_TASK_ID": task.task_id,
        # attribute events to the source repo, not the random worktree dir;
        # ARTERIES_REPO also anchors JSONL fallbacks somewhere that survives destroy
        "ARTERIES_PROJECT": repo.name, "ARTERIES_REPO": str(repo),
    }
    if memory_mode != "normal":
        env["ARTERIES_MEMORY"] = memory_mode
    if not retrieval:
        env["ARTERIES_RETRIEVAL"] = "off"
    if isolated:
        # parallel candidates must not feed each other memories mid-flight
        env["ARTERIES_EPHEMERAL"] = "discard"

    emit("heart", "episode.started", episode_id=episode_id, task_id=task.task_id,
         agent=agent, memory_mode=memory_mode, retrieval=retrieval,
         base_commit=task.base_commit[:12], fix_rounds=fix_rounds,
         pipeline=[r["name"] for r in roles] if roles else "solo")
    ws = Workspace(task.repo_path, task.base_commit, overlay=task.overlay_files)
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
                        ws, role_env, task, out, agent_cmd, runs_log, memory=mem)
            if role.get("verify_after") and can_fix:
                verify_rounds = _fix_loop(
                    task, ws, out, agent, env, fix_rounds, escalate, agent_cmd, runs_log
                )
        if any(r["role"] == "review" for r in runs_log):
            review_verdict = _review_verdict(out / "review.log")
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
                            agent_cmd, runs_log)
                verify_rounds += _fix_loop(
                    task, ws, out, agent, env, 0, None, agent_cmd, runs_log
                )
                # the gate must reflect the post-fix state: without re-review,
                # a resolved rejection still blocks --apply forever
                review_role = next(r for r in roles if r["name"] == "review")
                _agent_turn("review2", agent,
                            review_role["prompt"].format(prompt=task.prompt),
                            ws, {**env, "ARTERIES_MEMORY": "readonly"}, task, out,
                            agent_cmd, runs_log)
                review_verdict = _review_verdict(out / "review2.log") or review_verdict

        diff = ws.diff()
        (out / "diff.patch").write_text(diff)
        emit("heart", "diff.captured", episode_id=episode_id, task_id=task.task_id,
             diff_lines=reward_mod.diff_changed_lines(diff))

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
                verifier_results = run_verifiers(
                    task.public_verifiers, str(clean.path), task.timeout_seconds
                )
                if task.hidden_verifiers:
                    hidden_results = run_verifiers(
                        task.hidden_verifiers, str(clean.path), task.timeout_seconds
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
    elif outcome in ("blocked", "unverified"):
        # No correctness signal exists, so there is nothing to score. 0.0 would
        # assert the episode did badly; None says the axes don't apply. Scoring
        # these is actively harmful: compute() renormalizes over the surviving
        # components, and the survivors are diff_quality and efficiency — small
        # diff, finished fast — which is exactly what stopping early looks like.
        # A blocked episode would score ~0.97 for doing nothing.
        score = {"total": None, "components": {}}
    else:
        score = {"total": 0.0, "components": {}}

    def _sum(key: str) -> float | int | None:
        vals = [r[key] for r in runs_log if r.get(key) is not None]
        return sum(vals) if vals else None

    cost_total = _sum("cost_usd")
    usage = {
        "tokens_in": _sum("tokens_in"), "tokens_out": _sum("tokens_out"),
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
    emit("heart", "episode.finished", episode_id=episode_id, task_id=task.task_id,
         duration_ms=int(agent_result["duration_s"] * 1000), outcome=outcome,
         reward=score["total"], review_verdict=review_verdict,
         blocked_reason=blocked_reason, **usage_payload)
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


def _run_judge(
    task: TaskSpec, cand1: dict, cand2: dict, judge_agent: str,
    judge_cmd: str | None, runs_dir: str | Path,
) -> int | None:
    """One judge turn over the top-two diffs by reward. Returns 1, 2, or None
    (mute or unparseable judge) — a mute judge must never fail the swarm, the
    caller falls back to the reward ranking."""
    diff1 = (Path(runs_dir) / cand1["episode_id"] / "diff.patch").read_text(errors="replace")
    diff2 = (Path(runs_dir) / cand2["episode_id"] / "diff.patch").read_text(errors="replace")
    prompt = (
        f"Task: {task.prompt}\n\n"
        "Two candidate diffs both passed verification. Pick the better one on "
        "correctness, clarity, and scope.\n\n"
        f"DIFF 1:\n{diff1[:8000]}\n\nDIFF 2:\n{diff2[:8000]}\n\n"
        "Reply with brief reasoning, then a final line that is exactly "
        "`WINNER: 1` or `WINNER: 2`."
    )
    with tempfile.TemporaryDirectory() as tmp:
        # the judge only reads the diffs above, no workspace needed
        log_path = Path(tmp) / "judge.log"
        run_agent(judge_agent, prompt, tmp, {}, task.timeout_seconds, log_path,
                  agent_cmd=judge_cmd)
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
    hits = re.findall(r"WINNER:\s*([12])\b", text)
    return int(hits[-1]) if hits else None


def run_swarm(
    task: TaskSpec,
    agents: list[str],
    judge_agent: str | None = None,
    epsilon: float = 0.05,
    judge_cmd: str | None = None,
    **kwargs,
) -> dict:
    """Best-of-N with heterogeneous agents + one judge (STACK_READINESS §4.4).
    One candidate per listed agent, run in parallel and memory-isolated exactly
    like run_candidates. Ranked by reward total; when the top two are both
    outcome "pass" and within epsilon of each other, a judge breaks the tie —
    otherwise the reward ranking (best_episode's rule) picks the winner.

    judge_cmd: shell template for the judge, mirrors run_agent's agent_cmd.
    # ponytail: test/custom-judge escape hatch, exists for deterministic tests
    # and to let a caller supply its own judge without a new agent string.
    """
    kwargs.pop("agent", None)
    runs_dir = kwargs.get("runs_dir", "runs")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as pool:
        futures = [pool.submit(run_episode, task, agent=a, isolated=True, **kwargs)
                   for a in agents]
        episodes = [f.result() for f in futures]

    ranked = sorted(
        episodes,
        key=lambda e: (e["outcome"] == "pass", _score(e) if _score(e) is not None else -1.0),
        reverse=True,
    )
    winner = ranked[0]
    judged = False
    judge_used = None
    if (len(ranked) >= 2 and ranked[0]["outcome"] == "pass" and ranked[1]["outcome"] == "pass"
            and abs(_score(ranked[0]) - _score(ranked[1])) <= epsilon):
        judge_used = judge_agent or router.resolve("strong", default=None) or agents[0]
        pick = _run_judge(task, ranked[0], ranked[1], judge_used, judge_cmd, runs_dir)
        if pick is not None:
            judged = True
            winner = ranked[0] if pick == 1 else ranked[1]

    result = dict(winner)
    rewards = [round(_score(e), 4) if _score(e) is not None else None for e in episodes]
    result["swarm"] = {
        "agents": agents, "rewards": rewards, "judged": judged,
        "winner_agent": winner["agent"],
    }
    emit("heart", "swarm.judged", episode_id=result["episode_id"], task_id=task.task_id,
         agents=agents, rewards=rewards, judged=judged, winner_agent=winner["agent"],
         **({"judge_agent": judge_used} if judged else {}))
    return result
