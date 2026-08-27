"""Orchestration: build one task via sequential roles (A) or parallel workers
merged by git (B), deciding and recovering the way the design settled on.

Path A  — heart's role pipeline in one shared worktree, one diff, no merge.
          Right for coupled work; the safe default.
Path B  — a decomposer splits the task into workers, each run as an isolated
          episode routed to its own model + effort. Their diffs are 3-way merged
          by git (non-overlapping edits auto-resolve, even to the same file),
          the merged tree is verified, and failures recover at the cheapest rung
          that works:

  route (cheap guess: is B worth it?) ─ no ─▶ PATH A
     │ yes
     ▼
  [optional shared-scaffold pre-step]
  parallel workers (isolated worktrees, routed model+effort)
  git 3-way merge
     ├─ clean ───────────────▶ integration verify
     └─ textual conflict ─────▶ fall back to A        (v1: whole task; the
                                                        incremental-slice retry
                                                        is the documented seam)
  integration verify
     ├─ pass ────────────────▶ done (Path B)
     └─ semantic fail ───────▶ one repair pass (strong model)
                                 ├─ pass ─▶ done (Path B, repaired)
                                 └─ fail ─▶ fall back to A

Separability is never predicted rigidly: the decomposer *guesses*, git + the
tests *report*, and the outcome (clean_merge / integration) is emitted so a
caller can learn which task-kinds are worth B.

Decomposition is done by a real planning agent (`_llm_decompose`) that runs
*under the orchestration's parent id*, so its planning reasoning is written to
memory there and the workers inherit it — the downward half of arteries subagent
memory. Passing an explicit `decomposer` callable overrides the agent (tests
inject a fixed split); the merge/verify/recover engine below is model-free and is
what those tests exercise.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import reward as reward_mod
from . import route as route_mod
from .detect import detect_verifiers
from .env import Workspace
from .episode import DEFAULT_ROLES, run_episode
from .events import emit
from .runner import run_agent
from .verify import run_verifiers


@dataclass
class Subtask:
    name: str
    prompt: str
    skills: list[str] = field(default_factory=list)
    effort: str = "medium"
    allowed_paths: list[str] = field(default_factory=list)  # optional soft lane


Decomposer = Callable[[object], "list[Subtask] | None"]


_DECOMPOSE_PROMPT = """You are decomposing a coding task into INDEPENDENT subtasks that
will be built in PARALLEL by separate agents — each starting from the same commit,
none able to see another's work — and merged with git.

Task: {task}

Rules:
- Each subtask must touch a DISJOINT set of files (its "lane") so the diffs merge
  without conflicts. Read the repo to choose lanes.
- Workers run at the same time and CANNOT depend on each other's output. Anything
  they share — types, function signatures, an API, a file/module they integrate
  through — must be fixed UP FRONT in a single `contract` handed to every worker
  verbatim. Do NOT put shared code in one subtask for the others to depend on;
  they will never see it.
- Give each subtask the one skill it mainly needs and an effort level.
- If the task is tightly coupled and cannot be split into disjoint lanes behind a
  small stable contract, reply with {{"subtasks": []}} — one sequential build is
  better than a broken merge.

Reply with ONLY a JSON object:
{{"contract": "<the frozen shared interface every worker must honor: exact
  signatures, types, and the file/module names they integrate through. Empty
  string if the lanes are truly independent.>",
  "subtasks": [
    {{"name": "<slug>", "prompt": "<what to build, self-contained>",
      "skills": ["coding"], "effort": "medium",
      "allowed_paths": ["path/to/file.py", "dir/"]}}
  ]}}"""


def _subtasks_from_list(items: list) -> list[Subtask]:
    """Build Subtasks from parsed JSON dicts. Empty -> []. Raises on a subtask
    missing name/prompt."""
    out = []
    for d in items:
        if not d.get("name") or not d.get("prompt"):
            raise ValueError(f"subtask missing name/prompt: {d}")
        out.append(Subtask(
            name=str(d["name"]), prompt=str(d["prompt"]),
            skills=[s for s in (d.get("skills") or []) if isinstance(s, str)],
            effort=d.get("effort") or "medium",
            allowed_paths=[p for p in (d.get("allowed_paths") or []) if isinstance(p, str)]))
    return out


def _parse_decomposition(raw: str) -> tuple[str, list[Subtask]]:
    """Pull the decomposer's JSON out (fenced block preferred, else the outermost
    braces/brackets). Accepts the object form {"contract": "...", "subtasks": [...]}
    or a bare array (contract=""). Returns (contract, subtasks); an empty subtasks
    list means "not decomposable". Raises on malformed JSON or a bad subtask."""
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.DOTALL)
    text = fenced.group(1) if fenced else raw
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in decomposer output")
    start, end = min(starts), max(text.rfind("}"), text.rfind("]"))
    data = json.loads(text[start:end + 1])
    if isinstance(data, dict):
        return str(data.get("contract") or ""), _subtasks_from_list(data.get("subtasks") or [])
    return "", _subtasks_from_list(data)


def _parse_subtasks(raw: str) -> list[Subtask]:
    """Array-form parse (contract-free). Kept for the `decomposer` callable
    contract and direct tests."""
    return _parse_decomposition(raw)[1]


def _with_contract(contract: str, prompt: str) -> str:
    """Prefix a worker's prompt with the frozen shared contract. Parallel workers
    never see each other's code, so the only way disjoint lanes compose after the
    merge is if every one builds to the identical interface handed down here."""
    return (f"SHARED CONTRACT — every parallel worker builds to this EXACT interface; "
            f"do not change, rename, or reinvent it:\n{contract}\n\n"
            f"YOUR SLICE:\n{prompt}")


def _llm_decompose(task, agent: str, agent_cmd: str | None, runs_dir, parent: str,
                   manifest: dict) -> list[Subtask] | None:
    """Run a real planning agent under `parent` to split the task. Because it runs
    as the parent agent (`ARTERIES_AGENT_ID=parent`), its decomposition reasoning
    is written to memory there, and the workers (parent_agent_id=parent) inherit
    it — the downward half of subagent memory. Routed to a planning-capable model
    when a manifest exists. Returns None (Path A) when it declines to split, or
    on any failure."""
    dagent = agent
    if manifest:
        try:
            dagent = route_mod.route(
                dataclasses.replace(task, skills=["planning"]), manifest=manifest).agent
        except Exception:
            dagent = agent
    repo = Path(task.repo_path).resolve()
    out = Path(runs_dir) / f"{parent}-decompose"
    out.mkdir(parents=True, exist_ok=True)
    env = {"ARTERIES_AGENT_ID": parent, "ARTERIES_AGENT_ROLE": "parent",
           "ARTERIES_PROJECT": repo.name, "ARTERIES_REPO": str(repo)}
    ws = Workspace(task.repo_path, task.base_commit)
    try:
        res = run_agent(dagent, _DECOMPOSE_PROMPT.format(task=task.prompt),
                        cwd=str(ws.path), extra_env=env, timeout=task.timeout_seconds,
                        log_path=out / "decompose.log", agent_cmd=agent_cmd)
    finally:
        ws.destroy()
    if res["exit_code"] != 0:
        return None
    try:
        contract, subs = _parse_decomposition((out / "decompose.log").read_text(errors="replace"))
    except (ValueError, json.JSONDecodeError):
        return None
    if contract:
        # bake the frozen interface into every worker prompt — the only way
        # disjoint parallel lanes compose once merged (they never see each other)
        subs = [dataclasses.replace(s, prompt=_with_contract(contract, s.prompt)) for s in subs]
    emit("heart", "decompose.done", task_id=task.task_id, agent=dagent,
         parent=parent, subtasks=[s.name for s in subs], contract=bool(contract))
    return subs or None


def run_orchestrated(
    task,
    agent: str = "claude",
    decomposer: Decomposer | None = None,
    runs_dir: str | Path = "runs",
    agent_cmd: str | None = None,
    manifest: dict | None = None,
    roles: list[dict] | None = None,
) -> dict:
    """Build `task`, choosing Path A or B. Returns an episode-shaped dict
    (episode_id, outcome, review_verdict, usage) with diff.patch written under
    runs_dir/<id>/, so callers treat A and B results uniformly."""
    manifest = manifest if manifest is not None else route_mod.load_manifest()
    # Path B safety gate: it combines independently-built parts, so a clean merge
    # proves nothing unless a verifier actually exercises the seam. No integration
    # test => unsafe to split => build it as one coherent tree (A), where a single
    # agent holds the whole picture and composition is never in question. Gated
    # before the decompose call so a no-verifier task pays nothing for B.
    verifiers = task.public_verifiers or detect_verifiers(task.repo_path)
    if not verifiers:
        return _path_a(task, agent, roles, runs_dir, agent_cmd,
                       reason="no integration verifier")
    # one parent id for this orchestration. The decomposer runs as a real agent
    # under it, so its planning ephemeral lands there and the workers — which run
    # as its subagents (shared parent lineage, each its own agent id) — inherit
    # it, and their ephemeral compiles back up under the same id. Worktree
    # isolation is preserved; memory is subagent, not discard. An explicit
    # `decomposer` callable overrides the agent (tests / custom split).
    parent = f"orch-{task.task_id}"
    subs = decomposer(task) if decomposer is not None else _llm_decompose(
        task, agent, agent_cmd, runs_dir, parent, manifest)
    if not subs or len(subs) < 2:
        return _path_a(task, agent, roles, runs_dir, agent_cmd, reason="not decomposable")

    episode_ids, diffs = _run_workers(subs, task, task.base_commit, agent, manifest,
                                      parent, runs_dir, agent_cmd)
    _flush_subagent_memory(parent, task.repo_path)

    merged, conflicted, markers = _merge(task.repo_path, task.base_commit, diffs)
    if conflicted or markers:
        # incremental-slice retry: keep the lanes that merged clean, re-run only the
        # conflicting ones on top of them, re-merge. Whole-task A only if that still
        # can't produce a clean, passing tree.
        inc = _incremental_retry(task, agent, subs, diffs, episode_ids, conflicted,
                                 verifiers, manifest, runs_dir, agent_cmd, parent)
        if inc is not None:
            return inc
        return _fallback_a(task, agent, roles, runs_dir, agent_cmd,
                           episode_ids, "merge_conflict")

    done = _finish_b(task, merged, verifiers, manifest, agent, episode_ids,
                     runs_dir, agent_cmd, parent, "clean")
    if done is not None:
        return done
    return _fallback_a(task, agent, roles, runs_dir, agent_cmd,
                       episode_ids, "integration_failed")


def _run_workers(subs, task, base_commit, agent, manifest, parent, runs_dir, agent_cmd):
    """Build every subtask in parallel from `base_commit`, each in its own isolated
    worktree (run_episode) routed to its own model+effort. Threads suffice
    (subprocess/IO bound); global fan-out is capped by the runner's agent slots.
    map() preserves subs order so the merge is deterministic. Returns
    (episode_ids, diffs) aligned to subs."""
    def _build(sub):
        wagent, weffort = _route_worker(sub, task, agent, manifest)
        wtask = _worker_taskspec(dataclasses.replace(task, base_commit=base_commit),
                                 sub, weffort)
        ep = run_episode(wtask, agent=wagent, memory_mode="subagent",
                         parent_agent_id=parent, runs_dir=str(runs_dir), agent_cmd=agent_cmd)
        diff = (Path(runs_dir) / ep["episode_id"] / "diff.patch").read_text()
        return ep["episode_id"], diff

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subs)) as pool:
        results = list(pool.map(_build, subs))
    return [r[0] for r in results], [r[1] for r in results]


def _finish_b(task, merged, verifiers, manifest, agent, episode_ids, runs_dir,
              agent_cmd, parent, merge_label) -> dict | None:
    """Verify the merged tree, repair once if it fails (rejecting a repair that
    went green by weakening tests), and materialize a Path-B result. Returns None
    when repair can't save it, so the caller falls back to whole-task A. Shared by
    the clean-merge path and the incremental retry."""
    passed, _ = _integration_check(task, merged, verifiers)
    if passed is not False:
        outcome = "pass" if passed else "unverified"
        return _materialize(runs_dir, task, merged, outcome, episode_ids,
                            merge=merge_label, integration="pass" if passed else "none")
    strong = _strong_agent(manifest, agent)
    merged2, passed2 = _repair(task, merged, verifiers, strong, runs_dir, agent_cmd, parent)
    if passed2 and _repair_tampers_with_tests(merged2):
        emit("heart", "orchestration.repair_rejected", task_id=task.task_id,
             reason="repair_deleted_test_lines")
        passed2 = False
    if passed2:
        return _materialize(runs_dir, task, merged2, "pass", episode_ids,
                            merge=merge_label, integration="repaired")
    return None


def _incremental_retry(task, agent, subs, diffs, episode_ids, conflicted, verifiers,
                       manifest, runs_dir, agent_cmd, parent) -> dict | None:
    """Rebuild the merged tree from only the clean lanes, advance the base to
    include them, and re-run just the conflicting lanes on top so they build
    against the clean work instead of reproducing the overlap. Returns a
    materialized Path-B result, or None if the retry still can't produce a clean,
    passing tree (caller then does whole-task A)."""
    clean_idx = [i for i in range(len(subs)) if i not in conflicted]
    clean_diffs = [diffs[i] for i in clean_idx]
    base_clean, sc, mk = _merge(task.repo_path, task.base_commit, clean_diffs)
    if sc or mk:
        return None  # the clean lanes should always merge; bail if they somehow don't
    new_base = _commit_tree(task.repo_path, task.base_commit, base_clean)
    if new_base is None:
        return None
    retry_subs = [subs[i] for i in conflicted]
    retry_ids, retry_diffs = _run_workers(retry_subs, task, new_base, agent, manifest,
                                          parent, runs_dir, agent_cmd)
    _flush_subagent_memory(parent, task.repo_path)
    # clean lanes + retried lanes, applied to the ORIGINAL base -> full combined diff
    # (retried diffs were built on base+clean, so they apply on top of the clean ones)
    merged, c2, m2 = _merge(task.repo_path, task.base_commit, clean_diffs + retry_diffs)
    if c2 or m2:
        return None  # still overlapping -> whole-task A
    all_ids = [episode_ids[i] for i in clean_idx] + retry_ids
    emit("heart", "orchestration.incremental", task_id=task.task_id,
         retried=[subs[i].name for i in conflicted])
    return _finish_b(task, merged, verifiers, manifest, agent, all_ids, runs_dir,
                     agent_cmd, parent, "incremental")


def _commit_tree(repo: str, base_commit: str, diff: str) -> str | None:
    """Apply `diff` onto `base_commit` in a throwaway worktree and commit it,
    returning the new (dangling but valid for this run) commit sha — the base the
    retried workers build on. Objects survive worktree removal, so the sha stays
    checkoutable until GC, which won't run mid-process."""
    ws = Workspace(repo, base_commit)
    try:
        try:
            ws.apply(diff)
        except RuntimeError:
            return None
        p = str(ws.path)
        subprocess.run(["git", "-C", p, "add", "-A"], capture_output=True)
        subprocess.run(["git", "-C", p, "-c", "user.name=heart", "-c",
                        "user.email=heart@local", "commit", "-qm", "clean lanes"],
                       capture_output=True)
        sha = subprocess.run(["git", "-C", p, "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        return sha or None
    finally:
        ws.destroy()


# --- path A ---------------------------------------------------------------

def _path_a(task, agent, roles, runs_dir, agent_cmd, reason: str) -> dict:
    ep = run_episode(task, agent=agent, roles=roles, fix_rounds=1,
                     runs_dir=str(runs_dir), agent_cmd=agent_cmd)
    ep["orchestration"] = {"path": "A", "reason": reason}
    return ep


def _fallback_a(task, agent, roles, runs_dir, agent_cmd, episode_ids, reason) -> dict:
    # incremental (only the failing slices) is the enhancement; v1 re-runs the
    # whole task sequentially, which is always correct because A shares one tree.
    ep = run_episode(task, agent=agent, roles=roles or DEFAULT_ROLES, fix_rounds=1,
                     runs_dir=str(runs_dir), agent_cmd=agent_cmd)
    ep["orchestration"] = {"path": "B->A", "reason": reason, "worker_episodes": episode_ids}
    emit("heart", "orchestration.fallback", task_id=task.task_id, reason=reason,
         worker_episodes=episode_ids)
    return ep


# --- path B mechanics -----------------------------------------------------

def _flush_subagent_memory(parent_id: str, repo: str) -> None:
    """Compile the workers' subagent-tagged ephemeral up to project memory now.

    The workers are one-shot subprocesses that exit before arteries' async
    compile runs, so nothing would claim their records. This runs one compile
    pass under the parent id (which claims every `parent_agent_id = parent`
    record) via arteries' own CLI — a subprocess, so heart imports nothing from
    arteries here and stays runnable where arteries/Postgres are absent."""
    try:
        subprocess.run(
            [sys.executable, "-m", "arteries.compile"],
            env={**os.environ, "ARTERIES_AGENT_ID": parent_id,
                 "ARTERIES_PROJECT": Path(repo).resolve().name,
                 "ARTERIES_EPHEMERAL": "compile"},
            capture_output=True, timeout=120,
        )
    except Exception:
        pass


def _route_worker(sub: Subtask, task, agent: str, manifest: dict) -> tuple[str, str]:
    """Pick each worker's model + effort. With a manifest, route on the worker's
    declared skills; without one, inherit the base agent at the worker's effort."""
    if not manifest:
        return agent, sub.effort
    wtask = dataclasses.replace(task, prompt=sub.prompt, skills=sub.skills,
                                effort=sub.effort)
    d = route_mod.route(wtask, manifest=manifest)
    return d.agent, d.effort


def _worker_taskspec(task, sub: Subtask, effort: str):
    """A worker builds only its slice, structural checks only — a fragment can't
    be judged against whole-feature tests, so correctness is decided on the merge.
    allowed_paths is not a soft lane any more. Under HEART_SANDBOX=docker it
    becomes the container's mount table -- everything outside it is read-only --
    so a planner guessing paths from a prompt it has not verified against the
    code is guessing at a wall, not a hint. Worth revisiting whether a first
    attempt should run open and tighten only once plexus's ledger has seen the
    task before."""
    return dataclasses.replace(
        task,
        task_id=f"{task.task_id}-{sub.name}",
        prompt=sub.prompt,
        allowed_paths=sub.allowed_paths or task.allowed_paths,
        public_verifiers=[],
        hidden_verifiers=[],
        skills=sub.skills,
        effort=effort,
    )


def _merge(repo: str, base_commit: str, diffs: list[str]) -> tuple[str, list[int], bool]:
    """Apply worker diffs to a fresh base worktree with git's 3-way merge.

    Non-overlapping edits (even to the same file) auto-resolve; a true overlap
    leaves conflict markers and a non-zero apply. Returns (merged_diff,
    conflicted_worker_indices, markers_present)."""
    ws = Workspace(repo, base_commit)
    conflicted: list[int] = []
    try:
        for i, d in enumerate(diffs):
            if not d.strip():
                continue
            r = subprocess.run(
                ["git", "-C", str(ws.path), "apply", "--3way", "--whitespace=nowarn"],
                input=d, text=True, capture_output=True)
            if r.returncode != 0:
                conflicted.append(i)
        # --3way stages into the index, so ws.diff() (worktree-vs-index) is blank.
        # Stage everything and diff base->merged via the index instead.
        subprocess.run(["git", "-C", str(ws.path), "add", "-A"], capture_output=True)
        merged = subprocess.run(
            ["git", "-C", str(ws.path), "diff", "--staged", "--binary",
             "--", ".", *Workspace.DIFF_EXCLUDES],
            capture_output=True, text=True).stdout
        return merged, conflicted, ("<<<<<<<" in merged)
    finally:
        ws.destroy()


_TEST_PATH = re.compile(r"(^|/)tests?/|(^|/)(test_|conftest)|_test\.|\.test\.|\.spec\.", re.I)


def _integration_check(task, diff: str, verifiers: list) -> tuple[bool | None, dict]:
    """The single definition of "does the combined tree work": apply base+diff in
    a throwaway worktree and run the feature's verifiers — the net for semantic
    conflicts git can't see. Returns (verdict, results): verdict True/False, or
    None when the repo ships no verifiers (results empty). Path A funnels through
    run_episode's own verify; every Path-B verdict goes through here, so "pass"
    means the same thing on both paths and the raw results are available as
    evidence."""
    if not verifiers:
        return None, {}
    ws = Workspace(task.repo_path, task.base_commit)
    try:
        try:
            ws.apply(diff)
        except RuntimeError:
            return False, {}  # merged diff won't even apply cleanly
        results = run_verifiers(verifiers, str(ws.path), task.timeout_seconds)
        return all(r["passed"] for r in results.values()), results
    finally:
        ws.destroy()


def _repair_tampers_with_tests(diff: str) -> bool:
    """True if the repair diff removes any line from a test file. A strong model
    told "make the checks pass" can pass by deleting the failing test, so a repair
    that touches tests destructively is rejected and we fall back to A instead of
    shipping a green that lies.

    # ponytail: path + removed-line heuristic. It won't catch a weakened assert
    # edited in place (a changed value, no net line removal) — only a semantic
    # diff review would. Names the ceiling; upgrade to an assertion-count diff if
    # that bypass ever shows up in practice.
    """
    is_test = False
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            path = path[2:] if path.startswith("b/") else path
            is_test = bool(_TEST_PATH.search(path))
        elif is_test and line.startswith("-") and not line.startswith("---"):
            return True
    return False


def _repair(task, merged_diff: str, verifiers: list, agent: str,
            runs_dir, agent_cmd, parent: str) -> tuple[str, bool]:
    """One fix pass by a strong model over the merged state, in its own worktree.
    Returns (new_diff, passed).

    Runs under the orchestration's parent id: the workers' seam facts were already
    compiled up under `parent` (flush is pre-merge), so the repair agent's
    retrieval surfaces exactly those — "worker A exposed parse()->tuple, worker B
    expected a dict" — to diagnose the integration failure, and its own repair
    reasoning lands back under `parent` for the next attempt."""
    ws = Workspace(task.repo_path, task.base_commit)
    try:
        ws.apply(merged_diff)
        tail = _fail_tail(run_verifiers(verifiers, str(ws.path), task.timeout_seconds))
        out = Path(runs_dir) / f"{task.task_id}-repair"
        out.mkdir(parents=True, exist_ok=True)
        prompt = (f"Independently-built changes were merged and now fail verification:\n"
                  f"{tail}\nFix the integration so the checks pass. Do not weaken or "
                  f"delete tests.\nOriginal task: {task.prompt}")
        repo = Path(task.repo_path).resolve()
        env = {"ARTERIES_AGENT_ID": parent, "ARTERIES_AGENT_ROLE": "parent",
               "ARTERIES_PROJECT": repo.name, "ARTERIES_REPO": str(repo)}
        run_agent(agent, prompt, str(ws.path), env, task.timeout_seconds,
                  out / "repair.log", agent_cmd=agent_cmd)
        new_diff = ws.diff()
        results = run_verifiers(verifiers, str(ws.path), task.timeout_seconds)
        passed = all(r["passed"] for r in results.values()) if results else True
        return new_diff, passed
    finally:
        ws.destroy()


def _strong_agent(manifest: dict, default: str) -> str:
    if not manifest:
        return default
    best = max(manifest.values(), key=lambda m: route_mod.drank(m["max_difficulty"]))
    return best["agent"]


def _fail_tail(results: dict, limit: int = 2000) -> str:
    for r in results.values():
        if not r.get("passed"):
            return (r.get("output_tail") or "").strip()[-limit:]
    return "(no failing output captured)"


def _materialize(runs_dir, task, diff: str, outcome: str, worker_ids: list[str],
                 merge: str, integration: str) -> dict:
    """Write a Path-B result as an episode dir so callers read it like any
    episode (diff.patch, episode.json, episode-shaped return)."""
    ep_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-orch-" + uuid.uuid4().hex[:6]
    out = Path(runs_dir) / ep_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "diff.patch").write_text(diff)
    orchestration = {"path": "B", "merge": merge, "integration": integration,
                     "worker_episodes": worker_ids}
    ep = {
        "episode_id": ep_id, "task_id": task.task_id, "outcome": outcome,
        "diff_lines": reward_mod.diff_changed_lines(diff),
        "review_verdict": None, "blocked_reason": None,
        "reward": {"total": None, "components": {}}, "usage": {},
        "orchestration": orchestration,
    }
    (out / "episode.json").write_text(json.dumps(ep, indent=2))
    emit("heart", "orchestration.finished", episode_id=ep_id, task_id=task.task_id,
         outcome=outcome, **orchestration)
    return ep
