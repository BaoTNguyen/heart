"""Orchestration: build one task via sequential roles (A) or a dependency graph
of workers merged by git (B), deciding and recovering the way the design settled on.

Path A  — heart's role pipeline in one shared worktree, one diff, no merge.
          Right for work that cannot be split at all; the safe default.
Path B  — a decomposer splits the task into subtasks with `depends_on` edges.
          Subtasks with no unmet dependency form a wave and run at the same time,
          each as an isolated episode routed to its own model + effort. Their
          diffs are 3-way merged by git (non-overlapping edits auto-resolve, even
          to the same file), the merged tree is committed, and the next wave
          starts from it — so a downstream worker reads its upstream's real code
          instead of a promise about it. Failures recover at the cheapest rung:

  caller asked for B (--orchestrate) ─ no ─▶ PATH A, without reading this file
     │ yes
     ▼
  repo has an integration verifier ─ no ─▶ PATH A
     │ yes
     ▼
  decompose ─ declined / <2 subtasks / cycle / dangling edge ─▶ PATH A
     │
     ▼
  ┌─ for each wave ────────────────────────────────────────────┐
  │  workers in parallel (isolated worktrees, routed model)    │
  │  git 3-way merge of every diff so far, from the base       │
  │     ├─ clean ──────────────▶ commit ─▶ next wave's base    │
  │     └─ textual conflict ───▶ incremental retry: keep the   │
  │            clean lanes, re-run only the colliding ones on  │
  │            top, re-merge ─ still conflicting ─▶ PATH A     │
  └────────────────────────────────────────────────────────────┘
  integration verify (the whole merged tree)
     ├─ pass ────────────────▶ done (Path B)
     └─ semantic fail ───────▶ one repair pass (strong model)
                                 ├─ pass ─▶ done (Path B, repaired)
                                 └─ fail ─▶ fall back to A

Two levers, and the distinction is the point. A dependency EDGE orders work and
hands the downstream worker committed code. A CONTRACT is the weaker tool left
for subtasks that genuinely run at the same time and so can only share a promise
about an interface. Reach for the edge first.

Depth is not free: every wave is a round trip, so a chain of single-subtask waves
is a slow sequential build and Path A does it better. The decomposer is told this;
whether it listens is measured, not assumed.

Separability is never predicted rigidly: the decomposer *guesses* both the lanes
and the edges, git + the tests *report*, and the outcome (wave count, clean_merge,
incremental, integration) is emitted so a caller can learn which task-kinds are
worth B — and, later, which edges the decomposer keeps forgetting.

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
import graphlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import episode as episode_mod
from . import review as review_mod
from . import reward as reward_mod
from . import route as route_mod
from . import router as router_mod
from .detect import detect_verifiers
from .env import Workspace
from .episode import run_episode
from .events import emit
from .runner import run_agent
from .verify import run_probes, run_verifiers


@dataclass
class Subtask:
    name: str
    prompt: str
    skills: list[str] = field(default_factory=list)
    effort: str = "medium"
    allowed_paths: list[str] = field(default_factory=list)  # optional soft lane
    # Names of subtasks that must finish first. Same vocabulary as plexus's
    # feature-level `depends_on`, one scale down: there it orders the features of
    # a goal, here the subtasks of one feature. Empty -> runs in the first wave.
    depends_on: list[str] = field(default_factory=list)


Decomposer = Callable[[object], "list[Subtask] | None"]


_DECOMPOSE_PROMPT = """You are decomposing a coding task into subtasks built by
separate agents and merged with git. Subtasks form a dependency graph: everything
with no unmet dependency runs at the SAME TIME (a "wave"), then the next wave runs
on top of the committed result of the one before it.

Task: {task}

Rules:
- `depends_on` names the subtasks that must finish first. A subtask starts from a
  checkout that ALREADY CONTAINS its dependencies' work, so it can read and build
  on that code. Use this for genuinely sequential work: schema before the code
  that queries it, an interface before its second implementation.
- Subtasks in the SAME wave (nothing ordering them relative to each other) run
  concurrently and CANNOT see each other's output. They must touch a DISJOINT set
  of files (each one's "lane"), and anything they share — types, signatures, the
  module they integrate through — must be fixed UP FRONT in a single `contract`
  handed to every worker verbatim. Read the repo to choose lanes.
- Prefer an edge over a contract when the work is truly ordered: a dependency
  hands down real code, a contract only hands down a promise.
- Prefer edges over depth. Every wave is a round trip, so a chain of five
  one-subtask waves is just a slow sequential build — use Path A for that instead.
- Give each subtask ONE skill from exactly this list, and an effort level:
  {skills}
  These are the only words that route. Anything else -- "testing",
  "documentation", the name of a tool -- is dropped, and the subtask falls back
  to generic coding, losing the per-subtask model choice. Use `docs` for
  documentation work and `coding` for tests.
- The graph must be acyclic, and every `depends_on` name must be a subtask here.
- Reply with {{"subtasks": []}} only if the task cannot be split at all — not
  merely because the parts are ordered. Ordered parts are what `depends_on` is for.

Reply with ONLY a JSON object:
{{"contract": "<the frozen shared interface same-wave workers must honor: exact
  signatures, types, and the file/module names they integrate through. Empty
  string when nothing runs concurrently or the lanes are truly independent.>",
  "subtasks": [
    {{"name": "<slug>", "prompt": "<what to build, self-contained>",
      "skills": ["coding"], "effort": "medium",
      "depends_on": [],
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
            allowed_paths=[p for p in (d.get("allowed_paths") or []) if isinstance(p, str)],
            depends_on=[n for n in (d.get("depends_on") or []) if isinstance(n, str)]))
    return out


def _parse_decomposition(raw: str) -> tuple[str, list[Subtask]]:
    """Find the decomposer's plan in whatever the agent wrote to its log.

    Accepts the object form {"contract": "...", "subtasks": [...]} or a bare
    array (contract=""). Returns (contract, subtasks); an empty subtasks list
    means "not decomposable". Raises on a plan that parses but is malformed.

    This scans for JSON that actually decodes rather than guessing one span, and
    both halves of that were paid for in production. Slicing a ```fence``` broke
    on a good contract, which quotes the interface it freezes inside its own
    ```python block. Slicing first-brace-to-last-brace broke on codex, which
    writes a transcript full of braces before its answer. Both surfaced
    identically downstream -- as "declined to split", with the real plan sitting
    in the log -- so the parser now proves a candidate by decoding it.

    Skipping to the end of each successful decode is what keeps the `subtasks`
    array inside an object from being mistaken for a bare-array plan of its own,
    which would parse fine and silently drop the contract.

    # ponytail: a rescan from every brace is O(n * parse). Agent logs are KBs and
    # this runs once per decomposition; revisit if a log ever gets large enough
    # to notice.
    """
    decoder = json.JSONDecoder()
    plans: list[tuple[str, list]] = []
    i = 0
    while i < len(raw):
        if raw[i] not in "{[":
            i += 1
            continue
        try:
            data, end = decoder.raw_decode(raw, i)
        except ValueError:
            i += 1
            continue
        i = end
        if isinstance(data, dict) and isinstance(data.get("subtasks"), list):
            plans.append((str(data.get("contract") or ""), data["subtasks"]))
        elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
            plans.append(("", data))
    if not plans:
        raise ValueError("no decomposition JSON in decomposer output")
    # the last non-empty plan is the agent's final answer; an empty one only wins
    # when it is all there is, which is the decline the prompt asks for.
    contract, items = next((p for p in reversed(plans) if p[1]), plans[-1])
    return contract, _subtasks_from_list(items)


def _parse_subtasks(raw: str) -> list[Subtask]:
    """Array-form parse (contract-free). Kept for the `decomposer` callable
    contract and direct tests."""
    return _parse_decomposition(raw)[1]


def _with_contract(contract: str, prompt: str) -> str:
    """Prefix a worker's prompt with the frozen shared contract. Same-wave workers
    never see each other's code, so the only way concurrent lanes compose after the
    merge is if every one builds to the identical interface handed down here.

    A dependency edge is the stronger tool and should be preferred where the work
    is really ordered: it hands the downstream worker the actual code. The contract
    is what's left for work that genuinely runs at the same time.

    No line here may open with a bash keyword: the `shell` agent runs the prompt
    it is handed as a script, so a stray `do`/`done`/`then` is a syntax error that
    kills the subtask before it starts. Prose, but prose with one constraint."""
    return (f"SHARED CONTRACT — every parallel worker builds to this EXACT interface. "
            f"Never change, rename, or reinvent it:\n{contract}\n\n"
            f"YOUR SLICE:\n{prompt}")


def _with_upstream(deps: list[str], prompt: str) -> str:
    """Tell a downstream worker that its dependencies already landed. It starts
    from a commit containing their work — unlike a same-wave sibling, which never
    sees one — so the failure to guard against is reinventing what is already
    there, not missing it. Keyword-free for the same reason as _with_contract."""
    return (f"ALREADY DONE and present in this checkout: {', '.join(deps)}. "
            f"Read that code and build on it. Never re-create or duplicate it.\n\n"
            f"YOUR SLICE:\n{prompt}")


def _check_lanes(waves: list[list[Subtask]]) -> None:
    """Reject a wave that mixes lane-scoped and unscoped subtasks.

    `_worker_taskspec` falls back to the parent task's allowed_paths when a
    subtask declares none, and for `heart work` and plexus that is empty --
    meaning unrestricted. So one lane-less subtask beside four scoped ones does
    not get a narrow scope, it gets the whole tree: it can write over every lane
    running next to it, and both the mount table and the diff scan will agree
    that was allowed. The declared lanes look like a boundary and are not one.

    A wave where NOBODY declares a lane is a different thing and stays legal --
    that is the pre-lane behaviour, less isolated but not deceptive, and the
    merge still has to come out clean. What is refused is the mixture, because
    only the mixture makes a boundary that is not there look like one.
    """
    for index, wave in enumerate(waves):
        if len(wave) < 2:
            continue
        scoped = [s.name for s in wave if s.allowed_paths]
        unscoped = [s.name for s in wave if not s.allowed_paths]
        if scoped and unscoped:
            raise ValueError(
                f"wave {index} mixes scoped and unscoped subtasks: "
                f"{sorted(unscoped)} declare no lane while {sorted(scoped)} do, "
                f"so the unscoped ones inherit the whole tree and can write over them")


def _waves(subs: list[Subtask]) -> list[list[Subtask]]:
    """Dependency order as parallel waves: every subtask in a wave can run at the
    same time, and every edge points backwards into an earlier one.

    Order inside a wave follows the decomposer's own order, because _merge applies
    diffs in list order and a deterministic merge is worth more than any cleverer
    sort. Raises graphlib.CycleError on a cycle and ValueError on an edge to a name
    that isn't in the plan — both are invalid plans, and the caller answers them
    the same way it answers "not decomposable": build it sequentially instead.
    """
    rank = {s.name: i for i, s in enumerate(subs)}
    if len(rank) != len(subs):
        raise ValueError("duplicate subtask names")
    graph = {}
    for s in subs:
        unknown = [d for d in s.depends_on if d not in rank]
        if unknown:
            raise ValueError(f"{s.name} depends on unknown subtask(s): {unknown}")
        graph[s.name] = set(s.depends_on)
    sorter = graphlib.TopologicalSorter(graph)
    sorter.prepare()
    by_name = {s.name: s for s in subs}
    waves = []
    while sorter.is_active():
        ready = sorter.get_ready()
        waves.append([by_name[n] for n in sorted(ready, key=rank.get)])
        sorter.done(*ready)
    return waves


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
        res = run_agent(dagent, _DECOMPOSE_PROMPT.format(
                            task=task.prompt, skills=", ".join(route_mod.SKILLS)),
                        cwd=str(ws.path), extra_env=env, timeout=task.timeout_seconds,
                        log_path=out / "decompose.log", agent_cmd=agent_cmd)
    finally:
        ws.destroy()
    # Path A is the right answer whether the planner declined or the planner
    # broke, but they are not the same event and the difference is invisible in
    # the result. A codex CLI that rejected its own flag read downstream as a
    # considered "this task cannot be split", which is how a broken agent stays
    # broken for a week. Say which one happened.
    if res["exit_code"] != 0:
        emit("heart", "decompose.failed", task_id=task.task_id, agent=dagent,
             parent=parent, reason="agent exited nonzero", exit_code=res["exit_code"],
             log=str(out / "decompose.log"))
        return None
    try:
        contract, subs = _parse_decomposition((out / "decompose.log").read_text(errors="replace"))
    except (ValueError, json.JSONDecodeError) as exc:
        emit("heart", "decompose.failed", task_id=task.task_id, agent=dagent,
             parent=parent, reason=f"unparseable plan: {exc}",
             log=str(out / "decompose.log"))
        return None
    if contract:
        # bake the frozen interface into every worker prompt — the only way
        # disjoint parallel lanes compose once merged (they never see each other)
        subs = [dataclasses.replace(s, prompt=_with_contract(contract, s.prompt)) for s in subs]
    # route.classify() silently drops off-vocabulary skills and falls back to
    # ["coding"]. That is the right runtime behaviour -- a bad label must not
    # fail a plan -- but silence meant every docs and test lane routed as
    # generic coding and nothing said so. Name them here.
    unknown = sorted({k for s in subs for k in s.skills if k not in route_mod.SKILLS})
    emit("heart", "decompose.done", task_id=task.task_id, agent=dagent,
         parent=parent, subtasks=[s.name for s in subs], contract=bool(contract),
         unknown_skills=unknown,
         edges=[[d, s.name] for s in subs for d in s.depends_on],
         lanes={s.name: s.allowed_paths for s in subs if s.allowed_paths})
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
    started = time.monotonic()
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
    # Probe once, here, rather than once per worker. The planner is the party
    # whose assumptions cost the most -- it writes the subtask that assumes the
    # extension is already upgraded -- so it is the party that gets the
    # measurements. Workers inherit the facts in their prompts and re-probe
    # nothing.
    if task.probes:
        probe_ws = Workspace(task.repo_path, task.base_commit, overlay=task.overlay_files)
        try:
            _, probe_profile = episode_mod._sandbox_profiles(
                task, probe_ws.path, f"orch-{task.task_id}", Path(runs_dir))
            failure, facts = run_probes(task.probes, str(probe_ws.path),
                                        task.timeout_seconds, profile=probe_profile)
        finally:
            probe_ws.destroy()
        emit("heart", "probes.finished", task_id=task.task_id, count=len(task.probes),
             blocked=bool(failure), reason=failure)
        if failure:
            # Path A owns the blocked-episode shape; re-probing there is one
            # cheap command and keeps a single place that decides what a failed
            # precondition does to an episode.
            return _path_a(task, agent, roles, runs_dir, agent_cmd,
                           reason=f"precondition not met: {failure}")
        task = dataclasses.replace(task, prompt=task.prompt + facts, probes=[])

    parent = f"orch-{task.task_id}"
    subs = decomposer(task) if decomposer is not None else _llm_decompose(
        task, agent, agent_cmd, runs_dir, parent, manifest)
    if not subs or len(subs) < 2:
        return _path_a(task, agent, roles, runs_dir, agent_cmd, reason="not decomposable")
    try:
        waves = _waves(subs)
        _check_lanes(waves)
    except (graphlib.CycleError, ValueError) as exc:
        # An unbuildable graph is a bad plan, not a bad task. Same answer as a
        # decomposer that declined: build it sequentially, where order is implicit.
        return _path_a(task, agent, roles, runs_dir, agent_cmd,
                       reason=f"invalid subtask graph: {exc}")

    # Waves run in order; each one starts from a commit holding every wave before
    # it. Diffs stay measured from the ORIGINAL base and accumulate in wave order,
    # so the final merge is the same operation whether there was one wave or five.
    base = task.base_commit
    done_subs: list[Subtask] = []
    episode_ids: list[str] = []
    diffs: list[str] = []
    merged, label = "", "clean"
    for index, wave in enumerate(waves):
        emit("heart", "orchestration.wave", task_id=task.task_id, wave=index,
             of=len(waves), subtasks=[s.name for s in wave])
        # A worker that raises is a heart failure, not an agent failure -- a
        # worktree that will not commit, a sandbox that will not start. Path A
        # is still a correct answer, and the whole point of this file is that
        # every failure lands on the cheapest rung that works. Before this,
        # `pool.map` let the exception out of run_orchestrated entirely and the
        # command died with a traceback: no fallback, no event, no diff. One
        # worktree quirk on one lane threw away the other lanes' finished work.
        try:
            wave_ids, wave_diffs = _run_workers(wave, task, base, agent, manifest,
                                                parent, runs_dir, agent_cmd)
        except Exception as exc:
            _flush_subagent_memory(parent, task.repo_path)
            emit("heart", "orchestration.worker_failed", task_id=task.task_id,
                 wave=index, error=f"{type(exc).__name__}: {exc}"[:500])
            return _fallback_a(task, agent, roles, runs_dir, agent_cmd,
                               episode_ids, f"worker_failed: {type(exc).__name__}")
        # flushed per wave: the workers are subprocesses that have already exited,
        # and the next wave's retrieval should see what this one learned.
        _flush_subagent_memory(parent, task.repo_path)
        done_subs, episode_ids, diffs = (done_subs + wave, episode_ids + wave_ids,
                                         diffs + wave_diffs)
        merged, conflicted, markers = _merge(task.repo_path, task.base_commit, diffs)
        if conflicted or markers:
            # Overlap inside this wave — earlier waves are already committed and
            # cannot be the conflict. Keep the clean lanes, re-run only the
            # colliding ones on top of them, re-merge.
            inc = _incremental_merge(task, agent, done_subs, diffs, episode_ids,
                                     conflicted, manifest, runs_dir, agent_cmd, parent)
            if inc is None:
                return _fallback_a(task, agent, roles, runs_dir, agent_cmd,
                                   episode_ids, "merge_conflict")
            merged, done_subs, diffs, episode_ids = inc
            label = "incremental"
        if index + 1 < len(waves):
            base = _commit_tree(task.repo_path, task.base_commit, merged)
            if base is None:
                return _fallback_a(task, agent, roles, runs_dir, agent_cmd,
                                   episode_ids, "wave_base_failed")

    if len(waves) > 1:
        label = f"waves:{len(waves)}" + ("+incremental" if label == "incremental" else "")
    done = _finish_b(task, merged, verifiers, manifest, agent, episode_ids,
                     runs_dir, agent_cmd, parent, label,
                     elapsed=time.monotonic() - started,
                     # one sequential leg for the decompose, one per wave
                     budget=task.timeout_seconds * (1 + len(waves)),
                     roles=roles)
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
              agent_cmd, parent, merge_label, elapsed=None, budget=None,
              roles=None) -> dict | None:
    """Verify the merged tree, repair once if it fails (rejecting a repair that
    went green by weakening tests), and materialize a Path-B result. Returns None
    when repair can't save it, so the caller falls back to whole-task A. Shared by
    the clean-merge path and the incremental retry.

    The integration results are kept rather than dropped: the merged tree is real
    work with a real diff that just faced the real suite, so it gets a real
    reward. Only the workers stay unscored, because a fragment genuinely was not
    measured against anything."""
    passed, results = _integration_check(task, merged, verifiers)
    if passed is not False:
        outcome = "pass" if passed else "unverified"
        verdict, findings = _review_merged(task, merged, roles, agent,
                                           runs_dir, agent_cmd, parent)
        return _materialize(runs_dir, task, merged, outcome, episode_ids,
                            merge=merge_label, integration="pass" if passed else "none",
                            reward=_score(results, merged, elapsed, budget),
                            review_verdict=verdict, review_findings=findings)
    strong = _strong_agent(manifest, agent)
    merged2, passed2, results = _repair(task, merged, verifiers, strong, runs_dir,
                                        agent_cmd, parent)
    if passed2 and _repair_tampers_with_tests(merged2):
        emit("heart", "orchestration.repair_rejected", task_id=task.task_id,
             reason="repair_deleted_test_lines")
        passed2 = False
    if passed2:
        verdict, findings = _review_merged(task, merged2, roles, agent,
                                           runs_dir, agent_cmd, parent)
        return _materialize(runs_dir, task, merged2, "pass", episode_ids,
                            merge=merge_label, integration="repaired",
                            reward=_score(results, merged2, elapsed, budget),
                            review_verdict=verdict, review_findings=findings)
    return None


def _incremental_merge(task, agent, subs, diffs, episode_ids, conflicted,
                       manifest, runs_dir, agent_cmd, parent):
    """Rebuild the merged tree from only the clean lanes, advance the base to
    include them, and re-run just the conflicting lanes on top so they build
    against the clean work instead of reproducing the overlap.

    Returns (merged_diff, subs, diffs, episode_ids) — the last three reordered
    together as clean-then-retried, so the caller's parallel lists stay aligned
    for the next wave's conflict indices. None if the lanes still won't combine,
    and the caller falls back to whole-task A.
    """
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
    emit("heart", "orchestration.incremental", task_id=task.task_id,
         retried=[subs[i].name for i in conflicted])
    return (merged,
            [subs[i] for i in clean_idx] + retry_subs,
            clean_diffs + retry_diffs,
            [episode_ids[i] for i in clean_idx] + retry_ids)


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
    #
    # `roles` is passed through, not defaulted: `roles=None` is the caller
    # asking for a solo turn, and `roles or DEFAULT_ROLES` quietly overrode that
    # on the fallback path only -- so --solo was honoured when Path A was chosen
    # up front and ignored when Path B fell back to it.
    ep = run_episode(task, agent=agent, roles=roles, fix_rounds=1,
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
        prompt=_with_upstream(sub.depends_on, sub.prompt) if sub.depends_on else sub.prompt,
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
            runs_dir, agent_cmd, parent: str) -> tuple[str, bool, dict]:
    """One fix pass by a strong model over the merged state, in its own worktree.
    Returns (new_diff, passed, verifier_results) — the results travel because the
    repaired tree is what gets scored, and re-running the suite to find that out
    would be paying twice for an answer already in hand.

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
        return new_diff, passed, results
    finally:
        ws.destroy()


def _score(results: dict, diff: str, elapsed, budget) -> dict | None:
    """Reward for a merged Path-B tree, or None when nothing measured it.

    None is not zero. An unscored episode and a failed one are different facts,
    and collapsing them feeds marrow failures that never happened — the same
    distinction SPINE.md draws between `sandbox.denied` (reward None) and
    `guardrail.hit` (reward 0.0).

    The efficiency budget is the task timeout times the number of sequential
    legs — one decompose plus one per wave — because that is the wall clock the
    orchestration actually had. Scoring a multi-wave build against a single
    role's timeout would zero its efficiency for a structural reason and make
    Path B look worse than Path A at identical work.

    # ponytail: hidden verifiers are not run on the merged tree, so a Path-B
    # reward uses the no-hidden-tests weights while a Path-A reward on the same
    # task may not. Comparable within a path, not across one; fix by running the
    # hidden suite here once someone needs the two to line up.
    """
    if not results:
        return None
    return reward_mod.compute(results, diff, elapsed or 0.0, int(budget or 0))


def _review_merged(task, diff: str, roles, agent: str, runs_dir, agent_cmd,
                   parent: str) -> tuple[str | None, list]:
    """Assess the merged tree with the same three stages Path A uses.

    Path A reviews the tree an agent built; Path B had nothing reviewing the
    tree git assembled, so `--orchestrate` silently dropped the reviewer, and a
    caller gating on REJECT read Path B's None as "nobody objected" rather than
    "nobody looked".

    Assess only -- rounds=0. There is no coder here to hand findings to: the
    workers have exited and their worktrees are gone, and the repair pass that
    could act is driven by failing verifiers rather than by a reviewer. So this
    reports rather than negotiates, and a blocker sends the whole thing to
    Path A, where the findings loop does have someone to talk to.
    """
    role = next((r for r in (roles or []) if r.get("review")), None)
    if role is None:
        return None, []
    reviewer = role.get("agent") or router_mod.review_agent(agent)
    ws = Workspace(task.repo_path, task.base_commit)
    try:
        ws.apply(diff)
        out = Path(runs_dir) / f"{task.task_id}-review"
        out.mkdir(parents=True, exist_ok=True)
        repo = Path(task.repo_path).resolve()
        env = {"ARTERIES_AGENT_ID": parent, "ARTERIES_AGENT_ROLE": "parent",
               "ARTERIES_PROJECT": repo.name, "ARTERIES_REPO": str(repo)}

        def _assess(name, prompt):
            run_agent(reviewer, prompt, str(ws.path), env, task.timeout_seconds,
                      out / f"{name}.log", agent_cmd=agent_cmd)
            return out / f"{name}.log"

        result = review_mod.phase(
            task.prompt, assess=_assess, resolve=_assess, verify=lambda: None,
            legacy_verdict=episode_mod._review_verdict,
            assess_prompt=role["prompt"], rounds=0)
        emit("heart", "orchestration.reviewed", task_id=task.task_id,
             agent=reviewer, verdict=result.verdict,
             findings=[{"severity": f.severity, "file": f.file, "claim": f.claim}
                       for f in result.findings])
        return result.verdict, result.findings
    except Exception as exc:
        # A reviewer that could not run is not an approval -- say nothing rather
        # than manufacture one. But say it out loud: this swallowed a NameError
        # for a missing import once, and a silent None is indistinguishable from
        # "no review role configured".
        emit("heart", "orchestration.review_failed", task_id=task.task_id,
             error=f"{type(exc).__name__}: {exc}"[:300])
        return None, []
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


def _worker_usage(runs_dir, worker_ids: list[str]) -> dict:
    """Sum what the workers actually spent, so a Path-B episode can be priced.

    Materialized episodes used to report `usage: {}`, which reads as "unknown"
    to every consumer — plexus omits an unknown key from its ledger rather than
    calling it zero, so an orchestrated feature spent real money and showed up
    as free. The work happened in the workers; this is where it is added up.

    A key is omitted when no worker reported it, keeping the unknown/zero
    distinction the rest of the stack relies on.
    """
    totals: dict[str, float] = {}
    for wid in worker_ids:
        try:
            usage = json.loads(
                (Path(runs_dir) / wid / "episode.json").read_text()).get("usage") or {}
        except (OSError, json.JSONDecodeError):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    return {k: (round(v, 6) if k == "cost_usd" else v) for k, v in totals.items()}


def _materialize(runs_dir, task, diff: str, outcome: str, worker_ids: list[str],
                 merge: str, integration: str, reward: dict | None = None,
                 review_verdict: str | None = None,
                 review_findings: list | None = None) -> dict:
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
        "review_verdict": review_verdict, "blocked_reason": None,
        "review_findings": [{"severity": f.severity, "file": f.file,
                             "line": f.line, "claim": f.claim,
                             "evidence": f.evidence} for f in (review_findings or [])],
        "reward": reward or {"total": None, "components": {}},
        "usage": _worker_usage(runs_dir, worker_ids),
        "orchestration": orchestration,
    }
    (out / "episode.json").write_text(json.dumps(ep, indent=2))
    emit("heart", "orchestration.finished", episode_id=ep_id, task_id=task.task_id,
         outcome=outcome, reward=ep["reward"]["total"], **orchestration)
    return ep
