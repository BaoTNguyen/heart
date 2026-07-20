"""heart CLI: run | batch | check-task | mine | export | dataset."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from . import mine as mine_mod
from . import pulse as pulse_mod
from .detect import detect_verifiers
from .events import emit
from .episode import DEFAULT_ROLES, best_episode, run_candidates, run_episode
from .export import export_episodes
from .taskspec import TaskSpec, Verifier, load_task, load_tasks
from .training import datasets
from .verify import check_task

WORK_RUNS_DIR = Path.home() / ".local" / "share" / "heart" / "runs"


def _roles_for(args) -> list[dict] | None:
    if getattr(args, "roles", None):
        return json.loads(Path(args.roles).read_text())
    if getattr(args, "pipeline", False):
        return DEFAULT_ROLES
    return None


def _parse_variants(spec: str) -> list[tuple[str, bool]]:
    # "normal:on,clean:on,normal:off,clean:off" -> [(memory_mode, retrieval)]
    variants = []
    for part in spec.split(","):
        mem, _, ret = part.strip().partition(":")
        variants.append((mem, ret != "off"))
    return variants


def _episode_kwargs(args) -> dict:
    return dict(
        agent=args.agent, runs_dir=args.runs_dir, agent_cmd=args.agent_cmd,
        roles=_roles_for(args), fix_rounds=args.fix_rounds, escalate=args.escalate,
    )


def _ingest_rewards(runs_dir) -> None:
    """Best-effort credit-assignment bridge: hand finished episodes to arteries'
    reward ledger when its CLI is installed. A subprocess, not an import — heart
    stays stdlib-only and works fine without arteries. HEART_INGEST=off skips."""
    art = shutil.which("art")
    if not art or os.environ.get("HEART_INGEST") == "off":
        return
    try:
        proc = subprocess.run(
            [art, "ingest", str(runs_dir)], capture_output=True, text=True, timeout=120,
        )
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        if tail:
            print(f"art ingest: {tail[-1]}")
    except Exception as exc:  # rewards can always be re-ingested later
        print(f"art ingest skipped: {exc}", file=sys.stderr)


def cmd_run(args) -> int:
    task = load_task(args.task)
    eps = run_candidates(
        task, args.candidates,
        memory_mode=args.memory, retrieval=not args.no_retrieval,
        **_episode_kwargs(args),
    )
    ep = best_episode(eps)
    if len(eps) > 1:
        for e in eps:
            print(f"  candidate {e['episode_id']}: {e['outcome']} reward={e['reward']['total']}")
    print(json.dumps({k: ep[k] for k in ("episode_id", "task_id", "outcome", "reward")}, indent=2))
    _ingest_rewards(args.runs_dir)
    return 0 if ep["outcome"] == "pass" else 1


def cmd_work(args) -> int:
    """Daily driver: run a task against the current repo in an isolated worktree,
    orchestrated through the role pipeline, then optionally apply the diff."""
    repo = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    ).stdout.strip()
    if not repo:
        print("heart work must run inside a git repository", file=sys.stderr)
        return 2
    dirty = subprocess.run(
        ["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        print("note: repo has uncommitted changes; the worktree starts from HEAD without them")
    base = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    if args.verify:
        verifiers = [Verifier(name="verify", command=args.verify)]
    else:
        verifiers = detect_verifiers(repo)
        if not verifiers:
            print("no verifiers detected; pass --verify 'cmd' (episode will score without tests)")

    task = TaskSpec(
        task_id="work-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        repo_path=repo, base_commit=base, prompt=args.prompt,
        public_verifiers=verifiers, timeout_seconds=args.timeout,
    )
    kwargs = _episode_kwargs(args)
    if not args.solo:
        kwargs["roles"] = kwargs["roles"] or DEFAULT_ROLES
    ep = best_episode(run_candidates(task, args.candidates, **kwargs))

    ep_dir = Path(args.runs_dir) / ep["episode_id"]
    print(json.dumps({k: ep[k] for k in ("episode_id", "outcome", "review_verdict", "reward")}, indent=2))
    print(f"diff: {ep_dir / 'diff.patch'}  logs: {ep_dir}/")
    _ingest_rewards(args.runs_dir)

    if not args.apply:
        return 0 if ep["outcome"] == "pass" else 1
    if ep["outcome"] != "pass" or ep["review_verdict"] == "reject":
        print("not applying: episode did not pass verification + review")
        return 1
    diff = (ep_dir / "diff.patch").read_text()
    apply = subprocess.run(
        ["git", "-C", repo, "apply", "--whitespace=nowarn"],
        input=diff, capture_output=True, text=True,
    )
    if apply.returncode != 0:
        print(f"git apply failed:\n{apply.stderr}", file=sys.stderr)
        return 1
    print(f"applied to {repo} (working tree; review and commit yourself)")
    return 0


def cmd_batch(args) -> int:
    tasks = load_tasks(args.tasks_dir)
    variants = _parse_variants(args.variants)
    jobs = [
        (task, mem, ret, i)
        for task in tasks for mem, ret in variants for i in range(args.repeat)
    ]
    summary_path = Path(args.runs_dir) / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not summary_path.exists()
    if not new_file:
        # resume: a batch killed at 60/100 must not re-run (and re-pay for)
        # the 60. Start a fresh runs-dir to re-run everything.
        with open(summary_path, newline="") as f:
            done_keys = {(r[1], r[2], r[3], r[4]) for r in list(csv.reader(f))[1:] if len(r) >= 6}
        before = len(jobs)
        jobs = [j for j in jobs
                if (j[0].task_id, j[1], str(j[2]), str(j[3])) not in done_keys]
        if before - len(jobs):
            print(f"resume: {before - len(jobs)} episode(s) already in {summary_path}, "
                  f"{len(jobs)} to run")
    kwargs = _episode_kwargs(args)
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["episode_id", "task_id", "memory", "retrieval", "repeat", "outcome", "reward"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(run_episode, task, memory_mode=mem, retrieval=ret, **kwargs):
                    (task, mem, ret, i)
                for task, mem, ret, i in jobs
            }
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                task, mem, ret, i = futures[fut]
                ep = fut.result()
                done += 1
                emit("heart", "batch.progress", task_id=task.task_id,
                     done=done, total=len(jobs), outcome=ep["outcome"])
                row = [ep["episode_id"], task.task_id, mem, ret, i, ep["outcome"], ep["reward"]["total"]]
                writer.writerow(row)
                f.flush()
                print(",".join(str(x) for x in row))
    print(f"summary: {summary_path}")
    _ingest_rewards(args.runs_dir)
    return 0


def cmd_pulse(args) -> int:
    if args.what == "episode":
        if not args.id:
            print("usage: heart pulse episode <episode-id>", file=sys.stderr)
            return 2
        for line in pulse_mod.episode_timeline(args.id):
            print(line)
        return 0
    if args.what == "insights":
        for line in pulse_mod.insights(hours=args.hours):
            print(line)
        return 0
    if args.what == "health":
        lines, code = pulse_mod.health(hours=args.hours)
        for line in lines:
            print(line)
        return code
    pulse_mod.tail(
        n=args.n, episode=args.episode, task=args.task,
        source=args.source, follow=not args.once,
    )
    return 0


def cmd_stats(args) -> int:
    """Pass rates and mean reward grouped by ablation variant — the credit-
    assignment view the whole exercise exists for."""
    groups: dict[str, list[dict]] = defaultdict(list)
    outcomes: dict[str, int] = defaultdict(int)
    for ep_file in sorted(Path(args.runs_dir).glob("*/episode.json")):
        ep = json.loads(ep_file.read_text())
        key = f"{ep['memory_mode']}/{'ret-on' if ep['retrieval'] else 'ret-off'}"
        groups[key].append(ep)
        outcomes[ep["outcome"]] += 1
    if not groups:
        print(f"no episodes under {args.runs_dir}")
        return 1
    print(f"{'variant':<22}{'n':>4}{'pass%':>8}{'mean reward':>13}")
    for key in sorted(groups):
        eps = groups[key]
        passed = sum(e["outcome"] == "pass" for e in eps)
        mean_r = sum(e["reward"]["total"] for e in eps) / len(eps)
        print(f"{key:<22}{len(eps):>4}{100 * passed / len(eps):>7.0f}%{mean_r:>13.3f}")
    print("outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
    return 0


def cmd_check_task(args) -> int:
    verdict = check_task(load_task(args.task), n=args.n)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["ok"] else 1


def cmd_mine(args) -> int:
    written = mine_mod.mine(
        args.repo, args.out, limit=args.limit, scan=args.scan, test_cmd=args.test_cmd
    )
    for p in written:
        print(p)
    print(f"{len(written)} tasks written; validate with: heart check-task <task.json>")
    return 0


def cmd_export(args) -> int:
    n = export_episodes(args.runs_dir, args.out)
    print(f"{n} episodes -> {args.out}")
    return 0


def cmd_dataset(args) -> int:
    build = datasets.build_sft if args.kind == "sft" else datasets.build_dpo
    n = build(args.episodes, args.runs_dir, args.out)
    print(f"{n} {args.kind} rows -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="heart")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_agent_flags(p):
        p.add_argument(
            "--agent", default=os.environ.get("HEART_AGENT", "claude"),
            help="claude | codex | gemini | opencode | api[:profile] | shell | "
                 "auto (route by task complexity; tiers in models.json) "
                 "(default $HEART_AGENT or claude)",
        )
        p.add_argument("--agent-cmd", default=None, help="custom shell template; prompt in $HEART_PROMPT")
        p.add_argument("--runs-dir", default="runs")
        p.add_argument("--fix-rounds", type=int, default=0, help="verify-fix loop attempts in-workspace")
        p.add_argument("--escalate", default=None, help="stronger agent for the final fix attempt")

    def add_role_flags(p):
        p.add_argument("--pipeline", action="store_true", help="use implement/test/review roles")
        p.add_argument("--roles", default=None, help="JSON file with a custom role list")

    p = sub.add_parser("run", help="run one episode")
    p.add_argument("task")
    add_agent_flags(p)
    add_role_flags(p)
    p.add_argument("--memory", default="normal", choices=["normal", "readonly", "clean"])
    p.add_argument("--no-retrieval", action="store_true")
    p.add_argument("--candidates", type=int, default=1, help="best-of-N parallel attempts")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("work", help="orchestrated task against the current repo")
    p.add_argument("prompt")
    add_agent_flags(p)
    add_role_flags(p)
    p.add_argument("--solo", action="store_true", help="single agent, no role pipeline")
    p.add_argument("--verify", default=None, help="verifier command (else auto-detected)")
    p.add_argument("--timeout", type=int, default=600, help="per-role timeout seconds")
    p.add_argument("--apply", action="store_true", help="apply diff to the repo if pass+approve")
    p.add_argument("--candidates", type=int, default=1, help="best-of-N parallel attempts")
    p.set_defaults(func=cmd_work, runs_dir=str(WORK_RUNS_DIR), fix_rounds=2)

    p = sub.add_parser("batch", help="run tasks x variants x repeats")
    p.add_argument("tasks_dir")
    add_agent_flags(p)
    p.add_argument("--variants", default="normal:on", help="e.g. normal:on,clean:on,normal:off,clean:off")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--parallel", type=int, default=1, help="episodes to run concurrently")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("stats", help="pass rate / mean reward by ablation variant")
    p.add_argument("--runs-dir", default="runs")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("pulse", help="event spine: tail | episode <id> | insights | health")
    p.add_argument("what", nargs="?", default="tail",
                   choices=["tail", "episode", "insights", "health"])
    p.add_argument("id", nargs="?", help="episode id (for `pulse episode`)")
    p.add_argument("-n", type=int, default=20, help="history lines before following")
    p.add_argument("--hours", type=float, default=24, help="window for insights/health")
    p.add_argument("--episode", default=None, help="filter by episode id")
    p.add_argument("--task", default=None, help="filter by task id")
    p.add_argument("--source", default=None, help="filter by source (heart|arteries|marrow|agent)")
    p.add_argument("--once", action="store_true", help="print and exit; don't follow")
    p.set_defaults(func=cmd_pulse)

    p = sub.add_parser("check-task", help="verify determinism at base and pass at fix_commit")
    p.add_argument("task")
    p.add_argument("-n", type=int, default=3)
    p.set_defaults(func=cmd_check_task)

    p = sub.add_parser("mine", help="mine TaskSpecs from a repo's git history")
    p.add_argument("repo")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--scan", type=int, default=500)
    p.add_argument("--test-cmd", default="python3 -m pytest -x {files}")
    p.set_defaults(func=cmd_mine)

    p = sub.add_parser("export", help="episodes -> JSONL")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="episodes.jsonl")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("dataset", help="build SFT/DPO datasets from exported episodes")
    p.add_argument("kind", choices=["sft", "dpo"])
    p.add_argument("--episodes", default="episodes.jsonl")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_dataset)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
