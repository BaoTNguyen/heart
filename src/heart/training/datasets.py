"""Build training datasets from exported episodes.

SFT: passing episodes -> (prompt, diff). DPO: same-task pass/fail diff pairs.
ponytail: actual training scripts (torch/peft/trl) arrive when 100+ scored
episodes exist; gate/reranker datasets arrive with the arteries decision ledger.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path


def _load(episodes_path: str | Path) -> list[dict]:
    with open(episodes_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_diff(ep: dict, runs_dir: str | Path) -> str:
    return (Path(runs_dir) / ep["episode_id"] / "diff.patch").read_text()


def build_sft(episodes_path: str | Path, runs_dir: str | Path, out_path: str | Path) -> int:
    rows = 0
    with open(out_path, "w") as out:
        for ep in _load(episodes_path):
            if ep["outcome"] != "pass" or not ep.get("diff_lines"):
                continue
            out.write(json.dumps({
                "task_id": ep["task_id"],
                "prompt": ep["prompt"],
                "completion": _read_diff(ep, runs_dir),
                "reward": ep["reward"]["total"],
            }) + "\n")
            rows += 1
    return rows


def build_dpo(
    episodes_path: str | Path,
    runs_dir: str | Path,
    out_path: str | Path,
    max_pairs_per_task: int = 4,
) -> int:
    by_task: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"pass": [], "fail": []})
    for ep in _load(episodes_path):
        if ep["outcome"] in ("pass", "fail") and ep.get("diff_lines"):
            by_task[ep["task_id"]][ep["outcome"]].append(ep)

    rows = 0
    with open(out_path, "w") as out:
        for task_id, groups in by_task.items():
            pairs = itertools.islice(
                itertools.product(groups["pass"], groups["fail"]), max_pairs_per_task
            )
            for good, bad in pairs:
                out.write(json.dumps({
                    "task_id": task_id,
                    "prompt": good["prompt"],
                    "chosen": _read_diff(good, runs_dir),
                    "rejected": _read_diff(bad, runs_dir),
                }) + "\n")
                rows += 1
    return rows
