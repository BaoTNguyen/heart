"""TaskSpec: the environment definition for one coding task."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Verifier:
    name: str
    command: str
    # How to judge this verifier's stdout against the same command run at
    # base_commit. "" (default) judges the exit code alone, as verifiers always
    # have. "identical" demands byte-identical stdout; "no_worse" reads the
    # last float in each and demands head >= base, "no_more" head <= base. Both exist because the criteria
    # that actually gate a change -- recall, latency, index size -- have no
    # meaning as a single number: they are only ever better or worse than what
    # was there before, and a pass/fail verifier cannot say which.
    baseline: str = ""


@dataclass
class TaskSpec:
    task_id: str
    repo_path: str
    base_commit: str
    prompt: str
    # allowed_paths: empty list means no restriction. denied_paths always wins.
    allowed_paths: list[str] = field(default_factory=list)
    denied_paths: list[str] = field(default_factory=list)
    # Environment facts measured at base_commit BEFORE any agent runs. Each
    # probe does two jobs from one command: a non-zero exit blocks the episode
    # (reward None -- nothing was attempted, so there is nothing to score), and
    # its stdout is handed to the agent as measured fact. The second job is the
    # point. A plan written against an assumed extension version, an absent
    # binary or a group membership nobody has is wrong before the first token,
    # and the assumption used to be tested only by the agent failing.
    probes: list[Verifier] = field(default_factory=list)
    public_verifiers: list[Verifier] = field(default_factory=list)
    hidden_verifiers: list[Verifier] = field(default_factory=list)
    # files pinned into every workspace regardless of base_commit (path -> content):
    # mined tasks pin the fix-commit's tests so they actually fail at base.
    # Overlay paths are excluded from diffs, so agents can't edit them away.
    overlay_files: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 300
    difficulty: str = "unknown"
    # routing: which capabilities the task needs (route.SKILLS), how much context
    # it will touch, and how hard the chosen model should try. Empty/0/"" -> the
    # router infers from the prompt (route.classify).
    skills: list[str] = field(default_factory=list)
    min_context: int = 0
    # "" -> the router derives effort from difficulty (medium for most, high for
    # hard); set "low"/"medium"/"high" explicitly to override.
    effort: str = ""
    tags: list[str] = field(default_factory=list)
    fix_commit: str | None = None  # known-good commit; check-task verifies it passes
    # Marker the caller told the agent to emit when a decision is missing, e.g.
    # "PLEXUS_BLOCKED:". Seeing it makes the episode `blocked` — the agent chose
    # not to guess. Heart supplies the mechanism; the vocabulary is the caller's.
    blocked_marker: str | None = None
    # Which network the sandbox gets: "none" (default), "model" for the local
    # model servers, "build" when the task genuinely has to reach a package
    # registry. Default-deny because a task that cannot reach the network cannot
    # exfiltrate, and enforcing that costs nothing. Everything else about the
    # sandbox is derived from fields above -- see sandbox.profile_for.
    network: str = "none"


def load_task(path: str | Path) -> TaskSpec:
    path = Path(path)
    data = json.loads(path.read_text())
    missing = [k for k in ("task_id", "repo_path", "base_commit", "prompt") if not data.get(k)]
    if missing:
        raise ValueError(f"{path}: missing required fields {missing}")
    for key in ("probes", "public_verifiers", "hidden_verifiers"):
        data[key] = [Verifier(**v) for v in data.get(key, [])]
    known = {f.name for f in dataclasses.fields(TaskSpec)}
    return TaskSpec(**{k: v for k, v in data.items() if k in known})


def load_tasks(directory: str | Path) -> list[TaskSpec]:
    return [load_task(p) for p in sorted(Path(directory).glob("*.json"))]
