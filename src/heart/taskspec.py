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


@dataclass
class TaskSpec:
    task_id: str
    repo_path: str
    base_commit: str
    prompt: str
    # allowed_paths: empty list means no restriction. denied_paths always wins.
    allowed_paths: list[str] = field(default_factory=list)
    denied_paths: list[str] = field(default_factory=list)
    public_verifiers: list[Verifier] = field(default_factory=list)
    hidden_verifiers: list[Verifier] = field(default_factory=list)
    timeout_seconds: int = 300
    difficulty: str = "unknown"
    tags: list[str] = field(default_factory=list)
    fix_commit: str | None = None  # known-good commit; check-task verifies it passes


def load_task(path: str | Path) -> TaskSpec:
    path = Path(path)
    data = json.loads(path.read_text())
    missing = [k for k in ("task_id", "repo_path", "base_commit", "prompt") if not data.get(k)]
    if missing:
        raise ValueError(f"{path}: missing required fields {missing}")
    for key in ("public_verifiers", "hidden_verifiers"):
        data[key] = [Verifier(**v) for v in data.get(key, [])]
    known = {f.name for f in dataclasses.fields(TaskSpec)}
    return TaskSpec(**{k: v for k, v in data.items() if k in known})


def load_tasks(directory: str | Path) -> list[TaskSpec]:
    return [load_task(p) for p in sorted(Path(directory).glob("*.json"))]
