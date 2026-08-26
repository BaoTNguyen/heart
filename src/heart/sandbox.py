"""Least-privilege container profile derived from a TaskSpec.

The spec already says what a task is allowed to touch. Today `path_violations`
reads that back off the diff *after* the agent has finished, which detects a
breach it could have prevented. This module turns the same fields into the
mount table: a path the task may not write is mounted read-only, so writing it
fails at the syscall instead of at the review.

Nothing here is baked into an image. Every restriction is a `docker run` flag,
so two tasks in the same episode image can get genuinely different privileges.

The profile is deliberately shaped as data. `docker_args()` is the only place
that knows Docker exists, which is what keeps a second runtime -- gVisor, a
microVM, a plain worktree -- from being a rewrite.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .taskspec import TaskSpec

WORK = "/work"
CONTEXT = "/context"
JOURNAL = "/journal"

# Networks a task may ask for. "none" is the default because a task that cannot
# reach the network cannot exfiltrate, and that costs nothing to enforce.
# "model" reaches llama-server and the embedding server and nothing else; it is
# a docker network the operator creates, not something heart can conjure.
NETWORKS = {
    "none": "none",
    "model": os.getenv("HEART_MODEL_NETWORK", "heart-model"),
    "build": os.getenv("HEART_BUILD_NETWORK", "heart-build"),
}

# Inside a container, 127.0.0.1 is the container. A model server bound to host
# loopback is unreachable no matter which network the container joins, so an
# agent that resolves its endpoint from the host's config would quietly fail to
# connect. The container is told the address that reaches the host from its
# side; the host keeps its own.
#
# Pair this with binding the model server to the bridge address rather than
# 0.0.0.0 -- the lazy fix exposes the model to the whole LAN.
CONTAINER_MODEL_ENDPOINT = os.getenv(
    "HEART_CONTAINER_API_ENDPOINT", "http://host.docker.internal:8001/v1")


@dataclass(frozen=True)
class Mount:
    source: str
    target: str
    writable: bool = False

    def arg(self) -> str:
        return f"{self.source}:{self.target}:{'rw' if self.writable else 'ro'}"


@dataclass(frozen=True)
class SandboxProfile:
    image: str
    mounts: tuple[Mount, ...]
    network: str
    timeout_seconds: int
    env: dict[str, str] = field(default_factory=dict)
    memory: str = "4g"
    cpus: str = "2"
    pids: int = 512
    user: str = ""
    tmpfs: tuple[str, ...] = ("/tmp",)

    def docker_args(self) -> list[str]:
        args = [
            "--rm",
            "--read-only",                      # only the mounts below are writable
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--network", self.network,
            "--memory", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", str(self.pids),
            "--workdir", WORK,
        ]
        for path in self.tmpfs:
            args += ["--tmpfs", f"{path}:rw,nosuid,nodev,size=512m"]
        if self.user:
            args += ["--user", self.user]
        for mount in self.mounts:
            args += ["-v", mount.arg()]
        for key, value in sorted(self.env.items()):
            args += ["-e", f"{key}={value}"]
        args.append(self.image)
        return args


def profile_for(
    task: TaskSpec,
    workspace: str | Path,
    context_dir: str | Path,
    inbox_dir: str | Path,
    env: dict[str, str] | None = None,
    image: str | None = None,
) -> SandboxProfile:
    """The container a subtask of `task` is allowed to run in."""
    return SandboxProfile(
        image=image or os.getenv("HEART_SANDBOX_IMAGE", "heart-agent:latest"),
        mounts=_mounts(task, Path(workspace), Path(context_dir), Path(inbox_dir)),
        network=NETWORKS.get(_network(task), "none"),
        timeout_seconds=task.timeout_seconds,
        env=dict(env or {}) | _agent_env(task),
        memory=os.getenv("HEART_SANDBOX_MEMORY", "4g"),
        cpus=os.getenv("HEART_SANDBOX_CPUS", "2"),
        user=os.getenv("HEART_SANDBOX_USER", f"{os.getuid()}:{os.getgid()}"),
    )


def _agent_env(task: TaskSpec) -> dict[str, str]:
    env = {"EVENT_JOURNAL_DIR": JOURNAL}
    if _network(task) == "model":
        env["HEART_API_ENDPOINT"] = CONTAINER_MODEL_ENDPOINT
    return env


def _network(task: TaskSpec) -> str:
    return (getattr(task, "network", "") or "none").strip().lower()


def _mounts(task: TaskSpec, workspace: Path, context: Path, inbox: Path) -> tuple[Mount, ...]:
    """Worktree, context, journal -- and nothing else.

    allowed_paths inverts the default: the worktree goes in read-only and each
    allowed subtree is remounted writable on top. Saying "only these" is exact,
    whereas trying to deny everything else is a list you will get wrong.
    denied_paths then goes back over the top read-only, so it wins either way --
    the same precedence path_violations already documents.
    """
    restricted = bool(task.allowed_paths)
    mounts = [Mount(str(workspace), WORK, writable=not restricted)]

    for rel in task.allowed_paths:
        target = _under(WORK, rel)
        if target:
            mounts.append(Mount(str(workspace / rel), target, writable=True))

    for rel in task.denied_paths:
        target = _under(WORK, rel)
        if target:
            mounts.append(Mount(str(workspace / rel), target, writable=False))

    mounts += _git_mounts(task, workspace)

    # Read-only: the packet, retrieved memory and prompt are built on the host
    # so the container never needs a credential to fetch them.
    mounts.append(Mount(str(context), CONTEXT, writable=False))
    # The one outbound channel: this run's own journal inbox, not the shared
    # journal. Per-run means one container can neither read nor corrupt
    # another's record, and concurrent appends cannot interleave. The host
    # drains it -- appending needs no credential, a database connection would.
    mounts.append(Mount(str(inbox), JOURNAL, writable=True))
    return tuple(mounts)


def _git_mounts(task: TaskSpec, workspace: Path) -> list[Mount]:
    """What a worktree needs for git to work inside the container.

    A worktree's `.git` is a file, not a directory, holding an absolute path:
    `gitdir: <repo>/.git/worktrees/<name>`. Mount only the worktree and that
    path does not exist, so every git command fails -- including the `git diff`
    the review roles are told to run.

    Both mounts land at their host paths because the pointer is absolute. The
    worktree itself does not: git tolerates being read from somewhere other than
    where it was recorded, which is what lets the tree sit at /work.

    The object store goes in read-only, and that is the whole point. Measured:
    diff, status, log and blame all work against it; `git add` and `git commit`
    are refused with "insufficient permission for adding an object". The agent
    gets full history to debug with and cannot write a single object or move a
    ref. Heart commits afterwards on the host, where the store is writable.

    The per-worktree directory is writable because git updates its index and
    HEAD to answer those reads. It holds this worktree's metadata only -- no
    objects, no refs belonging to anything else.
    """
    git_dir = Path(task.repo_path) / ".git"
    if not git_dir.is_dir():
        return []  # a clone or a plain export needs none of this
    mounts = [Mount(str(git_dir), str(git_dir), writable=False)]
    per_worktree = git_dir / "worktrees" / workspace.name
    if per_worktree.is_dir():
        mounts.append(Mount(str(per_worktree), str(per_worktree), writable=True))
    return mounts


def _under(root: str, rel: str) -> str | None:
    """Join, refusing anything that escapes root. A task spec is data, and a
    denied_paths entry of '../..' must not become a mount of the host."""
    rel = rel.strip().lstrip("/")
    if not rel:
        return None
    target = os.path.normpath(os.path.join(root, rel))
    return target if target.startswith(root + "/") else None


# Caches a test run writes into the tree it is testing. On a read-only worktree
# they have to land in memory instead, or the verifier fails for reasons that
# have nothing to do with the code.
VERIFIER_TMPFS = ("/tmp", f"{WORK}/.pytest_cache", f"{WORK}/.ruff_cache",
                  f"{WORK}/node_modules/.cache")


def verifier_profile_for(
    task: TaskSpec,
    workspace: str | Path,
    inbox_dir: str | Path,
    env: dict[str, str] | None = None,
    image: str | None = None,
) -> SandboxProfile:
    """The container a verifier runs in: same mechanism, different role.

    Not a second sandbox. The profile is data, so "stricter" is a different set
    of arguments to the same machinery -- one image, one runtime, one set of
    flags rendered differently.

    Three things change, and none of them is containment strength:

    Writability. The agent must write the worktree; the verifier must not. If a
    verifier can edit the tree it is judging, "produce" and "judge" stop being
    separate steps -- which is a reward-integrity property, not a security one,
    and the reason this profile exists at all.

    Network. Always none, never derived from the task. verify.py already forces
    bwrap-nonet regardless of the agent's mode; this keeps that policy true when
    the runtime is a container instead.

    Context. Verifiers get no /context mount. They judge code, not memory, and a
    verifier that can read the continuity packet can be steered by it.
    """
    return SandboxProfile(
        image=image or os.getenv("HEART_SANDBOX_IMAGE", "heart-agent:latest"),
        mounts=(
            Mount(str(workspace), WORK, writable=False),
            Mount(str(inbox_dir), JOURNAL, writable=True),
        ),
        network="none",
        timeout_seconds=task.timeout_seconds,
        env=dict(env or {}) | {
            "EVENT_JOURNAL_DIR": JOURNAL,
            # same reasoning as verify.py: a stale .pyc from a same-second edit
            # must never decide whether a verifier passes
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        memory=os.getenv("HEART_SANDBOX_MEMORY", "4g"),
        cpus=os.getenv("HEART_SANDBOX_CPUS", "2"),
        user=os.getenv("HEART_SANDBOX_USER", f"{os.getuid()}:{os.getgid()}"),
        tmpfs=VERIFIER_TMPFS,
    )


def inbox_for(key: str) -> Path:
    """The journal inbox a container writes into, asked of arteries.

    Keyed by episode, not by subtask: the inbox is the isolation unit (one per
    container) while ARTERIES_RUN_ID is the identity unit (one per role). Every
    role in an episode appends to the same directory and the drain separates
    them by the run_id on each event.

    Asked, never computed. Guessing the path is the failure that cannot be
    noticed: a container writing where the drain does not read loses everything
    that run remembered, silently, and the episode looks like it simply had no
    memory. Raising here follows the same rule as a missing bwrap -- a requested
    sandbox must fail loudly rather than quietly degrade.
    """
    import subprocess
    try:
        out = subprocess.run(["python3", "-m", "arteries.cli", "journal", "inbox", key],
                             capture_output=True, text=True, timeout=30, check=True)
    except Exception as exc:
        raise RuntimeError(
            "HEART_SANDBOX=docker needs arteries to name the journal inbox "
            f"(`art journal inbox {key}`); it is not answering: {exc}"
        ) from exc
    path = Path(out.stdout.strip())
    if not path.is_absolute():
        raise RuntimeError(f"arteries returned a non-absolute inbox path: {path!r}")
    return path
