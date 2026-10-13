"""Write pulse events: one JSON line per event, in pulse's contract/envelope.json.

Vendored verbatim into every Python repo that emits (heart, arteries,
capillaries). Keep the copies identical: vascular's CI compares their hashes.
The canonical copy lives in pulse at emitters/python/pulse_emit.py.

Stdlib only, Python 3.10+. emit() never raises: observability must never take
down what it observes.

What every line gets, beyond what the caller passes:
  id              a fresh uuid4, so a replay can't duplicate it
  schema_version  envelope version this module writes
  emitter_seq     per-process counter, ordering events that share a timestamp
  context_id      names this process's provenance (component, commit, host,
                  sandbox, lane, experiment), announced once per context by a
                  `context.started` event
  project         $PULSE_PROJECT, else $PLEXUS_GOAL_ID, else the name of the git
                  checkout the process runs in (host only; in a sandbox the
                  checkout is /work, which names nothing)
  trace_id, parent_span_id
                  from $PULSE_TRACE_ID / $PULSE_PARENT_SPAN, when a parent
                  process set them
Fields the caller passes win over all of these.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import itertools
import json
import os
import socket
import threading
import uuid
from pathlib import Path

SCHEMA_VERSION = 1

_seq = itertools.count(1)
_lock = threading.Lock()
_announced: set[str] = set()


@functools.lru_cache(maxsize=None)  # fixed for the life of the process
def _git_commit(start: Path) -> str:
    """The commit checked out above `start`, read from .git without running git."""
    for d in (start, *start.parents):
        git = d / ".git"
        if git.is_file():  # a worktree or submodule: "gitdir: <path>"
            git = Path(git.read_text().split(":", 1)[1].strip())
            if not git.is_absolute():
                git = (d / git).resolve()
        if not git.is_dir():
            continue
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        common = git / "commondir"
        roots = [git] + ([(git / common.read_text().strip()).resolve()] if common.is_file() else [])
        for root in roots:
            if (root / ref).is_file():
                return (root / ref).read_text().strip()
            packed = root / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split(" ", 1)[0]
        return ""
    return ""


@functools.lru_cache(maxsize=None)
def _host() -> str:
    return socket.gethostname()


def _in_container() -> bool:
    return Path("/.dockerenv").exists()


def _project() -> str:
    name = os.environ.get("PULSE_PROJECT") or os.environ.get("PLEXUS_GOAL_ID")
    if name:
        return name
    if _in_container():
        return ""
    for d in (Path.cwd(), *Path.cwd().parents):
        if (d / ".git").exists():
            return d.name
    return ""


def context(component: str) -> dict:
    """This process's provenance. Its hash is the context_id."""
    ctx = {
        "component": component,
        "git_commit": _git_commit(Path(__file__).resolve().parent),
        "host": _host(),
        "sandbox": "docker" if _in_container() else "host",
        "lane": os.environ.get("ARTERIES_LANE", ""),
        "experiment_id": os.environ.get("PULSE_EXPERIMENT_ID", ""),
        "variant": os.environ.get("PULSE_VARIANT", ""),
    }
    return {k: v for k, v in ctx.items() if v}


def context_id(ctx: dict) -> str:
    return hashlib.sha256(json.dumps(ctx, sort_keys=True).encode()).hexdigest()[:16]


def event(source: str, kind: str, **fields) -> dict:
    """One envelope. None values are dropped (an absent field, per the contract)."""
    ctx = context(source)
    ev = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": source,
        "kind": kind,
        "id": str(uuid.uuid4()),
        "schema_version": SCHEMA_VERSION,
        "emitter_seq": next(_seq),
        "context_id": context_id(ctx),
        "project": _project(),
        "trace_id": os.environ.get("PULSE_TRACE_ID", ""),
        "parent_span_id": os.environ.get("PULSE_PARENT_SPAN", ""),
    }
    ev = {k: v for k, v in ev.items() if v != ""}
    ev.update({k: v for k, v in fields.items() if v is not None})
    return ev


def append(ev: dict, journal: Path) -> None:
    """Append one line with a single write(2) on an O_APPEND descriptor.

    One write call is what keeps concurrent writers' lines whole: the kernel
    positions and writes it in one step. Buffered file objects may split a long
    line into several writes.
    """
    day = ev["ts"][:10].replace("-", "")
    journal.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(ev, default=str, separators=(",", ":")) + "\n").encode()
    fd = os.open(journal / f"{day}.ndjson", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _announce(source: str, journal: Path, state: Path | None) -> None:
    """Write context.started once per context: once per process, and across
    processes too when `state` (a shared directory) is given. Short-lived
    processes such as hooks would otherwise announce on every run."""
    ctx = context(source)
    cid = context_id(ctx)
    with _lock:
        if cid in _announced:
            return
        _announced.add(cid)
    if state is not None:
        try:
            state.mkdir(parents=True, exist_ok=True)
            os.close(os.open(state / cid, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
        except FileExistsError:
            return
    append(event(source, "context.started", payload=ctx), journal)


def emit(source: str, kind: str, journal: Path, state: Path | None = None, **fields) -> None:
    """Write `kind` from `source` to the journal directory. Never raises.

    `fields` are envelope fields (episode_id, task_id, origin, ...) and/or
    `payload=` (a dict). `state` is a shared directory recording which contexts
    were already announced; pass vascular_paths.path("state", "pulse", "contexts").
    """
    try:
        _announce(source, journal, state)
        append(event(source, kind, **fields), journal)
    except Exception:
        pass
