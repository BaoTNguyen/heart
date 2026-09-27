"""Where the vascular stack keeps its state. Vendored verbatim into every repo.

One root for everything a component writes outside its checkout, so the
install route (the vascular umbrella or a single repo) never changes where
state lands, and uninstall has one place to look:

    ~/.vascular/<kind>/<component>/...      (VASCULAR_HOME overrides ~/.vascular)

kinds:  config   settings a person edits or a setup step writes
        state    regenerable runtime records (runs, events, usage, daemons)
        cache    safe to delete at any time (worktrees, downloads)
        data     things a person wrote; uninstall keeps them unless --purge
        backups  dumps taken before anything destructive

Inside a repo the stack works on, the same idea: <repo>/.vascular/<component>/.

Stdlib only, no I/O at import. Keep the copies identical: vascular's CI
compares their hashes.
"""
from __future__ import annotations

import os
from pathlib import Path

KINDS = ("config", "state", "cache", "data", "backups")


def home() -> Path:
    """The stack's root: $VASCULAR_HOME, else ~/.vascular."""
    return Path(os.environ.get("VASCULAR_HOME") or Path.home() / ".vascular")


def path(kind: str, component: str, *parts: str) -> Path:
    """home()/kind/component/parts. Creates nothing."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    return home().joinpath(kind, component, *parts)


def journal_dir() -> Path:
    """The event journal heart, arteries and capillaries all append to.

    Shared, so it has one definition: $EVENT_JOURNAL_DIR, else heart's state."""
    return Path(os.environ.get("EVENT_JOURNAL_DIR") or path("state", "heart", "events"))


def repo_dir(root: str | Path, component: str) -> Path:
    """<root>/.vascular/<component>: a component's files inside a repo it works on."""
    return Path(root) / ".vascular" / component
