"""Episode workspaces: one detached git worktree per episode."""
from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import subprocess
import uuid
from pathlib import Path

WS_ROOT = Path.home() / ".cache" / "heart-ws"


def _ws_root() -> Path:
    # env-aware so callers/tests that set HEART_WS_ROOT (like `heart clean`) and
    # this reclaimer agree on where worktrees live
    return Path(os.environ.get("HEART_WS_ROOT", str(WS_ROOT)))


def _force_rmtree(path: Path) -> bool:
    """Remove a tree, including directories an agent left unwritable.

    A worktree can come back with mode 0500 on a directory -- a test fixture, a
    build step, an agent doing whatever it liked inside its sandbox -- and
    rmtree cannot delete a child it has no write permission on the parent for.
    With ignore_errors that failure is silent, and a reclaimer that counts what
    it tried rather than what it achieved reports a clean sweep over a backlog
    that is still there. Measured: 48 reported, 48 still on disk.

    Returns whether the tree is actually gone.
    """
    shutil.rmtree(path, ignore_errors=True)
    if not path.exists():
        return True
    for child in sorted(path.rglob("*"), reverse=True):
        with contextlib.suppress(OSError):
            child.chmod(0o700)
    with contextlib.suppress(OSError):
        path.chmod(0o700)
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def _lock_path(worktree: Path) -> Path:
    """The liveness lock for a worktree, beside it rather than inside it.

    Inside, it would show up in `git status` and need another DIFF_EXCLUDES
    entry; beside, the worktree stays exactly what git thinks it is.
    """
    return worktree.parent / f"{worktree.name}.lock"


def worktree_source_repo(worktree: Path) -> str | None:
    """A worktree's `.git` file reads `gitdir: <repo>/.git/worktrees/<id>`;
    walk back up to the repo root. Best-effort -- None if unreadable."""
    git_file = worktree / ".git"
    if not git_file.is_file():
        return None
    try:
        line = git_file.read_text().strip()
    except OSError:
        return None
    if not line.startswith("gitdir:"):
        return None
    gitdir = Path(line.split(":", 1)[1].strip())
    # <repo>/.git/worktrees/<id> -> <repo>
    return str(gitdir.parent.parent.parent) if len(gitdir.parts) >= 3 else None


def _is_live(worktree: Path) -> bool:
    """True while a running Workspace holds this worktree.

    An advisory lock, not a heuristic. The kernel drops it when the owner dies
    however it dies -- SIGKILL, OOM, power loss -- so a lock that can be taken
    means nobody is using the tree. That is what lets reclaim run with no age
    cutoff and no external lock: `heart clean` used to guess that a recent
    worktree might be live, and prune_repo_worktrees relied on plexus holding a
    goal lock. This asks instead of guessing.

    A worktree with no lock file at all predates this and is not owned by
    anyone, so it is reclaimable too.

    (Advisory locks are unreliable on NFS. WS_ROOT is a local cache dir.)
    """
    lock = _lock_path(worktree)
    if not lock.exists():
        return False
    try:
        with open(lock, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh, fcntl.LOCK_UN)
        return False
    except OSError:
        return True


def reclaim(repo: str | Path | None = None, older_than: float | None = None) -> int:
    """Remove worktrees under WS_ROOT that no live Workspace owns.

    One reclaimer, because there used to be two. `heart clean` walked the disk
    with an age cutoff; prune_repo_worktrees walked a repo's own worktree list
    with none. They read the source repo from opposite directions and differed
    only in what made them safe to run, which is the shape that drifts apart
    and leaves the stale copy winning.

    Disk-first is the direction that matters. Asking a repo for its worktrees
    only ever finds leaks whose repo still exists -- measured on one real
    backlog, that was 1 of 86, the rest belonging to task clones and temp repos
    already deleted. Those are invisible from the repo side forever.

    `repo` narrows to one source repo, `older_than` to a mtime cutoff; neither
    is needed for safety any more (see _is_live) and both are just filters.

    Nothing is lost by reclaiming. A finished episode's work lives in
    refs/heart/, which survives worktree removal by design, and an episode
    killed before it committed never wrote an episode.json, so marrow has no
    record to join a worktree to.

    Best-effort: returns how many it removed.
    """
    ws_root = _ws_root()
    if not ws_root.is_dir():
        return 0
    repo = str(Path(repo).resolve()) if repo else None
    removed, repos = 0, set()
    for worktree in ws_root.iterdir():
        if not worktree.is_dir():
            continue
        try:
            if older_than is not None and worktree.stat().st_mtime >= older_than:
                continue
        except OSError:
            continue
        if _is_live(worktree):
            continue
        source = worktree_source_repo(worktree)
        if repo and (source is None or str(Path(source).resolve()) != repo):
            continue
        if not _force_rmtree(worktree):
            continue  # count what was achieved, not what was attempted
        _lock_path(worktree).unlink(missing_ok=True)
        removed += 1
        if source:
            repos.add(source)
    # a lock whose worktree is gone is litter: the tree was removed by
    # something that did not know about the lock (an older heart, a manual rm)
    for lock in ws_root.glob("*.lock"):
        if not (ws_root / lock.name[:-len(".lock")]).exists():
            lock.unlink(missing_ok=True)
    # one prune per repo, not one `git worktree remove` per leak: prune is what
    # deregisters a worktree whose directory is gone, and the per-leak removal
    # it replaces cost a subprocess each (measured ~11x on 20 worktrees).
    for source in repos:
        subprocess.run(["git", "-C", source, "worktree", "prune"], capture_output=True)
    return removed


def prune_repo_worktrees(repo: str | Path) -> int:
    """Reclaim one repo's leaked worktrees. Kept because plexus calls it at the
    top of a goal walk; `reclaim()` is the general form."""
    return reclaim(repo=repo)


# untracked integration files a worktree checkout doesn't carry; without them
# agents in the workspace run with no arteries memory/retrieval hooks at all
INTEGRATION_FILES = (".arteries", ".claude/settings.local.json", ".codex/config.toml")


def _run(args: list[str], cwd: str, input_text: str | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        args, cwd=cwd, input=input_text, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed in {cwd}:\n{proc.stderr.strip()}")
    return proc


@contextlib.contextmanager
def _worktree_lock(repo_path: str):
    """Serialise `git worktree add` and `remove` for one repo.

    Both take $GIT_DIR/worktrees.lock and write under .git/worktrees/<id>.
    Concurrent episodes in one repo therefore race, and the loser dies on
    "Unable to create '.../index.lock': File exists" -- a hard failure at
    episode start, not corruption, but a confusing one to read.

    A lock rather than a retry loop: the operation it serialises takes
    milliseconds, so the throughput cost is nil, and retries would turn real
    contention into intermittent flakiness with nothing naming the cause. The
    fd closes on process exit, so a crash cannot strand the lock.
    """
    lock_path = Path(repo_path) / ".git" / "heart-worktree.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "w")
    except OSError:
        yield  # not a git repo, or unwritable: nothing to serialise against
        return
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        handle.close()


_swept = False


def _sweep_once() -> None:
    """Reclaim leaks the first time this process makes a workspace.

    Here rather than in a command someone has to remember: only plexus ever
    called the old reclaimer, so driving heart directly leaked indefinitely and
    silently. Once per process because the cost is a WS_ROOT walk and leaks do
    not appear while we run -- our own trees are locked.
    """
    global _swept
    if _swept:
        return
    _swept = True
    try:
        reclaim()
    except Exception:
        pass  # never let cleanup fail an episode


class Workspace:
    def __init__(self, repo_path: str, commit: str, overlay: dict[str, str] | None = None):
        self.repo_path = str(repo_path)
        self.overlay = overlay or {}
        self.path = _ws_root() / uuid.uuid4().hex[:12]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _sweep_once()
        # Held for this object's lifetime, so a reclaim running alongside can
        # tell a live worktree from a leak without guessing at its age.
        self._lock = open(_lock_path(self.path), "w")
        fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with _worktree_lock(self.repo_path):
            _run(["git", "worktree", "add", "--detach", str(self.path), commit],
                 cwd=self.repo_path)
        for rel, content in self.overlay.items():
            target = self.path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        for rel in INTEGRATION_FILES:
            src = Path(self.repo_path) / rel
            if src.exists() and not (self.path / rel).exists():
                dst = self.path / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    # ponytail: runs/ and decisions/ excluded — fallback data
                    # written there would die with the worktree
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("runs", "decisions"))
                else:
                    shutil.copy2(src, dst)

    # agents run tests inside the workspace; their cache junk must not reach diffs
    EXCLUDED_PATHS = [
        "__pycache__", "*.pyc", ".pytest_cache", "node_modules",
        # integration files we copied in ourselves (INTEGRATION_FILES)
        ".arteries", ".claude", ".codex",
    ]
    DIFF_EXCLUDES = [f":(exclude){p}" for p in EXCLUDED_PATHS]

    def diff(self) -> str:
        # intent-to-add so untracked files created by the agent show up in the diff
        _run(["git", "add", "-A", "-N"], cwd=str(self.path))
        overlay_excludes = [f":(exclude){rel}" for rel in self.overlay]
        return _run(
            ["git", "diff", "--binary", "--", ".", *self.DIFF_EXCLUDES, *overlay_excludes],
            cwd=str(self.path),
        ).stdout

    def commit(self, message: str, ref: str | None = None) -> str | None:
        """Commit the worktree's changes and return the sha, or None if clean.

        A patch file is a description of work; a commit is the work itself --
        content-addressed, verifiable, and cherry-pickable without a fuzz factor.
        Binary files, renames and mode changes survive it intact, which
        `git apply` on a captured diff does not reliably manage.

        The worktree is detached, so this creates no branch. Passing `ref` writes
        the sha under refs/heart/ instead: outside refs/heads, so it never shows
        in `git branch` or gets pushed by accident, but referenced, so the commit
        survives `git worktree remove` and the next gc. That is the whole trick
        for keeping real commits without a thicket of branches.
        """
        # Stage with NO pathspec, then unstage what must not travel.
        #
        # `git add -A -- . :(exclude)X` fails outright (exit 1) when X is also in
        # .gitignore and present: the `.` names it, git refuses to add a named
        # ignored path, and the exclude does not suppress that check. heart
        # copies .claude and .arteries into every worktree and most real repos
        # gitignore both, so heart created the collision itself -- every commit
        # on such a repo raised, which on Path B took the whole orchestration
        # down. Toy repos with no .gitignore never saw it.
        #
        # `-f` would silence it and is the wrong fix: it stages EVERY ignored
        # file the excludes do not name, .env included, straight into the diff
        # heart scores and applies.
        #
        # Bare `git add -A` respects .gitignore on its own, so the excludes are
        # only here for paths a repo does NOT ignore. Unstaging them after is
        # exact, and `git reset` on a path that matched nothing is a no-op.
        _run(["git", "add", "-A"], cwd=str(self.path))
        _run(["git", "reset", "-q", "--", *self.EXCLUDED_PATHS, *self.overlay],
             cwd=str(self.path))
        # --quiet exits 1 when there is something staged, so _run's raise-on-
        # nonzero is the wrong helper here
        clean = subprocess.run(["git", "diff", "--cached", "--quiet"],
                               cwd=str(self.path), capture_output=True,
                               check=False).returncode == 0
        if clean:
            return None  # the agent changed nothing
        _run(["git", "-c", "user.name=heart", "-c", "user.email=heart@localhost",
              "commit", "--no-verify", "-q", "-m", message], cwd=str(self.path))
        sha = _run(["git", "rev-parse", "HEAD"], cwd=str(self.path)).stdout.strip()
        if ref:
            _run(["git", "update-ref", f"refs/heart/{ref}", sha], cwd=self.repo_path)
        return sha

    def apply(self, patch: str) -> None:
        _run(["git", "apply", "--whitespace=nowarn"], cwd=str(self.path), input_text=patch)

    def destroy(self) -> None:
        with _worktree_lock(self.repo_path):
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(self.path)],
                cwd=self.repo_path, capture_output=True, check=False,
            )
        # release last: while the lock is held the tree reads as live, so a
        # concurrent reclaim leaves it alone until the removal is finished
        try:
            self._lock.close()
        finally:
            _lock_path(self.path).unlink(missing_ok=True)
