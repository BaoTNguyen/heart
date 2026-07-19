"""Episode workspaces: one detached git worktree per episode."""
from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

WS_ROOT = Path.home() / ".cache" / "heart-ws"

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


class Workspace:
    def __init__(self, repo_path: str, commit: str, overlay: dict[str, str] | None = None):
        self.repo_path = str(repo_path)
        self.overlay = overlay or {}
        self.path = WS_ROOT / uuid.uuid4().hex[:12]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "worktree", "add", "--detach", str(self.path), commit], cwd=self.repo_path)
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
    DIFF_EXCLUDES = [
        ":(exclude)__pycache__", ":(exclude)*.pyc",
        ":(exclude).pytest_cache", ":(exclude)node_modules",
        # integration files we copied in ourselves (INTEGRATION_FILES)
        ":(exclude).arteries", ":(exclude).claude", ":(exclude).codex",
    ]

    def diff(self) -> str:
        # intent-to-add so untracked files created by the agent show up in the diff
        _run(["git", "add", "-A", "-N"], cwd=str(self.path))
        overlay_excludes = [f":(exclude){rel}" for rel in self.overlay]
        return _run(
            ["git", "diff", "--binary", "--", ".", *self.DIFF_EXCLUDES, *overlay_excludes],
            cwd=str(self.path),
        ).stdout

    def apply(self, patch: str) -> None:
        _run(["git", "apply", "--whitespace=nowarn"], cwd=str(self.path), input_text=patch)

    def destroy(self) -> None:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(self.path)],
            cwd=self.repo_path, capture_output=True,
        )
