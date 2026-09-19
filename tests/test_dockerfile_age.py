"""The Dockerfile staleness check reads commit time, not checkout time.

A fresh clone stamps every file with today's mtime. Before this, that made a
correctly built image look older than its Dockerfile, and every sandboxed
episode raised on a rebuild it did not need.
"""
import os
import subprocess
import time
from pathlib import Path

from heart.sandbox import _dockerfile_changed_at


def _repo(tmp_path: Path) -> Path:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    df = tmp_path / "Dockerfile"
    df.write_text("FROM scratch\n")
    subprocess.run(["git", "add", "Dockerfile"], cwd=tmp_path, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "add"], cwd=tmp_path, check=True, env=env)
    return df


def test_committed_dockerfile_uses_commit_time_not_mtime(tmp_path):
    df = _repo(tmp_path)
    # simulate a fresh checkout: mtime is now, the commit is older
    future = time.time() + 86_400
    os.utime(df, (future, future))
    assert _dockerfile_changed_at(df) < future, "read the checkout mtime, not the commit"


def test_uncommitted_edit_falls_back_to_mtime(tmp_path):
    df = _repo(tmp_path)
    df.write_text("FROM scratch\nRUN true\n")  # dirty, never committed
    future = time.time() + 86_400
    os.utime(df, (future, future))
    assert _dockerfile_changed_at(df) == future


def test_outside_git_falls_back_to_mtime(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM scratch\n")
    assert _dockerfile_changed_at(df) == df.stat().st_mtime
