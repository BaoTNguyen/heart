"""Repo-local arteries state lives at <repo>/.vascular/arteries: heart copies it
into every worktree and the API agent loop finds its observe hook there. The
old .arteries location is not a fallback -- a separate migration moves it."""
import shutil
import subprocess

import pytest

import heart.agents_api
import heart.env

pytestmark = pytest.mark.skipif(
    not (shutil.which("git") and shutil.which("bash")), reason="needs git and bash")

HOOK = "#!/bin/sh\necho remembered-$1\n"


def test_worktree_carries_vascular_arteries_hook(tmp_path, monkeypatch):
    monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / "a.txt").write_text("a\n")
    subprocess.run([*git, "add", "a.txt"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    hooks = repo / ".vascular" / "arteries" / "hooks"
    hooks.mkdir(parents=True)
    (hooks / "observe.sh").write_text(HOOK)

    ws = heart.env.Workspace(str(repo), head)
    try:
        assert (ws.path / ".vascular" / "arteries" / "hooks" / "observe.sh").exists()
        assert not (ws.path / ".arteries").exists()
        assert ws.diff() == ""
        monkeypatch.chdir(ws.path)
        assert "remembered-x" in heart.agents_api._arteries_context("x")
    finally:
        ws.destroy()


def test_old_arteries_location_is_not_a_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".arteries" / "hooks").mkdir(parents=True)
    (tmp_path / ".arteries" / "hooks" / "observe.sh").write_text(HOOK)
    assert heart.agents_api._arteries_context("x") == ""
