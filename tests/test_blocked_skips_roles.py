"""An implementer that writes PLEXUS_BLOCKED stops the pipeline: no verifier,
no fixer, no test or review turn -- and the episode says so."""
from __future__ import annotations

import subprocess
from pathlib import Path

from heart import episode
from heart.taskspec import TaskSpec, Verifier

ROLES = [
    {"name": "implement", "verify_after": True, "prompt": "{prompt}"},
    {"name": "test", "agent": "stub", "allowed_paths": ["tests"], "prompt": "test {prompt}"},
    {"name": "review", "agent": "stub", "review": True, "prompt": "review {prompt}"},
]


def _run(tmp_path, monkeypatch, block: bool):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVENT_JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.setenv("HEART_INGEST", "off")
    monkeypatch.setenv("ARTERIES_RETRIEVAL", "off")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("x = 1\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    base = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()

    calls: list[str] = []
    verifies: list[str] = []

    def fake_agent(agent, prompt, cwd, env, timeout, log_path, agent_cmd=None, profile=None):
        role = env["HEART_ROLE"]
        calls.append(role)
        if role == "implement":
            name, text = (("PLEXUS_BLOCKED", "PLEXUS_BLOCKED: sync or async?\n") if block
                          else ("calc.py", "x = 2\n"))
            (Path(cwd) / name).write_text(text)
        Path(log_path).write_text("APPROVE\n")
        return {"exit_code": 0, "timed_out": False, "duration_s": 0.0}

    def fake_verify(verifiers, cwd, timeout, profile=None):
        verifies.append(cwd)
        return {v.name: {"passed": True, "exit_code": 0, "output_tail": ""}
                for v in verifiers}

    monkeypatch.setattr(episode, "run_agent", fake_agent)
    monkeypatch.setattr(episode, "run_verifiers", fake_verify)
    task = TaskSpec(task_id="t", repo_path=str(repo), base_commit=base, prompt="go",
                    allowed_paths=["calc.py"],  # scoped: the marker is outside it
                    public_verifiers=[Verifier(name="unit", command="true")],
                    timeout_seconds=30)
    ep = episode.run_episode(task, agent="stub", runs_dir=tmp_path / "runs",
                             roles=ROLES, fix_rounds=1)
    return ep, calls, verifies


def test_blocked_implement_skips_remaining_roles(tmp_path, monkeypatch):
    ep, calls, verifies = _run(tmp_path, monkeypatch, block=True)
    assert calls == ["implement"]
    assert verifies == []
    assert ep["outcome"] == "blocked"
    assert ep["blocked_reason"] == "sync or async?"
    assert ep["roles"][1:] == [
        {"role": "test", "skipped": True, "reason": "implement blocked"},
        {"role": "review", "skipped": True, "reason": "implement blocked"},
    ]


def test_unblocked_implement_runs_every_role(tmp_path, monkeypatch):
    ep, calls, verifies = _run(tmp_path, monkeypatch, block=False)
    assert "test" in calls and "review.1" in calls
    assert verifies
    assert not any(r.get("skipped") for r in ep["roles"])
    assert ep["outcome"] == "pass"
