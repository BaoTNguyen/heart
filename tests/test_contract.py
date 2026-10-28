"""The black-box contract (DIRECTION §13): contract/cli.json run against the
heart executable ($HEART_BIN). The cases are data so a v2 in another language
can be judged by the same file; this module is only the runner."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
CLI = json.loads((HERE / "contract" / "cli.json").read_text())
BIN = (os.environ.get("HEART_BIN") or shutil.which("heart", path=str(Path(sys.executable).parent))
       or shutil.which("heart"))


@pytest.mark.parametrize("case", CLI["cases"], ids=[c["name"] for c in CLI["cases"]])
def test_cli_case(case, tmp_path):
    assert BIN, "no heart executable: set HEART_BIN"
    repo, runs = tmp_path / "repo", tmp_path / "runs"
    repo.mkdir()
    for name, text in CLI["repo"].items():
        (repo / name).write_text(text)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    base = subprocess.run([*git, "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    # the conftest fixtures already isolate VASCULAR_HOME and the journal;
    # no sandbox: the shell agent runs the prompt on the host, in a worktree
    env = {k: v for k, v in os.environ.items() if k != "HEART_SANDBOX"}
    env |= {"HEART_INGEST": "off", "ARTERIES_RETRIEVAL": "off", "HEART_PULSE": "file"}

    for step in case["steps"]:
        task = {**CLI["task"], "repo_path": str(repo), "base_commit": base,
                "prompt": step.get("prompt", "true")}
        (tmp_path / "task.json").write_text(json.dumps(task))
        args = [a.replace("{task}", "task.json").replace("{runs}", str(runs))
                for a in step["args"]]
        got = subprocess.run([BIN, *args], cwd=tmp_path, env=env,
                             capture_output=True, text=True)
        where = f"heart {' '.join(args)}\nstdout: {got.stdout}\nstderr: {got.stderr[-2000:]}"
        assert got.returncode == step["exit"], where
        for want in step.get("stdout_has", []):
            assert want in got.stdout, where
        if "stdout_json" in step:
            out = json.loads(got.stdout)
            assert {k: out.get(k) for k in step["stdout_json"]} == step["stdout_json"], where
        if args[0] == "run":
            run_dir = runs / json.loads(got.stdout)["episode_id"]
            assert [a for a in CLI["artifacts"] if not (run_dir / a).exists()] == [], where
            ep = json.loads((run_dir / "episode.json").read_text())
            assert [f for f in CLI["episode_fields"] if f not in ep] == [], where
