import hashlib
import json
import subprocess
from pathlib import Path

from heart.episode import run_episode
from heart.events import journal_dir
from heart.taskspec import TaskSpec, Verifier


def _make_repo(root: Path) -> tuple[Path, str]:
    repo = root / "toyrepo"
    repo.mkdir(parents=True)
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "test_calc.py").write_text(
        "import unittest\nfrom calc import add\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n"
    )
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git[:3], "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    head = subprocess.run([*git[:3], "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return repo, head


def _episode(root: Path) -> dict:
    repo, head = _make_repo(root)
    task = TaskSpec(
        task_id="artifacts", repo_path=str(repo), base_commit=head,
        prompt="sed -i 's/a - b/a + b/' calc.py",
        public_verifiers=[Verifier(name="unit", command="python3 -m unittest -q test_calc")],
    )
    # fix_rounds > 0 is what makes the episode verify in-workspace and emit verify.round
    return run_episode(task, agent="shell", runs_dir=root / "runs", fix_rounds=1)


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEART_INGEST", "off")
    monkeypatch.setenv("VASCULAR_HOME", str(tmp_path / "vascular"))


def _payload(event: dict) -> dict:
    return {**event, **(event.get("payload") or {})}


def test_artifacts_are_content_addressed(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("EVENT_JOURNAL_DIR", str(tmp_path / "journal"))
    ep = _episode(tmp_path)

    events = [_payload(json.loads(line))
              for f in sorted(journal_dir().glob("*.ndjson"))
              for line in f.read_text().splitlines() if line.strip()]
    events = [e for e in events if e.get("episode_id") == ep["episode_id"]]
    by_kind = {}
    for e in events:
        by_kind.setdefault(e["kind"], e)

    def check(kind, field):
        digest = by_kind[kind].get(field)
        assert digest, (kind, field, by_kind[kind])
        blob = (journal_dir() / "artifacts" / digest).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == digest
        return digest

    check("role.started", "prompt_artifact")
    check("role.finished", "output_artifact")
    check("verify.round", "output_artifact")
    diff_digest = check("diff.captured", "diff_artifact")
    patch = tmp_path / "runs" / ep["episode_id"] / "diff.patch"
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == diff_digest


def test_unwritable_journal_does_not_change_outcome(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("EVENT_JOURNAL_DIR", str(tmp_path / "control-journal"))
    control = _episode(tmp_path / "control")

    blocker = tmp_path / "journal-is-a-file"
    blocker.write_text("not a directory\n")
    monkeypatch.setenv("EVENT_JOURNAL_DIR", str(blocker))
    broken = _episode(tmp_path / "broken")

    assert control["outcome"] == broken["outcome"] == "pass"
