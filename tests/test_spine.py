"""Spine conformance + env-propagation contract tests (STACK_READINESS.md §1.2
items 1 and 3). Separate from test_heart.py by request.

Run: python3 -m unittest discover -s tests -q
"""
from __future__ import annotations

import datetime
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heart.episode import run_episode  # noqa: E402
from heart.taskspec import TaskSpec, Verifier  # noqa: E402

GOLDEN_DIR = Path(__file__).resolve().parent / "golden-events"

# Kinds that SPINE.md's "Event catalog" ties to a live episode (they live under
# the `episode.*` / `role.*` rows, plus `route.decided`/`verify.round`/
# `diff.captured`, all sourced from heart) and that src/heart/episode.py emits
# with an explicit episode_id= on every call site (never conditionally). Other
# sources' kinds only carry episode_id "when known" per SPINE.md's field
# table, so they are not required here.
KINDS_REQUIRING_EPISODE_ID = {
    "episode.started", "episode.finished", "episode.failed",
    "role.started", "role.finished",
    "route.decided", "verify.round", "diff.captured", "guardrail.hit",
    "steer.received", "swarm.judged",
}


def _iter_golden_events():
    for path in sorted(GOLDEN_DIR.glob("*.ndjson")):
        source = path.stem
        with open(path) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                yield source, lineno, line


def _assert_json_scalar(value) -> bool:
    """No NaN/Infinity anywhere in a decoded JSON value (json.loads accepts
    Python's non-standard NaN/Infinity tokens unless parse_constant is set)."""
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_assert_json_scalar(v) for v in value.values())
    if isinstance(value, list):
        return all(_assert_json_scalar(v) for v in value)
    return True


class TestSpineConformance(unittest.TestCase):
    def test_golden_dir_exists_and_nonempty(self):
        self.assertTrue(GOLDEN_DIR.is_dir())
        files = list(GOLDEN_DIR.glob("*.ndjson"))
        self.assertTrue(files)

    def test_every_golden_event_conforms(self):
        checked = 0
        for source, lineno, line in _iter_golden_events():
            with self.subTest(source=source, line=lineno):
                event = json.loads(line)  # must be valid JSON

                for field in ("ts", "source", "kind"):
                    self.assertIn(field, event)
                    self.assertTrue(event[field], f"{field} must be non-empty")

                ts = datetime.datetime.fromisoformat(event["ts"])
                self.assertIsNotNone(ts.tzinfo, "ts must be UTC-aware")
                self.assertEqual(ts.utcoffset(), datetime.timedelta(0), "ts must be UTC")

                self.assertEqual(event["source"], source,
                                  "source field must match the golden filename")

                if event["kind"] in KINDS_REQUIRING_EPISODE_ID:
                    self.assertIn("episode_id", event)
                    self.assertTrue(event["episode_id"], "episode_id must be non-empty")

                payload = event.get("payload", {})
                self.assertIsInstance(payload, dict)
                self.assertTrue(_assert_json_scalar(payload), "payload must contain no NaN/Infinity")
                checked += 1
        self.assertGreater(checked, 0)


class TestEnvPropagation(unittest.TestCase):
    """Pins the env contract a role subprocess actually receives, grounded in
    src/heart/episode.py (_run_episode/_agent_turn) and src/heart/runner.py
    (run_agent: `env = {**os.environ, **extra_env}`)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.commit = self._make_repo(self.root)
        self.runs = self.root / "runs"
        self._old_spool = os.environ.get("HEART_SPOOL_DIR")
        os.environ["HEART_SPOOL_DIR"] = str(self.root / "spool")
        self._old_ingest = os.environ.get("HEART_INGEST")
        os.environ["HEART_INGEST"] = "off"

    def tearDown(self):
        if self._old_spool is None:
            os.environ.pop("HEART_SPOOL_DIR", None)
        else:
            os.environ["HEART_SPOOL_DIR"] = self._old_spool
        if self._old_ingest is None:
            os.environ.pop("HEART_INGEST", None)
        else:
            os.environ["HEART_INGEST"] = self._old_ingest
        self.tmp.cleanup()

    @staticmethod
    def _make_repo(root: Path) -> str:
        repo = root / "toyrepo"
        repo.mkdir()
        (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git[:3], "init", "-q"], check=True)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", "init"], check=True)
        return subprocess.run(
            [*git[:3], "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()

    def test_role_subprocess_env(self):
        # ambient tier config that a *role subprocess* env should never carry
        # if heart scrubbed it (it doesn't — see the assertion below). Dump
        # env to a path outside the disposable worktree (under self.root) so
        # it survives Workspace.destroy(), matching how test_heart.py asserts
        # side effects that outlive the worktree (e.g. test_workspace_copies_
        # integration_files).
        old_tier = os.environ.get("HEART_TIER_CHEAP")
        os.environ["HEART_TIER_CHEAP"] = "must-be-scrubbed"
        try:
            dump = self.root / "envdump.txt"
            task = TaskSpec(
                task_id="envdump2",
                repo_path=str(self.root / "toyrepo"),
                base_commit=self.commit,
                prompt=f"env > {dump}",
            )
            ep = run_episode(task, agent="shell", runs_dir=self.runs)
        finally:
            if old_tier is None:
                os.environ.pop("HEART_TIER_CHEAP", None)
            else:
                os.environ["HEART_TIER_CHEAP"] = old_tier

        self.assertEqual(ep["outcome"], "no_change")
        lines = dump.read_text().splitlines()

        def has(prefix: str) -> bool:
            return any(line.startswith(prefix) for line in lines)

        # env vars episode.py/runner.py actually set for a role subprocess
        self.assertTrue(has("HEART_ROLE=solo"))
        self.assertTrue(has("ARTERIES_PROJECT=toyrepo"))
        self.assertTrue(has(f"ARTERIES_REPO={self.root / 'toyrepo'}"))
        self.assertTrue(has(f"ARTERIES_EPISODE_ID={ep['episode_id']}"))
        self.assertTrue(has(f"ARTERIES_TASK_ID={ep['task_id']}"))

        # ambient HEART_TIER_* is this process's routing config and must never
        # reach a role subprocess (runner.run_agent scrubs it — the leak broke
        # a real episode's verifiers once)
        self.assertFalse(has("HEART_TIER_CHEAP="))


if __name__ == "__main__":
    unittest.main()
