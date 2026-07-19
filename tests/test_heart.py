"""End-to-end self-check: toy repo -> episode -> verify -> reward -> export -> datasets.
Run: python3 tests/test_heart.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import contextlib  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402

from heart import reward as reward_mod  # noqa: E402
from heart.agents_api import resolve_config  # noqa: E402
from heart.cli import main as cli_main  # noqa: E402
from heart.detect import detect_verifiers  # noqa: E402
from heart.episode import best_episode, run_candidates, run_episode  # noqa: E402
from heart.export import export_episodes  # noqa: E402
from heart.taskspec import TaskSpec, Verifier  # noqa: E402
from heart.training import datasets  # noqa: E402
from heart.verify import check_task  # noqa: E402

BUGGY = "def add(a, b):\n    return a - b\n"
TEST = (
    "import unittest\nfrom calc import add\n\n"
    "class T(unittest.TestCase):\n"
    "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n"
    "if __name__ == '__main__':\n    unittest.main()\n"
)
FIX_CMD = "sed -i 's/a - b/a + b/' calc.py"


def make_repo(root: Path) -> str:
    repo = root / "toyrepo"
    repo.mkdir()
    (repo / "calc.py").write_text(BUGGY)
    (repo / "test_calc.py").write_text(TEST)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git[:3], "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "buggy add"], check=True)
    return subprocess.run(
        [*git[:3], "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


class TestHeart(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.commit = make_repo(self.root)
        self.runs = self.root / "runs"
        self._old_spool = os.environ.get("HEART_SPOOL_DIR")
        os.environ["HEART_SPOOL_DIR"] = str(self.root / "spool")
        self.task = TaskSpec(
            task_id="toy-add-fix",
            repo_path=str(self.root / "toyrepo"),
            base_commit=self.commit,
            prompt=FIX_CMD,  # shell agent executes the prompt as bash
            denied_paths=["test_calc.py"],
            public_verifiers=[Verifier(name="unit", command="python3 -m unittest -q test_calc")],
            timeout_seconds=60,
        )

    def tearDown(self):
        if self._old_spool is None:
            os.environ.pop("HEART_SPOOL_DIR", None)
        else:
            os.environ["HEART_SPOOL_DIR"] = self._old_spool
        self.tmp.cleanup()

    def run_ep(self, prompt: str):
        task = TaskSpec(**{**self.task.__dict__, "prompt": prompt})
        return run_episode(task, agent="shell", runs_dir=self.runs)

    def test_pass_episode(self):
        ep = self.run_ep(FIX_CMD)
        self.assertEqual(ep["outcome"], "pass")
        self.assertGreater(ep["reward"]["total"], 0.5)
        self.assertTrue((self.runs / ep["episode_id"] / "diff.patch").read_text().strip())

    def test_fail_no_change_violation(self):
        self.assertEqual(self.run_ep("sed -i 's/a - b/a * b/' calc.py")["outcome"], "fail")
        self.assertEqual(self.run_ep("true")["outcome"], "no_change")
        ep = self.run_ep("sed -i 's/add(2, 3), 5/add(2, 3), -1/' test_calc.py")
        self.assertEqual(ep["outcome"], "path_violation")
        self.assertEqual(ep["reward"]["total"], 0.0)

    def test_role_pipeline(self):
        roles = [
            {"name": "implement", "memory": "normal", "prompt": "{prompt}"},
            {"name": "review", "memory": "readonly",
             "prompt": "echo reviewing; echo APPROVE looks-correct"},
        ]
        ep = run_episode(self.task, agent="shell", runs_dir=self.runs, roles=roles)
        self.assertEqual(ep["outcome"], "pass")
        self.assertEqual(ep["review_verdict"], "approve")
        self.assertEqual([r["role"] for r in ep["roles"]], ["implement", "review"])
        self.assertEqual(ep["roles"][1]["memory"], "readonly")

    def test_detect_verifiers(self):
        names = [v.name for v in detect_verifiers(self.root / "toyrepo")]
        self.assertIn("pytest", names)

    def test_fix_loop(self):
        # same script every invocation: first call applies a wrong fix, the
        # verify-fix loop triggers a second call that applies the right one
        script = (
            "if [ -f .tried ]; then sed -i 's/a \\* b/a + b/' calc.py; rm .tried; "
            "else touch .tried; sed -i 's/a - b/a \\* b/' calc.py; fi"
        )
        ep = run_episode(
            self.task, agent="shell", runs_dir=self.runs,
            agent_cmd=script, fix_rounds=2,
        )
        self.assertEqual(ep["outcome"], "pass")
        self.assertEqual([r["passed"] for r in ep["verify_rounds"]], [False, True])
        self.assertIn("fix1", [r["role"] for r in ep["roles"]])

    def test_candidates(self):
        eps = run_candidates(self.task, 2, agent="shell", runs_dir=self.runs)
        self.assertEqual(len(eps), 2)
        self.assertEqual(best_episode(eps)["outcome"], "pass")
        # parallel candidates run memory-isolated so they can't cross-feed
        for e in eps:
            self.assertEqual(e["env_snapshot"]["ARTERIES_EPHEMERAL"], "discard")
        solo = self.run_ep(FIX_CMD)
        self.assertNotIn("ARTERIES_EPHEMERAL", solo["env_snapshot"])

    def test_router(self):
        from heart import router

        self.assertEqual(router.classify(self.task)[0], "cheap")
        hard = TaskSpec(**{**self.task.__dict__,
                           "prompt": "Refactor the threading and concurrency model " + "x " * 60})
        self.assertEqual(router.classify(hard)[0], "strong")
        by_difficulty = TaskSpec(**{**self.task.__dict__, "difficulty": "hard"})
        self.assertEqual(router.classify(by_difficulty)[0], "strong")

        old = dict(os.environ)
        try:
            os.environ["XDG_CONFIG_HOME"] = str(self.root / "cfg")  # no models.json
            os.environ["HEART_TIER_CHEAP"] = "shell"
            self.assertEqual(router.resolve("cheap"), "shell")
            self.assertEqual(router.resolve("strong", default="claude"), "claude")
            with self.assertRaises(ValueError):
                router.resolve("strong")
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_auto_routing_runs_episode(self):
        from heart import pulse

        old = dict(os.environ)
        try:
            os.environ["XDG_CONFIG_HOME"] = str(self.root / "cfg")
            os.environ["HEART_TIER_CHEAP"] = "shell"
            ep = run_episode(self.task, agent="auto", runs_dir=self.runs)
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(ep["outcome"], "pass")
        self.assertEqual(ep["agent"], "shell")  # routed, not the literal "auto"
        routed = [e for e in pulse.load_events(episode=ep["episode_id"])
                  if e["kind"] == "route.decided"]
        self.assertEqual(routed[0]["payload"]["tier"], "cheap")

    def test_workspace_copies_integration_files(self):
        from heart.env import Workspace

        repo = self.root / "toyrepo"
        (repo / ".arteries" / "hooks").mkdir(parents=True)
        (repo / ".arteries" / "hooks" / "observe.sh").write_text("#!/bin/sh\necho hi\n")
        (repo / ".arteries" / "runs").mkdir()
        (repo / ".arteries" / "runs" / "old.jsonl").write_text("{}\n")
        (repo / ".claude").mkdir()
        (repo / ".claude" / "settings.local.json").write_text("{}")
        ws = Workspace(str(repo), self.commit)
        try:
            self.assertTrue((ws.path / ".arteries" / "hooks" / "observe.sh").exists())
            self.assertFalse((ws.path / ".arteries" / "runs").exists())  # fallback data stays home
            self.assertTrue((ws.path / ".claude" / "settings.local.json").exists())
            self.assertEqual(ws.diff(), "")  # copied files never pollute the diff
        finally:
            ws.destroy()

    def test_mine_pins_fix_tests_as_overlay(self):
        from heart.mine import mine
        from heart.taskspec import load_task

        # the trap: base code passes the base tests; the fix commit strengthens
        # tests and code together. Without pinning the fix-commit tests, the
        # mined task scores a no-op diff as a pass.
        repo = self.root / "minerepo"
        repo.mkdir()
        weak = TEST.replace("add(2, 3), 5", "add(0, 0), 0")  # passes with a - b
        (repo / "calc.py").write_text(BUGGY)
        (repo / "test_calc.py").write_text(weak)
        git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git[:3], "init", "-q"], check=True)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", "weak"], check=True)
        (repo / "calc.py").write_text(BUGGY.replace("a - b", "a + b"))
        (repo / "test_calc.py").write_text(TEST)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", "fix add and strengthen test"], check=True)

        written = mine(str(repo), self.root / "mined",
                       test_cmd="python3 -m unittest -q test_calc")
        self.assertEqual(len(written), 1)
        task = load_task(written[0])
        self.assertIn("add(2, 3), 5", task.overlay_files["test_calc.py"])

        verdict = check_task(task, n=1)
        self.assertTrue(verdict["base_fails"])  # pinned tests fail at base
        self.assertTrue(verdict["ok"])

        ep = run_episode(TaskSpec(**{**task.__dict__, "prompt": FIX_CMD}),
                         agent="shell", runs_dir=self.runs)
        self.assertEqual(ep["outcome"], "pass")
        diff = (self.runs / ep["episode_id"] / "diff.patch").read_text()
        self.assertNotIn("test_calc", diff)  # overlay never leaks into the diff

    def test_review_reject_triggers_fix(self):
        roles = [
            {"name": "implement", "prompt": "{prompt}"},
            {"name": "review", "prompt": "echo REJECT needs-work"},
        ]
        ep = run_episode(self.task, agent="shell", runs_dir=self.runs,
                         roles=roles, fix_rounds=1)
        self.assertEqual(ep["review_verdict"], "reject")
        self.assertIn("review-fix", [r["role"] for r in ep["roles"]])
        self.assertEqual(ep["verify_rounds"][-1]["passed"], True)  # re-verified after fix
        self.assertEqual(ep["outcome"], "pass")

    def test_hidden_reward_weights(self):
        passed = {"passed": True, "exit_code": 0, "duration_s": 1, "output_tail": ""}
        failed = {**passed, "passed": False}
        with_hidden = reward_mod.compute({"unit": passed}, "", 1, 100, hidden_results={"h": failed})
        without = reward_mod.compute({"unit": passed}, "", 1, 100)
        self.assertIn("hidden_tests", with_hidden["components"])
        self.assertLess(with_hidden["total"], without["total"])

    def test_event_spine(self):
        from heart import pulse

        ep = self.run_ep(FIX_CMD)
        events = pulse.load_events(episode=ep["episode_id"])
        kinds = [e["kind"] for e in events]
        self.assertEqual(kinds[0], "episode.started")
        self.assertEqual(kinds[-1], "episode.finished")
        self.assertIn("role.started", kinds)
        self.assertIn("diff.captured", kinds)
        self.assertEqual(events[-1]["payload"]["outcome"], "pass")
        # cross-episode filtering: a second episode must not leak in
        self.run_ep("true")
        self.assertEqual(len(pulse.load_events(episode=ep["episode_id"])), len(events))

        timeline = pulse.episode_timeline(ep["episode_id"])
        self.assertTrue(timeline[0].lstrip().startswith("+"))
        self.assertIn("episode.finished", timeline[-1])

        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = cli_main(["pulse", "tail", "--once", "--episode", ep["episode_id"]])
        self.assertEqual(code, 0)
        self.assertIn("episode.started", buf.getvalue())

    def test_episode_crash_emits_failed(self):
        from heart import pulse

        bad = TaskSpec(**{**self.task.__dict__, "repo_path": str(self.root / "nonexistent")})
        with self.assertRaises((RuntimeError, OSError)):
            run_episode(bad, agent="shell", runs_dir=self.runs)
        crashed = [e for e in pulse.load_events() if e["kind"] == "episode.failed"]
        self.assertEqual(len(crashed), 1)
        self.assertIn("Error", crashed[0]["payload"]["error"])

    def test_insights_and_health(self):
        from heart import pulse

        self.run_ep(FIX_CMD)
        self.run_ep("sed -i 's/a - b/a * b/' calc.py")
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(cli_main(["pulse", "insights"]), 0)
        text = buf.getvalue()
        self.assertIn("traffic: episodes=2", text)
        self.assertIn("outcomes: ", text)
        self.assertIn("latency role=solo", text)
        self.assertIn("p95=", text)

        self.assertEqual(cli_main(["pulse", "health"]), 0)

        # a zombie episode and a degraded store write must each flip health to 1
        spool = Path(os.environ["HEART_SPOOL_DIR"])
        old_ts = "2026-01-01T00:00:00+00:00"
        with open(sorted(spool.glob("*.ndjson"))[0], "a") as f:
            f.write(json.dumps({"ts": old_ts, "source": "heart",
                                "kind": "episode.started", "episode_id": "ep-zombie"}) + "\n")
            f.write(json.dumps({"ts": old_ts, "source": "arteries", "kind": "turn.observed",
                                "payload": {"store": "jsonl"}}) + "\n")
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(cli_main(["pulse", "health", "--hours", "999999"]), 1)
        self.assertIn("never finished", buf.getvalue())
        self.assertIn("fell back from Postgres", buf.getvalue())

    def test_api_config_resolution(self):
        cfgdir = self.root / "cfg" / "heart"
        cfgdir.mkdir(parents=True)
        (cfgdir / "models.json").write_text(json.dumps({
            "profiles": {"gpt": {"endpoint": "https://api.openai.com/v1",
                                 "model": "gpt-5", "api_key_env": "TEST_KEY_VAR"}}
        }))
        old = dict(os.environ)
        try:
            os.environ.update({
                "XDG_CONFIG_HOME": str(self.root / "cfg"),
                "HEART_MODEL_PROFILE": "gpt", "TEST_KEY_VAR": "sk-test",
            })
            cfg = resolve_config()
            self.assertEqual(
                (cfg["endpoint"], cfg["model"], cfg["api_key"]),
                ("https://api.openai.com/v1", "gpt-5", "sk-test"),
            )
            os.environ.pop("HEART_MODEL_PROFILE")
            os.environ["HEART_API_ENDPOINT"] = "http://localhost:11434/v1"
            self.assertEqual(resolve_config()["endpoint"], "http://localhost:11434/v1")
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_check_task_and_datasets(self):
        verdict = check_task(self.task, n=2)
        self.assertTrue(verdict["deterministic"])
        self.assertFalse(verdict["base_results"]["unit"])  # bugfix task fails at base

        self.run_ep(FIX_CMD)
        self.run_ep("sed -i 's/a - b/a * b/' calc.py")
        episodes = self.root / "episodes.jsonl"
        self.assertEqual(export_episodes(self.runs, episodes), 2)
        sft = datasets.build_sft(episodes, self.runs, self.root / "sft.jsonl")
        dpo = datasets.build_dpo(episodes, self.runs, self.root / "dpo.jsonl")
        self.assertEqual((sft, dpo), (1, 1))
        row = json.loads((self.root / "dpo.jsonl").read_text())
        self.assertIn("a + b", row["chosen"])
        self.assertIn("a * b", row["rejected"])

        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = cli_main(["stats", "--runs-dir", str(self.runs)])
        self.assertEqual(code, 0)
        self.assertIn("normal/ret-on", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
