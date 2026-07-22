"""End-to-end self-check: toy repo -> episode -> verify -> reward -> export -> datasets.
Run: python3 tests/test_heart.py
"""
from __future__ import annotations

import datetime
import json
import shutil
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
from heart.episode import best_episode, run_candidates, run_episode, run_swarm  # noqa: E402
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
        self._old_ingest = os.environ.get("HEART_INGEST")
        os.environ["HEART_INGEST"] = "off"  # toy episodes must not hit the real ledger
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
        if self._old_ingest is None:
            os.environ.pop("HEART_INGEST", None)
        else:
            os.environ["HEART_INGEST"] = self._old_ingest
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

    def test_blocked_outcome_withholds_reward(self):
        """An agent that declines to guess must not be scored. Without this,
        reward.compute renormalizes over diff_quality + efficiency — small diff,
        finished fast — which is precisely what blocking looks like, so a block
        scored ~0.97 and taught the policy to block instead of work."""
        task = TaskSpec(**{**self.task.__dict__,
                           "prompt": "echo 'PLEXUS_BLOCKED: sync or async?' > PLEXUS_BLOCKED",
                           "blocked_marker": "PLEXUS_BLOCKED:"})
        ep = run_episode(task, agent="shell", runs_dir=self.runs)
        self.assertEqual(ep["outcome"], "blocked")
        self.assertEqual(ep["blocked_reason"], "sync or async?")
        self.assertIsNone(ep["reward"]["total"])
        # blocking beats no marker at all: "wrote nothing" and "asked instead of
        # writing" are different events even though both leave a tiny diff
        plain = run_episode(TaskSpec(**{**self.task.__dict__, "prompt": "true"}),
                            agent="shell", runs_dir=self.runs)
        self.assertEqual(plain["outcome"], "no_change")
        # an unscored episode never wins best-of-N over one with a real score
        self.assertEqual(best_episode([ep, self.run_ep(FIX_CMD)])["outcome"], "pass")

    def test_no_verifiers_is_unverified_not_pass(self):
        """all([]) is True, so an empty verifier set used to score a vacuous
        `pass` — the repo shipped no tests and heart claimed correctness."""
        task = TaskSpec(**{**self.task.__dict__, "public_verifiers": [],
                           "prompt": "echo 'x = 1' > newfile.py"})
        ep = run_episode(task, agent="shell", runs_dir=self.runs)
        self.assertEqual(ep["outcome"], "unverified")
        self.assertIsNone(ep["reward"]["total"])

    def test_guardrails_outrank_a_block(self):
        """A block must not launder a secret past the scanner."""
        task = TaskSpec(**{**self.task.__dict__, "blocked_marker": "PLEXUS_BLOCKED:",
                           "prompt": "printf 'PLEXUS_BLOCKED: q?\\n"
                                     "AKIAIOSFODNN7EXAMPLE\\n' > PLEXUS_BLOCKED"})
        ep = run_episode(task, agent="shell", runs_dir=self.runs)
        self.assertEqual(ep["outcome"], "guardrail_violation")

    def test_claude_envelope_survives_stdout_noise(self):
        """The CLI prints advisories before its JSON envelope. Whole-text
        json.loads then failed, which silently lost usage *and* left the raw
        envelope in the log for every downstream text consumer to choke on."""
        from heart.runner import _claude_envelope
        env = {"type": "result", "result": "```json\n[1]\n```", "usage":
               {"input_tokens": 7, "output_tokens": 9}}
        body = json.dumps(env)
        noisy = "Warning: no stdin data received in 3s, proceeding without it.\n" + body
        for text, expect in ((body, 7), (noisy, 7), ("no json here at all", None)):
            got = _claude_envelope(text)
            if expect is None:
                self.assertIsNone(got)
            else:
                self.assertEqual(got["usage"]["input_tokens"], expect)
        # the last envelope wins when the CLI emits more than one
        two = body + "\n" + json.dumps({**env, "result": "second"})
        self.assertEqual(_claude_envelope(two)["result"], "second")

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
            for k in list(os.environ):  # ambient tier config must not leak in
                if k.startswith("HEART_TIER_"):
                    del os.environ[k]
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
            for k in list(os.environ):
                if k.startswith("HEART_TIER_"):
                    del os.environ[k]
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
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(cli_main(["pulse", "insights"]), 0)
        self.assertIn("routing: cheap=1/1 pass", buf.getvalue())

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
        # reviewer rejects once, then approves on re-review after the fix;
        # the recorded verdict must be the post-fix one or --apply blocks forever
        marker = self.root / "reviewed-once"
        roles = [
            {"name": "implement", "prompt": "{prompt}"},
            {"name": "review",
             "prompt": f"if [ -f {marker} ]; then echo APPROVE ok; "
                       f"else touch {marker}; echo REJECT needs-work; fi"},
        ]
        ep = run_episode(self.task, agent="shell", runs_dir=self.runs,
                         roles=roles, fix_rounds=1)
        self.assertIn("review-fix", [r["role"] for r in ep["roles"]])
        self.assertIn("review2", [r["role"] for r in ep["roles"]])
        self.assertEqual(ep["review_verdict"], "approve")
        self.assertEqual(ep["verify_rounds"][-1]["passed"], True)  # re-verified after fix
        self.assertEqual(ep["outcome"], "pass")

    def test_review_reject_sticks_when_rereview_rejects(self):
        roles = [
            {"name": "implement", "prompt": "{prompt}"},
            {"name": "review", "prompt": "echo REJECT needs-work"},
        ]
        ep = run_episode(self.task, agent="shell", runs_dir=self.runs,
                         roles=roles, fix_rounds=1)
        self.assertEqual(ep["review_verdict"], "reject")

    def test_consume_steer_helper(self):
        from heart.episode import _consume_steer

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.assertIsNone(_consume_steer(out))  # no file yet
            (out / "steer.txt").write_text("   \n")
            self.assertIsNone(_consume_steer(out))  # whitespace-only counts as empty
            (out / "steer.txt").write_text("do X instead")
            self.assertEqual(_consume_steer(out), "do X instead")
            self.assertEqual((out / "steer.txt").read_text(), "")  # truncated after consuming

    def test_steer_mid_run_appends_to_next_role_and_emits_event(self):
        from heart import pulse

        # role1's own shell script drops a steer note into its own out dir
        # (known via $ARTERIES_EPISODE_ID + the fixed runs_dir); episode.py's
        # steer check runs before each subsequent role turn, so role2 must
        # pick it up and steer.received must be logged.
        roles = [
            {"name": "implement",
             "prompt": f"{FIX_CMD}; echo -n 'focus on edge cases' "
                       f"> {self.runs}/$ARTERIES_EPISODE_ID/steer.txt"},
            {"name": "test", "prompt": "true"},
        ]
        ep = run_episode(self.task, agent="shell", runs_dir=self.runs, roles=roles)
        events = pulse.load_events(episode=ep["episode_id"])
        steer_events = [e for e in events if e["kind"] == "steer.received"]
        self.assertEqual(len(steer_events), 1)
        self.assertEqual(steer_events[0]["episode_id"], ep["episode_id"])
        self.assertEqual(steer_events[0]["payload"]["chars"], len("focus on edge cases"))

    def test_batch_resume_skips_done(self):
        tasks_dir = self.root / "tasks"
        tasks_dir.mkdir()
        spec = {**self.task.__dict__,
                "public_verifiers": [{"name": "unit", "command": "python3 -m unittest -q test_calc"}]}
        (tasks_dir / "toy.json").write_text(json.dumps(spec))
        argv = ["batch", str(tasks_dir), "--agent", "shell",
                "--runs-dir", str(self.runs), "--repeat", "2"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main(argv), 0)
        rows = (self.runs / "summary.csv").read_text().strip().splitlines()
        self.assertEqual(len(rows), 3)  # header + 2 episodes
        # second invocation resumes: nothing left to run, no duplicate rows
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(cli_main(argv), 0)
        self.assertIn("resume: 2 episode(s) already", buf.getvalue())
        rows = (self.runs / "summary.csv").read_text().strip().splitlines()
        self.assertEqual(len(rows), 3)

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

    def test_render(self):
        from heart import pulse

        e = {"ts": "2026-07-19T10:00:01.500000+00:00", "source": "heart",
             "kind": "role.finished", "role": "dev", "duration_ms": 42,
             "payload": {"note": "x" * 100}}
        # relative-timestamp path uses module-level datetime (no local import)
        line = pulse.render(e, t0="2026-07-19T10:00:00+00:00")
        self.assertIn("+    1.5s", line)
        self.assertIn("role=dev", line)
        self.assertIn("42ms", line)
        self.assertIn("x" * 57 + "...", line)  # payload truncated at 60
        # unparseable ts falls back to raw string, not a crash
        self.assertTrue(pulse.render({"ts": "garbage"}, t0="also-garbage").startswith("garbage"))
        # no t0: wall-clock slice of the ISO timestamp
        self.assertIn("10:00:01", pulse.render(e))

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


def _bwrap_usable() -> bool:
    # Ubuntu 24.04+ AppArmor can block unprivileged user namespaces, in which
    # case bwrap exists but cannot start; the live test only runs where it can
    if not shutil.which("bwrap"):
        return False
    return subprocess.run(
        ["bwrap", "--ro-bind", "/", "/", "true"], capture_output=True
    ).returncode == 0


class TestSandbox(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("HEART_SANDBOX")
        os.environ["HEART_SANDBOX"] = "bwrap"

    def tearDown(self):
        if self._old is None:
            os.environ.pop("HEART_SANDBOX", None)
        else:
            os.environ["HEART_SANDBOX"] = self._old

    def test_off_by_default(self):
        from heart.runner import sandbox_wrap

        os.environ.pop("HEART_SANDBOX")
        self.assertEqual(sandbox_wrap(["echo", "hi"], False, "/tmp/ws", {}),
                         (["echo", "hi"], False))
        os.environ["HEART_SANDBOX"] = "chroot"
        with self.assertRaises(ValueError):
            sandbox_wrap(["echo", "hi"], False, "/tmp/ws", {})

    @unittest.skipUnless(shutil.which("bwrap"), "bubblewrap not installed")
    def test_wrap_builds_bwrap_argv(self):
        from heart.runner import sandbox_wrap

        cmd, shell = sandbox_wrap(["echo", "hi"], False, "/tmp/ws", {})
        self.assertFalse(shell)
        self.assertEqual(cmd[0], "bwrap")
        self.assertIn("/tmp/ws", cmd)  # worktree stays writable
        self.assertEqual(cmd[-2:], ["echo", "hi"])
        self.assertNotIn("--unshare-net", cmd)  # plain bwrap keeps egress
        os.environ["HEART_SANDBOX"] = "bwrap-nonet"
        cmd, _ = sandbox_wrap(["echo", "hi"], False, "/tmp/ws", {})
        self.assertIn("--unshare-net", cmd)
        os.environ["HEART_SANDBOX"] = "bwrap"
        # shell-template agents run under sh -c inside the sandbox
        cmd2, shell2 = sandbox_wrap("echo hi", True, "/tmp/ws", {})
        self.assertEqual(cmd2[-3:], ["sh", "-c", "echo hi"])
        self.assertFalse(shell2)

    @unittest.skipUnless(shutil.which("bwrap"), "bubblewrap not installed")
    def test_wrap_mode_override_forces_nonet(self):
        from heart.runner import sandbox_wrap

        # HEART_SANDBOX=bwrap in the env, but an explicit mode= must win —
        # this is how run_verifiers forces the no-network variant regardless
        # of what agents are configured to use.
        self.assertEqual(os.environ["HEART_SANDBOX"], "bwrap")
        cmd, shell = sandbox_wrap(["true"], False, "/tmp/ws", {}, mode="bwrap-nonet")
        self.assertFalse(shell)
        self.assertIn("--unshare-net", cmd)

    @unittest.skipUnless(_bwrap_usable(), "bwrap cannot create user namespaces here")
    def test_sandbox_blocks_stray_writes(self):
        from heart.runner import run_agent

        with tempfile.TemporaryDirectory() as ws:
            marker = Path.home() / f"heart-sbx-{os.getpid()}"
            log = Path(ws) / "agent.log"
            run_agent("shell", f"touch {marker}; touch {ws}/inside", ws, {}, 30, log)
            self.assertFalse(marker.exists())  # $HOME is read-only
            self.assertTrue((Path(ws) / "inside").exists())

    @unittest.skipUnless(_bwrap_usable(), "bwrap cannot create user namespaces here")
    def test_bwrap_nonet_blocks_network(self):
        from heart.runner import sandbox_wrap

        with tempfile.TemporaryDirectory() as ws:
            cmd, shell = sandbox_wrap(
                ["python3", "-c",
                 "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"],
                False, ws, {}, mode="bwrap-nonet",
            )
            proc = subprocess.run(cmd, shell=shell, capture_output=True, timeout=15)
            self.assertNotEqual(proc.returncode, 0)

    @unittest.skipUnless(_bwrap_usable(), "bwrap cannot create user namespaces here")
    def test_bwrap_hides_ssh(self):
        from heart.runner import sandbox_wrap

        ssh_dir = Path.home() / ".ssh"
        if not ssh_dir.exists():
            self.skipTest("no ~/.ssh on this host")
        with tempfile.TemporaryDirectory() as ws:
            cmd, shell = sandbox_wrap(
                ["sh", "-c", f"ls -A {ssh_dir} | grep -q ."], False, ws, {}, mode="bwrap",
            )
            proc = subprocess.run(cmd, shell=shell, capture_output=True, timeout=15)
            self.assertNotEqual(proc.returncode, 0)  # tmpfs-hidden: empty or unreadable

    @unittest.skipUnless(_bwrap_usable(), "bwrap cannot create user namespaces here")
    def test_bwrap_cwd_stays_writable(self):
        from heart.runner import sandbox_wrap

        with tempfile.TemporaryDirectory() as ws:
            cmd, shell = sandbox_wrap(["touch", f"{ws}/probe"], False, ws, {}, mode="bwrap")
            proc = subprocess.run(cmd, shell=shell, capture_output=True, timeout=15)
            self.assertEqual(proc.returncode, 0)
            self.assertTrue((Path(ws) / "probe").exists())

    @unittest.skipUnless(_bwrap_usable(), "bwrap cannot create user namespaces here")
    def test_bwrap_git_commit_works(self):
        from heart.runner import sandbox_wrap

        with tempfile.TemporaryDirectory() as ws:
            git = ["git", "-C", ws, "-c", "user.name=t", "-c", "user.email=t@t"]
            subprocess.run([*git[:3], "init", "-q"], check=True)
            (Path(ws) / "file.txt").write_text("hello\n")
            subprocess.run([*git, "add", "-A"], check=True)
            cmd, shell = sandbox_wrap(
                [*git, "commit", "-qm", "wip"], False, ws, {}, mode="bwrap",
            )
            proc = subprocess.run(cmd, shell=shell, capture_output=True, timeout=15)
            self.assertEqual(proc.returncode, 0, proc.stderr)


class TestCost(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_usage_claude_envelope(self):
        from heart.runner import _extract_usage

        log = self.root / "implement.log"
        log.write_text(json.dumps({
            "result": "did the thing",
            "usage": {"input_tokens": 100, "output_tokens": 40},
            "total_cost_usd": 1.23,
        }))
        usage = _extract_usage(log, "claude")
        self.assertEqual(usage, {"tokens_in": 100, "tokens_out": 40})
        # downstream code greps the log for plain text (verdicts, failure tails)
        self.assertEqual(log.read_text(), "did the thing")

    def test_extract_usage_garbage_log(self):
        from heart.runner import _extract_usage

        log = self.root / "implement.log"
        log.write_text("not json at all\njust agent chatter\n")
        self.assertEqual(_extract_usage(log, "claude"), {"tokens_in": None, "tokens_out": None})

        missing = self.root / "missing.log"
        self.assertEqual(_extract_usage(missing, "claude"), {"tokens_in": None, "tokens_out": None})

    def test_extract_usage_api_heart_usage_line(self):
        from heart.runner import _extract_usage

        log = self.root / "solo.log"
        log.write_text("$ echo hi\nhi\nHEART_USAGE={\"tokens_in\": 12, \"tokens_out\": 34}\n")
        self.assertEqual(_extract_usage(log, "api"), {"tokens_in": 12, "tokens_out": 34})

        log2 = self.root / "no-usage.log"
        log2.write_text("$ echo hi\nhi\n")
        self.assertEqual(_extract_usage(log2, "api"), {"tokens_in": None, "tokens_out": None})

        self.assertEqual(_extract_usage(log, "shell"), {"tokens_in": None, "tokens_out": None})

    def test_price(self):
        from heart.runner import _price

        cfgdir = self.root / "cfg" / "heart"
        cfgdir.mkdir(parents=True)
        (cfgdir / "models.json").write_text(json.dumps({
            "pricing": {
                "api:qwen": {"in_per_mtok": 1.0, "out_per_mtok": 2.0},
                "claude": {"in_per_mtok": 3.0, "out_per_mtok": 15.0},
            }
        }))
        old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = str(self.root / "cfg")
        try:
            # exact match
            self.assertEqual(_price("api:qwen", 1_000_000, 1_000_000), 3.0)
            # base fallback: "claude:opus" -> "claude" entry
            self.assertEqual(_price("claude:opus", 1_000_000, 1_000_000), 18.0)
            # no pricing entry for this agent
            self.assertIsNone(_price("gemini", 1_000_000, 1_000_000))
            # missing tokens
            self.assertIsNone(_price("claude", None, 100))
        finally:
            if old is None:
                os.environ.pop("XDG_CONFIG_HOME", None)
            else:
                os.environ["XDG_CONFIG_HOME"] = old

    def test_episode_usage_is_none_for_shell_agent_and_insights_survives(self):
        from heart import pulse
        from heart.episode import run_episode

        old_spool = os.environ.get("HEART_SPOOL_DIR")
        old_ingest = os.environ.get("HEART_INGEST")
        os.environ["HEART_SPOOL_DIR"] = str(self.root / "spool")
        os.environ["HEART_INGEST"] = "off"
        try:
            commit = make_repo(self.root)
            task = TaskSpec(
                task_id="toy-add-fix",
                repo_path=str(self.root / "toyrepo"),
                base_commit=commit,
                prompt=FIX_CMD,
                denied_paths=["test_calc.py"],
                public_verifiers=[Verifier(name="unit", command="python3 -m unittest -q test_calc")],
                timeout_seconds=60,
            )
            ep = run_episode(task, agent="shell", runs_dir=self.root / "runs")
            for r in ep["roles"]:
                self.assertIn("tokens_in", r)
                self.assertIsNone(r["tokens_in"])
                self.assertIsNone(r["tokens_out"])
                self.assertIsNone(r["cost_usd"])
            self.assertEqual(ep["usage"], {"tokens_in": None, "tokens_out": None, "cost_usd": None})
            # must not crash even though no episode in this window carries cost
            pulse.insights(hours=24)
        finally:
            if old_spool is None:
                os.environ.pop("HEART_SPOOL_DIR", None)
            else:
                os.environ["HEART_SPOOL_DIR"] = old_spool
            if old_ingest is None:
                os.environ.pop("HEART_INGEST", None)
            else:
                os.environ["HEART_INGEST"] = old_ingest


class TestGuardrails(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.commit = make_repo(self.root)
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

    def test_scan_secrets_true_positives(self):
        from heart.guard import scan_secrets

        cases = {
            "aws_access_key": '+AWS_KEY = "AKIAEXAMPLE0EXAMPLE0"',
            "private_key": "+-----BEGIN RSA PRIVATE KEY-----",
            "github_token": '+token = "ghp_' + "x" * 36 + '"',
            "slack_token": '+SLACK_TOKEN = "xoxb-1234567890-abcdefghij"',
            "generic_secret_assignment": '+api_key: "abcdefghijklmnopqrstuvwx"',
        }
        for rule, line in cases.items():
            with self.subTest(rule=rule):
                hits = scan_secrets(line)
                self.assertTrue(hits, f"expected a hit for {rule}: {line}")
                self.assertTrue(any(h.startswith(rule) for h in hits), hits)
                # never leak the full secret value into the hit description:
                # each hit is "<rule>: <snippet<=60 chars>", never the raw line
                for h in hits:
                    _, _, snippet = h.partition(": ")
                    self.assertLessEqual(len(snippet), 60)

    def test_scan_secrets_ignores_benign_lookalikes(self):
        from heart.guard import scan_secrets

        diff = "\n".join([
            '+sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"',
            "+# the user's password is stored in the vault, not here",
            "+thisisnotanAKIAsecretjustalongword",
            "+++ b/some/file.py",  # file header, not an added line
        ])
        self.assertEqual(scan_secrets(diff), [])

    def test_secret_in_diff_yields_guardrail_violation(self):
        prompt = 'printf \'AWS_KEY = "AKIAEXAMPLE0EXAMPLE0"\\n\' >> calc.py'
        task = TaskSpec(
            task_id="toy-secret",
            repo_path=str(self.root / "toyrepo"),
            base_commit=self.commit,
            prompt=prompt,
            public_verifiers=[Verifier(name="unit", command="python3 -m unittest -q test_calc")],
            timeout_seconds=60,
        )
        ep = run_episode(task, agent="shell", runs_dir=self.runs)
        self.assertEqual(ep["outcome"], "guardrail_violation")
        self.assertEqual(ep["reward"]["total"], 0.0)
        self.assertTrue(ep["violations"])


class TestDetectStatic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tsc_requires_installed_binary(self):
        repo = self.root / "tsrepo"
        repo.mkdir()
        (repo / "tsconfig.json").write_text("{}")
        # tsconfig.json present but no node_modules/.bin/tsc: must not detect
        names = [v.name for v in detect_verifiers(repo)]
        self.assertNotIn("tsc", names)

    def test_tsc_detected_when_binary_present(self):
        repo = self.root / "tsrepo2"
        binp = repo / "node_modules" / ".bin"
        binp.mkdir(parents=True)
        (repo / "tsconfig.json").write_text("{}")
        (binp / "tsc").write_text("#!/bin/sh\n")
        (binp / "tsc").chmod(0o755)
        names = [v.name for v in detect_verifiers(repo)]
        self.assertIn("tsc", names)

    @unittest.skipUnless(shutil.which("ruff"), "ruff not installed")
    def test_ruff_detected_when_config_and_tool_present(self):
        repo = self.root / "pyrepo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n")
        names = [v.name for v in detect_verifiers(repo)]
        self.assertIn("ruff", names)

    def test_ruff_not_detected_without_config(self):
        repo = self.root / "pyrepo2"
        repo.mkdir()
        names = [v.name for v in detect_verifiers(repo)]
        self.assertNotIn("ruff", names)


class TestClean(unittest.TestCase):
    def test_clean_removes_old_episodes_keeps_fresh_and_summary(self):
        import time

        tmp = tempfile.TemporaryDirectory()
        try:
            runs_dir = Path(tmp.name) / "runs"
            old_dir = runs_dir / "old-episode"
            fresh_dir = runs_dir / "fresh-episode"
            old_dir.mkdir(parents=True)
            fresh_dir.mkdir(parents=True)
            (old_dir / "episode.json").write_text("{}")
            (fresh_dir / "episode.json").write_text("{}")
            summary = runs_dir / "summary.csv"
            summary.write_text("episode_id,task_id\n")

            old_ts = time.time() - 20 * 86400  # 20 days old
            os.utime(old_dir / "episode.json", (old_ts, old_ts))

            old_ws_root = os.environ.get("HEART_WS_ROOT")
            os.environ["HEART_WS_ROOT"] = str(Path(tmp.name) / "no-such-ws-root")
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    rc = cli_main(["clean", "--runs-dir", str(runs_dir), "--days", "7"])
            finally:
                if old_ws_root is None:
                    os.environ.pop("HEART_WS_ROOT", None)
                else:
                    os.environ["HEART_WS_ROOT"] = old_ws_root

            self.assertEqual(rc, 0)
            self.assertFalse(old_dir.exists())
            self.assertTrue(fresh_dir.exists())
            self.assertTrue(summary.exists())
            self.assertIn("1 run(s) removed", buf.getvalue())
        finally:
            tmp.cleanup()


class TestGoalLineage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._old_spool = os.environ.get("HEART_SPOOL_DIR")
        os.environ["HEART_SPOOL_DIR"] = str(self.root / "spool")

    def tearDown(self):
        if self._old_spool is None:
            os.environ.pop("HEART_SPOOL_DIR", None)
        else:
            os.environ["HEART_SPOOL_DIR"] = self._old_spool
        self.tmp.cleanup()

    def _write(self, **event):
        from heart.events import spool_dir

        d = spool_dir()
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "20260101.ndjson", "a") as f:
            f.write(json.dumps(event) + "\n")

    def test_emit_stamps_goal_lineage_from_env(self):
        from heart.events import emit, spool_dir

        old_goal = os.environ.get("PLEXUS_GOAL_ID")
        old_feat = os.environ.get("PLEXUS_FEATURE_ID")
        os.environ["PLEXUS_GOAL_ID"] = "g-env"
        os.environ["PLEXUS_FEATURE_ID"] = "f-env"
        try:
            emit("heart", "episode.started", episode_id="ep-env", task_id="t-env")
            # explicit payload kwargs are never overwritten by the env stamp
            emit("heart", "episode.finished", episode_id="ep-env2", goal_id="explicit-g")
        finally:
            if old_goal is None:
                os.environ.pop("PLEXUS_GOAL_ID", None)
            else:
                os.environ["PLEXUS_GOAL_ID"] = old_goal
            if old_feat is None:
                os.environ.pop("PLEXUS_FEATURE_ID", None)
            else:
                os.environ["PLEXUS_FEATURE_ID"] = old_feat

        lines = sorted(spool_dir().glob("*.ndjson"))[0].read_text().splitlines()
        events = {json.loads(line)["episode_id"]: json.loads(line) for line in lines}
        self.assertEqual(events["ep-env"]["payload"]["goal_id"], "g-env")
        self.assertEqual(events["ep-env"]["payload"]["feature_id"], "f-env")
        self.assertEqual(events["ep-env2"]["payload"]["goal_id"], "explicit-g")

    def test_goal_timeline_groups_by_feature_then_episode(self):
        from heart import pulse

        self._write(ts="2026-01-01T00:00:00+00:00", source="heart", kind="episode.finished",
                    episode_id="ep1", task_id="t1",
                    payload={"outcome": "pass", "reward": 0.9, "cost_usd": 0.12,
                             "goal_id": "g1", "feature_id": "f1"})
        self._write(ts="2026-01-01T00:01:00+00:00", source="heart", kind="episode.finished",
                    episode_id="ep2", task_id="t2",
                    payload={"outcome": "fail", "reward": 0.0, "cost_usd": 0.05,
                             "goal_id": "g1", "feature_id": "f2"})
        self._write(ts="2026-01-01T00:02:00+00:00", source="heart", kind="episode.finished",
                    episode_id="ep-other", task_id="t3",
                    payload={"outcome": "pass", "goal_id": "g-other", "feature_id": "fx"})

        lines = pulse.goal_timeline("g1")
        self.assertIn("goal g1: features=2 episodes=2", lines[0])
        self.assertIn("pass=1", lines[0])
        self.assertIn("fail=1", lines[0])
        self.assertTrue(any(
            "feature f1: episode ep1 outcome=pass reward=0.9 cost=$0.12" in l for l in lines))
        self.assertTrue(any("feature f2: episode ep2 outcome=fail" in l for l in lines))
        self.assertFalse(any("ep-other" in l for l in lines))

    def test_goal_timeline_empty(self):
        from heart import pulse

        self.assertEqual(pulse.goal_timeline("nope"), ["no events for goal nope"])

    def test_cli_pulse_goal(self):
        self._write(ts="2026-01-01T00:00:00+00:00", source="heart", kind="episode.finished",
                    episode_id="ep1", task_id="t1",
                    payload={"outcome": "pass", "reward": 1.0, "cost_usd": 0.1,
                             "goal_id": "g2", "feature_id": "f1"})
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(cli_main(["pulse", "goal", "g2"]), 0)
        self.assertIn("goal g2:", buf.getvalue())


class TestHealthRules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._old_spool = os.environ.get("HEART_SPOOL_DIR")
        os.environ["HEART_SPOOL_DIR"] = str(self.root / "spool")

    def tearDown(self):
        if self._old_spool is None:
            os.environ.pop("HEART_SPOOL_DIR", None)
        else:
            os.environ["HEART_SPOOL_DIR"] = self._old_spool
        self.tmp.cleanup()

    def _write(self, **event):
        from heart.events import spool_dir

        d = spool_dir()
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "20260101.ndjson", "a") as f:
            f.write(json.dumps(event) + "\n")

    def test_review2_reject_streak_warns(self):
        from heart import pulse

        now = datetime.datetime.now(datetime.timezone.utc)
        for i, verdict in enumerate(["approve", "reject", "reject", "reject"]):
            ts = (now - datetime.timedelta(minutes=10 - i)).isoformat()
            self._write(ts=ts, source="heart", kind="episode.finished", episode_id=f"ep{i}",
                        payload={"outcome": "fail", "review_verdict": verdict})
        lines, code = pulse.health(hours=1)
        self.assertEqual(code, 1)
        self.assertTrue(any("reject" in l.lower() and "streak" in l.lower() for l in lines))

    def test_review2_reject_streak_ok_when_mixed(self):
        from heart import pulse

        now = datetime.datetime.now(datetime.timezone.utc)
        for i, verdict in enumerate(["reject", "approve", "reject"]):
            ts = (now - datetime.timedelta(minutes=10 - i)).isoformat()
            self._write(ts=ts, source="heart", kind="episode.finished", episode_id=f"ep{i}",
                        payload={"outcome": "fail", "review_verdict": verdict})
        lines, code = pulse.health(hours=1)
        self.assertFalse(any("streak" in l.lower() for l in lines))

    def test_cost_alert(self):
        from heart import pulse

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._write(ts=now, source="heart", kind="episode.finished", episode_id="ep1",
                    payload={"outcome": "pass", "cost_usd": 5.0})
        self._write(ts=now, source="heart", kind="episode.finished", episode_id="ep2",
                    payload={"outcome": "pass", "cost_usd": 6.0})
        old = os.environ.get("HEART_COST_ALERT")
        os.environ["HEART_COST_ALERT"] = "10"
        try:
            lines, code = pulse.health(hours=1)
        finally:
            if old is None:
                os.environ.pop("HEART_COST_ALERT", None)
            else:
                os.environ["HEART_COST_ALERT"] = old
        self.assertEqual(code, 1)
        self.assertTrue(any("cost" in l.lower() and "10.00" in l for l in lines))

    def test_silent_stall_warns_when_goal_active(self):
        from heart import pulse

        old_ts = "2026-01-01T00:00:00+00:00"
        self._write(ts=old_ts, source="heart", kind="episode.started", episode_id="ep-old")
        old = os.environ.get("PLEXUS_GOAL_ACTIVE")
        os.environ["PLEXUS_GOAL_ACTIVE"] = "1"
        try:
            lines, code = pulse.health(hours=999999)
        finally:
            if old is None:
                os.environ.pop("PLEXUS_GOAL_ACTIVE", None)
            else:
                os.environ["PLEXUS_GOAL_ACTIVE"] = old
        self.assertEqual(code, 1)
        self.assertTrue(any("stall" in l.lower() for l in lines))

    def test_silent_stall_inert_without_env(self):
        from heart import pulse

        old_ts = "2026-01-01T00:00:00+00:00"
        self._write(ts=old_ts, source="heart", kind="episode.started", episode_id="ep-old")
        os.environ.pop("PLEXUS_GOAL_ACTIVE", None)
        lines, code = pulse.health(hours=999999)
        self.assertFalse(any("stall" in l.lower() for l in lines))


class TestPulseServe(unittest.TestCase):
    def test_page_and_insights_endpoints(self):
        import threading
        import urllib.request
        from http.server import ThreadingHTTPServer

        from heart.serve import Handler

        old = os.environ.get("HEART_SPOOL_DIR")
        tmp = tempfile.mkdtemp()
        os.environ["HEART_SPOOL_DIR"] = tmp
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            page = urllib.request.urlopen(base + "/").read().decode()
            self.assertIn("heart pulse", page)
            api = json.loads(urllib.request.urlopen(base + "/api/insights?hours=1").read())
            self.assertIn("insights", api)
            self.assertIn("health", api)
        finally:
            httpd.shutdown()
            if old is None:
                os.environ.pop("HEART_SPOOL_DIR", None)
            else:
                os.environ["HEART_SPOOL_DIR"] = old

    def test_steer_and_episode_drilldown_endpoints(self):
        import threading
        import urllib.error
        import urllib.request
        from http.server import ThreadingHTTPServer

        from heart import serve as serve_mod
        from heart.serve import Handler

        old_spool = os.environ.get("HEART_SPOOL_DIR")
        os.environ["HEART_SPOOL_DIR"] = tempfile.mkdtemp()
        runs_dir = Path(tempfile.mkdtemp())
        ep_dir = runs_dir / "ep-drill"
        ep_dir.mkdir()
        (ep_dir / "diff.patch").write_text("--- a/x\n+++ b/x\n")
        (ep_dir / "implement.log").write_text("did the thing\n")

        old_runs_dir = serve_mod.RUNS_DIR
        serve_mod.RUNS_DIR = runs_dir
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"

            req = urllib.request.Request(
                base + "/api/steer?episode=ep-drill", data=b"focus on edge cases", method="POST")
            resp = urllib.request.urlopen(req)
            self.assertEqual(resp.status, 204)
            self.assertEqual((ep_dir / "steer.txt").read_text(), "focus on edge cases")

            req2 = urllib.request.Request(
                base + "/api/steer?episode=nope", data=b"x", method="POST")
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req2)
            self.assertEqual(ctx.exception.code, 404)

            api = json.loads(urllib.request.urlopen(base + "/api/episode?id=ep-drill").read())
            self.assertEqual(api["diff"], "--- a/x\n+++ b/x\n")
            self.assertIn("implement", api["logs"])
            self.assertIn("did the thing", api["logs"]["implement"])
        finally:
            httpd.shutdown()
            serve_mod.RUNS_DIR = old_runs_dir
            if old_spool is None:
                os.environ.pop("HEART_SPOOL_DIR", None)
            else:
                os.environ["HEART_SPOOL_DIR"] = old_spool


class TestSwarm(unittest.TestCase):
    """Best-of-N with heterogeneous agents + one judge (STACK_READINESS §4.4)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.commit = make_repo(self.root)
        self.runs = self.root / "runs"
        self._old_spool = os.environ.get("HEART_SPOOL_DIR")
        os.environ["HEART_SPOOL_DIR"] = str(self.root / "spool")
        self._old_ingest = os.environ.get("HEART_INGEST")
        os.environ["HEART_INGEST"] = "off"
        self.task = TaskSpec(
            task_id="toy-swarm",
            repo_path=str(self.root / "toyrepo"),
            base_commit=self.commit,
            prompt=FIX_CMD,
            denied_paths=["test_calc.py"],
            public_verifiers=[Verifier(name="unit", command="python3 -m unittest -q test_calc")],
            timeout_seconds=60,
        )

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

    def test_wrong_candidate_loses_no_judge(self):
        # both candidates are "shell", branching on the profile string
        # (run_agent stamps HEART_MODEL_PROFILE from "agent:profile") so each
        # candidate applies a different fix without touching AGENT_COMMANDS
        task = TaskSpec(**{**self.task.__dict__, "prompt": (
            'if [ "$HEART_MODEL_PROFILE" = good ]; then ' + FIX_CMD + "; "
            "else sed -i 's/a - b/a * b/' calc.py; fi"
        )})
        ep = run_swarm(task, ["shell:good", "shell:bad"], runs_dir=self.runs)
        self.assertEqual(ep["outcome"], "pass")
        self.assertEqual(ep["agent"], "shell:good")
        self.assertEqual(ep["swarm"]["agents"], ["shell:good", "shell:bad"])
        self.assertEqual(len(ep["swarm"]["rewards"]), 2)
        self.assertFalse(ep["swarm"]["judged"])
        self.assertEqual(ep["swarm"]["winner_agent"], "shell:good")

    def test_judge_breaks_tie_between_two_passes(self):
        # both candidates apply the identical correct fix -> rewards within
        # epsilon -> the judge decides; judge_cmd is the test-door escape
        # hatch since a real judge agent can't be made to print WINNER here
        ep = run_swarm(
            self.task, ["shell:cand1", "shell:cand2"],
            judge_cmd="echo 'WINNER: 2'", runs_dir=self.runs,
        )
        self.assertTrue(ep["swarm"]["judged"])
        self.assertEqual(ep["agent"], "shell:cand2")
        self.assertEqual(ep["swarm"]["winner_agent"], "shell:cand2")
        self.assertEqual(ep["swarm"]["agents"], ["shell:cand1", "shell:cand2"])

    def test_mute_judge_falls_back_to_reward_ranking(self):
        ep = run_swarm(
            self.task, ["shell:cand1", "shell:cand2"],
            judge_cmd="true", runs_dir=self.runs,
        )
        self.assertFalse(ep["swarm"]["judged"])
        self.assertEqual(ep["outcome"], "pass")
        self.assertIn(ep["agent"], ("shell:cand1", "shell:cand2"))

    def test_cli_run_swarm(self):
        task_path = self.root / "task.json"
        task_path.write_text(json.dumps({
            "task_id": "toy-swarm-cli",
            "repo_path": str(self.root / "toyrepo"),
            "base_commit": self.commit,
            "prompt": FIX_CMD,
            "public_verifiers": [{"name": "unit", "command": "python3 -m unittest -q test_calc"}],
            "timeout_seconds": 60,
        }))
        code = cli_main([
            "run", str(task_path), "--swarm", "shell,shell", "--runs-dir", str(self.runs),
        ])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
