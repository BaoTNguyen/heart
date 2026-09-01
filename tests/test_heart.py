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
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import contextlib  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402

from heart import reward as reward_mod  # noqa: E402
from heart.agents_api import resolve_config  # noqa: E402
from heart.cli import main as cli_main  # noqa: E402
from heart.detect import detect_verifiers  # noqa: E402
from heart.episode import (  # noqa: E402
    DEFAULT_ROLES, best_episode, run_candidates, run_episode)
from heart.export import export_episodes  # noqa: E402
from heart.taskspec import TaskSpec, Verifier  # noqa: E402
from heart.verify import compare_baseline  # noqa: E402
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
        self._old_journal = os.environ.get("EVENT_JOURNAL_DIR")
        os.environ["EVENT_JOURNAL_DIR"] = str(self.root / "journal")
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
        if self._old_journal is None:
            os.environ.pop("EVENT_JOURNAL_DIR", None)
        else:
            os.environ["EVENT_JOURNAL_DIR"] = self._old_journal
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

    # a real refusal, no container needed: a directory the agent itself made
    # unwritable. What matters is that the refusal reaches the log.
    _REFUSED = "mkdir -p locked && chmod 500 locked && touch locked/out.txt"

    def test_a_refusal_is_reported_even_when_the_episode_still_scores(self):
        """The refusal scan used to run only when the diff came back empty, so
        the only scopes anyone heard about were the ones so tight the agent
        produced nothing. A scope tight enough to stop an agent finishing but
        not starting was recorded nowhere -- and scored as a model failure."""
        ep = self.run_ep(f"{FIX_CMD}; {self._REFUSED}")
        self.assertEqual(ep["outcome"], "pass")
        self.assertGreater(ep["reward"]["total"], 0.5)  # scored on its merits
        self.assertTrue(ep["scope_suspect"])            # and flagged anyway
        self.assertEqual(ep["scope_refused_paths"], ["locked/out.txt"])

    def test_the_tag_is_for_episodes_that_carry_a_number(self):
        # scope_denied already says it in the outcome and withholds the reward;
        # tagging it too would just be the same fact twice
        ep = self.run_ep(self._REFUSED)
        self.assertEqual(ep["outcome"], "scope_denied")
        self.assertIsNone(ep["reward"]["total"])
        self.assertFalse(ep["scope_suspect"])
        self.assertEqual(ep["scope_refused_paths"], ["locked/out.txt"])

    def test_a_forbidden_probe_is_never_reported_as_our_misconfiguration(self):
        # denied_paths is test_calc.py here; a refusal naming it is the agent's
        # doing, and must not reach the ledger as ground the task needed
        ep = self.run_ep("chmod 500 . && touch test_calc.py/x")
        self.assertEqual(ep["scope_refused_paths"], [])
        self.assertFalse(ep["scope_suspect"])

    def test_the_episode_records_which_build_of_the_agent_ran(self):
        """`agent: "claude:opus"` does not say which claude. A CLI that
        auto-updates mid-batch splits the run across two agents with nothing in
        the data saying so."""
        from heart.runner import _agent_version

        ep = self.run_ep(FIX_CMD)
        self.assertIn("agent_version", ep["roles"][0])
        # shell is bash, so None here; a real CLI reports a string
        self.assertIsNone(ep["roles"][0]["agent_version"])
        if shutil.which("claude"):
            self.assertTrue(_agent_version("claude"))

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
        # `agent` pins the reviewer; without it a review role rotates to another
        # model family, which is the point but not what this test measures.
        roles = [
            {"name": "implement", "memory": "normal", "prompt": "{prompt}"},
            {"name": "review", "memory": "readonly", "agent": "shell",
             "prompt": "echo reviewing; echo APPROVE looks-correct"},
        ]
        ep = run_episode(self.task, agent="shell", runs_dir=self.runs, roles=roles)
        self.assertEqual(ep["outcome"], "pass")
        self.assertEqual(ep["review_verdict"], "approve")
        # the review stage is numbered: one analysis, possibly repeated
        self.assertEqual([r["role"] for r in ep["roles"]], ["implement", "review.1"])
        self.assertEqual(ep["roles"][1]["memory"], "readonly")

    def test_a_prompt_without_findings_falls_back_to_the_verdict_word(self):
        # a --roles file written before findings existed keeps working: no JSON
        # means the legacy APPROVE/REJECT read, not a silent approval.
        roles = [{"name": "implement", "prompt": "{prompt}"},
                 {"name": "review", "agent": "shell", "prompt": "echo REJECT nope"}]
        ep = run_episode(self.task, agent="shell", runs_dir=self.runs, roles=roles)
        self.assertEqual(ep["review_verdict"], "reject")
        self.assertEqual(ep["review_findings"], [])

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
        blocker = ('{"findings":[{"severity":"blocker","file":"calc.py",'
                   '"line":2,"claim":"needs work"}]}')
        roles = [
            {"name": "implement", "prompt": "{prompt}"},
            {"name": "review", "agent": "shell",
             "prompt": f"if [ -f {marker} ]; then echo '{{\"findings\":[]}}'; "
                       f"else touch {marker}; echo '{blocker}'; fi"},
        ]
        ep = run_episode(self.task, agent="shell", runs_dir=self.runs,
                         roles=roles, fix_rounds=1)
        # a blocker sends the diff to resolve, then to a confirm stage that reads
        # the findings and the claims rather than re-reviewing cold
        self.assertIn("review-fix.1", [r["role"] for r in ep["roles"]])
        self.assertIn("review-confirm.2", [r["role"] for r in ep["roles"]])
        self.assertEqual(ep["verify_rounds"][-1]["passed"], True)  # re-verified after fix
        self.assertEqual(ep["outcome"], "pass")

    def test_review_reject_sticks_when_rereview_rejects(self):
        blocker = ('{"findings":[{"severity":"blocker","file":"calc.py",'
                   '"line":2,"claim":"still broken"}]}')
        roles = [
            {"name": "implement", "prompt": "{prompt}"},
            {"name": "review", "agent": "shell", "prompt": f"echo '{blocker}'"},
        ]
        ep = run_episode(self.task, agent="shell", runs_dir=self.runs,
                         roles=roles, fix_rounds=1)
        self.assertEqual(ep["review_verdict"], "reject")

    def test_a_concern_records_without_blocking(self):
        # the old shape threw away everything an APPROVE noticed; severity is
        # now the gate, so a non-blocking finding survives the approval.
        note = ('{"findings":[{"severity":"concern","file":"calc.py","line":2,'
                '"claim":"no docstring"}]}')
        roles = [{"name": "implement", "prompt": "{prompt}"},
                 {"name": "review", "agent": "shell", "prompt": f"echo '{note}'"}]
        ep = run_episode(self.task, agent="shell", runs_dir=self.runs, roles=roles)
        self.assertEqual(ep["review_verdict"], "approve")
        self.assertEqual([f["severity"] for f in ep["review_findings"]], ["concern"])
        self.assertNotIn("review-fix.1", [r["role"] for r in ep["roles"]])

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

    def test_path_violations_match_at_boundaries(self):
        from heart.episode import path_violations
        diff = ("diff --git a/src_gen/x.py b/src_gen/x.py\n"
                "--- a/src_gen/x.py\n+++ b/src_gen/x.py\n@@ -0,0 +1 @@\n+x\n")
        # allowed=["src"] must NOT admit src_gen/ (a bare startswith would)
        self.assertEqual(path_violations(diff, ["src"], []), ["src_gen/x.py"])
        # a real child of an allowed dir is fine
        ok = ("diff --git a/src/x.py b/src/x.py\n"
              "--- a/src/x.py\n+++ b/src/x.py\n@@ -0,0 +1 @@\n+x\n")
        self.assertEqual(path_violations(ok, ["src"], []), [])
        # denied is boundary-matched too: "secret" must not trip "secretly/"
        d2 = ("diff --git a/secretly/x.py b/secretly/x.py\n"
              "--- a/secretly/x.py\n+++ b/secretly/x.py\n@@ -0,0 +1 @@\n+x\n")
        self.assertEqual(path_violations(d2, [], ["secret"]), [])

    def test_efficiency_only_credited_when_all_checks_pass(self):
        passed = {"passed": True, "exit_code": 0, "duration_s": 1, "output_tail": ""}
        failed = {**passed, "passed": False}
        # a fast *failing* episode earns no speed credit — no reward for bailing early
        self.assertEqual(
            reward_mod.compute({"unit": failed}, "", 1, 100)["components"]["efficiency"], 0.0)
        # a fast *passing* episode is rewarded for finishing quickly
        self.assertGreater(
            reward_mod.compute({"unit": passed}, "", 1, 100)["components"]["efficiency"], 0.0)
        # public passes but hidden fails -> not correct -> still no speed credit
        self.assertEqual(
            reward_mod.compute({"unit": passed}, "", 1, 100,
                               hidden_results={"h": failed})["components"]["efficiency"], 0.0)

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
        journal = Path(os.environ["EVENT_JOURNAL_DIR"])
        old_ts = "2026-01-01T00:00:00+00:00"
        with open(sorted(journal.glob("*.ndjson"))[0], "a") as f:
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


def _docker_usable() -> bool:
    """A daemon AND the sandbox image, present locally.

    The image is checked, never pulled: a test suite that silently downloads
    gigabytes is a test suite people stop running. Build it with
    `docker build -t heart-agent:latest .` and these un-skip."""
    if not shutil.which("docker"):
        return False
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from heart.sandbox import DEFAULT_IMAGE

    image = os.environ.get("HEART_SANDBOX_IMAGE", DEFAULT_IMAGE)
    return subprocess.run(["docker", "image", "inspect", image],
                          capture_output=True).returncode == 0


def _run_in(profile, script: str, timeout: int = 120):
    """Run a shell script through sandbox_wrap exactly as an agent would."""
    from heart.runner import sandbox_wrap

    cmd, shell = sandbox_wrap(script, True, "/unused", {}, mode="docker-sbx", profile=profile)
    return subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=timeout)


class TestSandboxWrap(unittest.TestCase):
    """The dispatch, without a container."""

    def setUp(self):
        self._old = os.environ.get("HEART_SANDBOX")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("HEART_SANDBOX", None)
        else:
            os.environ["HEART_SANDBOX"] = self._old

    def test_off_by_default(self):
        from heart.runner import sandbox_wrap

        os.environ.pop("HEART_SANDBOX", None)
        self.assertEqual(sandbox_wrap(["echo", "hi"], False, "/tmp/ws", {}),
                         (["echo", "hi"], False))

    def test_an_unknown_mode_is_refused_not_ignored(self):
        from heart.runner import sandbox_wrap

        # including the modes that used to work: a stale HEART_SANDBOX=bwrap in
        # someone's shell must stop the run, never silently run unsandboxed
        for mode in ("chroot", "bwrap", "bwrap-nonet", "docker"):
            os.environ["HEART_SANDBOX"] = mode
            with self.assertRaises(ValueError):
                sandbox_wrap(["echo", "hi"], False, "/tmp/ws", {})

    def test_an_agent_that_needs_a_model_is_not_given_network_none(self):
        """Denying a model-using agent the network yields a connection failure,
        an empty diff, and an episode that reads as 'the agent did nothing'."""
        from heart.runner import run_agent
        from heart.sandbox import profile_for
        from heart.taskspec import TaskSpec

        task = TaskSpec(task_id="t", repo_path="/repo", base_commit="c", prompt="p")
        prof = profile_for(task, "/ws", "/ctx", "/jr")
        self.assertEqual(prof.network, "none")
        with tempfile.TemporaryDirectory() as ws:
            with self.assertRaises(RuntimeError) as cm:
                run_agent("claude", "p", ws, {}, 5, Path(ws) / "a.log", profile=prof)
            self.assertIn("network", str(cm.exception))
            # shell needs no model, so it stays allowed
            run_agent("shell", "true", ws, {}, 10, Path(ws) / "b.log", profile=prof)

    def test_docker_without_a_profile_fails_loudly(self):
        from heart.runner import sandbox_wrap

        os.environ["HEART_SANDBOX"] = "docker-sbx"
        with self.assertRaises(RuntimeError):
            sandbox_wrap(["echo", "hi"], False, "/tmp/ws", {})

    def test_the_caller_env_reaches_the_container_but_cannot_override_the_profile(self):
        from heart.runner import sandbox_wrap
        from heart.sandbox import JOURNAL, profile_for
        from heart.taskspec import TaskSpec

        task = TaskSpec(task_id="t", repo_path="/repo", base_commit="c", prompt="p")
        prof = profile_for(task, "/ws", "/ctx", "/jr")
        cmd, _ = sandbox_wrap("run", True, "/ws", {"HEART_PROMPT": "do it",
                                                   "EVENT_JOURNAL_DIR": "/hijack"},
                              mode="docker-sbx", profile=prof)
        script = cmd[2]
        self.assertIn("HEART_PROMPT=do it", script)
        self.assertIn(f"EVENT_JOURNAL_DIR={JOURNAL}", script)
        self.assertNotIn("EVENT_JOURNAL_DIR=/hijack", script)

    def test_run_agent_passes_the_sandbox_profile_not_the_model_profile(self):
        """`claude:opus` splits into a model profile named `profile`, which used
        to shadow the sandbox profile parameter of the same name -- so docker
        mode raised "no sandbox profile was supplied" for every agent carrying
        one. The bug is invisible unless something asserts on what arrives."""
        from heart import runner

        os.environ["HEART_SANDBOX"] = "docker-sbx"
        seen = {}
        real = runner.sandbox_wrap

        def spy(cmd, shell, cwd, extra_env, *, mode=None, profile=None):
            seen["profile"] = profile
            seen["env"] = extra_env
            return ["true"], False

        from heart.sandbox import profile_for
        from heart.taskspec import TaskSpec

        sentinel = profile_for(
            TaskSpec(task_id="t", repo_path="/repo", base_commit="c", prompt="p",
                     network="api"), "/ws", "/ctx", "/jr")
        runner.sandbox_wrap = spy
        try:
            with tempfile.TemporaryDirectory() as ws:
                runner.run_agent("claude:opus", "p", ws, {}, 5, Path(ws) / "a.log",
                                 agent_cmd="true", profile=sentinel)
        finally:
            runner.sandbox_wrap = real
        self.assertIs(seen["profile"], sentinel)
        self.assertEqual(seen["env"]["HEART_MODEL_PROFILE"], "opus")


@unittest.skipUnless(_docker_usable(), "no docker daemon, or the sandbox image is not built (docker build -t heart-agent:latest .)")
class TestSandboxLive(unittest.TestCase):
    """The container, for real. These are the negative-space tests: what the
    sandbox must refuse. A green unit test on the mount table proves the flags
    were rendered, not that the kernel enforced them."""

    def setUp(self):
        from heart.taskspec import TaskSpec

        self.tmp = tempfile.TemporaryDirectory(dir=Path.home() / ".cache")
        root = Path(self.tmp.name)
        self.ws, self.ctx, self.jr = root / "ws", root / "ctx", root / "jr"
        for d in (self.ws / "src", self.ws / "secrets", self.ctx, self.jr):
            d.mkdir(parents=True)
        (self.ws / "src" / "a.py").write_text("x = 1\n")
        (self.ws / "secrets" / "k.txt").write_text("hunter2\n")
        self.task = TaskSpec(task_id="t", repo_path=str(root / "norepo"),
                             base_commit="c", prompt="p", allowed_paths=["src"],
                             denied_paths=["secrets"], timeout_seconds=60)

    def tearDown(self):
        self.tmp.cleanup()

    def _agent(self, **kw):
        from heart.sandbox import profile_for

        return profile_for(self.task, self.ws, self.ctx, self.jr, **kw)

    def test_allowed_paths_are_writable_and_everything_else_is_not(self):
        r = _run_in(self._agent(), "touch /work/src/new.py; touch /work/top; "
                                   "touch /work/secrets/x; true")
        self.assertTrue((self.ws / "src" / "new.py").exists(), r.stderr)
        self.assertFalse((self.ws / "top").exists())
        self.assertFalse((self.ws / "secrets" / "x").exists())
        self.assertIn("read-only file system", r.stderr.lower())

    def test_a_denied_path_stays_readable(self):
        # denied means "may not change", not "may not see" -- a task that
        # cannot read a config it must not edit is a sandbox drawn too tight
        r = _run_in(self._agent(), "cat /work/secrets/k.txt")
        self.assertIn("hunter2", r.stdout)

    def test_the_default_network_is_none(self):
        """The plugin has no --network, and its sandboxes come up on bridge.
        heart detaches and re-attaches between creating the sandbox and running
        anything in it, so `none` means none again."""
        self.assertEqual(self._agent().network, "none")
        r = _run_in(self._agent(), "getent hosts example.com >/dev/null && echo REACHED")
        self.assertNotIn("REACHED", r.stdout)

    def test_resource_limits_are_applied_after_creation_too(self):
        # --memory/--cpus/--pids-limit are not flags the plugin takes either;
        # `docker update` accepts them on the running sandbox
        r = _run_in(self._agent(), "cat /sys/fs/cgroup/memory.max 2>/dev/null | head -1")
        self.assertEqual(r.stdout.strip(), str(4 * 1024 ** 3), r.stderr)

    def test_a_task_that_asks_for_egress_gets_the_network_the_operator_named(self):
        """"api" no longer means the open internet. It means whichever network
        HEART_API_NETWORK points at -- by default the --internal one where the
        egress proxy's allowlist decides what leaves. Pointed at bridge, the
        old unrestricted behaviour is still there for anyone who wants it."""
        import dataclasses
        import importlib

        from heart import sandbox

        self.task = dataclasses.replace(self.task, network="api")
        with unittest.mock.patch.dict(os.environ, {"HEART_API_NETWORK": "bridge"}):
            importlib.reload(sandbox)
            try:
                prof = sandbox.profile_for(self.task, self.ws, self.ctx, self.jr)
                self.assertEqual(prof.network, "bridge")
                r = _run_in(prof, "getent hosts example.com >/dev/null && echo REACHED")
                self.assertIn("REACHED", r.stdout, r.stderr)
            finally:
                importlib.reload(sandbox)

    def test_the_host_home_is_not_visible(self):
        r = _run_in(self._agent(), f"ls {Path.home()}/.ssh 2>&1 | head -3; "
                                   f"ls {Path.home()} 2>&1 | head -3")
        self.assertNotIn("id_", r.stdout)
        self.assertIn("No such file", r.stdout)

    def test_the_agent_cli_has_a_writable_home(self):
        # under --read-only the image's own /home/agent is not writable, and a
        # claude CLI that cannot write its config never gets as far as a turn
        r = _run_in(self._agent(),
                    'echo probe > "$HOME/x" && cat "$HOME/x" && claude --version')
        self.assertIn("probe", r.stdout)
        self.assertIn("Claude Code", r.stdout, r.stderr)

    def test_tool_caches_land_somewhere_writable_under_a_tight_scope(self):
        """A tight scope makes the whole tree read-only, so every incidental
        write a toolchain makes -- pytest's cache, ruff's, npm's -- is refused
        for reasons that have nothing to do with the task. Those refusals match
        _DENIAL_SIGNS, so without this they are recorded as paths the task
        needed and accumulate in plexus's ledger, which never retracts."""
        from heart.sandbox import CACHE_ENV

        dirs = [v for k, v in CACHE_ENV.items() if v.startswith("/tmp/")]
        script = "; ".join(f'mkdir -p {d} && touch {d}/probe' for d in dirs)
        r = _run_in(self._agent(), script + "; echo ALL_WRITABLE")
        self.assertIn("ALL_WRITABLE", r.stdout, r.stderr)
        self.assertEqual(r.stderr.strip(), "")

    def test_the_journal_is_the_only_way_out(self):
        r = _run_in(self._agent(), 'echo "{}" > /journal/e.jsonl; touch /context/x; true')
        self.assertTrue((self.jr / "e.jsonl").exists(), r.stderr)
        self.assertFalse((self.ctx / "x").exists())

    def test_a_verifier_cannot_write_the_tree_it_judges_or_reach_the_network(self):
        import dataclasses

        from heart.sandbox import verifier_profile_for

        task = dataclasses.replace(self.task, network="api")  # even then
        prof = verifier_profile_for(task, self.ws, self.jr)
        r = _run_in(prof, "touch /work/src/cheat.py; ls /context; "
                          "getent hosts example.com && echo REACHED; true")
        self.assertFalse((self.ws / "src" / "cheat.py").exists())
        self.assertNotIn("REACHED", r.stdout)
        self.assertIn("No such file", r.stderr + r.stdout)  # /context not mounted

    def test_a_verifier_can_still_run_a_test_suite_on_a_read_only_tree(self):
        """A test runner writes: caches, .pyc, temp dirs. On a tree it may not
        write, every one of those has to land somewhere else or the verifier
        fails for reasons that have nothing to do with the code under test.

        stdlib unittest, not pytest: the default image is Docker's agent
        template, which carries the agent CLIs and not a repo's test
        dependencies. A real task points HEART_SANDBOX_IMAGE at an image that
        can run its own verifiers."""
        from heart.sandbox import verifier_profile_for

        (self.ws / "test_ok.py").write_text(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_ok(self):\n        self.assertTrue(True)\n")
        prof = verifier_profile_for(self.task, self.ws, self.jr)
        r = _run_in(prof, "python3 -m unittest -q test_ok")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_git_reads_the_worktree_but_cannot_write_the_object_store(self):
        import dataclasses

        repo = Path(self.tmp.name) / "repo"
        repo.mkdir()
        git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git[:3], "init", "-q", "-b", "main"], check=True)
        (repo / "src").mkdir()
        (repo / "src" / "a.py").write_text("x = 1\n")
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", "base"], check=True)
        wt = Path(self.tmp.name) / "wt"
        subprocess.run([*git, "worktree", "add", "-q", "--detach", str(wt)], check=True)
        self.task = dataclasses.replace(self.task, repo_path=str(repo))
        from heart.sandbox import profile_for

        prof = profile_for(self.task, wt, self.ctx, self.jr)
        r = _run_in(prof, 'git -C /work log --oneline | head -1; '
                          'git -C /work status --porcelain >/dev/null && echo STATUS_OK; '
                          'echo y >> /work/src/a.py; '
                          'git -C /work -c user.name=t -c user.email=t@t commit -aqm x '
                          '2>&1 | head -2')
        self.assertIn("base", r.stdout, r.stderr)
        self.assertIn("STATUS_OK", r.stdout)
        self.assertIn("read-only file system", r.stdout.lower())
        self.assertIn("failed to insert into database", r.stdout.lower())
        # and the edit itself survived for heart to commit on the host
        self.assertIn("y", (wt / "src" / "a.py").read_text())
        subprocess.run([*git, "worktree", "remove", "--force", str(wt)], check=True)

    def test_the_container_kills_itself_when_the_task_timeout_expires(self):
        import dataclasses
        import time as _t

        self.task = dataclasses.replace(self.task, timeout_seconds=3)
        t0 = _t.monotonic()
        r = _run_in(self._agent(), "sleep 60", timeout=60)
        self.assertLess(_t.monotonic() - t0, 45)
        self.assertNotEqual(r.returncode, 0)


class TestRewardBridge(unittest.TestCase):
    def test_heart_calls_the_reward_ingest_not_the_document_ingest(self):
        """`art ingest` is arteries' document ingester: it globs *.md and embeds
        what it finds. heart called it for as long as the bridge existed, which
        did nothing -- runs dirs held no markdown, so no reward was ever
        ingested and nothing said so. The day heart wrote memory packets to
        runs/<id>/context/*.md, the glob matched and arteries tried to embed an
        agent's memory into the corpus as documentation."""
        import inspect

        from heart import cli

        src = inspect.getsource(cli._ingest_rewards)
        assert '"rewards"' in src, "the reward ledger lives behind `art rewards`"
        assert '"ingest"' not in src.split('"""')[-1], \
            "`art ingest` embeds documents; it is not the credit-assignment bridge"


class TestReviewerRotation(unittest.TestCase):
    """A reviewer must not be the family that wrote the code: the same lineage
    brings the same blind spots to finding the bug it brought to writing it."""

    def setUp(self):
        # pin the pool: review_pool() reads ~/.config/heart/models.json, and a
        # unit test that asserts on the operator's live config fails whenever
        # they change a model -- which is exactly what it is meant to let them do
        self._patch = unittest.mock.patch.dict(
            os.environ, {"HEART_REVIEW_MODELS": "claude:opus,codex:sol"})
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_the_reviewer_is_a_different_family_than_the_coder(self):
        from heart.router import review_agent

        self.assertEqual(review_agent("claude:opus"), "codex:sol")
        self.assertEqual(review_agent("codex:sol"), "claude:opus")
        # a different profile in the same family is still the same lineage
        self.assertEqual(review_agent("claude:sonnet"), "codex:sol")
        self.assertEqual(review_agent("codex:terra"), "claude:opus")

    def test_a_coder_outside_the_pool_still_gets_a_reviewer(self):
        # a local model or a subscription seat is not in the pool; failing the
        # episode over that would be a config choice breaking a run
        from heart.router import review_agent

        self.assertEqual(review_agent("api:local"), "claude:opus")

    def test_the_pool_is_data_not_code(self):
        """Adding or changing a model must not need a code change, and the
        rotation has to work for any number of entries."""
        from heart.router import review_agent

        pool = ["a:one", "b:two", "c:three"]
        with unittest.mock.patch.dict(os.environ, {"HEART_REVIEW_MODELS": ",".join(pool)}):
            self.assertEqual(review_agent("a:one"), "b:two")
            self.assertEqual(review_agent("c:three"), "a:one")
            self.assertEqual(review_agent("z:none"), "a:one")

    def test_an_explicit_agent_on_the_role_still_wins(self):
        # rotation fills in what nobody chose; it does not overrule a caller
        from heart.episode import DEFAULT_ROLES

        review = next(r for r in DEFAULT_ROLES if r["name"] == "review")
        self.assertTrue(review.get("review"))
        self.assertNotIn("agent", review, "the default must rotate, not pin")


class TestReclaim(unittest.TestCase):
    """One reclaimer, and liveness asked rather than guessed at.

    There used to be two: `heart clean` walked the disk with an age cutoff,
    prune_repo_worktrees walked a repo's worktree list with none. Both are
    filters on the same walk now, and neither is what makes it safe."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.ws_root = root / "ws"
        self.ws_root.mkdir()
        self._old = os.environ.get("HEART_WS_ROOT")
        os.environ["HEART_WS_ROOT"] = str(self.ws_root)
        self.repo = root / "repo"
        self.repo.mkdir()
        git = ["git", "-C", str(self.repo), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git[:3], "init", "-q", "-b", "main"], check=True)
        (self.repo / "f.txt").write_text("x\n")
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", "base"], check=True)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("HEART_WS_ROOT", None)
        else:
            os.environ["HEART_WS_ROOT"] = self._old
        self.tmp.cleanup()

    def test_a_live_workspace_is_never_reclaimed(self):
        from heart.env import Workspace, reclaim

        ws = Workspace(str(self.repo), "HEAD")
        try:
            self.assertEqual(reclaim(), 0)
            self.assertTrue(ws.path.is_dir())
        finally:
            ws.destroy()

    def test_a_worktree_whose_owner_died_is_reclaimed_at_once(self):
        """No age cutoff. The kernel drops the lock however the owner dies, so
        a lock that can be taken means the tree is nobody's."""
        from heart.env import Workspace, reclaim

        ws = Workspace(str(self.repo), "HEAD")
        path = ws.path
        ws._lock.close()          # what a SIGKILL does, minus the process exit
        self.assertEqual(reclaim(), 1)
        self.assertFalse(path.exists())
        listed = subprocess.run(["git", "-C", str(self.repo), "worktree", "list"],
                                capture_output=True, text=True).stdout
        self.assertNotIn(str(path), listed, "prune should have deregistered it")

    def test_a_worktree_with_no_lock_at_all_is_a_leak(self):
        # everything created before this existed, including one real 86-strong
        # backlog, has no lock file
        from heart.env import reclaim

        stray = self.ws_root / "deadbeefcafe"
        stray.mkdir()
        (stray / "junk.txt").write_text("x")
        self.assertEqual(reclaim(), 1)
        self.assertFalse(stray.exists())

    def test_an_orphan_whose_repo_is_gone_is_still_reclaimed(self):
        """85 of 86 leaks in one real backlog belonged to task clones and temp
        repos already deleted. Walking from a repo means someone has to name it,
        and nobody can name a repo that no longer exists -- so the walk starts
        from the disk and asks each worktree where it came from.

        `git -C <gone> worktree list` returns nothing, which is why the old
        repo-first version could not see these at all."""
        from heart.env import Workspace, reclaim

        ws = Workspace(str(self.repo), "HEAD")
        path = ws.path
        ws._lock.close()
        shutil.rmtree(self.repo)          # the repo disappears
        probe = subprocess.run(["git", "-C", str(self.repo), "worktree", "list"],
                               capture_output=True, text=True)
        self.assertNotEqual(probe.returncode, 0, "the repo really is gone")
        self.assertEqual(reclaim(), 1)    # found without being told a repo
        self.assertFalse(path.exists())

    def test_a_tree_an_agent_left_unwritable_is_still_removed(self):
        """rmtree cannot delete a child when the parent is mode 0500, and with
        ignore_errors that failure is silent. A reclaimer counting attempts
        rather than results reported 48 removed over 48 still on disk."""
        from heart.env import reclaim

        stray = self.ws_root / "aaaabbbbcccc"
        (stray / "locked").mkdir(parents=True)
        (stray / "locked" / "f.txt").write_text("x")
        (stray / "locked").chmod(0o500)
        try:
            self.assertEqual(reclaim(), 1)
            self.assertFalse(stray.exists())
        finally:
            if stray.exists():
                (stray / "locked").chmod(0o700)

    def test_a_lock_whose_worktree_is_gone_is_litter(self):
        from heart.env import reclaim

        (self.ws_root / "ffffeeeedddd.lock").write_text("")
        reclaim()
        self.assertFalse((self.ws_root / "ffffeeeedddd.lock").exists())

    def test_the_filters_narrow_without_being_the_safety(self):
        from heart.env import Workspace, reclaim

        ws = Workspace(str(self.repo), "HEAD")
        ws._lock.close()
        other = Path(self.tmp.name) / "elsewhere"
        self.assertEqual(reclaim(repo=other), 0)      # different repo: skipped
        self.assertEqual(reclaim(older_than=0), 0)    # nothing is that old
        self.assertEqual(reclaim(repo=self.repo), 1)

    def test_making_a_workspace_sweeps_leaks_left_by_earlier_runs(self):
        """Only plexus ever called the old reclaimer, so driving heart directly
        leaked indefinitely and silently."""
        import heart.env as env

        stray = self.ws_root / "0123456789ab"
        stray.mkdir()
        env._swept = False
        ws = env.Workspace(str(self.repo), "HEAD")
        try:
            self.assertFalse(stray.exists())
        finally:
            ws.destroy()


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
        # no cache fields in the envelope means no cache traffic -> 0, not None:
        # zero is a measurement, None is "we could not read the log"
        self.assertEqual(usage, {"tokens_in": 100, "tokens_out": 40,
                                 "cache_read": 0, "cache_write_5m": 0,
                                 "cache_write_1h": 0})
        # downstream code greps the log for plain text (verdicts, failure tails)
        self.assertEqual(log.read_text(), "did the thing")

    def test_accumulate_usage_normalises_openai_cache_convention(self):
        """OpenAI's `prompt_tokens` INCLUDES cached tokens; Anthropic's
        `input_tokens` excludes them. The stack has one convention — tokens_in
        is uncached — so the OpenAI subset is carved out here, at the edge.
        Passing it through billed every cached token at the full input rate
        instead of a tenth, which at real agent cache-hit rates is a ~6x
        overcharge, not a rounding error."""
        from heart.agents_api import _accumulate_usage

        t = {}
        _accumulate_usage(t, {"usage": {
            "prompt_tokens": 4293, "completion_tokens": 520,
            "prompt_tokens_details": {"cached_tokens": 3200}}})
        self.assertEqual(t, {"tokens_in": 1093, "tokens_out": 520,
                             "cache_read": 3200})

        # an Anthropic-shaped response already separates them: subtracting
        # would under-report input as badly as adding over-reports it
        t = {}
        _accumulate_usage(t, {"usage": {
            "input_tokens": 15, "output_tokens": 2692,
            "cache_read_input_tokens": 48_000}})
        self.assertEqual(t, {"tokens_in": 15, "tokens_out": 2692,
                             "cache_read": 48_000})

        # no cache traffic at all -> no cache key, not a zero
        t = {}
        _accumulate_usage(t, {"usage": {"prompt_tokens": 100, "completion_tokens": 40}})
        self.assertEqual(t, {"tokens_in": 100, "tokens_out": 40})

        # accumulation across a multi-turn tool loop stays normalised
        t = {}
        for _ in range(3):
            _accumulate_usage(t, {"usage": {
                "prompt_tokens": 1000, "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 900}}})
        self.assertEqual(t, {"tokens_in": 300, "tokens_out": 30, "cache_read": 2700})

    def test_extract_usage_counts_cache_tokens(self):
        """The bug this exists for: an agent turn re-reads a cached prefix every
        time, so `input_tokens` alone can report 15 against 2,692 out while the
        real prompt was tens of thousands of cached tokens. Reading only that
        field understated every Claude turn's cost."""
        from heart.runner import _extract_usage

        log = self.root / "implement.log"
        log.write_text(json.dumps({
            "result": "did the thing",
            "usage": {
                "input_tokens": 15, "output_tokens": 2692,
                "cache_read_input_tokens": 48_000,
                "cache_creation_input_tokens": 12_000,
                "cache_creation": {"ephemeral_5m_input_tokens": 9_000,
                                   "ephemeral_1h_input_tokens": 3_000},
            },
        }))
        self.assertEqual(_extract_usage(log, "claude"), {
            "tokens_in": 15, "tokens_out": 2692, "cache_read": 48_000,
            "cache_write_5m": 9_000, "cache_write_1h": 3_000})

    def test_extract_usage_flat_cache_total_when_ttl_absent(self):
        """An older CLI reports only the flat creation total. Treat it as the
        5m bucket rather than dropping it: 1.25x is the common case and the
        cheaper of the two, so the error is small and never an overcharge."""
        from heart.runner import _extract_usage

        log = self.root / "implement.log"
        log.write_text(json.dumps({
            "result": "ok",
            "usage": {"input_tokens": 10, "output_tokens": 20,
                      "cache_read_input_tokens": 500,
                      "cache_creation_input_tokens": 300},
        }))
        usage = _extract_usage(log, "claude")
        self.assertEqual(usage["cache_write_5m"], 300)
        self.assertEqual(usage["cache_write_1h"], 0)

    def test_model_id_keys_and_effective_dating(self):
        """A rate entered against the exact model id beats one inferred through
        a profile, and a scheduled change takes effect on its own date."""
        from heart.runner import model_pricing

        cfg = self.root / "heart"
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / "models.json").write_text(json.dumps({
            "profiles": {"sonnet": {"model": "claude-sonnet-5"}},
            "pricing": {"claude:sonnet": {"in_per_mtok": 99.0, "out_per_mtok": 99.0}},
            "model_pricing": {
                "claude-opus-5": {"in_per_mtok": 5.0, "out_per_mtok": 25.0,
                                  "source": "https://example/pricing",
                                  "verified": "2026-08-05"},
                "claude-sonnet-5": [
                    {"in_per_mtok": 2.0, "out_per_mtok": 10.0,
                     "effective_from": "2026-01-01"},
                    {"in_per_mtok": 3.0, "out_per_mtok": 15.0,
                     "effective_from": "2026-09-01"},
                ],
            },
        }))
        old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = str(self.root)
        try:
            # model-id entry wins over the profile join, which said $99
            day = model_pricing(on="2026-08-05")
            self.assertEqual(day["claude-opus-5"], {"input": 5.0, "output": 25.0})
            self.assertEqual(day["claude-sonnet-5"], {"input": 2.0, "output": 10.0})
            # the announced change lands on its date without anyone editing
            self.assertEqual(model_pricing(on="2026-08-31")["claude-sonnet-5"],
                             {"input": 2.0, "output": 10.0})
            self.assertEqual(model_pricing(on="2026-09-01")["claude-sonnet-5"],
                             {"input": 3.0, "output": 15.0})
            # a future-only row must not apply early
            self.assertEqual(model_pricing(on="2025-06-01").get("claude-sonnet-5"),
                             {"input": 99.0, "output": 99.0})
        finally:
            os.environ.pop("XDG_CONFIG_HOME", None) if old is None \
                else os.environ.__setitem__("XDG_CONFIG_HOME", old)

    def test_set_model_price_demands_provenance(self):
        """A rate with no source cannot be re-verified later, so it is refused
        rather than written — an unauditable cost is one people stop asking
        about, which is how a stale number survives a price change."""
        from heart.runner import model_pricing, set_model_price

        old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = str(self.root / "prov")
        try:
            with self.assertRaises(ValueError):
                set_model_price("m", 1.0, 2.0, source="")
            with self.assertRaises(ValueError):
                set_model_price("m", -1.0, 2.0, source="https://example/pricing")
            row = set_model_price("claude-x", 4.0, 8.0,
                                  source="https://example/pricing",
                                  verified="2026-08-05")
            self.assertEqual(row["source"], "https://example/pricing")
            self.assertEqual(model_pricing()["claude-x"],
                             {"input": 4.0, "output": 8.0})
            # a scheduled row is added beside the current one, not on top of it
            set_model_price("claude-x", 6.0, 12.0,
                            source="https://example/pricing",
                            effective_from="2099-01-01")
            self.assertEqual(model_pricing()["claude-x"],
                             {"input": 4.0, "output": 8.0})
            self.assertEqual(model_pricing(on="2099-06-01")["claude-x"],
                             {"input": 6.0, "output": 12.0})
        finally:
            os.environ.pop("XDG_CONFIG_HOME", None) if old is None \
                else os.environ.__setitem__("XDG_CONFIG_HOME", old)

    def test_model_pricing_joins_profiles_to_rates(self):
        """models.json keys rates by provider:profile and models by profile.
        A CLI transcript reports neither — only a concrete model id — so the
        two have to be joined before an interactive turn can be priced."""
        from heart.runner import model_pricing

        cfg = self.root / "heart"
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / "models.json").write_text(json.dumps({
            "profiles": {
                "haiku": {"model": "claude-haiku-4-5"},
                "terra": {"model": "gpt-5.6-terra"},
                "local": {"model": "qwen3.6-27b"},
                "orphan": {"model": "no-rate-for-this"},   # profile, no rate
                "nomodel": {},                              # rate, no model id
            },
            "pricing": {
                "claude:haiku": {"in_per_mtok": 1.0, "out_per_mtok": 5.0},
                "codex:terra": {"in_per_mtok": 2.0, "out_per_mtok": 12.0},
                "api:local": {"in_per_mtok": 0.0, "out_per_mtok": 0.0},
                "claude:nomodel": {"in_per_mtok": 9.0, "out_per_mtok": 9.0},
                "claude": {"in_per_mtok": 5.0, "out_per_mtok": 25.0},
            },
        }))
        old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = str(self.root)
        try:
            rates = model_pricing()
            self.assertEqual(rates["claude-haiku-4-5"], {"input": 1.0, "output": 5.0})
            self.assertEqual(rates["gpt-5.6-terra"], {"input": 2.0, "output": 12.0})
            # a free local model is 0/0, not absent: the caller must bill it as
            # zero rather than falling back to a provider rate
            self.assertEqual(rates["qwen3.6-27b"], {"input": 0.0, "output": 0.0})
            # unjoinable entries are absent, so the caller falls back instead
            # of billing a model at a rate that was never meant for it
            self.assertNotIn("no-rate-for-this", rates)
            # the provider-wide key has no profile and must not become a model
            self.assertNotIn("claude", rates)
        finally:
            os.environ.pop("XDG_CONFIG_HOME", None) if old is None \
                else os.environ.__setitem__("XDG_CONFIG_HOME", old)

    def test_model_pricing_survives_a_missing_or_broken_config(self):
        """No card is "we cannot price this", never "everything is free"."""
        from heart.runner import model_pricing

        old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = str(self.root / "nonexistent")
        try:
            self.assertEqual(model_pricing(), {})
            cfg = self.root / "broken" / "heart"
            cfg.mkdir(parents=True, exist_ok=True)
            (cfg / "models.json").write_text("{not json")
            os.environ["XDG_CONFIG_HOME"] = str(self.root / "broken")
            self.assertEqual(model_pricing(), {})
        finally:
            os.environ.pop("XDG_CONFIG_HOME", None) if old is None \
                else os.environ.__setitem__("XDG_CONFIG_HOME", old)

    def test_price_applies_cache_multipliers(self):
        """Read 0.1x, 5m write 1.25x, 1h write 2x, all off the base input rate."""
        from heart.runner import _price

        cfg = self.root / "heart"
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / "models.json").write_text(json.dumps({"pricing": {
            "claude": {"in_per_mtok": 5.0, "out_per_mtok": 25.0}}}))
        old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = str(self.root)
        try:
            # 1M uncached in = $5; 1M read = $0.50; 1M 5m write = $6.25;
            # 1M 1h write = $10; 1M out = $25
            self.assertAlmostEqual(
                _price("claude", 1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000),
                5.0 + 25.0 + 0.5 + 6.25 + 10.0, places=6)
            # the two-argument call every existing caller makes is unchanged
            self.assertAlmostEqual(_price("claude", 1_000_000, 0), 5.0, places=6)
            # cache-only traffic still costs money
            self.assertAlmostEqual(_price("claude", 0, 0, 1_000_000), 0.5, places=6)
            # the understated case, priced both ways
            bare = _price("claude", 15, 2692)
            full = _price("claude", 15, 2692, 48_000, 9_000, 3_000)
            self.assertGreater(full, bare)
        finally:
            os.environ.pop("XDG_CONFIG_HOME", None) if old is None \
                else os.environ.__setitem__("XDG_CONFIG_HOME", old)

    def test_extract_usage_garbage_log(self):
        from heart.runner import _extract_usage

        log = self.root / "implement.log"
        log.write_text("not json at all\njust agent chatter\n")
        self.assertEqual(_extract_usage(log, "claude"),
                         {"tokens_in": None, "tokens_out": None, "cache_read": None,
                          "cache_write_5m": None, "cache_write_1h": None})

        missing = self.root / "missing.log"
        self.assertEqual(_extract_usage(missing, "claude"),
                         {"tokens_in": None, "tokens_out": None, "cache_read": None,
                          "cache_write_5m": None, "cache_write_1h": None})

    def test_extract_usage_api_heart_usage_line(self):
        from heart.runner import _extract_usage

        log = self.root / "solo.log"
        log.write_text("$ echo hi\nhi\nHEART_USAGE={\"tokens_in\": 12, \"tokens_out\": 34}\n")
        self.assertEqual(_extract_usage(log, "api"),
                         {"tokens_in": 12, "tokens_out": 34, "cache_read": 0,
                          "cache_write_5m": 0, "cache_write_1h": 0})

        log2 = self.root / "no-usage.log"
        log2.write_text("$ echo hi\nhi\n")
        none = {"tokens_in": None, "tokens_out": None, "cache_read": None,
                "cache_write_5m": None, "cache_write_1h": None}
        self.assertEqual(_extract_usage(log2, "api"), none)

        self.assertEqual(_extract_usage(log, "shell"), none)

    def test_price(self):
        from heart.runner import _price

        cfgdir = self.root / "cfg" / "heart"
        cfgdir.mkdir(parents=True)
        (cfgdir / "models.json").write_text(json.dumps({
            "profiles": {
                "qwen": {"endpoint": "https://api.example.com/v1", "model": "qwen"},
                "local": {"endpoint": "http://127.0.0.1:8000/v1", "model": "q"},
            },
            "pricing": {
                "api": {"in_per_mtok": 1.0, "out_per_mtok": 2.0},
                "claude": {"in_per_mtok": 3.0, "out_per_mtok": 15.0},
            }
        }))
        old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = str(self.root / "cfg")
        try:
            # metered profile prices via the base "api" entry
            self.assertEqual(_price("api:qwen", 1_000_000, 1_000_000), 3.0)
            # a local endpoint is free even under a broad "api" entry
            self.assertEqual(_price("api:local", 1_000_000, 1_000_000), 0.0)
            # base fallback: "claude:opus" -> "claude" entry (subscription seat priced)
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

        old_journal = os.environ.get("EVENT_JOURNAL_DIR")
        old_ingest = os.environ.get("HEART_INGEST")
        os.environ["EVENT_JOURNAL_DIR"] = str(self.root / "journal")
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
            self.assertEqual(ep["usage"], {"tokens_in": None, "tokens_out": None,
                                       "cache_read": None, "cache_write_5m": None,
                                       "cache_write_1h": None, "cost_usd": None})
            # must not crash even though no episode in this window carries cost
            pulse.insights(hours=24)
        finally:
            if old_journal is None:
                os.environ.pop("EVENT_JOURNAL_DIR", None)
            else:
                os.environ["EVENT_JOURNAL_DIR"] = old_journal
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
        self._old_journal = os.environ.get("EVENT_JOURNAL_DIR")
        os.environ["EVENT_JOURNAL_DIR"] = str(self.root / "journal")
        self._old_ingest = os.environ.get("HEART_INGEST")
        os.environ["HEART_INGEST"] = "off"

    def tearDown(self):
        if self._old_journal is None:
            os.environ.pop("EVENT_JOURNAL_DIR", None)
        else:
            os.environ["EVENT_JOURNAL_DIR"] = self._old_journal
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
        self._old_journal = os.environ.get("EVENT_JOURNAL_DIR")
        os.environ["EVENT_JOURNAL_DIR"] = str(self.root / "journal")

    def tearDown(self):
        if self._old_journal is None:
            os.environ.pop("EVENT_JOURNAL_DIR", None)
        else:
            os.environ["EVENT_JOURNAL_DIR"] = self._old_journal
        self.tmp.cleanup()

    def _write(self, **event):
        from heart.events import journal_dir

        d = journal_dir()
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "20260101.ndjson", "a") as f:
            f.write(json.dumps(event) + "\n")

    def test_emit_stamps_goal_lineage_from_env(self):
        from heart.events import emit, journal_dir

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

        lines = sorted(journal_dir().glob("*.ndjson"))[0].read_text().splitlines()
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
        self._old_journal = os.environ.get("EVENT_JOURNAL_DIR")
        os.environ["EVENT_JOURNAL_DIR"] = str(self.root / "journal")

    def tearDown(self):
        if self._old_journal is None:
            os.environ.pop("EVENT_JOURNAL_DIR", None)
        else:
            os.environ["EVENT_JOURNAL_DIR"] = self._old_journal
        self.tmp.cleanup()

    def _write(self, **event):
        from heart.events import journal_dir

        d = journal_dir()
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

        old = os.environ.get("EVENT_JOURNAL_DIR")
        tmp = tempfile.mkdtemp()
        os.environ["EVENT_JOURNAL_DIR"] = tmp
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
                os.environ.pop("EVENT_JOURNAL_DIR", None)
            else:
                os.environ["EVENT_JOURNAL_DIR"] = old

    def test_steer_and_episode_drilldown_endpoints(self):
        import threading
        import urllib.error
        import urllib.request
        from http.server import ThreadingHTTPServer

        from heart import serve as serve_mod
        from heart.serve import Handler

        old_journal = os.environ.get("EVENT_JOURNAL_DIR")
        os.environ["EVENT_JOURNAL_DIR"] = tempfile.mkdtemp()
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
            if old_journal is None:
                os.environ.pop("EVENT_JOURNAL_DIR", None)
            else:
                os.environ["EVENT_JOURNAL_DIR"] = old_journal


class TestUnrestrictedTasksStayUnrestricted(unittest.TestCase):
    """An empty allowed_paths means no restriction. A role that declares its own
    paths must not turn that into a restriction -- doing so made every pipeline
    run score path_violation for editing the file it was told to edit, while the
    same task run solo passed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.head = make_repo(self.root)
        self.repo = str(self.root / "toyrepo")
        self.runs = str(self.root / "runs")

    def tearDown(self):
        self.tmp.cleanup()

    def _task(self, **kw):
        return TaskSpec(
            task_id="scope", repo_path=self.repo, base_commit=self.head,
            prompt=FIX_CMD, timeout_seconds=60,
            public_verifiers=[Verifier(name="t", command="python3 -m pytest -q")],
            **kw)

    def test_role_paths_do_not_invent_a_lane(self):
        ep = run_episode(self._task(), agent="shell", roles=DEFAULT_ROLES,
                         runs_dir=self.runs)
        self.assertNotEqual(ep["outcome"], "path_violation")

    def test_a_declared_lane_is_still_enforced(self):
        ep = run_episode(self._task(allowed_paths=["docs"]), agent="shell",
                         roles=DEFAULT_ROLES, runs_dir=self.runs)
        self.assertEqual(ep["outcome"], "path_violation")


class TestGitignoredIntegrationFiles(unittest.TestCase):
    """heart copies .claude/.arteries into every worktree, and most real repos
    gitignore both. `git add -A -- . :(exclude).claude` then exits 1, because the
    `.` names an ignored path and the exclude does not suppress that check --
    so every commit failed on any repo with a .gitignore, and on Path B the
    exception escaped and killed the whole orchestration."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.head = make_repo(self.root)
        self.repo = self.root / "toyrepo"
        (self.repo / ".gitignore").write_text(".claude/\n.arteries/\n.env\n")
        git = ["git", "-C", str(self.repo), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", "add gitignore"], check=True)
        self.head = subprocess.run(
            [*git[:3], "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True).stdout.strip()

    def tearDown(self):
        self.tmp.cleanup()

    def test_commit_survives_gitignored_integration_dirs(self):
        from heart.env import Workspace
        ws = Workspace(str(self.repo), self.head)
        try:
            (ws.path / ".claude").mkdir(exist_ok=True)
            (ws.path / ".claude" / "settings.json").write_text("{}")
            (ws.path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
            sha = ws.commit("heart test")
            self.assertIsNotNone(sha)
            files = subprocess.run(
                ["git", "-C", str(ws.path), "show", "--name-only", "--format=", sha],
                capture_output=True, text=True, check=True).stdout.split()
            self.assertIn("calc.py", files)
            self.assertNotIn(".claude/settings.json", files)
        finally:
            ws.destroy()

    def test_an_ignored_secret_never_reaches_the_commit(self):
        # the reason `git add -f` is not the fix: it would stage every ignored
        # file the excludes do not name, .env included.
        from heart.env import Workspace
        ws = Workspace(str(self.repo), self.head)
        try:
            (ws.path / ".env").write_text("API_KEY=sk-live-not-a-real-key\n")
            (ws.path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
            sha = ws.commit("heart test")
            files = subprocess.run(
                ["git", "-C", str(ws.path), "show", "--name-only", "--format=", sha],
                capture_output=True, text=True, check=True).stdout.split()
            self.assertIn("calc.py", files)
            self.assertNotIn(".env", files)
        finally:
            ws.destroy()


class TestReviewPhase(unittest.TestCase):
    """The loop itself, with the agent turns injected. The confirm stage has its
    own prompt by design, so a shell `reviewer` cannot script a verdict flip --
    which is why the round trip is tested here rather than through an episode."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def _writer(self, scripted):
        def turn(name, prompt):
            self.calls.append((name, prompt))
            log = self.out / f"{name}.log"
            log.write_text(scripted.get(name, ""))
            return log
        return turn

    def test_a_resolved_blocker_flips_the_verdict(self):
        from heart import review
        blocker = ('{"findings":[{"severity":"blocker","file":"a.py","line":1,'
                   '"claim":"drops only one index"}]}')
        turn = self._writer({
            "review.1": blocker,
            "review-fix.1": '[{"id":0,"status":"fixed","how":"query pg_index"}]',
            "review-confirm.2": '{"findings":[]}',
        })
        r = review.phase("do the thing", assess=turn, resolve=turn,
                         verify=lambda: None, legacy_verdict=lambda p: None)
        self.assertEqual(r.verdict, "approve")
        self.assertEqual(r.rounds, 2)
        self.assertEqual([n for n, _ in self.calls],
                         ["review.1", "review-fix.1", "review-confirm.2"])

    def test_the_fixer_is_handed_every_finding_not_a_log_tail(self):
        # the defect this replaces: review.log[-1500:] cut a 3471-char review at
        # char 1971, and the blocker sat at char 13.
        from heart import review
        blocker = ('{"findings":[{"severity":"blocker","file":"a.py","line":1,'
                   '"claim":"THE BLOCKER"},{"severity":"note","claim":"cosmetic"}]}')
        turn = self._writer({"review.1": blocker, "review-confirm.2": '{"findings":[]}'})
        review.phase("t", assess=turn, resolve=turn, verify=lambda: None,
                     legacy_verdict=lambda p: None)
        fix_prompt = next(p for n, p in self.calls if n == "review-fix.1")
        self.assertIn("THE BLOCKER", fix_prompt)
        self.assertIn("cosmetic", fix_prompt)

    def test_confirm_is_told_what_was_claimed(self):
        from heart import review
        turn = self._writer({
            "review.1": '{"findings":[{"severity":"blocker","claim":"X broken"}]}',
            "review-fix.1": '[{"id":0,"status":"fixed","how":"rewrote X"}]',
            "review-confirm.2": '{"findings":[]}',
        })
        review.phase("t", assess=turn, resolve=turn, verify=lambda: None,
                     legacy_verdict=lambda p: None)
        confirm = next(p for n, p in self.calls if n == "review-confirm.2")
        self.assertIn("X broken", confirm)
        self.assertIn("rewrote X", confirm)

    def test_an_unfixed_blocker_stops_at_the_round_budget(self):
        from heart import review
        blocker = '{"findings":[{"severity":"blocker","claim":"still broken"}]}'
        turn = self._writer({"review.1": blocker, "review-confirm.2": blocker})
        r = review.phase("t", assess=turn, resolve=turn, verify=lambda: None,
                         legacy_verdict=lambda p: None, rounds=1)
        self.assertEqual(r.verdict, "reject")
        self.assertEqual([n for n, _ in self.calls],
                         ["review.1", "review-fix.1", "review-confirm.2"])

    def test_rounds_zero_assesses_without_ever_resolving(self):
        from heart import review
        turn = self._writer(
            {"review.1": '{"findings":[{"severity":"blocker","claim":"x"}]}'})
        r = review.phase("t", assess=turn, resolve=turn, verify=lambda: None,
                         legacy_verdict=lambda p: None, rounds=0)
        self.assertEqual(r.verdict, "reject")
        self.assertEqual([n for n, _ in self.calls], ["review.1"])

    def test_no_json_falls_back_and_says_so(self):
        from heart import review
        turn = self._writer({"review.1": "looks fine to me, APPROVE"})
        r = review.phase("t", assess=turn, resolve=turn, verify=lambda: None,
                         legacy_verdict=lambda p: "approve")
        self.assertTrue(r.fell_back)
        self.assertEqual(r.verdict, "approve")
        self.assertEqual(r.findings, [])


class TestProbesAndBaselines(unittest.TestCase):
    """Preconditions measured before planning, and criteria judged against base."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.commit = make_repo(self.root)
        self.runs = self.root / "runs"
        os.environ["EVENT_JOURNAL_DIR"] = str(self.root / "journal")
        os.environ["HEART_INGEST"] = "off"
        self.task = TaskSpec(
            task_id="probe-toy",
            repo_path=str(self.root / "toyrepo"),
            base_commit=self.commit,
            prompt=FIX_CMD,
            public_verifiers=[Verifier(name="unit",
                                       command="python3 -m unittest -q test_calc")],
            timeout_seconds=60,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, **over):
        return run_episode(TaskSpec(**{**self.task.__dict__, **over}),
                           agent="shell", runs_dir=self.runs)

    def test_failing_probe_blocks_before_the_agent_runs(self):
        ep = self._run(probes=[Verifier(name="pgvector", command="exit 3")])
        self.assertEqual(ep["outcome"], "blocked")
        self.assertIsNone(ep["reward"]["total"])
        self.assertIn("precondition not met", ep["blocked_reason"])
        # the whole point: nothing was spent
        self.assertEqual(ep["roles"], [])
        self.assertEqual(ep["diff_lines"], 0)

    def test_passing_probe_hands_its_measurement_to_the_agent(self):
        ep = self._run(probes=[Verifier(name="version", command="echo 0.8.6")])
        self.assertEqual(ep["outcome"], "pass")
        self.assertIn("Measured environment", ep["prompt"])
        self.assertIn("0.8.6", ep["prompt"])

    def test_baseline_identical_fails_when_the_output_moved(self):
        ep = self._run(public_verifiers=[
            Verifier(name="unit", command="python3 -m unittest -q test_calc"),
            Verifier(name="answers", command="cat calc.py", baseline="identical")])
        self.assertEqual(ep["outcome"], "fail")   # unit passed; the criterion did not
        answers = ep["verifier_results"]["answers"]
        self.assertTrue(answers["exit_code"] == 0)
        self.assertFalse(answers["passed"])       # the comparison overrides the exit
        self.assertFalse(answers["baseline"]["passed"])

    def test_baseline_no_worse_accepts_an_unchanged_measurement(self):
        ep = self._run(public_verifiers=[
            Verifier(name="recall", command="echo recall 0.955", baseline="no_worse")])
        self.assertEqual(ep["outcome"], "pass")
        self.assertIn(">= base", ep["verifier_results"]["recall"]["baseline"]["detail"])

    def test_compare_baseline_refuses_what_it_cannot_read(self):
        self.assertFalse(compare_baseline("no_worse", "nothing", "numeric")[0])
        self.assertFalse(compare_baseline("bogus", "1", "1")[0])
        # the last number is the answer, and latency gates the other way
        self.assertTrue(compare_baseline("no_worse", "ran 5 recall 0.94", "ran 5 recall 0.96")[0])
        self.assertTrue(compare_baseline("no_more", "p50 1402", "p50 1397")[0])
        self.assertFalse(compare_baseline("no_more", "p50 1402", "p50 1900")[0])


if __name__ == "__main__":
    unittest.main()
