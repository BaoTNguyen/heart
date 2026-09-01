"""Routing + orchestration. Router tests are pure; orchestration tests drive the
real merge/verify/fallback engine with `shell` workers and real git — no model.

Run: python3 -m unittest tests.test_route_orchestrate
"""
from __future__ import annotations

import dataclasses
import graphlib
import json
import os
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from heart import orchestrate, route
from heart.orchestrate import Subtask, run_orchestrated
from heart.taskspec import TaskSpec, Verifier

MANIFEST = {
    "local":  {"agent": "api:local", "skills": {"coding": 0.5, "frontend": 0.4},
               "context": 32_000, "max_difficulty": "easy", "cost": 0.1},
    "claude": {"agent": "claude", "skills": {"coding": 0.9, "frontend": 0.7},
               "context": 200_000, "max_difficulty": "hard", "cost": 3.0},
    "twin":   {"agent": "api:twin", "skills": {"coding": 0.9, "frontend": 0.7},
               "context": 200_000, "max_difficulty": "hard", "cost": 0.2},
}


def _task(prompt="add a function", **kw):
    return TaskSpec(task_id="t", repo_path=".", base_commit="x", prompt=prompt, **kw)


# route.route() emits route.decided to the event journal. Without this, synthetic
# test decisions (api:twin, task_id="t") leak into the real ~/.local journal and
# show up on `pulse serve`. Isolate the journal for the whole module, mirroring the
# XDG_CONFIG_HOME isolation the orchestration classes already do for config.
_journal_tmp: tempfile.TemporaryDirectory | None = None


def setUpModule():
    global _journal_tmp
    _journal_tmp = tempfile.TemporaryDirectory()
    os.environ["EVENT_JOURNAL_DIR"] = _journal_tmp.name


def tearDownModule():
    os.environ.pop("EVENT_JOURNAL_DIR", None)
    if _journal_tmp is not None:
        _journal_tmp.cleanup()


class TestRouter(unittest.TestCase):
    def test_classify_infers_skills_and_difficulty(self):
        skills, difficulty, _ = route.classify(_task("fix the failing race condition in the server"))
        self.assertIn("debug", skills)
        self.assertEqual(difficulty, "hard")  # "race" is a hard word

    def test_declared_skills_win_over_inference(self):
        skills, _, _ = route.classify(_task("whatever", skills=["frontend"]))
        self.assertEqual(skills, ["frontend"])

    def test_context_constraint_filters(self):
        d = route.route(_task(skills=["coding"], difficulty="easy", min_context=100_000),
                        manifest=MANIFEST, stats={})
        self.assertNotEqual(d.agent, "api:local")  # local's 32k window is too small

    def test_difficulty_ceiling_filters(self):
        d = route.route(_task(skills=["coding"], difficulty="hard", min_context=1000),
                        manifest=MANIFEST, stats={})
        self.assertNotEqual(d.agent, "api:local")  # local caps at "easy"

    def test_ties_break_toward_cheaper(self):
        # claude and twin have identical skills; twin is far cheaper -> twin wins
        d = route.route(_task(skills=["coding"], difficulty="hard", min_context=1000),
                        manifest=MANIFEST, stats={})
        self.assertEqual(d.agent, "api:twin")

    def test_capability_beats_cost_when_scores_differ(self):
        # easy coding: local (0.5) is cheapest but claude/twin (0.9) score higher
        d = route.route(_task(skills=["coding"], difficulty="easy", min_context=1000),
                        manifest=MANIFEST, stats={})
        self.assertIn(d.agent, ("claude", "api:twin"))
        self.assertEqual(d.candidates[0]["score"], 0.9)

    def test_measured_stats_override_declared_prior(self):
        # local declared 0.5 for coding, but measured 0.95 over enough samples on
        # easy tasks -> its blended score should overtake the 0.9 declared models
        stats = {"api:local": {"coding|easy": {"n": 40, "mean": 0.98}}}
        d = route.route(_task(skills=["coding"], difficulty="easy", min_context=1000),
                        manifest=MANIFEST, stats=stats)
        self.assertEqual(d.agent, "api:local")

    def test_blend_shrinkage(self):
        self.assertEqual(route._blend(0.9, None), 0.9)             # no evidence -> prior
        self.assertAlmostEqual(route._blend(0.5, {"n": 8, "mean": 1.0}), 0.75)  # w=0.5

    def test_aggregate_builds_reward_stats(self):
        events = [
            {"kind": "episode.finished", "payload":
                {"agent": "claude", "reward": 0.8, "skills": ["coding"], "difficulty": "hard"}},
            {"kind": "episode.finished", "payload":
                {"agent": "claude", "reward": 0.6, "skills": ["coding"], "difficulty": "hard"}},
            {"kind": "episode.finished", "payload":  # no skills -> ignored
                {"agent": "claude", "reward": 1.0, "difficulty": "hard"}},
        ]
        stats = route.aggregate(events)
        self.assertEqual(stats["claude"]["coding|hard"], {"n": 2, "mean": 0.7})

    def test_ordinal_manifest_no_floats(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "models.json"
            p.write_text(json.dumps({"models": {
                "big":   {"agent": "claude", "tier": "frontier",
                          "skills": {"planning": "strong", "vision": "weak"}, "cost": 3.0},
                "small": {"agent": "api:local", "tier": "small",
                          "skills": {"coding": "capable"}, "context": 32000, "cost": 0.1},
            }}))
            m = route.load_manifest(p)
            # tier sets baseline + ceiling; ordinal words become coarse priors
            self.assertEqual(m["big"]["max_difficulty"], "hard")     # frontier default
            self.assertEqual(m["small"]["max_difficulty"], "easy")   # small default
            self.assertAlmostEqual(m["big"]["skills"]["planning"], route._LEVEL["strong"])
            self.assertAlmostEqual(m["big"]["baseline"], route._TIER_BASELINE["frontier"])
            # a hard planning task: small is filtered (ceiling), big wins on the
            # strong note; an unlisted skill would fall back to big's baseline
            d1 = route.route(_task(skills=["planning"], difficulty="hard", min_context=1000),
                             manifest=m, stats={})
            self.assertEqual(d1.agent, "claude")

    def test_legacy_tiers_synthesize_a_manifest(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "models.json"
            p.write_text(json.dumps({"tiers": {"cheap": "api:q", "strong": "claude"}}))
            m = route.load_manifest(p)
            self.assertEqual(m["strong"]["agent"], "claude")
            self.assertEqual(m["cheap"]["max_difficulty"], "easy")


class TestModelPin(unittest.TestCase):
    """CLI agents pin a specific model via `--model`, so opus/sonnet/haiku run on
    a subscription seat instead of the metered API."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = Path(self.tmp.name) / "heart"
        cfg.mkdir()
        (cfg / "models.json").write_text(json.dumps({"profiles": {
            "sonnet": {"model": "claude-sonnet-5"},
            "gpt": {"model": "gpt-5"},
        }}))
        self._old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = self.tmp.name

    def tearDown(self):
        if self._old is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self._old
        self.tmp.cleanup()

    def _model_of(self, cmd):
        return cmd[cmd.index("--model") + 1]

    def test_claude_profile_pins_resolved_model(self):
        from heart.runner import _agent_command
        cmd, shell = _agent_command("claude:sonnet", "do it")
        self.assertEqual(self._model_of(cmd), "claude-sonnet-5")
        # the flag sits before the positional prompt
        self.assertLess(cmd.index("--model"), cmd.index("do it"))
        self.assertFalse(shell)

    def test_codex_profile_pins_model(self):
        from heart.runner import _agent_command
        cmd, _ = _agent_command("codex:gpt", "do it")
        self.assertEqual(self._model_of(cmd), "gpt-5")
        self.assertLess(cmd.index("--model"), cmd.index("do it"))  # before the prompt

    def test_unknown_token_used_as_literal_model_id(self):
        from heart.runner import _agent_command
        cmd, _ = _agent_command("claude:claude-opus-4-8", "x")
        self.assertEqual(self._model_of(cmd), "claude-opus-4-8")

    def test_bare_agent_gets_no_model_flag(self):
        from heart.runner import _agent_command
        cmd, _ = _agent_command("claude", "x")
        self.assertNotIn("--model", cmd)

    def test_api_agent_selects_model_itself_not_via_flag(self):
        from heart.runner import _agent_command
        cmd, _ = _agent_command("api:local", "x")
        self.assertNotIn("--model", cmd)  # agents_api reads the profile


class TestDecompose(unittest.TestCase):
    def test_parse_subtasks_fenced_and_bare(self):
        from heart.orchestrate import _parse_subtasks, Subtask
        payload = ('[{"name":"api","prompt":"build the endpoint","skills":["backend"],'
                   '"effort":"high","allowed_paths":["api/"]},'
                   '{"name":"ui","prompt":"build the page","skills":["frontend"]}]')
        for raw in (f"```json\n{payload}\n```\nDone.", f"here:\n{payload}"):
            subs = _parse_subtasks(raw)
            self.assertEqual([s.name for s in subs], ["api", "ui"])
            self.assertIsInstance(subs[0], Subtask)
            self.assertEqual(subs[0].allowed_paths, ["api/"])
            self.assertEqual(subs[1].effort, "medium")  # default when omitted

    def test_parse_subtasks_rejects_missing_fields_and_empty(self):
        from heart.orchestrate import _parse_subtasks
        self.assertEqual(_parse_subtasks("[]"), [])          # decline -> empty
        with self.assertRaises(ValueError):
            _parse_subtasks('[{"name":"x"}]')                # no prompt
        with self.assertRaises(ValueError):
            _parse_subtasks("no array here")

    def test_a_contract_that_quotes_code_does_not_hide_the_plan(self):
        # A good contract freezes an interface by showing it, in its own fence.
        # That fence used to win the regex, leaving no JSON to find -- which the
        # caller reported as "declined to split" while the plan sat in the log.
        raw = ('''Here is the split.\n\n```json\n'''
               '''{"contract": "`store.py`:\\n```python\\nclass Store: ...\\n```",'''
               ''' "subtasks": ['''
               '''{"name": "models", "prompt": "build models"},'''
               '''{"name": "store", "prompt": "build store", "depends_on": ["models"]}]}'''
               '''\n```\n''')
        contract, subs = orchestrate._parse_decomposition(raw)
        self.assertIn("class Store", contract)
        self.assertEqual([s.name for s in subs], ["models", "store"])
        self.assertEqual([[s.name for s in w] for w in orchestrate._waves(subs)],
                         [["models"], ["store"]])

    def test_a_transcript_full_of_braces_does_not_hide_the_plan(self):
        # codex writes its reasoning, then a token count, then the answer. The
        # first brace belongs to the transcript and the last to the plan, so the
        # old first-to-last span parsed neither -- and read as "declined".
        raw = ('thinking {"step": 1} and {"step": 2}\ntokens used\n7127\n'
               '{"contract":"","subtasks":['
               '{"name":"models","prompt":"build models"},'
               '{"name":"store","prompt":"build store","depends_on":["models"]}]}')
        contract, subs = orchestrate._parse_decomposition(raw)
        self.assertEqual(contract, "")
        self.assertEqual([[s.name for s in w] for w in orchestrate._waves(subs)],
                         [["models"], ["store"]])

    def test_the_subtasks_array_is_not_mistaken_for_a_bare_plan(self):
        # the inner array parses on its own and would win as the "last" plan,
        # dropping the contract the same-wave workers need.
        raw = ('{"contract":"def parse(x) -> Result","subtasks":'
               '[{"name":"a","prompt":"build"},{"name":"b","prompt":"call"}]}')
        contract, subs = orchestrate._parse_decomposition(raw)
        self.assertEqual(contract, "def parse(x) -> Result")
        self.assertEqual([s.name for s in subs], ["a", "b"])

    def test_parse_decomposition_object_with_contract(self):
        from heart.orchestrate import _parse_decomposition, _with_contract
        raw = ('prose\n{"contract":"def parse(x)->Result","subtasks":'
               '[{"name":"a","prompt":"build parse"},{"name":"b","prompt":"call parse"}]}')
        contract, subs = _parse_decomposition(raw)
        self.assertEqual(contract, "def parse(x)->Result")
        self.assertEqual([s.name for s in subs], ["a", "b"])
        # injection puts the frozen interface in front of every worker's slice
        injected = _with_contract(contract, subs[1].prompt)
        self.assertIn("def parse(x)->Result", injected)
        self.assertIn("call parse", injected)

    def test_parse_decomposition_bare_array_has_no_contract(self):
        from heart.orchestrate import _parse_decomposition
        contract, subs = _parse_decomposition('[{"name":"a","prompt":"x"}]')
        self.assertEqual(contract, "")
        self.assertEqual(subs[0].name, "a")

    def test_decline_object_form_is_not_decomposable(self):
        from heart.orchestrate import _parse_decomposition
        contract, subs = _parse_decomposition('{"subtasks": []}')
        self.assertEqual((contract, subs), ("", []))


class TestSubagentMemory(unittest.TestCase):
    """heart marks orchestration workers as arteries subagents, using arteries'
    own identity contract (import direction: heart -> arteries only)."""

    def test_identity_env_matches_arteries_contract(self):
        from heart.episode import _subagent_env
        from arteries.subagent import subagent_env  # source of truth
        env = _subagent_env("orch-goal-f1-a1")
        self.assertEqual(env["ARTERIES_PARENT_AGENT_ID"], "orch-goal-f1-a1")
        self.assertEqual(env["ARTERIES_AGENT_ROLE"], "subagent")
        self.assertTrue(env["ARTERIES_AGENT_ID"].startswith("orch-goal-f1-a1-sub-"))
        # identity only — memory mode is set separately by the launcher
        self.assertNotIn("ARTERIES_MEMORY", env)
        self.assertEqual(set(env), set(subagent_env("p")))

    def test_subagent_never_discards_ephemeral(self):
        # the critical branch: a worker (parent set, isolated worktree) inherits
        # parent lineage and compiles up — it must NEVER get ephemeral=discard,
        # or every worker's memory is silently thrown away. A best-of-N candidate
        # (isolated, no parent) is the opposite and must discard.
        from heart.episode import _memory_env
        worker = _memory_env("subagent", retrieval=True, isolated=True,
                             parent_agent_id="orch-x")
        self.assertEqual(worker["ARTERIES_PARENT_AGENT_ID"], "orch-x")
        self.assertEqual(worker["ARTERIES_MEMORY"], "subagent")
        self.assertNotIn("ARTERIES_EPHEMERAL", worker)  # must not discard
        candidate = _memory_env("normal", retrieval=True, isolated=True,
                                parent_agent_id=None)
        self.assertEqual(candidate["ARTERIES_EPHEMERAL"], "discard")
        self.assertNotIn("ARTERIES_PARENT_AGENT_ID", candidate)

    def test_arteries_does_not_import_heart(self):
        # the whole point of the direction rule: nothing heart leaks into arteries
        import arteries.subagent
        import arteries.config
        import arteries.compile
        for mod in (arteries.subagent, arteries.config, arteries.compile):
            src = Path(mod.__file__).read_text()
            self.assertNotIn("import heart", src, f"{mod.__name__} must not import heart")


def _make_repo(root: Path) -> tuple[str, str]:
    repo = root / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("start\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git[:3], "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    head = subprocess.run([*git[:3], "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return str(repo), head


class TestOrchestrate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        # save what setUpModule set so tearDown restores it instead of deleting
        # the module-level journal isolation (which the router tests rely on).
        self._saved_env = {k: os.environ.get(k) for k in ("EVENT_JOURNAL_DIR", "HEART_INGEST")}
        os.environ["EVENT_JOURNAL_DIR"] = str(root / "journal")
        os.environ["HEART_INGEST"] = "off"
        self.repo, self.head = _make_repo(root)
        self.runs = str(root / "runs")

    def tearDown(self):
        self.tmp.cleanup()
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _task(self, prompt, verifier_cmd):
        return TaskSpec(task_id="orch", repo_path=self.repo, base_commit=self.head,
                        prompt=prompt, timeout_seconds=60,
                        public_verifiers=[Verifier(name="files", command=verifier_cmd)])

    def test_decomposer_declines_falls_to_path_a(self):
        # real decompose path: a shell "planner" can't emit JSON -> declines ->
        # Path A. manifest={} keeps the decomposer on shell (not a routed model).
        task = self._task("printf 'a\\n' > a.txt; printf 'b\\n' > b.txt",
                          "test -f a.txt && test -f b.txt")
        ep = run_orchestrated(task, agent="shell", runs_dir=self.runs, manifest={})
        self.assertEqual(ep["orchestration"]["path"], "A")
        self.assertEqual(ep["outcome"], "pass")

    def test_a_failed_probe_stops_orchestration_before_decompose(self):
        # The planner is the expensive party to let plan on a false assumption,
        # so the probe runs before it and nothing downstream is spent.
        task = dataclasses.replace(
            self._task("(orchestrated)", "true"),
            probes=[Verifier(name="pgvector", command="exit 3")])
        called = []
        ep = run_orchestrated(task, agent="shell",
                              decomposer=lambda t: called.append(t) or [],
                              runs_dir=self.runs, manifest={})
        self.assertEqual(called, [])
        self.assertEqual(ep["outcome"], "blocked")
        self.assertIsNone(ep["reward"]["total"])

    def test_workers_inherit_the_measurements_and_do_not_re_probe(self):
        task = dataclasses.replace(
            self._task("(orchestrated)", "test -f a.txt && test -f b.txt"),
            probes=[Verifier(name="version", command="echo 0.8.6")])
        seen = {}

        def _plan(t):
            seen["prompt"] = t.prompt
            return [Subtask("wa", "printf 'a\\n' > a.txt"),
                    Subtask("wb", "printf 'b\\n' > b.txt")]

        ep = run_orchestrated(task, agent="shell", decomposer=_plan,
                              runs_dir=self.runs, manifest={})
        self.assertIn("0.8.6", seen["prompt"])   # the planner was told
        self.assertEqual(ep["outcome"], "pass")

    def test_path_b_disjoint_workers_merge_clean_and_verify(self):
        task = self._task("(orchestrated)", "test -f a.txt && test -f b.txt")
        subs = [Subtask("wa", "printf 'a\\n' > a.txt"),
                Subtask("wb", "printf 'b\\n' > b.txt")]
        ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                              runs_dir=self.runs, manifest={})  # {} -> shell workers
        self.assertEqual(ep["orchestration"]["path"], "B")
        self.assertEqual(ep["orchestration"]["merge"], "clean")
        self.assertEqual(ep["orchestration"]["integration"], "pass")
        self.assertEqual(ep["outcome"], "pass")
        # the merged diff carries both workers' files
        diff = (Path(self.runs) / ep["episode_id"] / "diff.patch").read_text()
        self.assertIn("a.txt", diff)
        self.assertIn("b.txt", diff)

    def test_the_merged_episode_is_scored_and_the_workers_are_not(self):
        # The merged tree is real work: a real diff that faced the real suite,
        # so it carries a real reward. A worker built a fragment nothing could
        # judge, so its None stays None -- unscored is not the same as failed.
        task = self._task("(orchestrated)", "test -f a.txt && test -f b.txt")
        subs = [Subtask("wa", "printf 'a\\n' > a.txt"),
                Subtask("wb", "printf 'b\\n' > b.txt")]
        ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                              runs_dir=self.runs, manifest={})
        self.assertEqual(ep["outcome"], "pass")
        self.assertIsNotNone(ep["reward"]["total"])
        self.assertGreater(ep["reward"]["total"], 0.0)
        self.assertIn("public_tests", ep["reward"]["components"])
        for wid in ep["orchestration"]["worker_episodes"]:
            worker = json.loads((Path(self.runs) / wid / "episode.json").read_text())
            self.assertIsNone(worker["reward"]["total"])

    def test_an_unmeasured_merge_scores_none_not_zero(self):
        # _score refuses to invent a number when nothing ran. Feeding marrow a
        # 0.0 here would be a failure that never happened.
        self.assertIsNone(orchestrate._score({}, "diff", 1.0, 60))

    def test_the_merged_tree_is_reviewed_when_a_review_role_is_configured(self):
        # No worker ever sees the merged tree, and it is what lands. Without
        # this, --orchestrate silently dropped the reviewer and a caller gating
        # on REJECT read None as "nobody objected" instead of "nobody looked".
        task = self._task("(orchestrated)", "test -f a.txt && test -f b.txt")
        subs = [Subtask("wa", "printf 'a\\n' > a.txt"),
                Subtask("wb", "printf 'b\\n' > b.txt")]
        blocker = ('{"findings":[{"severity":"blocker","file":"a.txt","line":1,'
                   '"claim":"lanes disagree"}]}')
        roles = [{"name": "review", "review": True, "agent": "shell",
                  "prompt": f"echo '{blocker}'"}]
        ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                              runs_dir=self.runs, manifest={}, roles=roles)
        self.assertEqual(ep["orchestration"]["path"], "B")
        self.assertEqual(ep["review_verdict"], "reject")
        self.assertEqual([f["claim"] for f in ep["review_findings"]],
                         ["lanes disagree"])

    def test_no_review_role_means_no_verdict_not_a_silent_approval(self):
        task = self._task("(orchestrated)", "test -f a.txt && test -f b.txt")
        subs = [Subtask("wa", "printf 'a\\n' > a.txt"),
                Subtask("wb", "printf 'b\\n' > b.txt")]
        ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                              runs_dir=self.runs, manifest={}, roles=None)
        self.assertIsNone(ep["review_verdict"])

    def test_a_path_b_episode_is_priced_from_its_workers(self):
        # usage {} reads as "unknown" to plexus, so an orchestrated feature spent
        # real money and landed in the ledger as free.
        task = self._task("(orchestrated)", "test -f a.txt && test -f b.txt")
        subs = [Subtask("wa", "printf 'a\\n' > a.txt"),
                Subtask("wb", "printf 'b\\n' > b.txt")]
        ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                              runs_dir=self.runs, manifest={})
        workers = [json.loads((Path(self.runs) / w / "episode.json").read_text())
                   for w in ep["orchestration"]["worker_episodes"]]
        for key in ("tokens_in", "tokens_out", "cost_usd"):
            vals = [(w.get("usage") or {}).get(key) for w in workers]
            numeric = [v for v in vals if isinstance(v, (int, float))]
            if numeric:
                self.assertEqual(ep["usage"][key], sum(numeric), key)
            else:
                # unpriceable stays absent -- a missing key means unknown, and
                # plexus reads a present zero as "this was free"
                self.assertNotIn(key, ep["usage"], key)

    def test_no_integration_verifier_forces_path_a(self):
        # Path B is unsafe without a seam-crossing test: a clean merge would be a
        # false green. With no verifier the split is refused before decompose.
        task = TaskSpec(task_id="orch", repo_path=self.repo, base_commit=self.head,
                        prompt="printf 'a\\n' > a.txt", timeout_seconds=60,
                        public_verifiers=[])
        subs = [Subtask("wa", "printf 'a\\n' > a.txt"),
                Subtask("wb", "printf 'b\\n' > b.txt")]
        ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                              runs_dir=self.runs, manifest={})
        self.assertEqual(ep["orchestration"]["path"], "A")
        self.assertEqual(ep["orchestration"]["reason"], "no integration verifier")

    def test_repair_tamper_guard_flags_test_deletions(self):
        from heart.orchestrate import _repair_tampers_with_tests
        # fixing prod code is fine
        self.assertFalse(_repair_tampers_with_tests(
            "--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-old\n+new\n"))
        # removing a line from a test file is not
        self.assertTrue(_repair_tampers_with_tests(
            "--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1 +0 @@\n-    assert f() == 1\n"))
        # adding a test assertion is fine
        self.assertFalse(_repair_tampers_with_tests(
            "--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1 +2 @@\n+    assert g() == 2\n"))

    def _spy_flush(self):
        """Replace _flush_subagent_memory with a recorder; returns (calls, restore).
        Isolates 'do memory-merge and verification conflict' from Postgres being up."""
        import heart.orchestrate as orch
        calls = []
        orig = orch._flush_subagent_memory
        orch._flush_subagent_memory = lambda parent, repo: calls.append(parent)
        return calls, (lambda: setattr(orch, "_flush_subagent_memory", orig))

    def test_memory_flush_and_verification_coexist_on_pass(self):
        # the full Path-B run does BOTH: compiles workers' memory up (flush) AND
        # runs the unified integration check — neither breaks the other.
        calls, restore = self._spy_flush()
        try:
            task = self._task("(orchestrated)", "test -f a.txt && test -f b.txt")
            subs = [Subtask("wa", "printf 'a\\n' > a.txt"),
                    Subtask("wb", "printf 'b\\n' > b.txt")]
            ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                                  runs_dir=self.runs, manifest={})
        finally:
            restore()
        self.assertEqual(calls, ["orch-orch"])  # flushed once, under the parent id
        self.assertEqual(ep["orchestration"]["integration"], "pass")
        self.assertEqual(ep["outcome"], "pass")

    def test_no_verifier_gate_creates_no_subagent_memory(self):
        # gate fires before decompose/workers -> no subagents spawn -> nothing to
        # flush. The two features agree: an unsafe-to-split task leaves no orphan
        # parent memory behind.
        calls, restore = self._spy_flush()
        try:
            task = TaskSpec(task_id="orch", repo_path=self.repo, base_commit=self.head,
                            prompt="printf 'a\\n' > a.txt", timeout_seconds=60,
                            public_verifiers=[])
            subs = [Subtask("wa", "printf 'a\\n' > a.txt"),
                    Subtask("wb", "printf 'b\\n' > b.txt")]
            ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                                  runs_dir=self.runs, manifest={})
        finally:
            restore()
        self.assertEqual(ep["orchestration"]["path"], "A")
        self.assertEqual(calls, [])

    def test_memory_flush_runs_even_on_fallback(self):
        # the flush happens before the merge, so a Path-B run that FAILS to merge
        # still compiled its workers' seam facts up first — exactly what makes a
        # failed integration troubleshootable. Verification (fallback) and memory
        # (flush) both did their jobs.
        calls, restore = self._spy_flush()
        try:
            # conflict + an unsatisfiable verifier: incremental re-merges the lanes
            # clean, but integration can't pass and shell repair can't fix it, so it
            # ultimately falls back to whole-task A.
            task = self._task("(orchestrated)", "test -f nope.txt")
            subs = [Subtask("wa", "printf 'aaa\\n' > x.txt"),
                    Subtask("wb", "printf 'bbb\\n' > x.txt")]  # same file -> conflict
            ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                                  runs_dir=self.runs, manifest={})
        finally:
            restore()
        self.assertEqual(ep["orchestration"]["path"], "B->A")
        self.assertTrue(calls and all(c == "orch-orch" for c in calls))

    def test_repair_runs_under_parent_for_memory_retrieval(self):
        # clean merge (disjoint files) but the verifier wants a file no worker made
        # -> integration fails -> repair fires. The repair agent must run under the
        # orchestration parent id with project env, so arteries retrieval surfaces
        # the workers' compiled seam facts. Workers run as orch-orch-sub-*, so the
        # exact "orch-orch" id isolates the repair call.
        import heart.orchestrate as orch
        captured = []
        orig = orch.run_agent

        def spy(*a, **k):
            env = k.get("extra_env", a[3] if len(a) > 3 else None)
            captured.append(dict(env) if isinstance(env, dict) else {})
            return orig(*a, **k)

        orch.run_agent = spy
        try:
            task = self._task("(orchestrated)", "test -f fixed.txt")  # never created
            subs = [Subtask("wa", "printf 'a\\n' > a.txt"),
                    Subtask("wb", "printf 'b\\n' > b.txt")]
            run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                             runs_dir=self.runs, manifest={})
        finally:
            orch.run_agent = orig
        repair_envs = [e for e in captured if e.get("ARTERIES_AGENT_ID") == "orch-orch"]
        self.assertTrue(repair_envs, "repair did not run under the parent id")
        self.assertEqual(repair_envs[0]["ARTERIES_AGENT_ROLE"], "parent")
        self.assertIn("ARTERIES_PROJECT", repair_envs[0])
        self.assertIn("ARTERIES_REPO", repair_envs[0])

    def test_conflict_recovered_by_incremental_retry(self):
        # both workers create the SAME new file -> second patch conflicts.
        # incremental keeps the first lane, re-runs the second on top (now a modify,
        # not a colliding create), re-merges clean -> stays Path B.
        task = self._task("(orchestrated)", "test -f x.txt")
        subs = [Subtask("wa", "printf 'aaa\\n' > x.txt"),
                Subtask("wb", "printf 'bbb\\n' > x.txt")]
        ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                              runs_dir=self.runs, manifest={})
        self.assertEqual(ep["orchestration"]["path"], "B")
        self.assertEqual(ep["orchestration"]["merge"], "incremental")
        self.assertEqual(ep["outcome"], "pass")

    def test_incremental_reruns_only_the_conflicting_lane(self):
        # wa/wb are disjoint (a.txt, b.txt); wc collides with wa on a.txt. Only wc
        # is re-run; the clean a.txt and b.txt lanes are kept and all three land.
        task = self._task("(orchestrated)", "test -f a.txt && test -f b.txt")
        subs = [Subtask("wa", "printf 'a\\n' > a.txt"),
                Subtask("wb", "printf 'b\\n' > b.txt"),
                Subtask("wc", "printf 'ccc\\n' > a.txt")]  # collides with wa
        ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                              runs_dir=self.runs, manifest={})
        self.assertEqual(ep["orchestration"]["path"], "B")
        self.assertEqual(ep["orchestration"]["merge"], "incremental")
        self.assertEqual(ep["outcome"], "pass")
        diff = (Path(self.runs) / ep["episode_id"] / "diff.patch").read_text()
        self.assertIn("a.txt", diff)
        self.assertIn("b.txt", diff)

    def test_waves_group_by_dependency(self):
        # a and c are unordered relative to each other -> same wave; b waits.
        subs = [Subtask("a", "x"), Subtask("b", "y", depends_on=["a"]), Subtask("c", "z")]
        self.assertEqual([[s.name for s in w] for w in orchestrate._waves(subs)],
                         [["a", "c"], ["b"]])

    def test_waves_reject_a_cycle(self):
        subs = [Subtask("a", "x", depends_on=["b"]), Subtask("b", "y", depends_on=["a"])]
        with self.assertRaises(graphlib.CycleError):
            orchestrate._waves(subs)

    def test_waves_reject_an_edge_to_a_name_thats_not_in_the_plan(self):
        with self.assertRaises(ValueError):
            orchestrate._waves([Subtask("a", "x", depends_on=["ghost"]),
                                Subtask("b", "y")])

    def test_a_wave_mixing_scoped_and_unscoped_subtasks_is_rejected(self):
        # The unscoped one inherits the task's scope -- unrestricted for
        # `heart work` and plexus -- so it can write over the lanes beside it
        # while both the mount table and the diff scan call that allowed.
        waves = [[Subtask("scoped", "x", allowed_paths=["src/a.py"]),
                  Subtask("unscoped", "y")]]
        with self.assertRaises(ValueError) as ctx:
            orchestrate._check_lanes(waves)
        self.assertIn("unscoped", str(ctx.exception))

    def test_a_wave_where_nobody_declares_a_lane_stays_legal(self):
        # Not deceptive: no lane claims a boundary, and the merge still has to
        # come out clean. Refusing this would break every lane-free decomposer.
        orchestrate._check_lanes([[Subtask("a", "x"), Subtask("b", "y")]])

    def test_lanes_in_different_waves_may_differ_in_scoping(self):
        # Waves are sequential, so a later subtask cannot race an earlier one.
        orchestrate._check_lanes([[Subtask("a", "x", allowed_paths=["src/a.py"])],
                                  [Subtask("b", "y")]])

    def test_mixed_lanes_fall_back_to_path_a(self):
        task = self._task("printf 'a\\n' > a.txt", "test -f a.txt")
        subs = [Subtask("wa", "printf 'a\\n' > a.txt", allowed_paths=["a.txt"]),
                Subtask("wb", "printf 'b\\n' > b.txt")]
        ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                              runs_dir=self.runs, manifest={})
        self.assertEqual(ep["orchestration"]["path"], "A")
        self.assertIn("declare no lane", ep["orchestration"]["reason"])

    def test_off_vocabulary_skills_are_named_not_silently_dropped(self):
        # route.classify() drops them and falls back to ["coding"], which is the
        # right runtime behaviour and the wrong thing to stay quiet about.
        from heart import route
        subs = orchestrate._subtasks_from_list([
            {"name": "a", "prompt": "p", "skills": ["coding"]},
            {"name": "b", "prompt": "p", "skills": ["testing", "documentation"]}])
        unknown = sorted({k for s in subs for k in s.skills if k not in route.SKILLS})
        self.assertEqual(unknown, ["documentation", "testing"])
        self.assertIn("docs", route.SKILLS)   # the word the prompt now tells it to use

    def test_a_worker_that_raises_falls_back_instead_of_killing_the_run(self):
        # A crashing worker is heart failing, not the agent failing. It used to
        # escape run_orchestrated through pool.map and take the command with it.
        task = self._task("printf 'a\\n' > a.txt", "test -f a.txt")
        subs = [Subtask("wa", "printf 'a\\n' > a.txt"),
                Subtask("wb", "printf 'b\\n' > b.txt")]
        boom = unittest.mock.patch.object(
            orchestrate, "_run_workers", side_effect=RuntimeError("git add failed"))
        with boom:
            ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                                  runs_dir=self.runs, manifest={})
        self.assertEqual(ep["orchestration"]["path"], "B->A")
        self.assertIn("worker_failed", ep["orchestration"]["reason"])
        self.assertEqual(ep["outcome"], "pass")

    def test_invalid_graph_falls_back_to_path_a(self):
        # a bad plan is not a bad task: build it sequentially rather than failing.
        task = self._task("printf 'a\\n' > a.txt", "test -f a.txt")
        subs = [Subtask("wa", "x", depends_on=["wb"]),
                Subtask("wb", "y", depends_on=["wa"])]
        ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                              runs_dir=self.runs, manifest={})
        self.assertEqual(ep["orchestration"]["path"], "A")
        self.assertIn("invalid subtask graph", ep["orchestration"]["reason"])

    def test_second_wave_builds_on_the_first_ones_committed_work(self):
        # wb can only produce b.txt by reading a.txt, which exists solely because
        # wave 1 was committed and became wave 2's base. If the base never
        # advanced, cat finds nothing, b.txt is empty, and integration fails.
        task = self._task("(orchestrated)", "test -f a.txt && grep -q hello b.txt")
        subs = [Subtask("wa", "printf 'hello\\n' > a.txt"),
                Subtask("wb", "cat a.txt > b.txt", depends_on=["wa"])]
        ep = run_orchestrated(task, agent="shell", decomposer=lambda _t: subs,
                              runs_dir=self.runs, manifest={})
        self.assertEqual(ep["orchestration"]["path"], "B")
        self.assertEqual(ep["orchestration"]["merge"], "waves:2")
        self.assertEqual(ep["outcome"], "pass")
        diff = (Path(self.runs) / ep["episode_id"] / "diff.patch").read_text()
        self.assertIn("a.txt", diff)
        self.assertIn("b.txt", diff)

    def test_only_a_dependent_worker_is_told_its_upstream_landed(self):
        # a same-wave sibling must not be told to go read code it cannot see.
        task = self._task("(orchestrated)", "true")
        first = orchestrate._worker_taskspec(task, Subtask("wa", "build a"), "medium")
        later = orchestrate._worker_taskspec(
            task, Subtask("wb", "build b", depends_on=["wa"]), "medium")
        self.assertEqual(first.prompt, "build a")
        self.assertIn("wa", later.prompt)
        self.assertIn("build b", later.prompt)


if __name__ == "__main__":
    unittest.main()


class ReviewAgentHarnessTests(unittest.TestCase):
    """`shell` is the test harness, not a model."""

    def test_shell_is_never_rotated_to_a_real_cli(self):
        """Rotating it made every role-pipeline test spawn a billed CLI locally
        and fail on CI, where the binary does not exist."""
        with unittest.mock.patch.object(orchestrate.router_mod, "review_pool",
                                        return_value=["claude", "codex"]):
            self.assertEqual(orchestrate.router_mod.review_agent("shell"), "shell")

    def test_a_real_agent_still_rotates_family(self):
        with unittest.mock.patch.object(orchestrate.router_mod, "review_pool",
                                        return_value=["claude:opus", "codex"]):
            self.assertEqual(orchestrate.router_mod.review_agent("claude:sonnet"), "codex")
