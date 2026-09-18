# heart cleanup: import placement and duplication

Branch `dev`. Two commits: the import audit, then the dedup.

## Counts

| | |
|---|---|
| function-local imports audited | 23 |
| hoisted to module top | 21 |
| kept local | 2 |
| comments added to KEEPs | 2 (both KEEPs had none) |
| lines removed by dedup | 88 removed / 43 added across 8 files |

Two of the 21 hoists were deletions rather than moves: `runner.py` imported
`json as _json` inside a function that sits in a module already importing
`json` at top, and imported `urllib.request` in two separate functions.

## KEEP categories

| category | count | files | reason |
|---|---|---|---|
| breaks an intra-package cycle | 1 | `src/heart/cli.py:370` | `serve.py:20` does `from .cli import WORK_RUNS_DIR` at module scope, so a top-level `from . import serve` in cli deadlocks the import |
| undeclared third-party | 1 | `src/heart/episode.py:629` | `arteries` is not in `pyproject.toml` and heart declares no dependencies at all; hoisting turns an optional fallback into a hard install failure |

Everything else was stdlib or an intra-heart relative import with no cycle and
no stated reason to be deferred. The package import graph has exactly one
cycle (`cli` <-> `serve`); nothing else in `src/heart` needed laziness.

## UNDECLARED DEPENDENCIES

heart's `pyproject.toml` has no `[project.dependencies]` at all. Every
non-stdlib import is therefore undeclared by construction. There is exactly
one in the runtime package:

| module | file:line | path | verdict |
|---|---|---|---|
| `arteries` | `src/heart/episode.py:629` (`from arteries.subagent import subagent_env`) | optional — inside `try/except Exception` with an inline fallback that keeps the identity contract working | correct as-is. Must stay lazy. |

Two more references to arteries exist but are not imports, so they cannot
break an install:

- `src/heart/sandbox.py:779` shells out to `python3 -m arteries.cli journal inbox`.
  This one is **not** optional: it raises `RuntimeError` when arteries does not
  answer, and it is on the hot path for every `HEART_SANDBOX=docker` episode.
  Deliberate (the docstring says so: a requested sandbox must fail loudly), but
  it means docker-mode heart has a hard runtime dependency on arteries that no
  packaging metadata records.
- Test-only: `pytest` (`tests/conftest.py:13`, `tests/test_sandbox.py:13`) and
  `arteries` (5 sites under `tests/`). There is no dev extra or test extra in
  `pyproject.toml`, so CI has to know to install these out of band.

Verified: heart imports end to end with `arteries` blocked from `sys.meta_path`.

## Within-repo duplication

### Extracted

**1. The models.json path, nine copies.** The literal
`Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "heart" / "models.json"`
appeared nine times in five modules — twice already wrapped in a named function
(`runner.models_json_path`, `route._config_path`) and seven times inline.
`agents_api` now owns `models_json_path()` (`src/heart/agents_api.py:49`) and
`load_models_json()` (`:54`); it is the lowest module in the graph and imports
nothing first-party, so `runner`, `router` and `route` can all reach it without
a cycle. `runner` re-exports both names because `cli` reads the rate card
through `runner`.

Removed with it: five near-identical `try: json.loads(path.read_text()) except
(OSError, json.JSONDecodeError)` blocks in `agents_api`, `router` and `runner`.
`agents_api.endpoint_for` and `runner._resolve_model` now just call
`profile_config()`, which is what they were open-coding.
`agents_api.resolve_config` keeps its own strict read — it is the one caller
that wants the exception text for its `sys.exit` message.

**2. `orchestrate._plans_from` re-scanning JSON out of an agent log**
(`src/heart/orchestrate.py:187`). `review._json_objects`
(`src/heart/review.py:105`) is the same `JSONDecoder().raw_decode` walk, and
its docstring already said "same lesson as the decomposer's parser". They are
now one generator; `orchestrate` already imported `review` at module scope.
13 lines gone.

**3. `env._is_live` re-implementing `env._lock_is_held`**
(`src/heart/env.py:106`). The two bodies were byte-identical `flock` probes;
`_lock_is_held`'s own docstring said "same test". `_is_live` is now
`lock.exists() and _lock_is_held(lock)`. One behaviour change: a non-`OSError`
exception from the probe now reads as "not held" instead of propagating, which
is the decision `_lock_is_held` had already made.

**4. `orchestrate._commit_tree` re-implementing `Workspace.commit`**
(`src/heart/orchestrate.py:592`). See the bug note below — this one was not
just duplication.

### Deliberately left

- **`episode._assess` / `orchestrate._assess`.** Same name, same two-line
  shape, different bodies: one goes through `_agent_turn` with role memory and
  a sandbox profile, the other calls `run_agent` directly. Similar shape, not
  the same logic.
- **`route.classify` / `router.classify`.** Unrelated functions that collided
  on a name — one returns skills+tier+difficulty, the other tier+manifest.
- **Per-call-site `json.loads(path.read_text())` on files that are not
  models.json** (taskspec, detect, export, cli roles). Each reads a different
  file with a different failure policy; a shared loader would need a policy
  argument, which is the abstraction this pass is supposed to avoid.
- **`verify.py:203` / `verify.py:209` output-tail formatting.** Two lines, two
  different truncation widths, two different message shapes. Extracting saves
  nothing.

## Cross-repo duplication candidates

### (i) heart <-> plexus / marrow — has a home post-monorepo

Both repos already import heart directly, so anything here can move into heart
today, without waiting for a monorepo.

| # | logic | files | ~lines | worth it? |
|---|---|---|---|---|
| 1 | Reading local model endpoints out of `models.json` and deciding which are loopback. `plexus.sandbox.local_model_hosts` opens heart's `models.json` by hand and re-implements the loopback test (`hostname in ("127.0.0.1", "localhost", "::1", "0.0.0.0")`) that heart already exports as `agents_api.is_local_endpoint`. | `plexus/src/plexus/sandbox.py:50-69` vs `heart/src/heart/agents_api.py:49` (`models_json_path`/`load_models_json`) and `is_local_endpoint` | ~14 | **Yes.** plexus already imports heart elsewhere in the same file's neighbourhood (`registry.heart_model_rates` does exactly this correctly). The duplicate loopback list is the risk: heart's version also accepts private LAN ranges, plexus's does not, so the two disagree about what "local" means on a box with a model server on the LAN. Small change, removes a real semantic fork. |
| 2 | The diff-size reward curve. `marrow.reward.score_patch` hard-codes `1.0 if changed <= 50 else max(0.0, 1.0 - (changed - 50) / 450)` — character for character what `heart.reward.compute` uses for `diff_quality`. | `marrow/marrow/reward.py:76-77` vs `heart/src/heart/reward.py:56-57` | 2 | **Yes, despite being two lines.** This is the number the trainer optimises against and the number the runtime scores with. If they drift, GRPO learns to satisfy a reward heart no longer pays. marrow already imports `heart.reward.diff_changed_lines` from the module next door, so exporting the curve as `heart.reward.diff_quality(diff_text)` is a one-line addition and a one-line deletion. |
| 3 | Knowing how to invoke the `claude` and `codex` CLIs. `marrow.collect.build_command` builds `["codex", "exec", "--json", ...]` and `["claude", "-p", prompt, "--output-format", "stream-json", ...]`; `heart.runner.AGENT_COMMANDS` + `_agent_command` build the same two command lines with the same `--json`/`--output-format` reasoning, and carry ~30 lines of hard-won comments about why each flag is there. | `marrow/marrow/collect.py:32-47` vs `heart/src/heart/runner.py:32-50` and `_agent_command` | ~16 | **Marginal.** The flag sets genuinely differ — collect wants `stream-json` plus debug tracing for capture, heart wants a single JSON result for scoring. What is actually shared is smaller: the agent-name allowlist and the `--model` pinning rule. I would export only `heart.runner.resolve_model` and leave the command tables separate. Do not force these into one builder. |
| 4 | Knowledge of the `runs/<episode_id>/episode.json` record shape. `plexus.export._episode_facts` reads `outcome`, `reward.total`, `diff_lines`, `agent`, `blocked_reason` out of the file `heart.episode` writes, by string key, with no shared definition. | `plexus/src/plexus/export.py:29-45` vs `heart/src/heart/episode.py:1090` | ~15 | **Not as a dedup, as a contract.** The logic is not duplicated; the schema is. A `heart.export.load_episode(path) -> dict` (or a documented key list) would give the join one place to break loudly instead of silently returning `{}`. Low priority, low risk. |

Worth recording as the counter-example: `plexus/src/plexus/events.py` and
`plexus/src/plexus/registry.py:279` already do this right — they import
`heart.events.emit` and `heart.runner.model_pricing` rather than copying, and
say why in the docstring. The pattern exists; items 1-3 are the places it was
not applied.

### (ii) heart <-> capillaries / arteries — would need a new shared package

heart does not depend on either, and neither can be made a dependency of heart
without inverting the stack. Anything here has no legal home short of a new
shared package, so the bar is much higher.

| # | logic | files | ~lines | worth it? |
|---|---|---|---|---|
| 5 | The event-spine emitter. `capillaries.spine.emit` is a hand-copied `heart.events.emit`: same `EVENT_JOURNAL_DIR` default, same `~/.local/share/heart/events`, same `%Y%m%d.ndjson` filename, same swallow-everything contract, same closing comment ("observability must never take down the observed"). | `capillaries/src/capillaries/spine.py:12-24` vs `heart/src/heart/events.py:25-64` | ~15 | **No — leave it.** The file says so itself: "No shared library by design — mirrors heart/events.py's emit() in ~15 lines of stdlib." The duplicated surface is the journal *format*, which is specified in `heart/SPINE.md`, and a spec is the right shared artifact here rather than a package. Fifteen stdlib lines is cheaper than a fifth repo. If a shared package ever exists for other reasons, this is the first thing to move into it. |
| 6 | Nothing else. A scan of `capillaries/src` and `arteries/src` for heart's helper shapes (`JSONDecoder.raw_decode` log scanning, `git -C` worktree management, `XDG_CONFIG_HOME` config paths, `flock` liveness probes) found no hits. | — | — | — |

## Suspected bugs

Reported, not fixed — except #1, which the dedup removed as a side effect.

1. **`orchestrate._commit_tree` staged excluded paths into the retry base.**
   It ran a bare `git add -A` + `git commit` in a `Workspace` instead of
   calling `Workspace.commit`, which exists precisely to `git reset` the
   `EXCLUDED_PATHS` (`.env` and friends) and the overlay files back out before
   committing. The sha it returned is handed to the retried Path-B workers as
   their new `base_commit`, so anything `git add -A` swept up became part of
   the tree every retried subtask built on. It also signed as
   `heart@local` where every other commit in the codebase signs as
   `heart@localhost`. Fixed as part of the dedup
   (`src/heart/orchestrate.py:592`), since the fix *is* calling the existing
   method. One behaviour note: `Workspace.commit` returns `None` when nothing
   staged, where the inline version returned the base HEAD sha, so the new code
   falls back to `base_commit` explicitly to preserve the caller's contract.

2. **`sandbox.image_is_stale` compares the Dockerfile's mtime, not its
   content or its git history** (`src/heart/sandbox.py:277`). The docstring
   describes diagnosing the original problem with `git log -1 Dockerfile`, and
   argues for timestamps over a content hash — but `stat().st_mtime` is the
   *checkout* time, not the commit time. On a fresh clone or a CI checkout
   every file gets today's mtime, so a correctly-built image will read as
   "older than its Dockerfile" and `sandbox_wrap` will `raise RuntimeError` on
   every sandboxed episode. `git log -1 --format=%ct -- Dockerfile` with an
   mtime fallback would match what the docstring already claims.

3. **`route.refresh_stats` used to swallow an import error.** The
   `from .pulse import load_events` was inside the same `try:` as the
   aggregation, so a broken `pulse` module silently produced empty routing
   stats rather than an error. Hoisting it (`src/heart/route.py`) makes that
   failure loud. Behaviour change, deliberate, and the same class of bug
   `orchestrate.py`'s `review_failed` handler has a comment about ("this
   swallowed a NameError for a missing import once").

4. **No test or dev extra in `pyproject.toml`.** `tests/` needs `pytest` and
   five test modules need `arteries`, and nothing in the packaging metadata
   says so. Not a runtime bug; it is why the stdlib-only claim has to be
   checked by hand rather than by installing the package.

## Verification

| check | before | after |
|---|---|---|
| `PYTHONPATH=src python3 -m pytest -q` | 293 passed, 1 skipped, 38 subtests | 293 passed, 1 skipped, 38 subtests |
| full `pkgutil.walk_packages` import of every `heart.*` module | clean | clean |
| `import heart; import heart.cli` with `arteries` blocked at `sys.meta_path` | n/a | passes |
