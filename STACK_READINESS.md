# Stack readiness: integration testing + the five feature gaps

Working plan for taking plexus/heart/arteries/capillaries from "each repo works"
to "the factory works," plus ownership and implementation detail for the five
feature gaps: cost tracking, sandboxing, orchestration hardening, DSPy
self-improvement, and full observability. Marrow is out of scope for now by
decision; every section notes what it will consume later so nothing here has to
be redone.

Written 2026-07-20. Each item ends with a **Done when** line — that is the
acceptance gate, not the commit message.

---

## 0. Ownership doctrine

One rule prevents most redundancy: **a feature lives in the repo whose question
it answers.**

| Question | Owner | Consequence |
| --- | --- | --- |
| What should be built next; is the scope satisfied? | plexus | goals, plans, acceptance, escalation, retry budgets |
| Was this task done correctly, and how do agents run? | heart | worktrees, roles, verifiers, sandbox, routing, reward, cost *capture* |
| What happened, and what do we remember? | arteries | ledger (episodes/decisions/rewards/costs), memory tiers, turn observation |
| Which prompt/skill fits, and is it getting better? | capillaries | retrieval, skill lifecycle, DSPy optimization, promotion gates |
| Read side of everything | heart `pulse` | insights, health, dashboard — readers only, no state of their own |

Two deliberate near-duplications that are **not** redundancies — do not merge:

- **plexus acceptance vs heart review/verify.** Heart judges the task with its
  own verifiers; plexus judges the goal against ground truth. The
  "heart passed / acceptance failed" cell is a hard negative marrow can't
  otherwise see (plexus LEDGER law 5). Keep both.
- **capillaries feedback vs arteries rewards.** `capillaries_feedback` is a
  local relevance signal ("this prompt fit"); arteries rewards are episode
  outcomes. §5 joins them by `episode_id`; neither replaces the other.

One true redundancy to remove when touched: capillaries' `llm_judge` metric as
the *default* optimizer signal — once §5 grounds metrics in episode rewards,
the judge becomes the fallback for prompts with no episode traffic, not the
primary.

---

## 1. Cross-stack integration testing

### 1.1 Seam inventory (what can actually break)

| Seam | Contract | Known risk |
| --- | --- | --- |
| plexus → heart | Python import (`heart.episode`, `heart.taskspec`, `heart.env`) | heart API drift breaks plexus silently — no pinning, same-machine source checkouts |
| heart → arteries | env (`ARTERIES_PROJECT/REPO/MEMORY/EPHEMERAL`, `HEART_EPISODE_ID`) + `art ingest` subprocess + `.arteries/hooks/observe.sh` | env leakage between roles; ingest schema drift; hooks absent in a fresh worktree |
| arteries → capillaries | Python import (retrieval gate) + shared Postgres | capillaries import failure takes down arteries eval (seen: `No module named 'capillaries'`) |
| everyone → spine | NDJSON spool, SPINE.md contract, additive-only fields | a malformed emitter corrupts nothing (tolerant readers) but silently drops observability |
| everyone → Postgres | arteries owns `episodes/decisions/rewards`; capillaries owns its retrieval + skills + optimize tables | schema ownership is clean today; keep migrations in the owning repo only |

### 1.2 Contract tests (per seam, cheap, run in each repo's suite)

1. **Spine conformance (heart owns).** `tests/golden-events/` in heart holds one
   real sample event per kind per source (heart, arteries, capillaries, plexus,
   marrow-stub). A heart test asserts: parseable JSON, `ts/source/kind`
   present, `episode_id` present where SPINE.md requires it, UTC isoformat
   timestamps. When any repo adds an event kind, it adds a golden sample here.
   This is the shared-contract test that replaces the shared library we
   deliberately don't have.
2. **plexus → heart API pin (plexus owns).** A plexus test imports exactly the
   heart symbols it uses (`TaskSpec`, `run_candidates`, `best_episode`,
   `Workspace`, `detect_verifiers`) and calls each with the signature it
   depends on against a stub task. Heart refactors then fail plexus's suite
   loudly instead of at goal-run time.
3. **Env-propagation test (heart owns, exists partially).** Assert the full
   env contract a role subprocess receives: `HEART_ROLE`, `HEART_EPISODE_ID`,
   `ARTERIES_*` — and assert what it must NOT receive (ambient `HEART_TIER_*`,
   parallel candidates forcing `ARTERIES_EPHEMERAL=discard`). The rehearsal
   bug (tier env leaking into verifiers) becomes a pinned regression test.
4. **Ingest round-trip (arteries owns, extend).** `heart export` JSONL of a toy
   run → `art ingest` → assert ledger rows, then ingest twice → assert dedup.
   Extend with cost columns when §2 lands.
5. **Degradation drills (arteries owns).** Stop Postgres (or point at a dead
   port): arteries must fall back to JSONL with `store=jsonl` on the spine,
   `pulse health` must go non-zero, and `art ingest` later reconciles. The
   fallback path only counts as existing once a test kills the DB.

### 1.3 The full-stack smoke: `stack-smoke.sh` (heart owns, new)

One script, run before anything is declared integrated. Uses a throwaway git
repo + a trivial plexus goal ("make this function return 4") so it costs one
cheap-tier episode:

```
plexus run goal.md            # decompose → dispatch → accept
  └─ heart episode (auto-routed, sandboxed once AppArmor is on)
       └─ arteries observes turns + decisions (store=db)
       └─ capillaries gate fires (search or abstain — either, but decided)
art ingest                     # rewards land
heart pulse insights --hours 1 # scorecard shows the episode
heart pulse health             # exit 0
```

Assertions: goal reaches `satisfied` in the plexus ledger; one `episode_id`
appears in **all four** places (plexus ledger, heart runs dir, arteries
rewards, spine events); no `store=jsonl`; health exit 0. That single
"same id in four places" check is the integration test — everything else is
detail.

**Conflict audit (run once, then guarded by the smoke):** env prefixes are
already disjoint (`HEART_/ARTERIES_/CAPILLARIES_/PLEXUS_`); ports must be
declared in one place — adopt: capillaries HTTP 8100, vLLM 8000, pulse serve
7717, document in each README; Postgres schemas per owner (§1.1); spool is
append-only per-source files so concurrent writers are safe; config files
(`~/.config/heart/models.json`, capillaries config, `.arteries/`) never shared.

**Done when:** `stack-smoke.sh` passes on this machine from a cold shell, and
each repo's suite carries its contract tests from §1.2.

---## 2. Token & cost tracking

**Owner: heart captures, arteries stores, pulse renders.** Nobody else touches
money.

### 2.1 Capture (heart)

- `runner.run_agent` learns to extract usage after the subprocess exits, per
  agent family:
  - **api agent**: the OpenAI-compatible response carries `usage` — already in
    hand, just plumb it out (easiest; do first).
  - **claude CLI**: run with `--output-format json` in a capture-friendly mode
    or parse the final `usage` block from stream output; tokens in/out + model.
  - **codex/gemini/opencode**: parse what each exposes; where a CLI exposes
    nothing, record `tokens=null` honestly rather than estimating.
- Pricing map lives in `~/.config/heart/models.json` next to the tiers:
  `"pricing": {"<profile-or-model>": {"in_per_mtok": 3.0, "out_per_mtok": 15.0}}`.
  Subscription CLIs get `null` pricing (quota, not dollars) — cost shows as
  tokens only. This encodes the existing pricing-mix stance directly.
- Emit on the spine: `role.finished` payload gains
  `tokens_in, tokens_out, cost_usd` (additive — SPINE.md-legal). Episode total
  goes into `episode.finished` payload and `episode.json`.

### 2.2 Store (arteries)

- `rewards` table (or a sibling `costs` table if migrations are cleaner) gains
  `tokens_in, tokens_out, cost_usd`; `art ingest` reads them from
  `episode.json`. Dedup semantics unchanged.

### 2.3 Render (pulse)

- `insights`: `cost: total=$X.XX  per-pass=$Y.YY` and per-tier cost appended to
  the routing scorecard — `cheap=8/8 pass $0.04/ep, standard=1/2 pass $0.31/ep`.
  That line is the routing policy's actual objective function, visible.
- Dashboard: reward chip on the episode card gains cost; insights panel gets
  the cost lines for free.

**Tests:** unit — usage-parse per agent family from canned outputs; pricing
math; null-pricing path. Integration — one api-agent episode against a stub
endpoint returning a fixed `usage`, assert cost flows spine → episode.json →
ingest → insights.

**Done when:** a meridian batch prints a per-tier cost line in insights and the
same totals sit in Postgres.

---

## 3. Sandboxing as the default, not the demo

**Owner: heart.** Everything heart spawns goes through `sandbox_wrap`; the only
question per subprocess type is which mode.

### 3.1 Activation (user, once)

AppArmor profile for bwrap userns (already prepared) — until it's loaded,
everything below stays skip-not-fail.

### 3.2 Policy matrix (code)

| Subprocess | Mode | Why |
| --- | --- | --- |
| agent roles (implement/test/review/fix) | `bwrap` | needs API egress; fs contained |
| verifiers | `bwrap-nonet` | tests have no business on the network; kills exfil-via-test and flaky-network tests in one move |
| `art ingest`, git plumbing | none | trusted local tooling touching real state on purpose |

Implementation: `HEART_SANDBOX` stays the master switch; when it's `bwrap`,
verifier subprocesses (`verify.py` / episode verify calls) upgrade themselves
to `bwrap-nonet` automatically. One env var, no per-role config until a real
need appears.

### 3.3 Testing it ("full sandboxing for each feature to implement and test with")

- The existing containment test (stray write to `$HOME` fails) un-skips.
- Add the negative-space tests: verifier under `bwrap-nonet` cannot `curl`;
  agent under `bwrap` cannot read `~/.ssh`; worktree and `/tmp` remain
  writable; a `git commit` inside the worktree still works (bwrap keeps uid).
- Acceptance: re-run the 10-task meridian shakedown with `HEART_SANDBOX=bwrap`
  — same pass rate as unsandboxed. Bun installing under bwrap is the likely
  friction point (cache dir binds); the shakedown finds it.
- **Every feature in this document is developed and tested with the sandbox
  on** once activated — sandbox-off becomes the exception you type, not the
  default you forget.

**Done when:** shakedown passes sandboxed at parity, and the policy matrix
above is enforced by tests, not convention.

---

## 4. Orchestration hardening: lint/type/guardrails/worktrees/swarms

**Owner: heart** (mechanism). Plexus decides *when* to spend more; heart
provides the knobs.

### 4.1 Static verifiers (lint + type-check)

- `detect.py` grows static checks alongside test detection: `ruff check` /
  `mypy` (pyproject presence), `tsc --noEmit` (tsconfig), `eslint`/`biome`
  (config presence). Emitted as verifiers named `static:*`.
- Scoring: static failures gate at a small reward weight — they must not
  drown the test signal, but a diff that fails lint/type never scores full
  marks and `--apply` warns. Threshold, not tyranny.
- Runs `bwrap-nonet` like all verifiers.

### 4.2 Guardrails on the diff (pre-apply, pre-reward)

All in one place — `reward.py`/apply path already zero-rewards denied paths:

- **Secrets scan**: regex set (AWS keys, private key headers, `ghp_`, generic
  high-entropy assignment) over the diff; hit → zero reward + `guardrail.hit`
  spine event + never apply. ~20 lines, no dependency.
- **Size fuse**: diff > N lines (default 2000) → flag, require explicit
  `--allow-large` to apply. Catches runaway rewrites and vendored junk.
- **Protected paths**: `.github/workflows`, lockfiles unless task mentions
  deps — extend the existing denied-paths mechanism, config per TaskSpec.

### 4.3 Worktrees

Already the isolation unit. Two gaps: `heart clean` (GC stale worktrees +
runs older than N days) and a crash test asserting a killed episode leaves no
locked worktree behind. Nothing else — resist worktree pooling until slowness
is measured.

### 4.4 Swarms — the narrow definition

Not free-form multi-agent chatter (surveyed and rejected already). A swarm is:
**best-of-N candidates, heterogeneous agents, one judge.**

- `--candidates N` exists; add per-candidate agent assignment
  (`--swarm claude,codex,api:qwen` → N candidates, one per listed agent).
- Judge = the existing review role run once over the top-2 diffs by reward,
  picking the winner; `best_episode` already breaks ties, the judge only runs
  when rewards are within epsilon.
- **When**: not a default. Plexus escalation is the trigger — a feature that
  failed its retry budget re-dispatches as a swarm before reaching the human
  queue. That's plexus deciding intent, heart providing mechanism — the
  boundary holds.

**Tests:** static-verifier detection per fixture repo; secrets scan corpus
(true hits + benign lookalikes — test fixtures with fake keys); size fuse;
swarm returns the judge's pick and emits `swarm.judged`.

**Done when:** a `heart work --swarm` run on a real repo picks a winner with
the judge's reasoning in the runs dir, and a planted fake AWS key zero-rewards
the episode.

---

## 5. Self-improving prompts & skills (DSPy) — capillaries

**Owner: capillaries** — `optimize/` (capture, metrics, dspy_optimize, resolve)
and `skills/promote.py` already exist; this grounds them in real outcomes and
handles the text/code split. DSPy stays a capillaries dependency; heart stays
stdlib.

### 5.1 Ground the metric in episode rewards

The missing piece is the join: capillaries logs `prompt.gate.decided` /
selection decisions with `episode_id`; arteries holds `rewards` by
`episode_id`. New `optimize/harvest.py`:

- Query: for each prompt/skill served into an episode, pair (query, served
  content) with that episode's reward. Feed through the existing
  `ExampleCapture.capture_external` path.
- Metric precedence: episode-reward-weighted > `capillaries_feedback` signal >
  `llm_judge` (fallback only, per §0). Contrastive pairs (same query, served
  A rewarded high / B low) use the existing `capture_contrastive`.
- Prereq from Phase 3: **top-k candidate logging** — without the candidates
  that *weren't* served, the optimizer can't learn ranking. Implement it now
  (log candidate ids + scores in the gate decision payload, additive field).

### 5.2 The text/code split for skills

Skills are text + code and only text is DSPy-optimizable:

| Component | Treatment |
| --- | --- |
| routing description | DSPy-optimized (retrieval hit-rate metric from harvest) |
| procedure prose / step instructions | DSPy-optimized (episode-reward metric) |
| embedded commands / code blocks | **never touched by DSPy** — code is verified by execution: run the skill's own check (or a heart episode exercising it) after any edit |
| frontmatter/schema | validated, not optimized |

Concretely: skill markdown gets fenced sections; the optimizer edits only
prose sections, a validator asserts code fences byte-identical pre/post
optimization, and any code change goes through a human or a heart episode with
verifiers — not through a prompt optimizer.

### 5.3 Promotion gate (A/B before replace)

`skills/promote.py` extends to: candidate version runs shadow (served to X% or
to a replay of the last N harvested queries), compare episode-reward /
feedback deltas, promote only on a win, keep every version (rollback is a row
flip). Emit `skill.promoted` / `skill.rejected` on the spine so pulse shows
optimizer activity.

**Tests:** harvest join on fixture ledger rows; code-fence-immutability
(optimizer output with a mutated fence → hard fail); promotion gate with a
rigged winner and a rigged loser; one end-to-end offline `dspy_optimize` run
on captured examples with a stub LM.

**Done when:** one real prompt improves its harvested-reward metric through
the full loop (harvest → optimize → A/B → promote) with code fences untouched,
visible as `skill.promoted` on the dashboard.

---

## 6. Full observability: metrics, alerts, tracing

**Owner: emitters stay per-repo; heart `pulse` owns the read side.** v0
dashboard shipped (`pulse serve`); this section is the delta to "I can see and
be told."

### 6.1 Tracing the full causal chain

The chain is goal → feature → task → episode → decisions/rewards. Episode→down
already correlates. The gap is plexus→down:

- plexus sets `PLEXUS_GOAL_ID` / `PLEXUS_FEATURE_ID` env when dispatching and
  stamps both into the TaskSpec (or task_id prefix); heart copies them into
  every event it emits for that episode (additive `payload` fields — one dict
  merge in `events.emit` via env, mirroring how arteries stamps episode ids).
- `pulse episode <id>` then shows the goal lineage; a `pulse goal <goal-id>`
  view lists its features → episodes → outcomes (a groupby over existing
  events, ~30 lines).
- Contract test in §1.2's golden events: plexus-dispatched episode events
  carry goal/feature ids.

### 6.2 Alerts (symptoms page someone)

`pulse health` already computes symptoms with an exit code — alerting is
delivery, not detection:

- systemd **user** timer (no sudo), every 10 min:
  `heart pulse health --hours 1 || notify-send + ntfy.sh push`. One unit file
  + one shell line; ships in heart's `contrib/` with an install one-liner.
- Health rules to add: episode cost spike (needs §2), zero events during an
  active plexus goal (the "factory silently stalled" case), review2 reject
  streak (the §0/heart review ceiling's tripwire).

### 6.3 Metrics

`pulse insights` is the metrics surface; resist Prometheus/Grafana until
something external needs to scrape. Additions, all from existing events +
§2: cost lines, per-goal progress (features done/total from plexus events),
optimizer activity (promotions/rejections). If scraping is ever needed, a
`/metrics` textfile endpoint on `pulse serve` is an afternoon — note it, don't
build it.

### 6.4 Dashboard v1 (from the visual plan, now with owners)

1. steering + needs-input (heart: steer file between roles; UI text box)
2. episode drill-down page (pulse: timeline + logs + rendered diff)
3. goal lane (plexus events → per-goal progress bar on the board)
4. cost chips (§2)

**Done when:** killing Postgres mid-goal produces a phone/desktop notification
within 10 minutes, and `pulse goal <id>` traces a finished plexus goal down to
every reward row.

---

## 7. Execution order

Dependencies, not preferences. Each milestone ends green before the next
starts; everything after M1 is developed sandbox-on.

| M | What | Why this order |
| --- | --- | --- |
| **M1** | AppArmor on → sandbox parity shakedown (§3) | everything else should be built inside the sandbox, not retrofitted |
| **M2** | Contract tests + golden events + `stack-smoke.sh` (§1) | locks the seams before new features widen them |
| **M3** | Cost capture → ingest → insights (§2) | small, self-contained, and §6's alerts + §4's routing economics need it |
| **M4** | Static verifiers + guardrails + `heart clean` (§4.1–4.3) | pure heart, no cross-repo coordination |
| **M5** | Tracing ids + alerts timer + dashboard v1 items 1–2 (§6) | needs M2's contract discipline and M3's cost events |
| **M6** | Capillaries harvest + top-k logging + text/code split + promotion gate (§5) | needs M2 (join contract) and real traffic from M1–M5 to harvest |
| **M7** | Swarm mode + plexus escalation hook (§4.4) | last: spends the most tokens, needs M3 to price itself |

Marrow re-enters after M6: it consumes cost-annotated episodes, goal-lineage
traces, and the plexus hard negatives — all produced above without it.
