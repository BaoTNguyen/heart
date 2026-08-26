# heart

Agent orchestration and RL environment runtime for coding. Heart runs coding
agents in isolated git worktrees, orchestrates them through coding-specific
workflows, verifies their work against real test suites, scores it, and exports
the results as training data.

It sits on top of two sibling systems and below two:

```text
capillaries  prompt/skill retrieval   (rides along via arteries)
arteries     memory + trace substrate (rides along via CLI hooks)
heart        orchestration + environment + reward   <- this repo
plexus       goal decomposition + acceptance loop (imports heart as a library)
marrow       RL training on heart's exported episodes (separate repo)
```

Stdlib-only on purpose: it installs anywhere in seconds, and marrow imports it
as a library for reward computation.

## What sets it apart

Most agent frameworks give you a graph DSL and a way to make models talk to each
other. Heart makes them ship code you can trust, and turns every run into
training data:

- **Any agent, one interface.** Claude Code, Codex, Gemini, OpenCode, any
  OpenAI-compatible endpoint, or a plain shell script — all the same `--agent`
  flag. `--agent auto` routes per task and per role, and because a tier is just
  an agent string, the same routing minimizes dollars on metered APIs, protects
  quota on subscription seats, or feeds a local model that doubles as the RL data
  engine.
- **Ground truth is a test suite, not a vibe.** Heart auto-detects verifiers
  (pytest, npm, cargo, go, plus ruff/mypy/tsc/eslint when configured), runs the
  agent's diff on a clean checkout, and scores it. `--apply` lands the change
  only when verification passes and the reviewer approves.
- **Reward hacking is blocked by construction.** Verification happens on a fresh
  worktree with only the diff applied; editing a test to make it pass doesn't
  travel. `allowed_paths`/`denied_paths` violations and a secret scanner both
  zero the reward. `.github/workflows/` is never editable in a work run.
- **It is an RL environment, not just an orchestrator.** `heart mine` builds
  TaskSpecs from git history, `heart batch --candidates N` is best-of-N that
  doubles as a data engine, and `heart export`/`heart dataset` emit SFT and DPO
  files. Marrow trains on exactly what heart ran.
- **Observability built for unattended runs.** An append-only NDJSON spine feeds
  `pulse health` (exit code as the alert primitive), a systemd health timer, and
  `pulse serve` — the whole factory floor as one localhost HTML page with live
  episode cards you can steer mid-run, no build step.
- **Routing that learns.** `--agent auto` scores each model in a manifest against
  the task's inferred skills, difficulty, and context size, filters to the ones
  that can actually run it, and picks the cheapest capable match — with declared
  scores corrected by a measured-reward sidecar so they can't drift unchecked.
- **Orchestrator-worker for tasks that split.** A decomposer breaks a task into
  workers, each routed to its own model and effort and run as an isolated episode;
  their diffs are 3-way merged by git, the merged tree is verified, and failures
  recover at the cheapest rung that works before falling back to the single-worktree
  role pipeline. Coupled work stays on the safe sequential path.

## Install

```bash
pip install -e .        # provides the `heart` command
python3 tests/test_heart.py   # self-check, no network or GPUs needed
```

## Daily use

```bash
heart work "fix the flaky retry logic in client.py"            # full pipeline
heart work "add a --json flag" --solo --agent api:deepseek     # one agent, cheap model
heart work "refactor the parser" --candidates 4 --apply        # best-of-4, apply winner
```

`heart work` runs against the current repo: isolated worktree from HEAD,
auto-detected verifiers (pytest/npm/cargo/go, or `--verify 'cmd'`), the
implement→test→review role pipeline with a verify-fix loop (default 2 rounds),
verification on a clean checkout, and `--apply` to land the diff only when
verification passes and the reviewer approves.

## Agents — any paid or local model

| `--agent` | What it drives |
|---|---|
| `claude` | Claude Code headless (`claude -p`) |
| `codex` | Codex CLI (`codex exec --full-auto`) |
| `gemini` | Gemini CLI (`gemini --yolo -p`) |
| `opencode` | OpenCode (`opencode run`) |
| `api` / `api:<profile>` | Any OpenAI-compatible endpoint: OpenAI, Anthropic, OpenRouter, Together, DeepSeek, Groq, vLLM, Ollama, llama.cpp… |
| `shell` | Runs the prompt as bash — scripted baselines and tests |
| `auto` | Route by task complexity: cheap tier for routine tasks, strong for hard ones |
| `--agent-cmd 'tmpl'` | Escape hatch: any shell template, prompt in `$HEART_PROMPT` |

Set a default with `HEART_AGENT`. The `api` agent resolves config from
`HEART_API_ENDPOINT` / `HEART_API_MODEL` / `HEART_API_KEY`, or named profiles in
`~/.config/heart/models.json`:

```json
{"profiles": {
  "gpt":       {"endpoint": "https://api.openai.com/v1", "model": "gpt-5", "api_key_env": "OPENAI_API_KEY"},
  "deepseek":  {"endpoint": "https://api.deepseek.com/v1", "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"},
  "local":     {"endpoint": "http://127.0.0.1:8001/v1", "model": "default"},
  "local-think": {"endpoint": "http://127.0.0.1:8001/v1", "model": "default", "reasoning": true}
}}
```

**Per-request reasoning.** A local server (llama.cpp/vLLM) is launched with
thinking off; `"reasoning": true` on a profile flips `enable_thinking` per call,
so `local` and `local-think` share one loaded model on one port — a fast
executor and a thinking planner without a reload. `HEART_API_REASONING=1`
forces it for a single run. When on, the token floor rises to 4000 so a
reasoning trace can't consume the whole budget and return an empty answer.
Point these at the [hot-swap model manager](../plexus/local_model_ideas/) and
the model behind the fixed port swaps underneath without the profiles changing.

CLI agents keep arteries memory/retrieval hooks alive if installed in the
target repo (episode worktrees carry the repo's `.arteries/` and CLI hook
config); the `api` agent calls the repo's observe hook itself, and is also how
a marrow-trained model acts as the coding agent.

### Model routing (`--agent auto`)

`heart` picks the model per task and per role (test-writing routes cheap). The
richer form is a **capability manifest**: each model in `models.json` declares a
`tier` and the few `skills` it's notably strong or weak at, plus a context window
and difficulty ceiling; `route` filters to the models that can actually run the
task (skills, difficulty, context) and picks the cheapest capable match, with
declared scores corrected by a measured-reward sidecar so they can't drift
unchecked. A bare tier map still works and is synthesized into a manifest:

```json
{"tiers": {"cheap": "api:local7b", "standard": "claude", "strong": "api:gpt"}}
```

Every routing call emits a `route.decided` event with its signals, so routing
quality is auditable in the journal and can later train a learned gate.
`--escalate` defaults to the strong tier when routing. `HEART_MAX_AGENTS`
(default 8) caps concurrent agents across parallel episodes and candidates.

### Fleet concurrency: one local model, many callers

`HEART_MAX_AGENTS` bounds one process. When several processes run at once — a
`heart batch`, or a few `plexus run` goals working different repos — they each
get their own cap, and if they route cheap-tier work to the *same* local model
server they sum to N×8 requests against one GPU. Two knobs bound the total,
both opt-in (unset or `0` changes nothing):

- **`HEART_MAX_AGENTS_GLOBAL`** — a machine-wide cap on *every* agent, for when
  the shared limit is a paid API key or aggregate load.
- **`HEART_LOCAL_SLOTS`** — a cap on agents hitting a *local* model server
  (loopback or LAN address), and nothing else. Paid APIs and subscription CLIs
  skip it and keep running, so a fleet's frontier roles aren't throttled to
  protect the GPU. Slots are keyed by **host:port**, so every profile pointing
  at the *same* server shares one pool, while two servers get independent pools.
  That is the multi-GPU story without a supervisor: run one llama.cpp (or vLLM)
  per GPU — `127.0.0.1:8000` on card 0, `:8001` on card 1 — point a profile at
  each, and `HEART_LOCAL_SLOTS` becomes the *per-GPU* ceiling, so two GPUs give
  you `2 × N` local agents, N per card, each queue bounded on its own. Profiles
  sharing one endpoint (`local7b`, `local-coder`, a trained checkpoint all on
  `:8000`) contend as one resource — which is why a single server that keeps
  swapping models between them just thrashes; pin one model per endpoint and
  make a swap a deliberate restart, not a routing accident. A tensor-parallel
  server spanning both cards on one port is a single pool, correctly: it is one
  queue.

Size `HEART_LOCAL_SLOTS` to what one server batches well (2–4 is a sane start);
excess callers wait on a flock'd slot rather than piling onto the endpoint. A
crashed agent's slot frees immediately — the kernel drops the lock when the
holder dies.

Tiers mix pricing models freely — a tier is an agent string, so it can be a
local server (`api:local7b`), a metered API (`api:gpt`), or a subscription CLI
seat (`claude`, `codex`). What routing optimizes depends on that mix:

- **Metered APIs**: routing minimizes dollars — routine traffic lands on the
  local model, frontier spend concentrates on hard tasks.
- **Subscription CLIs**: marginal cost is zero but quota isn't; routing
  preserves usage-window headroom. Keep batch/best-of-N traffic on the local
  tier (no rate windows) and let interactive work ride the subscription.
  Keep concurrency modest on subscription tiers — window throttling surfaces
  as slow/failing turns, not clean retryable errors.
- **Local models**: near-free capacity that doubles as the RL data engine —
  cheap-tier episodes are marrow's training traffic, and trained checkpoints
  redeploy into the same profile slot.

Routing can't see remaining subscription quota (no CLI exposes it); when a
window runs hot, re-point a tier for the day: `HEART_TIER_STANDARD=api:local7b`.

## Orchestration

Four coding-specific mechanisms, composable per run:

- **Verify-fix loop** (`--fix-rounds N`): verifiers run in the workspace after
  implementation; failures are fed back to a fix agent. `--escalate <agent>`
  uses a stronger model for the final attempt.
- **Role pipeline** (`--pipeline`, default for `work`): implement (normal
  memory) → test-writer (clean memory) → reviewer (readonly memory, must end
  APPROVE/REJECT). Custom pipelines via `--roles roles.json`; each role may set
  its own `agent`, `memory`, `prompt`, `verify_after`.
- **Candidates** (`--candidates N`): N independent attempts in parallel
  worktrees, best reward wins. Doubles as the RL data engine.
- **Orchestrator-worker** (`orchestrate.py`, Path A/B): a decomposer either keeps
  the task on the sequential role pipeline in one worktree (Path A, the safe
  default for coupled work) or splits it into workers — each run as an isolated
  episode routed to its own model and effort — whose diffs are 3-way merged by
  git, verified as one tree, and recovered at the cheapest rung that works before
  falling back to Path A. It replaces the old heterogeneous swarm: same
  best-of-many payoff, but the work is decomposed rather than duplicated, so the
  spend buys coverage instead of redundant attempts.

Operational switches:

- **Sandbox** (`HEART_SANDBOX=bwrap`): agent subprocesses run under bubblewrap
  — filesystem read-only except the worktree, /tmp, and agent config/cache
  dirs; `~/.ssh`/`~/.aws`/`~/.gnupg` hidden; network open (agents call APIs).
  Off by default; turn it on for unattended batches. Containment for accidents,
  not a boundary against a hostile model. `HEART_SANDBOX=bwrap-nonet` also cuts
  the network — for verifier runs and local-model agents needing no egress.
  Whenever sandboxing is on at all (`bwrap` or `bwrap-nonet`), verifier
  subprocesses always run under `bwrap-nonet` regardless of the agent's mode:
  agents get network, verifiers never do. Ubuntu 24.04+ needs a one-time
  AppArmor profile allowing bwrap to create user namespaces.
- **Resume**: `heart batch` skips episodes already recorded in the runs dir's
  `summary.csv`, so an interrupted batch continues where it died. Fresh runs
  dir = full re-run.
- **Reward ingest**: after `run`/`work`/`batch`, heart calls `art ingest` on
  the runs dir when arteries' CLI is on PATH (best-effort subprocess; heart
  stays stdlib-only). `HEART_INGEST=off` disables. `heart ingest [runs-dir]`
  re-runs the sweep any time (dedup makes it safe).
- `heart pulse insights` includes a routing scorecard (pass rate per tier) —
  a cheap tier that keeps failing means the classifier thresholds need moving.
- **Cost capture**: `runner.run_agent` extracts tokens and, from a
  `~/.config/heart/models.json` `"pricing"` map keyed by agent string, dollars
  per role, rolled into `episode["usage"]` and the spine. Two rules make the
  dollar figure a usable routing signal rather than a billing artifact:
  - **A local model server is always free** — `0.0`, even if a broad `"api"`
    pricing entry exists — because you pay nothing to run it.
  - **Everything else prices at the map's API rates, including subscription
    seats.** A Claude Pro/Max seat's marginal cost is zero, but pricing `claude`
    at Anthropic's published rates records the spend it *would* cost, so routing
    and `pulse insights` compare tiers on real dollars. Give each seat its own
    API-equivalent entry:
    ```json
    "pricing": {
      "api":    {"in_per_mtok": 0.15, "out_per_mtok": 0.60},
      "claude": {"in_per_mtok": 3.00, "out_per_mtok": 15.00},
      "codex":  {"in_per_mtok": 2.50, "out_per_mtok": 10.00}
    }
    ```
    (rates are illustrative — set them to the current published API price of the
    model behind each seat.)
- **`heart pulse serve`**: the factory floor as a local web page
  (http://127.0.0.1:7717) — live episode board, event stream, and insights,
  all from the same NDJSON journal the terminal tools read. Stdlib HTTP + SSE,
  one HTML file, no build step, localhost-only.
- **`heart pulse goal <goal-id>`**: goal lineage — features → episodes →
  outcome/reward/cost, grouped from `episode.finished` events whose payload
  carries `goal_id`/`feature_id` (stamped automatically by `events.emit()`
  from `PLEXUS_GOAL_ID`/`PLEXUS_FEATURE_ID` env when plexus is dispatching).
- **Steering + episode drill-down** (`pulse serve`): running-episode cards get
  a one-line text box — `POST /api/steer?episode=<id>` drops a note into that
  episode's runs dir, which the next role turn (or fix-loop attempt) appends
  to its prompt as an operator note and logs a `steer.received` event.
  Clicking a card's episode id opens a drill-down overlay with the full
  timeline, the captured diff, and per-role log tails.
- **Health alerting timer**: `contrib/heart-health.timer` + `.service` are
  systemd **user** units (no sudo) that run `heart pulse health --hours 1`
  every 10 minutes and fire a `notify-send` (plus an optional `ntfy.sh` push
  via `NTFY_TOPIC`) on the first WARN line. `pulse health` also now flags a
  review2-reject streak (3 most recent reviewed episodes all rejected), a
  `HEART_COST_ALERT` dollar ceiling on window spend, and (once plexus wires
  `PLEXUS_GOAL_ACTIVE`) a silently-stalled factory.
- **Static verifiers**: `detect_verifiers` also picks up `ruff`, `mypy`,
  `tsc`, `eslint`, and `biome` when both the tool's config and the tool
  itself are locally installed — never anything that downloads (no `npx`,
  no `pip install`), since sandboxed verifiers have no network. They run
  after the test verifiers and score under `lint_typecheck`.
- **Guardrails**: `heart work` refuses to hand back or apply dangerous diffs.
  `guard.scan_secrets` scans added lines for AWS keys, private key headers,
  GitHub/Slack tokens, and generic `key/secret/token/password = "..."`
  assignments — a hit zeroes reward with outcome `guardrail_violation`,
  mirroring a path violation. A size fuse in `cmd_work --apply` refuses diffs
  over `HEART_MAX_DIFF_LINES` (default 2000) changed lines unless
  `--allow-large` is passed. `.github/workflows/` is always a denied path in
  `heart work` — agents never edit CI config in a work run.
- **`heart clean [--days N] [--runs-dir DIR]`**: deletes episode directories
  older than N days (default 7; `summary.csv` is kept) and removes stale
  heart worktrees under `~/.cache/heart-ws`, pruning them from their source
  repo with `git worktree prune` when the source repo is discoverable.

## RL environment

```bash
heart mine ~/some/repo --out tasks/        # TaskSpecs from git history
heart check-task tasks/foo.json            # verifier determinism gate
heart batch tasks/ --variants normal:on,clean:on,normal:off,clean:off --repeat 3 --parallel 4
heart stats                                # pass rate / reward by ablation variant
heart export --out episodes.jsonl
heart dataset sft --out sft.jsonl && heart dataset dpo --out dpo.jsonl
```

Reward hacking is blocked structurally: verification happens on a fresh
worktree with only the agent's diff applied, and `allowed_paths` /
`denied_paths` violations zero the reward. `tasks/holdout/` is never trained
on — it's the evaluation set for marrow checkpoints.

## Design notes

Surveyed LangGraph, AutoGen/AG2, CrewAI, OpenAI Agents SDK, smolagents,
MetaGPT, SWE-agent, OpenHands, and aider before writing this. Kept: verifier
feedback loops (SWE-agent/OpenHands), repo-map context and edit-format
pragmatism (aider), role workflows with gates (MetaGPT, Anthropic's
workflow patterns), best-of-N orchestrator-workers. Skipped: graph DSLs,
conversation-driven multi-agent chatter, and framework dependencies — a coding
orchestrator's control flow is short and its ground truth is a test suite, so
plain Python stays debuggable.
