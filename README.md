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
  "gpt":      {"endpoint": "https://api.openai.com/v1", "model": "gpt-5", "api_key_env": "OPENAI_API_KEY"},
  "deepseek": {"endpoint": "https://api.deepseek.com/v1", "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"},
  "local7b":  {"endpoint": "http://127.0.0.1:8000/v1", "model": "default"}
}}
```

CLI agents keep arteries memory/retrieval hooks alive if installed in the
target repo (episode worktrees carry the repo's `.arteries/` and CLI hook
config); the `api` agent calls the repo's observe hook itself, and is also how
a marrow-trained model acts as the coding agent.

### Model routing (`--agent auto`)

`heart` picks the model tier per task — explicit `task.difficulty` first, else
a keyword/size heuristic — and per role (test-writing routes to the cheap
tier). Configure tiers in `models.json` (or `HEART_TIER_<TIER>` env):

```json
{"tiers": {"cheap": "api:local7b", "standard": "claude", "strong": "api:gpt"}}
```

Every routing call emits a `route.decided` event with its signals, so routing
quality is auditable in the spool and can later train a learned gate.
`--escalate` defaults to the strong tier when routing. `HEART_MAX_AGENTS`
(default 8) caps concurrent agents across parallel episodes and candidates.

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

Three coding-specific mechanisms, composable per run:

- **Verify-fix loop** (`--fix-rounds N`): verifiers run in the workspace after
  implementation; failures are fed back to a fix agent. `--escalate <agent>`
  uses a stronger model for the final attempt.
- **Role pipeline** (`--pipeline`, default for `work`): implement (normal
  memory) → test-writer (clean memory) → reviewer (readonly memory, must end
  APPROVE/REJECT). Custom pipelines via `--roles roles.json`; each role may set
  its own `agent`, `memory`, `prompt`, `verify_after`.
- **Candidates** (`--candidates N`): N independent attempts in parallel
  worktrees, best reward wins. Doubles as the RL data engine.

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
- **`heart pulse serve`**: the factory floor as a local web page
  (http://127.0.0.1:7717) — live episode board, event stream, and insights,
  all from the same NDJSON spool the terminal tools read. Stdlib HTTP + SSE,
  one HTML file, no build step, localhost-only.

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
