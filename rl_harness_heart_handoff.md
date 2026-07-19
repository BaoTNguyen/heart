# RL harness handoff for Arteries and Heart

This document is meant to move with the future `heart` repo. It captures the current state of `arteries`, the related role of `capillaries`, and the missing pieces needed to build a proper RL harness for coding workflows.

## Working thesis

Do not start by training everything at once. Build the data and action substrate first.

The useful sequence is:

```text
instrument decisions -> collect episodes -> run verifiers -> assign rewards -> train components -> orchestrate with heart -> run joint optimization
```

`arteries` should remain the memory, trace, and action-ledger layer. `capillaries` should remain the prompt and skill retrieval layer. `heart` should become the orchestration and environment runtime that drives episodes, subagents, verifiers, rollouts, reward aggregation, and dataset export.

## Current projects

### Arteries

`arteries` is the memory and tracing layer around `capillaries`.

Current responsibilities:

- Observes user turns from agent CLI hooks.
- Extracts short-lived memory from prompts.
- Stores memory in three tiers: ephemeral, persistent, evergreen.
- Builds a `MemoryFrame` consumed by `capillaries`.
- Logs runs, observed turns, memory extraction, frame construction, gate decisions, retrieved prompts, and compilation.
- Produces compaction packets.
- Supports multiple CLI adapters, including Codex, Claude Code, Pi, OpenCode, Hermes, and Cursor.
- Supports memory isolation modes for subagents.

Current storage:

- `arteries.ephemeral`
- `arteries.persistent`
- `arteries.evergreen`
- `arteries.retrievals`
- `arteries.agent_runs`
- `arteries.agent_events`

Current per-turn flow:

```text
user prompt
  -> turn.observed
  -> heuristic memory extraction
  -> MemoryFrame assembly
  -> capillaries gate
  -> capillaries retrieval if gate opens
  -> prompt text returned to hook
  -> background ephemeral-to-persistent compilation
```

Current memory modes:

| Mode | Persistent read | Ephemeral behavior | Use |
|---|---|---|---|
| unset | relevance-filtered | compile | normal agent |
| `ARTERIES_MEMORY=readonly` | relevance-filtered | discard | subagent can read memory but leaves no trace |
| `ARTERIES_MEMORY=clean` | none | discard | clean-room subagent or ablation |
| `ARTERIES_EPHEMERAL=discard` | unchanged | in-process only | temporary analysis |
| `ARTERIES_PERSISTENT_READ=none` | none | unchanged | memory ablation |

The important point: these are already proto-actions. RL can later learn when to use clean memory, readonly memory, persistent read, or discard mode only because those modes exist.

### Capillaries

`capillaries` is the retrieval engine.

Current responsibilities:

- Prompt gating.
- Prompt search and reranking.
- Private prompt corpus access.
- Skill recall for multi-step procedures.
- MCP, HTTP, Python, and CLI interfaces.
- Feedback endpoint for prompt or skill usefulness.
- Shared memory contract types such as `MemoryFrame`.

Current retrieval pipeline:

```text
query
  -> dense pgvector search
  -> sparse pg_trgm search
  -> reciprocal rank fusion
  -> cross-encoder reranking
  -> skill recall
  -> single prompt, skill, or none
```

Capillaries currently owns semantic prompt selection, but it does not own environment reset, coding actions, verifiers, rollouts, or reward distribution.

### Heart

`heart` does not exist yet. It should become the orchestration and RL runtime layer.

Heart should not replace `arteries` or `capillaries`. It should coordinate them.

Target dependency shape:

```text
heart
  -> agents/models
  -> repo environments
  -> verifier runners
  -> arteries for memory, traces, decisions, rewards
  -> capillaries for prompt and skill retrieval
```

## What exists today versus what is missing

| Capability | Arteries today | Capillaries today | Missing for RL harness |
|---|---|---|---|
| User turn observation | yes | no | link observations to formal episodes |
| Memory extraction | heuristic extractor | no | trainable extractor policy and reward labels |
| Memory write/discard | yes, via modes | no | explicit decision log with available actions |
| Persistent memory read | relevance-filtered | consumed in retrieval | actionized read budgets and ablations |
| Evergreen memory | yes | consumed as intent/context | promotion/rejection policy |
| Prompt gate | logs result | owns gate | actionized gate decision with reward attribution |
| Prompt retrieval | logs retrieved prompt | owns retrieval | top-k, reject, use-cache actions |
| Skill retrieval | no direct ownership | yes | reward attribution by skill step |
| Run telemetry | yes | partial | episode/step/reward schema |
| Coding environment | no | no | task specs, repo reset, tool actions |
| Verifiers | no | no | public tests, hidden tests, lint, typecheck, security |
| Reward distribution | no | feedback only | decision-level reward assignment |
| Subagent orchestration | memory modes only | no | spawn, assign, merge, terminate |
| Parallel rollouts | no | no | heart runtime |
| Dataset export | trace data only | feedback/retrieval data | SFT, DPO, RL trajectory export |

## The main gap

Arteries currently has:

```text
state -> hard-coded behavior -> event log
```

A proper RL harness needs:

```text
state -> available actions -> chosen action -> cost -> outcome -> reward
```

The missing object is a decision/action ledger.

Events say what happened. Decisions say what could have happened, what was chosen, and what outcome followed. RL needs decisions.

## Action surfaces already present

These action surfaces exist as functionality, but most are not yet logged as explicit decisions.

| Surface | Existing action/functionality | Current limitation |
|---|---|---|
| Memory extraction | extract candidate memories from prompt | heuristic, not actionized |
| Ephemeral write | write extracted facts to DB | no available-action log |
| Ephemeral discard | `ARTERIES_EPHEMERAL=discard` | mode-level, not per-decision |
| Persistent read | relevance-filtered retrieval | no read budget action |
| Persistent read off | `ARTERIES_PERSISTENT_READ=none` | useful for ablation |
| Clean memory | `ARTERIES_MEMORY=clean` | useful for subagents and ablations |
| Readonly memory | `ARTERIES_MEMORY=readonly` | useful for subagents |
| Memory compilation | ephemeral to persistent | background process, not policy-driven |
| Prompt gate | search or abstain | owned by capillaries, logged as event |
| Prompt retrieval | retrieve prompt | no top-k/reject action ledger |
| Skill retrieval | retrieve multi-step skill | no step-level reward attribution |
| Retrieval cache | recent retrievals in `MemoryFrame` | not used as explicit cache action |
| Compaction | compaction packet | not tied to context budget reward |

## Action surfaces still needed

### Memory actions

Add these as explicit decisions before trying to train memory behavior:

```text
read_none
read_ephemeral
read_persistent_relevant
read_evergreen
read_recent_retrievals
write_ephemeral
discard_ephemeral
compile_to_persistent
delay_compile
dedupe_memory
supersede_memory
promote_evergreen_candidate
reject_evergreen_candidate
suppress_memory
expire_memory
summarize_memory
redact_memory
```

Why this matters:

- RL can learn to discard scratch work.
- RL can learn when project memory helps versus contaminates the task.
- RL can learn which facts deserve promotion.
- RL can learn memory budgets instead of flooding context.

### Retrieval actions

Add these around capillaries calls:

```text
abstain
search_single
search_top_k
search_skill
search_prompt_chain
use_cached_retrieval
force_fresh_search
rerank_with_memory
rerank_without_memory
reject_retrieval
accept_retrieval
```

Useful knobs:

```text
k
confidence_threshold
domain_filter
intent_filter
task_type_filter
skill_vs_prompt_preference
novelty_penalty
recent_prompt_penalty
memory_filter_enabled
```

Reward retrieval by downstream usefulness, not just similarity:

```text
retrieval_reward = task_success_delta - token_cost - distraction_penalty
```

### Context actions

The model only acts on what reaches its context window. This is a major missing surface.

```text
include_memory
exclude_memory
include_retrieved_prompt
exclude_retrieved_prompt
include_prior_turns
include_file_snippets
compress_history
summarize_test_output
order_by_relevance
order_by_recency
reserve_budget_for_repo
reserve_budget_for_tests
reserve_budget_for_instructions
```

Context actions should record token cost and downstream result.

### Coding workflow actions

These belong mostly in `heart`, but `arteries` should be able to log them.

```text
inspect_file
search_repo
run_focused_test
run_full_test_suite
run_lint
run_typecheck
edit_file
revert_edit
checkpoint_patch
submit_patch
continue_work
ask_for_clarification
declare_blocked
```

For coding RL, these are often better action units than raw tokens.

### Verifier actions

The harness needs explicit verifier choices:

```text
run_public_tests
run_hidden_tests
run_lint
run_typecheck
run_security_scan
run_mutation_tests
run_diff_quality_review
stop_without_verifier
```

Verifier actions need cost:

```json
{
  "latency_ms": 4820,
  "tokens": 0,
  "exit_code": 0,
  "tests_run": 42
}
```

### Subagent and orchestration actions

These are heart's main action surfaces:

```text
spawn_subagent
choose_subagent_role
choose_subagent_model
choose_subagent_memory_mode
assign_task_slice
set_subagent_budget
merge_subagent_result
reject_subagent_result
terminate_subagent
retry_with_clean_context
retry_with_more_memory
retry_with_stronger_model
```

The existing memory modes make this possible:

```text
test-writer subagent -> ARTERIES_MEMORY=clean
reviewer subagent -> ARTERIES_MEMORY=readonly
main implementation agent -> normal memory
scratch researcher -> ARTERIES_EPHEMERAL=discard
```

### Recovery actions

Failures are where good policies learn. Add explicit recovery choices:

```text
retry_same_strategy
retry_clean_memory
retry_with_retrieval
retry_without_retrieval
revert_last_patch
run_more_tests
inspect_failure_logs
escalate_model
spawn_debugger
stop_and_report_failure
```

## Proposed Arteries additions

Keep this small. The first milestone is not an RL framework. It is a decision ledger.

### 1. Add schema

Add three tables to `src/arteries/schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS arteries.episodes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      TEXT NOT NULL,
    agent_id        TEXT,
    task_id         TEXT,
    run_id          UUID,
    status          TEXT NOT NULL DEFAULT 'running',
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at        TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS arteries.decisions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id        UUID,
    run_id            UUID,
    turn_id           TEXT,
    project_id        TEXT NOT NULL,
    agent_id          TEXT,
    decision_type     TEXT NOT NULL,
    observation       JSONB NOT NULL DEFAULT '{}',
    available_actions JSONB NOT NULL DEFAULT '[]',
    chosen_action     TEXT NOT NULL,
    cost              JSONB NOT NULL DEFAULT '{}',
    metadata          JSONB NOT NULL DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS arteries.rewards (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id     UUID,
    decision_id    UUID,
    run_id         UUID,
    project_id     TEXT NOT NULL,
    reward_type    TEXT NOT NULL,
    value          REAL NOT NULL,
    components     JSONB NOT NULL DEFAULT '{}',
    source         TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_episodes_project_created
    ON arteries.episodes (project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_decisions_episode_created
    ON arteries.decisions (episode_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_decisions_type_created
    ON arteries.decisions (decision_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_rewards_episode_created
    ON arteries.rewards (episode_id, created_at ASC);
```

### 2. Add `src/arteries/actionlog.py`

Mirror the style of `runlog.py`:

```python
start_episode(...)
current_episode(...)
log_decision(...)
log_reward(...)
recent_decisions(...)
episode_trace(...)
```

Use Postgres when available. Fall back to repo-local JSONL when the DB is unavailable.

Support these environment variables:

```text
ARTERIES_EPISODE_ID
ARTERIES_TASK_ID
ARTERIES_REWARD_SOURCE
```

### 3. Instrument existing decisions

Start with existing behavior. Do not change policy yet.

In `eval.py`:

- Log `memory.extract` before or after extraction.
- Log `memory.frame_build`.
- Log `retrieval.gate` with available actions `["abstain", "search"]`.
- Log `retrieval.select` when capillaries returns a prompt or skill.

In `extract.py`:

- Log `memory.write_policy` for each extraction.
- Available actions should include `write_ephemeral` and `discard_ephemeral`.
- Chosen action depends on `EPHEMERAL_MODE`.

In `frame.py` or `memory_select.py`:

- Log `memory.read_policy`.
- Available actions should include `read_none`, `read_relevant_persistent`, `read_recent_ephemeral`, and `read_evergreen`.
- Chosen action should reflect env and selected memory tiers.

In compaction packet generation:

- Log `context.compact`.
- Include token or character counts when available.

### 4. Add CLI commands

Add lightweight commands under `art`:

```bash
art episode start --task TASK_ID
art episode reward --type task_success --value 1.0
art episode show EPISODE_ID
art decisions recent --project PROJECT_ID
art rewards recent --project PROJECT_ID
```

Avoid building a large CLI tree at first. These commands only need to prove the ledger works.

### 5. Add dataset export

Add a JSONL export command:

```bash
art episode export --project arteries --out episodes.jsonl
```

One exported record should include:

```json
{
  "episode": {},
  "events": [],
  "decisions": [],
  "rewards": [],
  "retrievals": []
}
```

This is enough to produce SFT, DPO, and RL training datasets later.

## Proposed Heart responsibilities

Heart should own the runtime pieces that do not belong in a memory layer.

### Minimum viable heart

```text
TaskSpec loader
repo reset/checkpoint manager
agent runner
subagent manager
verifier runner
reward calculator
episode lifecycle manager
rollout queue
parallel worker pool
dataset exporter
experiment metadata tracker
```

### TaskSpec

Start with a file format like:

```json
{
  "task_id": "repair-parser-001",
  "repo_path": "/path/to/repo",
  "base_commit": "abc123",
  "prompt": "Fix the failing parser test.",
  "allowed_paths": ["src/", "tests/"],
  "public_verifiers": [
    {"name": "unit", "command": "python3 -m unittest discover -s tests"}
  ],
  "hidden_verifiers": [],
  "timeout_seconds": 300,
  "difficulty": "easy",
  "tags": ["bugfix", "python"]
}
```

### Episode object

Heart should create the episode and pass IDs into Arteries:

```text
ARTERIES_EPISODE_ID=<episode-id>
ARTERIES_TASK_ID=<task-id>
ARTERIES_MEMORY=<normal|readonly|clean>
```

Episode record:

```json
{
  "episode_id": "ep_123",
  "task_id": "repair-parser-001",
  "repo": "example",
  "base_commit": "abc123",
  "agent_id": "codex-main",
  "model": "gpt-5-codex",
  "memory_mode": "normal",
  "started_at": "...",
  "ended_at": "...",
  "outcome": "pass"
}
```

### Verifier result

Heart should produce structured verifier output:

```json
{
  "public_tests": {"passed": true, "score": 1.0, "command": "python3 -m unittest"},
  "hidden_tests": {"passed": false, "score": 0.0},
  "lint": {"passed": true, "score": 1.0},
  "typecheck": {"passed": true, "score": 1.0},
  "diff_quality": {"score": 0.7, "notes": "Large but acceptable"},
  "security": {"passed": true, "score": 1.0}
}
```

### Reward calculation

Initial reward formula:

```text
R_total =
  0.45 hidden_tests
+ 0.25 public_tests
+ 0.10 lint_typecheck
+ 0.10 diff_quality
+ 0.05 efficiency
+ 0.05 retrieval_memory_usefulness
```

When hidden tests are unavailable:

```text
R_total =
  0.45 public_tests
+ 0.15 lint_typecheck
+ 0.15 diff_quality
+ 0.10 regression_safety
+ 0.10 efficiency
+ 0.05 retrieval_memory_usefulness
```

### Reward distribution

Use both local rewards and end-to-end rewards.

| Component | Local reward | End-to-end reward |
|---|---|---|
| Memory extraction | memory later reused or validated | task success delta with memory |
| Memory read policy | relevant context, low token cost | success delta versus clean memory |
| Retrieval gate | search only when useful | success delta versus no retrieval |
| Retriever/reranker | selected prompt helped | success delta versus alternate prompt |
| Context assembly | useful context per token | pass rate and fewer retries |
| Coding agent | tests, lint, diff quality | final task reward |
| Subagent policy | useful delegated result | success delta versus no subagent |
| Orchestrator | cost, time, reliability | rollout success rate |

Difference rewards should be preferred:

```text
component_reward = outcome_with_component - outcome_without_component - component_cost
```

Examples:

```text
memory_reward = success_with_memory - success_clean_memory - token_cost
retrieval_reward = success_with_prompt - success_no_prompt - prompt_token_cost
subagent_reward = success_with_subagent - success_single_agent - subagent_cost
```

## Ablations that must exist

Ablations are not optional. They are how the system assigns credit.

Required modes:

```text
normal memory vs clean memory
persistent read vs no persistent read
retrieval on vs retrieval off
top-1 prompt vs top-k prompts
skill mode vs single prompt mode
single agent vs subagent run
cheap verifier only vs full verifier stack
```

Arteries already supports some of these:

```bash
ARTERIES_MEMORY=clean
ARTERIES_MEMORY=readonly
ARTERIES_EPHEMERAL=discard
ARTERIES_PERSISTENT_READ=none
```

Heart should formalize these as rollout variants.

## Data to collect

Each episode should collect:

| Data | Owner | Purpose |
|---|---|---|
| task spec | heart | environment definition |
| repo base commit | heart | reproducibility |
| initial diff | heart | sanity check |
| memory mode | heart/arteries | ablation and attribution |
| MemoryFrame summary | arteries | observation state |
| retrieved prompts | arteries/capillaries | retrieval attribution |
| decisions | arteries | RL action data |
| tool calls | heart or agent adapter | trajectory actions |
| file diffs | heart | final artifact |
| verifier outputs | heart | reward source |
| rewards | heart/arteries | training target |
| final outcome | heart | success label |

Minimum useful episode record:

```json
{
  "episode_id": "ep_123",
  "task_id": "repair-parser-001",
  "run_id": "run_456",
  "base_commit": "abc123",
  "memory_mode": "normal",
  "retrieved_prompt_ids": ["prompt_1"],
  "decisions": [],
  "final_diff": "...",
  "verifier_results": {},
  "rewards": {},
  "outcome": "pass"
}
```

## Training path

Do component training first. Keep an end-to-end benchmark running the whole time.

| Stage | What to train or optimize | Data needed |
|---|---|---|
| 1 | verifier reliability | task specs, expected outcomes |
| 2 | prompt gate and retrieval | retrieval decisions, task outcomes |
| 3 | memory extraction and selection | memory decisions, reuse, ablations |
| 4 | context assembly | context decisions, token costs, outcomes |
| 5 | coding behavior | successful trajectories, failed trajectories |
| 6 | orchestration policy | subagent decisions, costs, outcomes |
| 7 | joint policy | stable episodes and rewards |

Algorithm sequence:

```text
offline eval
best-of-N and rejection sampling
SFT on successful traces
DPO/IPO from pass/fail pairs
context/retrieval policy tuning
GRPO/PPO only after the runtime is stable
```

Do not start with PPO. Coding rewards are expensive and sparse. Verifier-guided search, SFT, and DPO will likely pay off sooner.

## First useful milestones

### Milestone 1: decision ledger in Arteries

Done when one normal user turn produces:

```text
turn.observed
memory.extract decision
memory.write_policy decision
memory.read_policy decision
retrieval.gate decision
retrieval.select decision if retrieval happens
```

### Milestone 2: single-task episode

Done when one coding task can be run as:

```text
start episode
reset repo
run agent
capture final diff
run verifier
log reward
export episode JSONL
```

### Milestone 3: ablation pair

Done when the same task can run as:

```text
normal memory + retrieval
clean memory + retrieval
normal memory + retrieval off
clean memory + retrieval off
```

The exported data should show which decisions changed and how reward changed.

### Milestone 4: first training dataset

Done when there are:

```text
100+ scored episodes
20+ pass/fail pairs for the same or similar tasks
decision logs for memory and retrieval
verifier outputs stored as structured JSON
```

At this point, SFT and DPO experiments become reasonable.

## 30/60/90 day plan

### First 30 days

Build the substrate:

- Add `episodes`, `decisions`, and `rewards` tables.
- Add `actionlog.py`.
- Instrument existing memory and retrieval decisions.
- Add episode/task ID environment support.
- Export episode JSONL.
- Create 20 to 50 deterministic coding tasks.
- Add a simple verifier runner.

Success criterion:

```text
For one task, the system can explain what context was used, what prompt was retrieved, what actions were chosen, what changed in the repo, and why the reward was assigned.
```

### Days 31 to 60

Build the data flywheel:

- Run 100 to 500 episodes.
- Add ablation variants.
- Create comparison pairs for DPO.
- Track retrieval usefulness.
- Track memory usefulness.
- Add failure taxonomy: syntax, logic, timeout, overfit, excessive diff, bad retrieval, bad memory.

Success criterion:

```text
Memory and retrieval can be measured against clean/no-retrieval baselines.
```

### Days 61 to 90

Start heart if orchestration pain is real:

- Add task queue.
- Add parallel rollout workers.
- Add subagent roles.
- Add verifier scheduling.
- Add reward aggregation.
- Add experiment tracking.
- Add dataset export for SFT/DPO/RL.

Success criterion:

```text
Heart can run a batch of tasks across variants and produce scored trajectories without manual coordination.
```

## What not to build yet

Avoid these until the basics work:

- A large RL framework.
- PPO/GRPO integration.
- Complex multi-agent planning.
- Learned memory promotion.
- Learned reward models.
- Automatic hidden test generation.
- Distributed workers across machines.

These only make sense after the ledger, tasks, verifiers, and ablations are stable.

## Open design questions

- Should `heart` store episode metadata itself, or should it write all episode records into Arteries?
- Should action logging be synchronous on every turn, or buffered to avoid hook latency?
- How much prompt text should be stored versus content hashes for privacy and storage control?
- Should hidden verifier results ever be visible to agents, or only to the reward calculator?
- What is the minimum task format that covers repo reset, verifier commands, allowed paths, and timeouts?
- Which agent adapters should heart support first: Codex only, or Codex plus Claude/Pi/OpenCode?
- Should capillaries expose top-k retrieval directly to heart, or should heart call lower-level search APIs?

## Recommended ownership split

| System | Owns | Does not own |
|---|---|---|
| arteries | memory, trace, decisions, rewards, episode linkage, export | repo reset, subagent scheduling, verifier execution |
| capillaries | prompt retrieval, skill recall, retrieval feedback | coding environment, reward calculation, orchestration |
| heart | episodes, tasks, subagents, verifiers, rollouts, reward aggregation, datasets | durable memory storage, prompt corpus search |

## Immediate next implementation step

Add the Arteries decision ledger first.

That means:

```text
schema.sql: episodes, decisions, rewards
src/arteries/actionlog.py
instrument eval.py, extract.py, frame.py or memory_select.py
add episode export command
```

Once that exists, Heart has something concrete to orchestrate. Without it, Heart would have to infer decisions from generic event logs, which makes reward attribution weak.

