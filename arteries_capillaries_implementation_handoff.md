# Arteries and Capillaries implementation handoff

This document explains how `arteries` and `capillaries` are currently implemented, what each system does today, and how the workflows fit together.

It is separate from `rl_harness_heart_handoff.md`, which focuses on the future RL harness and the planned `heart` orchestration project.

## System summary

`arteries` and `capillaries` are designed as two separate layers:

```text
agent CLI hook or extension
  -> arteries
  -> capillaries
  -> prompt or skill returned to the agent
```

`arteries` owns memory, tracing, CLI integration, and continuity packets.

`capillaries` owns prompt and skill retrieval.

The direction of dependency is one-way:

```text
arteries imports and calls capillaries
capillaries does not write to arteries
```

That split matters. Arteries can observe agent sessions across multiple CLIs and build memory frames. Capillaries can stay focused on deciding which prompt or skill is relevant for the current situation.

## Current repo layout

The expected local layout is:

```text
/home/bao-tn/Coding/Projects/arteries
/home/bao-tn/Coding/Projects/capillaries
```

Arteries setup scripts assume `capillaries` is a sibling repo unless the caller passes `--capillaries-root`.

Both systems use the shared `capillaries` Postgres database. Arteries stores its own tables in the `arteries` schema. Capillaries owns prompt, search, and skill tables outside that schema.

## Arteries implementation

### What Arteries does today

Arteries is an always-on memory and tracing layer for coding CLIs.

It currently does these jobs:

- Installs hooks or adapters into supported agent CLIs.
- Starts an Arteries run at session activation.
- Observes user turns.
- Extracts candidate memories from user messages.
- Stores memory in ephemeral, persistent, and evergreen tiers.
- Builds a `MemoryFrame`.
- Calls Capillaries gate and retrieval.
- Prints retrieved prompt text back to the host CLI hook.
- Logs telemetry to Postgres or JSONL fallback files.
- Builds continuity packets on compaction.
- Provides trace and inspection commands.
- Supports subagent memory isolation modes.

### Main Arteries modules

| Module | Current role |
|---|---|
| `src/arteries/cli.py` | Top-level `art` command router. |
| `src/arteries/eval.py` | Per-turn evaluation path called by hooks. |
| `src/arteries/extract.py` | Fast heuristic memory extraction from user messages. |
| `src/arteries/frame.py` | Builds `MemoryFrame` from storage and retrieval logs. |
| `src/arteries/memory_select.py` | Selects ephemeral and persistent memories for a frame. |
| `src/arteries/memory_types.py` | Local dataclasses for `MemoryFrame` and related types. |
| `src/arteries/storage.py` | Postgres access for memory tiers and retrieval logs. |
| `src/arteries/compile.py` | Compiles ephemeral memories into persistent memories. |
| `src/arteries/runlog.py` | Run/event telemetry with Postgres and JSONL fallback. |
| `src/arteries/packet.py` | Continuity packet builder for compaction. |
| `src/arteries/setup_cli.py` | Installs CLI-specific adapters into target repos. |
| `src/arteries/cli_normalize.py` | Normalizes CLI event JSON into Arteries env/context fields. |
| `src/arteries/trace.py` | Produces central trace output for runs, memories, and retrievals. |
| `src/arteries/runs.py` | Starts, shows, and summarizes agent runs. |
| `src/arteries/evergreen.py` | Extracts, reviews, imports, edits, and removes evergreen memory. |
| `src/arteries/doctor.py` | Checks DB and runtime setup health. |
| `src/arteries/setup_db.py` | Initializes the Arteries schema. |

### Arteries CLI

The top-level command is `art`.

Current commands:

```text
art setup
art evergreen
art setup-db
art eval
art inspect
art runs
art doctor
art packet
art trace
art backfill-embeddings
```

The CLI router lives in `src/arteries/cli.py`. It forwards work to smaller modules rather than putting all behavior in one large command file.

### Arteries setup workflow

The setup command installs Arteries runtime files into a target project:

```bash
bash scripts/art.sh setup add codex
bash scripts/art.sh setup add claude
bash scripts/art.sh setup add pi
bash scripts/art.sh setup add opencode
bash scripts/art.sh setup add cursor
bash scripts/art.sh setup add hermes
```

Supported providers:

| Provider | Integration style |
|---|---|
| `pi` | Native extension with prompt/context hooks and compaction replacement. |
| `codex` | Native hooks plus `AGENTS.md` context. |
| `claude` | Native hooks. |
| `opencode` | Native plugin with compaction context injection. |
| `hermes` | Conservative MCP/context-file adapter. |
| `cursor` | MCP plus Cursor rules adapter. |

Setup writes a `.arteries/` runtime directory into the target repo. That directory includes:

- `.arteries/config.json`
- `.arteries/hooks/observe.sh`
- `.arteries/hooks/generic-observe.sh`
- `.arteries/hooks/activate.sh`
- `.arteries/hooks/compact-packet.sh`
- CLI-specific hook wrappers
- smoke scripts

The generated runtime scripts set:

```text
ARTERIES_ROOT
CAPILLARIES_ROOT
PROJECT_ROOT
PYTHONPATH
ARTERIES_PROJECT
ARTERIES_AGENT_ID
ARTERIES_CLI
ARTERIES_REPO
```

Then they call Python modules such as:

```bash
python3 -m arteries.eval "$prompt"
python3 -m arteries.runs start ...
python3 -m arteries.packet ...
```

### Per-turn Arteries workflow

The core per-turn workflow lives in `src/arteries/eval.py`.

Flow:

```text
1. Create a turn id.
2. Log `turn.observed`.
3. Extract candidate ephemeral memories.
4. Log `memory.ephemeral.extracted`.
5. Build a MemoryFrame.
6. Log `memory.frame.built`.
7. Start background memory compilation when allowed.
8. Call capillaries gate.
9. Log `prompt.gate.decided`.
10. If the gate opens, call capillaries find.
11. Log `prompt.retrieved`.
12. Store retrieval metadata in `arteries.retrievals`.
13. Return prompt text to the hook.
```

If Capillaries is not importable, Arteries still performs memory extraction, frame construction, and telemetry. Retrieval is simply skipped.

### Memory extraction

The fast memory extractor lives in `src/arteries/extract.py`.

It is intentionally heuristic. It does not call a model.

It looks for:

- preferences
- current working context
- project/team facts
- corrections
- domain signals

The extractor uses regex patterns and a shared-ish domain taxonomy. Example domains include:

```text
technical
AI
business
strategy
product
finance
career
learning
personal
writing
```

The extractor returns `Extraction` objects:

```text
fact
domains
confidence
signal_type
```

Then `extract_and_store()` either:

- inserts each extraction into `arteries.ephemeral`, or
- keeps it in an in-process buffer if `ARTERIES_EPHEMERAL=discard`.

This is a bootstrap extractor. The code comments already call out that it can later be replaced or supplemented by a trained small model once enough signal exists.

### Memory tiers

Arteries stores three memory tiers.

| Tier | Scope | Purpose |
|---|---|---|
| Ephemeral | project + agent process | Recent, high-churn observations from the current session. |
| Persistent | project | Compiled project facts, preferences, and decisions. |
| Evergreen | global | Human-reviewed cross-project facts and durable preferences. |

The schema lives in `src/arteries/schema.sql`.

#### Ephemeral

Table:

```text
arteries.ephemeral
```

Important fields:

```text
fact
embedding
domains
confidence
project_id
agent_process_id
parent_agent_id
status
valid_from
valid_until
```

Ephemeral records are usually `uncompiled` until the compiler claims and processes them.

#### Persistent

Table:

```text
arteries.persistent
```

Important fields:

```text
fact
embedding
domains
confidence
project_id
source_project_id
parent_ids
child_ids
valid_from
valid_until
```

Persistent memories are embedded and retrieved by relevance using pgvector HNSW cosine search.

#### Evergreen

Table:

```text
arteries.evergreen
```

Important fields:

```text
fact
embedding
domains
confidence
parent_ids
superseded_by
source_meta
```

Evergreen memory is meant to be human-reviewed. The `evergreen` CLI can extract candidate memories into a Markdown review file and then import accepted entries.

### Memory selection

The frame builder uses `memory_select.select_for_frame(message)`.

Selection behavior:

- Ephemeral memory comes from the current project and agent process.
- Persistent memory is relevance-filtered when embeddings are available.
- If embeddings are missing or the embedding server is unavailable, selection falls back to recency.
- `ARTERIES_PERSISTENT_READ=none` disables persistent memory reads.
- `ARTERIES_MEMORY=clean` disables persistent memory and discards ephemeral writes.
- `ARTERIES_MEMORY=readonly` allows memory reads but discards new ephemeral writes.

The relevance threshold defaults to `0.3` and can be adjusted with:

```text
ARTERIES_RELEVANCE_THRESHOLD
```

### MemoryFrame

Arteries owns local dataclasses for the frame shape:

```text
MemoryFrame
  ephemeral: EphemeralMemory
  persistent: PersistentMemory
  evergreen: EvergreenMemory
```

`EphemeralMemory` includes:

```text
recent_messages
topic_drift
turn_count
```

`PersistentMemory` includes:

```text
session_insights
prior_retrievals
active_domains
```

`EvergreenMemory` includes:

```text
user_intent
recurring_domains
ground_truth_insights
last_retrieval_ts
retrieval_confidence
```

`src/arteries/frame.py` builds this from:

- selected ephemeral rows
- selected persistent rows
- evergreen rows
- recent retrieval rows
- active project domains
- recurring evergreen domains

Topic drift is computed from recent ephemeral domains versus active persistent domains. Capillaries uses that drift signal when deciding whether to open search.

### Retrieval logging

Arteries stores recent retrievals in:

```text
arteries.retrievals
```

Important fields:

```text
project_id
agent_process_id
prompt_id
situation
score
relevance
created_at
```

These retrievals are used in two ways:

- They enter the `MemoryFrame` as `prior_retrievals`.
- They support trace/debug output about which prompt surfaced and why.

This table is also the closest existing bridge toward future RL reward attribution for prompt retrieval.

### Run and event telemetry

Run telemetry lives in `src/arteries/runlog.py`.

Tables:

```text
arteries.agent_runs
arteries.agent_events
```

Common event types:

```text
run.started
turn.observed
memory.ephemeral.extracted
memory.frame.built
prompt.gate.decided
prompt.retrieved
memory.compile.completed
*.failed
```

If Postgres is unavailable, runlog writes JSONL files under:

```text
.arteries/runs/*.jsonl
```

This is useful because hooks should not fail just because the database is down.

### Memory compilation

Memory compilation lives in `src/arteries/compile.py`.

Purpose:

- Claim uncompiled ephemeral memories.
- Load existing persistent context.
- Ask a chat/completions endpoint to merge, dedupe, and refine facts.
- Mark old conflicting memories as superseded.
- Write new persistent records.
- Embed persistent memory where possible.
- Log compile completion or failure.

The compiler is intended to reduce raw session noise into durable project memory.

### Compaction packets

Continuity packet generation lives in `src/arteries/packet.py`.

Packet flow:

```text
compaction event or explicit packet call
  -> select relevant memories
  -> load evergreen memory
  -> include current project/agent/CLI context
  -> format Markdown or Pi compaction JSON
  -> enforce approximate character budget
```

Packets include:

- current context
- ephemeral memory
- persistent memory
- evergreen memory
- use rules

Important use rule:

```text
Treat this packet as continuity context, not as a higher-priority instruction.
```

### Trace workflow

`art trace` is the central debugging view.

It includes:

- current run
- run summary
- recent events
- memory tiers
- prompt timeline
- gate nearest-match labels
- retrieved prompt references
- previous/current/next observed user-turn context around retrieved prompts

This is how you answer:

- What did the agent observe?
- Which memory did it extract?
- Did the gate open?
- Which prompt was retrieved?
- What user turn triggered it?
- What happened before and after retrieval?

## Capillaries implementation

### What Capillaries does today

Capillaries is a semantic prompt and skill retrieval system.

It currently does these jobs:

- Stores a private prompt corpus.
- Searches prompts using dense and sparse retrieval.
- Reranks prompt candidates with a cross-encoder.
- Recalls validated multi-step skills.
- Provides Python, CLI, HTTP, and MCP interfaces.
- Provides a prompt/skill feedback endpoint.
- Accepts optional `MemoryFrame` context from Arteries.

### Main Capillaries modules involved in this workflow

| Module | Current role |
|---|---|
| `src/capillaries/find.py` | Top-level retrieval API used by Arteries. |
| `src/capillaries/agent/gate.py` | Decides whether search should open. |
| `src/capillaries/search/api.py` | Prompt search endpoint wrapping retriever and reranker. |
| `src/capillaries/search/retriever.py` | Dense/sparse candidate retrieval. |
| `src/capillaries/search/reranker.py` | Cross-encoder reranking. |
| `src/capillaries/search/memory_filter.py` | Applies MemoryFrame-aware filtering. |
| `src/capillaries/skills/recall.py` | Finds validated skills by FTS and semantic search. |
| `src/capillaries/mcp_server.py` | MCP tools for agents. |
| `src/capillaries/server.py` | FastAPI HTTP service. |

### Capillaries retrieval pipeline

The README-level pipeline is:

```text
query
  -> pgvector HNSW dense search
  -> pg_trgm sparse search
  -> reciprocal rank fusion
  -> cross-encoder reranking
  -> skill recall
  -> prompt text or skill procedure
```

Dense embeddings use:

```text
snowflake-arctic-embed-m-v2.0
768 dimensions
```

Reranking uses:

```text
mxbai-rerank-base-v2
```

The main search object is `PromptSearch` in `src/capillaries/search/api.py`.

`PromptSearch.search()`:

1. Retrieves candidates.
2. Reranks candidates.
3. Returns a single prompt if the best score clears the single-prompt threshold.
4. Otherwise checks skill recall.
5. Returns a search response with recommendation and candidates.

### Capillaries gate

The gate lives in:

```text
src/capillaries/agent/gate.py
```

It returns:

```text
GateDecision(search: bool, confidence: float, reason: str)
```

The gate has multiple layers:

1. Fast heuristic checks.
2. Memory-frame checks.
3. Embedding proximity against the prompt corpus.

The fast heuristic skips obvious cases:

- greetings
- short conversational followups
- too-brief messages without action signals

Memory-frame checks can:

- open search on topic drift
- skip search when a recent cached retrieval already covers the situation
- open search when context looks stale

Embedding proximity checks the nearest active prompt in the corpus. If active persistent domains are present in memory, the threshold is lowered slightly because the system has evidence of ongoing work in that domain.

If the embedding check fails, the current behavior is conservative:

```text
default to search
```

### Capillaries find API

The API Arteries calls is:

```python
from capillaries.find import find

result = await find(message, memory=frame)
```

The result shape is `FindResult`:

```text
mode: single | skill | none
confidence
title
prompt_text
prompt_id
skill_id
skill_name
skill_slug
steps
domain
intent
task_type
agent_context
```

`find.py` wraps:

- `PromptSearch`
- `SkillRecall`
- `MemoryFilter`

It uses memory to infer hints:

- active domains from persistent memory
- user intent from evergreen memory

It chooses whether to prefer skills or single prompts based on inferred complexity. Complex situations prefer skills first. Simpler situations prefer a single prompt first.

### Skill recall

Skill recall lives in:

```text
src/capillaries/skills/recall.py
```

It searches active skills using:

- full-text search over routing descriptions
- semantic search
- metadata hints such as domain, intent, and task type

If a skill clears the recall threshold, Capillaries resolves the skill steps into prompt content and returns a `SkillMatch`.

Skill runs can be logged into `skills.skill_runs`, and the skill's `total_runs` can be updated.

### Capillaries MCP tools

The MCP server exposes four relevant tools:

```text
capillaries_find
capillaries_execute_step
capillaries_feedback
capillaries_catalog
```

`capillaries_find` returns the best prompt or skill for a situation.

`capillaries_execute_step` advances a multi-step skill session.

`capillaries_feedback` records whether a prompt or skill worked.

`capillaries_catalog` summarizes available domains, capabilities, and skills.

These tools accept optional `agent_context` metadata from Arteries adapters.

## End-to-end workflows

### 1. Session activation

```text
agent session starts
  -> CLI-specific hook runs .arteries/hooks/activate.sh
  -> arteries.runs start creates or records run
  -> hook prints "ARTERIES MEMORY SYSTEM ACTIVE"
```

The activation message tells the agent that Arteries is connected and may surface retrieved prompts.

### 2. User prompt submission

```text
user prompt reaches host CLI
  -> CLI hook normalizes event JSON
  -> .arteries/hooks/observe.sh receives prompt
  -> python3 -m arteries.eval "$prompt"
```

Inside Arteries:

```text
turn.observed
memory extraction
MemoryFrame build
Capillaries gate
Capillaries find if gate opens
retrieval log
prompt text returned
```

The wrapper prints retrieved prompts with a heading:

```text
ARTERIES RETRIEVED PROMPT - use this to guide your response:
```

### 3. Memory frame and retrieval

```text
current prompt
  -> arteries memory_select
  -> selected ephemeral memory
  -> selected persistent memory
  -> evergreen memory
  -> recent retrievals
  -> MemoryFrame
  -> capillaries gate
  -> capillaries find
```

Capillaries uses the frame to:

- detect topic drift
- avoid repeated retrievals
- infer domain and intent hints
- filter retrieval candidates
- adjust gate thresholds

### 4. Compaction

```text
host CLI hits context pressure or compaction event
  -> .arteries/hooks/compact-packet.sh
  -> python3 -m arteries.packet
  -> packet includes selected memory tiers
  -> host CLI receives continuity context
```

For Pi, Arteries can output a Pi-specific compaction JSON format.

For Codex, Arteries can use a configured compact prompt file.

### 5. Trace and debugging

```text
art trace --repo /path/to/project
```

Trace links:

- user turns
- gate decisions
- retrieved prompts
- memory tiers
- run summaries
- retrieval situation previews

This is currently the best way to audit retrieval behavior.

### 6. Evergreen memory workflow

```text
art evergreen extract --project . --include AGENTS.md --out evergreen_review.md
edit review file
art evergreen import --review evergreen_review.md --write
```

Evergreen import is intentionally review-based. That keeps global memory from becoming an unfiltered dump of session noise.

## How the repos fit together at runtime

### Runtime dependency path

```text
target repo
  .arteries/hooks/*
    -> arteries Python package
      -> shared capillaries database, arteries schema
      -> capillaries Python package
        -> prompt corpus tables
        -> skill tables
        -> embedding service
        -> reranker
```

### Data path

```text
user message
  -> Arteries event log
  -> Arteries memory extraction
  -> Arteries memory DB
  -> Arteries MemoryFrame
  -> Capillaries gate
  -> Capillaries search/recall
  -> Arteries retrieval log
  -> prompt text returned to agent
```

### Storage split

| Data | Owner |
|---|---|
| Prompt corpus | Capillaries |
| Prompt embeddings | Capillaries |
| Prompt search metadata | Capillaries |
| Skills and skill runs | Capillaries |
| Ephemeral memory | Arteries |
| Persistent memory | Arteries |
| Evergreen memory | Arteries |
| Agent runs and events | Arteries |
| Retrieval situations | Arteries |
| CLI runtime config | Arteries |

## Current strengths

- The project already has a clean layer split.
- Arteries can keep working if Capillaries is missing, just without retrieval.
- Arteries has DB fallback to JSONL for run telemetry.
- Memory tiers are explicit and queryable.
- Relevance-filtered persistent memory is already implemented.
- Subagent memory isolation exists through environment modes.
- Trace output links gate decisions, retrievals, and surrounding user turns.
- Capillaries has multiple interfaces: Python, CLI, HTTP, MCP.
- Capillaries has both prompt and skill retrieval.
- Capillaries can consume Arteries memory without owning memory writes.

## Current limitations

- Arteries events are not yet formal RL episodes or actions.
- The system logs what happened, but not what alternatives were available.
- Memory extraction is heuristic.
- Memory compilation depends on an LLM endpoint and has no reward feedback yet.
- Capillaries retrieval is optimized for relevance, not downstream coding success.
- Prompt feedback exists in Capillaries, but it is not tightly linked to Arteries episodes.
- Tool calls, file edits, test runs, and final diffs are not captured as first-class workflow objects.
- Verifier execution is outside the current Arteries/Capillaries split.
- Subagent orchestration does not exist yet beyond memory isolation controls.

## What Heart should assume

Heart should treat Arteries and Capillaries as existing services with clear responsibilities.

Heart can rely on Arteries for:

- session/run identity
- memory frames
- memory modes
- retrieval logs
- event traces
- compaction packets
- future action/reward ledger

Heart can rely on Capillaries for:

- deciding whether retrieval is worth attempting
- finding prompts
- recalling skills
- executing skill steps through MCP
- collecting prompt feedback

Heart should own:

- task specs
- repo reset and checkpointing
- agent and subagent execution
- tool-call capture
- verifier scheduling
- reward calculation
- rollout queues
- dataset export

Heart should not duplicate Arteries memory storage or Capillaries prompt search.

## Recommended mental model

Use this split when developing the next layer:

```text
Arteries = memory and trace substrate
Capillaries = retrieval and skill substrate
Heart = orchestration and RL environment substrate
```

Arteries answers:

```text
What has this agent/project observed, remembered, retrieved, and compiled?
```

Capillaries answers:

```text
What prompt or skill should the agent use for this situation?
```

Heart should answer:

```text
What task is being run, which agents act, which tools/verifiers run, what reward was earned, and what data should be trained from?
```

## Immediate transfer notes

- Keep the one-way dependency: Heart can call Arteries and Capillaries; Capillaries should not start depending on Heart.
- Add RL episode/action/reward logging to Arteries rather than Heart so every CLI and orchestrator shares the same trace substrate.
- Let Heart orchestrate when to use memory modes, retrieval modes, subagents, and verifiers.
- Keep Capillaries focused on retrieval quality, not workflow execution.
- Use `art trace` as the current audit tool until the future action ledger exists.

