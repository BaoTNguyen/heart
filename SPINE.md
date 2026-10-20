# The event spine

## Canonical contract: pulse

pulse ([github.com/BaoTNguyen/pulse](https://github.com/BaoTNguyen/pulse)) now owns
the contract. The canonical wire format is pulse's `contract/envelope.json`, and the
kind catalog is pulse's `contract/catalog.json`. The new optional fields (`id`,
`schema_version`, `project`, trace ids, `context_id`, `emitter_seq`, `trust`) are
defined there. This file keeps the prose: the two drift rules, the
`sandbox.denied` vs `guardrail.hit` distinction, and the SRE mapping. The catalog
table below is a summary and may lag behind pulse's catalog.

The contract for cross-stack observability. This file is the canon; emitters in
other repos (arteries `journal.py`, marrow via `heart.events`) conform to it.
There is deliberately **no shared library** — the standard is this wire format,
like syslog. Emitters are ~40 lines of stdlib each and stay that dumb.

## Wire format

Append-only NDJSON, one event per line:

```json
{"ts": "2026-07-04T23:22:21.480221+00:00", "source": "arteries",
 "kind": "decision.retrieval.gate", "episode_id": "ep-demo-1",
 "task_id": "demo-task", "turn_id": "…",
 "payload": {"chosen": "abstain", "available": ["abstain", "search"], "store": "db"}}
```

| Field | Required | Meaning |
|---|---|---|
| `ts` | yes | UTC ISO-8601, emitter's clock |
| `source` | yes | `heart` \| `arteries` \| `capillaries` \| `marrow` \| `plexus` \| `agent` |
| `kind` | yes | dotted event name, `noun.verb` or `decision.<type>` |
| `episode_id` | when known | heart-generated episode id (also via `ARTERIES_EPISODE_ID` env) |
| `task_id` | when known | TaskSpec id (also via `ARTERIES_TASK_ID` env) |
| `turn_id` / `role` | when known | finer correlation |
| `duration_ms` | when known | elapsed time of the thing the event closes |
| `payload` | optional | kind-specific fields |

Correlation model: `episode_id > run_id > role / turn_id > decision_id`.
Above episodes, plexus threads goal lineage through `task_id` by convention —
`<goal_id>-<feature_id>-a<attempt>` — plus `goal_id`/`feature_id` payload
fields on its own events; no field changes here were needed. heart's own
`events.emit()` mirrors this: when `PLEXUS_GOAL_ID`/`PLEXUS_FEATURE_ID` are set
in the environment, it stamps `payload.goal_id`/`payload.feature_id` onto
every event it emits (optional, additive, skipped if the call site already
set them) — `pulse goal <goal-id>` reads this back. They are also set as
the envelope's top-level `goal_id`/`feature_id`, which is what pulse queries
by; the payload copies stay for older readers.

## Journal location

`$EVENT_JOURNAL_DIR`, else `~/.vascular/spool/events/`. One file per UTC day:
`YYYYMMDD.ndjson`. Writers append single lines (atomic enough on Linux);
readers tolerate torn or malformed lines by skipping them.
`$VASCULAR_HOME` moves the whole `~/.vascular` root, journal included.

## The two rules that prevent drift

1. **Additive-only.** Fields and kinds are added, never renamed or removed.
2. **Tolerant readers.** Readers ignore unknown fields and unknown kinds.

With these, independent emitters cannot meaningfully diverge, which is why a
shared library isn't needed. Emission never raises: observability must never
take down the observed.

## Event catalog (current)

| Kind | Source | Notable payload |
|---|---|---|
| `episode.started/finished/failed` | heart | agent, memory_mode, retrieval, pipeline; outcome; error |
| `role.started/finished` | heart | agent, memory, exit_code, timed_out, prompt_artifact (started), output_artifact (finished); envelope `origin` on finished only (`agent`, `external` if network=web) |
| `route.decided` | heart | tier, agent, reason (heuristic signals or task.difficulty) |
| `route.failover` | heart | role, wanted (the tier's local agent), agent (used instead), reason |
| `verify.round` | heart | attempt, passed, output_artifact; envelope `origin` (`agent`/`external`) |
| `diff.captured` | heart | diff_lines, diff_artifact; envelope `origin` (`agent`/`external`) |
| `sandbox.denied` | heart | `paths` (worktree-relative, refused), allowed_paths, denied_paths, network, `evidence` (the raw refusal lines) |
| `guardrail.hit` | heart | `rules` (`denied_path_probe`, or the secret-scan rules), `paths` (worktree-relative) |
| `batch.progress` | heart | done, total, outcome |
| `turn.observed`, `memory.*`, `prompt.*`, `*.failed` | arteries/capillaries | runlog tee; `store: db\|jsonl` |
| `decision.<type>` | arteries | chosen, available, cost, `store` |
| `training.started/progress/finished` | marrow | stage (sft/dpo/grpo), step, loss and other numeric trainer logs |
| `goal.started/finished` | plexus | goal_id; outcome |
| `plan.created/approved` | plexus | goal_id, feature count |
| `feature.started/failed/landed` | plexus | goal_id, feature_id, reason, episode_ids |
| `acceptance.round` | plexus | attempt, passed (goal-level mirror of `verify.round`) |
| `escalation.raised/resolved` | plexus | goal_id, feature_id, reason, episode_ids |

`*_artifact` values are sha256 hex digests of blobs at <journal>/artifacts/<sha256>.

`sandbox.denied` and `guardrail.hit` are the two halves of one distinction, and
consumers must keep them apart: the first says the sandbox refused ground the
spec permitted (heart drew the mount table too tight, and the episode is
excused as `scope_denied`), the second says the agent reached for ground the
spec forbade (its violation, `path_violation`). Collapse them and `scope_denied`
becomes an escape from failing.

`sandbox.denied` fires whenever a refusal on permitted ground appears in the
episode's logs, whatever the outcome — a passing episode can carry one. It is a
report that the mount table was tighter than the work needed, not a verdict on
the episode, and it never moves the outcome. Consumers that only looked at it
alongside `scope_denied` were seeing the tail of the distribution: the scopes so
tight the agent produced nothing, never the far more common scope that let an
agent start and not finish. Episodes in the second group carry `scope_suspect:
true` and `scope_refused_paths` on the episode record.

Both carry `paths` already parsed and worktree-relative — the vocabulary
`allowed_paths` speaks, not the container's `/work` mount point. Read that
field; do not re-parse `evidence`. It exists for a human reading a trace, and a
second parser for the same prose is how the two drifted apart the first time.

`store` reports where the durable write landed: `jsonl` means the Postgres
write failed and the record fell back to repo-local JSONL — a silent-degradation
signal (`pulse health` watches it).

## SRE mapping (sre.google/sre-book/monitoring-distributed-systems)

| Golden signal | Where it lives |
|---|---|
| Latency | `role.finished`/`episode.finished` durations → `pulse insights` p50/p95, split pass vs fail (slow failures are worst) |
| Traffic | episodes + turns per window → `pulse insights` |
| Errors | `episode.failed`, `*.failed`, outcomes (`apply_failed`, `path_violation`, `guardrail_violation`) |
| Saturation | verifier timeouts, fix-round exhaustion, duration vs timeout budget |

Symptoms vs causes: `pulse health` is the symptom check (few, simple, reliable
rules with an exit code); `pulse episode <id>` is the cause drill-down.
Percentiles, never averages, for anything latency-shaped. No paging exists at
this scale — `pulse health`'s exit code is the alert primitive; keep its rules
to ones that are actionable, and delete any rule that never fires.
