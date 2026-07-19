# The event spine

The contract for cross-stack observability. This file is the canon; emitters in
other repos (arteries `spool.py`, marrow via `heart.events`) conform to it.
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
| `kind` | yes | dotted event name, `noun.verb` or `decision.<type>` / `reward.<type>` |
| `episode_id` | when known | heart-generated episode id (also via `ARTERIES_EPISODE_ID` env) |
| `task_id` | when known | TaskSpec id (also via `ARTERIES_TASK_ID` env) |
| `turn_id` / `role` | when known | finer correlation |
| `duration_ms` | when known | elapsed time of the thing the event closes |
| `payload` | optional | kind-specific fields |

Correlation model: `episode_id > run_id > role / turn_id > decision_id`.
Above episodes, plexus threads goal lineage through `task_id` by convention —
`<goal_id>-<feature_id>-a<attempt>` — plus `goal_id`/`feature_id` payload
fields on its own events; no field changes here were needed.

## Spool location

`$HEART_SPOOL_DIR`, else `~/.local/share/heart/events/`. One file per UTC day:
`YYYYMMDD.ndjson`. Writers append single lines (atomic enough on Linux);
readers tolerate torn or malformed lines by skipping them.

## The two rules that prevent drift

1. **Additive-only.** Fields and kinds are added, never renamed or removed.
2. **Tolerant readers.** Readers ignore unknown fields and unknown kinds.

With these, independent emitters cannot meaningfully diverge, which is why a
shared library isn't needed. Emission never raises: observability must never
take down the observed.

## Event catalog (current)

| Kind | Source | Notable payload |
|---|---|---|
| `episode.started/finished/failed` | heart | agent, memory_mode, retrieval, pipeline; outcome, reward; error |
| `role.started/finished` | heart | agent, memory, exit_code, timed_out |
| `route.decided` | heart | tier, agent, reason (heuristic signals or task.difficulty) |
| `verify.round` | heart | attempt, passed |
| `diff.captured` | heart | diff_lines |
| `batch.progress` | heart | done, total, outcome |
| `turn.observed`, `memory.*`, `prompt.*`, `*.failed` | arteries/capillaries | runlog tee; `store: db\|jsonl` |
| `decision.<type>` | arteries | chosen, available, cost, `store` |
| `reward.<type>` | arteries | value, reward_source, `store` |
| `training.started/progress/finished` | marrow | stage (sft/dpo/grpo), step, loss and other numeric trainer logs |
| `goal.started/finished` | plexus | goal_id; outcome |
| `plan.created/approved` | plexus | goal_id, feature count |
| `feature.started/failed/landed` | plexus | goal_id, feature_id, reason, episode_ids |
| `acceptance.round` | plexus | attempt, passed (goal-level mirror of `verify.round`) |
| `escalation.raised/resolved` | plexus | goal_id, feature_id, reason, episode_ids |

`store` reports where the durable write landed: `jsonl` means the Postgres
write failed and the record fell back to repo-local JSONL — a silent-degradation
signal (`pulse health` watches it).

## SRE mapping (sre.google/sre-book/monitoring-distributed-systems)

| Golden signal | Where it lives |
|---|---|
| Latency | `role.finished`/`episode.finished` durations → `pulse insights` p50/p95, split pass vs fail (slow failures are worst) |
| Traffic | episodes + turns per window → `pulse insights` |
| Errors | `episode.failed`, `*.failed`, outcomes (`apply_failed`, `path_violation`), reward=0 classes |
| Saturation | verifier timeouts, fix-round exhaustion, duration vs timeout budget |

Symptoms vs causes: `pulse health` is the symptom check (few, simple, reliable
rules with an exit code); `pulse episode <id>` is the cause drill-down.
Percentiles, never averages, for anything latency-shaped. No paging exists at
this scale — `pulse health`'s exit code is the alert primitive; keep its rules
to ones that are actionable, and delete any rule that never fires.
