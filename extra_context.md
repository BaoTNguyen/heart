# Extra context — heart

Session handoff, written 2026-08-26. `SPINE.md` is the event contract,
`STACK_READINESS.md` the integration plan, `PRICING.md` the cost model. This
file carries what those don't: current state, live decisions, and traps.

## State

Branch `dev`, in sync with origin, at `9b8c226` (merge of
`event-journal-and-sandbox`). Tests: `python3 -m pytest -q` → 145 passed,
1 skipped, 38 subtests, about 5 seconds. Clean.

The merge brought in dev-side work that has **not been reviewed** — notably
`tests/test_sandbox.py` and `tests/test_local_slots.py`. They pass, but passing
and reviewed are different claims.

## The sandbox is the live thread

`src/heart/sandbox.py` derives a least-privilege container profile from the
`TaskSpec`: `allowed_paths` become writable mounts, `denied_paths` get layered
back read-only. Enforcement moves from post-hoc diff inspection to the syscall.

Decisions already made — don't relitigate without new reasons:

- **The profile is data.** `docker_args()` is the only function that knows
  Docker exists. That's what keeps gVisor, a microVM, or a plain worktree from
  being a rewrite.
- **Network defaults to `none`.** A task that can't reach the network can't
  exfiltrate, and that costs nothing.
- **The `sandbox.py` fallback was removed** and replaced by a contract test.
- **bwrap is removed.** It could not express a per-task mount table, and two
  mechanisms enforcing overlapping halves of one policy drift apart. Modes are
  now `off` and `docker`; a stale `HEART_SANDBOX=bwrap` raises.
- **Workspace diffs become real commits.** Verified end to end under docker:
  the episode writes `runs/<id>/commit` and the sha is in the repo. The
  workspace concept persists because the container needs a tree to mount that
  is not the user's checkout.
- **docker-sbx carries context into each sandbox** in least-privilege style.
  Multiple sandboxes per episode; episodes are sequential.

The unresolved design problem: a mount table derived from a spec is a
*prediction*, and predictions are wrong. Two failure classes must stay
distinguishable, and the distinction is load-bearing:

| Event | Meaning |
|---|---|
| `sandbox.denied` | refused ground the spec **permitted** — our misconfiguration |
| `guardrail.hit` + `denied_path_probe` | agent reached for ground the spec **forbade** — its violation |

Collapse them and `scope_denied` becomes a reward hack: an agent facing a bad
score writes one byte to a denied path, produces an empty diff plus a refusal,
and escapes scoring. `scope_denied` therefore carries **no reward** — `None`,
not `0.0`. Zero blames the model for a sandbox we drew too tight.

Open, unresolved: review agents can generate insights needing scopes wider than
the least-privilege profile predicted. A two-phase idea is on the table — run
relaxed first, then spawn a secondary sandboxed run for the correction — but
nothing is built.

## Ownership doctrine

A feature lives in the repo whose question it answers. Heart's question is
*"was this task done correctly, and how do agents run?"* — worktrees, roles,
verifiers, sandbox, routing, reward, and cost **capture**. Not cost policy, not
goal acceptance, not memory.

Two near-duplications that are deliberate and must not be merged:

- **plexus acceptance vs heart review/verify.** Heart judges the task with its
  own verifiers; plexus judges the goal against ground truth. The "heart passed
  / acceptance failed" cell is a hard negative marrow cannot otherwise see.
- **capillaries feedback vs arteries rewards.** One is a local relevance signal,
  the other an episode outcome. Joined by `episode_id`; neither replaces the
  other.

## Identity flows downward

Plexus names episodes and passes them down. Heart assigns run IDs to arteries
per subagent session; arteries then operates independently within that session
scope. Architecture is unidirectional — capillaries feeds arteries feeds heart.
**Arteries never imports heart.**

## Traps

- **`.arteries/hooks/` is generated and gitignored.** Capillaries lost its
  entire hook install this way and ran blind for a week. Verify the directory
  exists; don't trust `settings.local.json` referencing it.
- Inside a container, `127.0.0.1` is the container. A model server on host
  loopback is unreachable regardless of network — see the comment at the top of
  `sandbox.py`.
- The spine is telemetry, never a system of record. Written best-effort after
  the authoritative write, never read back as state.
