# Sandboxed end-to-end run — findings

Date: 2026-09-13 (runs launched 2026-09-12 21:01 and 21:07 local)
Host repo: `/home/bao-tn/Coding/Projects/heart` @ `470c1ba`
Stack: arteries + capillaries + heart, all on `main`, live Postgres `capillaries` (schema `arteries`)

Three identical sandboxed sessions, same prompt, run twice. Round one failed
before reaching a model. Round two completed the full loop. This file records
both, because the round-one failure is the one that hides.

---

## How the runs were launched

```bash
HEART=/home/bao-tn/Coding/Projects/plexus/.venv/bin/heart

# round one (failed)
HEART_SANDBOX=docker-sbx $HEART work \
  "Find a real bug in this repo's Python source and fix it. Keep the change small and explain what was wrong." \
  --agent api:local --network model --solo --timeout 900

# round two (worked) -- the only difference
HEART_SANDBOX=docker-sbx HEART_SANDBOX_PROXY=http://egress:8888 $HEART work …
```

Environment at the time:

- `llama-server` on `0.0.0.0:8001` (pid 2381674)
- `egress` container up, `ALLOW=api.anthropic.com,host.docker.internal`,
  attached to both `bridge` and `heart-egress`
- `heart-agent:latest` built ~23h earlier, not stale
- `~/.config/heart/models.json` profiles `local` / `local-think` →
  `http://127.0.0.1:8001/v1`

Episode ids:

| round | episode | outcome | reward |
|---|---|---|---|
| 1 | `20260912-210157-3876b101` | `no_change` | 0.0 |
| 1 | `20260912-210157-21cd92e7` | `no_change` | 0.0 |
| 1 | `20260912-210157-6f5f491a` | `no_change` | 0.0 |
| 2 | `20260912-210719-abce2c5d` | `fail` | 0.2143 |
| 2 | `20260912-210719-740c585a` | `fail` | 0.0 |
| 2 | `20260912-210719-54297091` | `fail` | 0.2143 |

Run artifacts: `~/.local/share/heart/runs/<episode>/`
(`episode.json`, `diff.patch`, `solo.log`, `fix1.log`, `fix2.log`, `context/`).

---

## Blocker: `--network model` without a proxy is unsatisfiable, and says so illegibly

Round one died in every role with:

```
urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
  at heart/agents_api.py:159 in _chat
```

recorded as `outcome: no_change`, `reward.total: 0.0` — indistinguishable in the
summary from an agent that looked and found nothing.

### Chain

1. `sandbox.py:117-118` — `NETWORKS["model"]` and `NETWORKS["api"]` both resolve
   to `heart-egress`, which is an `--internal` docker network.
2. `sandbox.py:378 container_endpoint()` rewrites a loopback model URL to
   `host.docker.internal:<port>`.
3. On an `--internal` network Docker does not publish `host.docker.internal`.
   Verified directly:

   ```
   docker run --rm --network heart-egress python:3.13-slim \
     → host.docker.internal : resolve FAIL [Errno -3]
     → egress               : 172.18.0.2  (resolves)
   ```

   From the `egress` container (which is also on `bridge`) it does resolve, to
   `192.168.65.2`, and `http://host.docker.internal:8001/v1/models` returns 200.
   So the address is right; the *network* cannot see the name.
4. The only route out is therefore the proxy — and `sandbox.py:480 proxy_env()`
   populates `HTTP_PROXY`/`HTTPS_PROXY` **only if `HEART_SANDBOX_PROXY` is set on
   the host**. Unset, the container gets no proxy, no gateway, and no DNS.

### Resolved 2026-09-14

`sandbox.proxy_env(network)` now takes the network it is being asked about and
finds the proxy instead of waiting to be told about one. `docker network
inspect` gives `Internal` and the container list: on an internal network it
picks the container named by `HEART_EGRESS_CONTAINER` (default `egress`), or the
sole container if there is exactly one, and refuses at launch otherwise with the
`docker run` line that fixes it. Ambiguity is a refusal rather than a guess --
concurrent runs put agent containers on that same network, and picking the first
would proxy an agent through another agent.

Verified: the round-one command, with `HEART_SANDBOX_PROXY` unset, now completes
and produces a diff (`20260914-094204-ec506b67`, 45 `allow POST
host.docker.internal:8001` lines in the proxy log).

### What was missing

A refusal at launch. "Internal network requested and no proxy named" is known
before the container starts and cannot succeed. Today it surfaces 250s later as
a DNS error three roles deep. Same shape as the stale-image bug that read as a
plugin incompatibility for two weeks.

Note `172.17.0.1:8001` is **refused** from a container (the old models.json
value); `host.docker.internal` works. The earlier endpoint change was correct.

Why, measured 2026-09-14: `host.docker.internal` resolves to `192.168.65.2`,
the Docker Desktop VM's gateway, and no host interface carries that address.
So the standing advice in `sandbox.py` -- bind llama-server to the docker0
address rather than `0.0.0.0` -- would have made the model unreachable from
every container. Corrected in place. `0.0.0.0` is right here and its cost is
real: the model server answers on `tailscale0` and `proxmox0` too. That is a
host firewall job.

## Resolved alongside: the allowlist ignored ports

`contrib/egress-proxy.py permitted()` matched on hostname only. `ALLOW` carries
`host.docker.internal` so an agent can call the model on :8001 -- and that same
entry tunnelled `CONNECT host.docker.internal:5432` to the host's Postgres and
`:8000` to heart's own server. An entry may now name a port. Bare names still
allow every port, because a silent narrowing would read as the network being
broken; the running `egress` container was recreated with
`ALLOW=api.anthropic.com,host.docker.internal:8001`. Verified from inside the
network: :8001 → 200, :8000 → 403, :5432 CONNECT → 403, pypi → 403,
api.anthropic.com → 401 (reached, no key).

## Resolved alongside: 103 abandoned containers

`docker ps -a` held 103 `heart-*` containers, all `Exited (1)`, all from the
round-one batch two days earlier, each still holding its worktree mount. The
removal was the last line of the generated script, so anything that exited
early -- the outer timeout killing the client, a failed network step -- skipped
it. It is a `trap ... EXIT INT TERM` now, set the moment the sandbox is created.
The 103 are still there; `docker rm $(docker ps -aq --filter name=heart-)`
clears them.

---

## Round two: the loop works

```
abce2c5d  fail  0.2143   31 lines   public_tests 0.0  diff_quality 1.0  efficiency 0.0
740c585a  fail  0.0     919 lines   public_tests 0.0  diff_quality 0.0  efficiency 0.0
54297091  fail  0.2143   16 lines   public_tests 0.0  diff_quality 1.0  efficiency 0.0
```

Real diffs against `src/heart/runner.py` and `src/heart/sandbox.py`, three fix
rounds each, all rejected by the verifier. All three broke
`TestSandboxWrap.test_the_caller_env_reaches_the_container_but_cannot_override_the_profile`,
which passes at baseline (`python3 -m pytest tests/test_heart.py -k caller_env_reaches -q` → 1 passed).
So `fail` is honest, not infrastructural.

Token accounting, agent reachability, verifier, reward components, diff capture
and the host journal all behaved.

### Allowlist proved itself

`docker logs egress`:

```
allow POST host.docker.internal     (×many)
deny  GET  example.com
deny  CONNECT pypi.org
deny  CONNECT raw.githubusercontent.com
```

Five `sandbox.denied` events reached the host journal. The agents did try to
reach the open internet; the proxy stopped them.

---

## Issue 1 — `arteries.episodes` is a write-only table

```
select status, count(*) from arteries.episodes group by 1;
 running | 299
running older than 1 hour: 287
```

299 of 299 rows are `running`. `ended_at` is never set.

- `arteries/actionlog.py:360` — `INSERT INTO arteries.episodes (id, project_id, agent_id, task_id, run_id)`
- No `UPDATE` of `status` or `ended_at` exists anywhere in `src/arteries/`.
- `arteries.agent_runs` *does* have close logic (`arteries/runlog.py:480-507`,
  `SET ended_at = now() … WHERE id = %s AND ended_at IS NULL`). `episodes` never
  got the equivalent.

Heart emits `episode.finished` (`heart/episode.py:1037`) to the journal, so the
terminal fact exists — nothing carries it into the table.

Consequence: any query that asks "what finished, and how" is unanswerable from
the database. Retention, maturity and activity-day logic that reads episode
state is reading a column that has one value.

---

## Issue 2 — RESOLVED, and it was never about the sandbox

Corrected 2026-09-15. The framing below is wrong in a way worth keeping on the
page: it reads as a sandbox problem, and the sandbox had nothing to do with it.

Three faults, stacked, all in the drain:

1. The filter took only `source == "arteries"` events carrying an `id`. Heart
   writes `source: "heart"` with no id. Ids are synthesised from the event's
   bytes now -- deterministic, so a re-drain conflicts with itself.
2. The drain read `incoming/*/*.ndjson` and nothing else. An inbox is where a
   writer puts a line when it cannot reach the database; every host process
   writes straight to the daily file, which nothing had ever read. So no event
   heart emitted had ever reached Postgres -- sandboxed or not. There is one
   event stream, not two.
3. `agent_events.run_id` is a foreign key and `execute_batch` sends the batch
   as one statement, so one event naming a pruned run failed all of them. The
   except swallowed it as `stored: 0`. 30,707 lines sat behind one bad id.

After: 2,341 heart events across 296 episodes, plus plexus and capillaries
events that had been sitting unread in the same file. `episode_id` rides in the
payload, so per-episode queries work.

Still open, and now stated without the sandbox in it: nothing writes an
episode's *work* into the memory tiers. Events are facts about a run; ephemeral
rows come from `observe.py`/`extract.py` watching turns, and no heart episode is
watched by anything. That is a decision about what memory is for, not a bug.

## Original note (kept for the record)


Zero rows in every tier for the three round-two episodes:

```
ephemeral  0
persistent 0
evergreen  0   (for episode_id in the three)
```

(The host session was ingesting normally at the same time: 18 ephemeral,
21 persistent, 5 evergreen in the same 50-minute window, all
`episode_id = NULL`, `source` in `user`/`assistant`.)

`art journal drain` reads the inboxes and stores nothing:

```json
{"files": 2, "lines": 54, "stored": 0}
```

Cause — `arteries/journal.py:110`:

```python
if isinstance(event, dict) and event.get("source") == "arteries" and event.get("id"):
    rows.append(event)
```

Only `source == "arteries"` events carrying an `id` are stored. Every event a
sandbox writes is `source: "heart"` with no `id`, so all 54 lines are filtered
out, merged into the daily file, and the inbox is **deleted**
(`drain(delete=True)` default — the `incoming/` directory is now empty).

This is the channel chosen over mounting the Postgres socket. The mechanism
works (files appeared, were read, were folded in). What is missing is any writer
inside the container that emits the one event shape the drain accepts. Until
something does, "sandboxed work is logged" is true only of the flat file, and
nothing sandboxed ever reaches memory.

Open design question, unresolved: whether heart should write host-side
`source: "arteries"` events after a sandboxed run (mechanical episode facts —
events, not claims; transcripts referenced by path), or whether the in-container
agent should emit them.

---

## Issue 3 — the agent's own test run polluted the journal and the scope detector

The prompt sent agents into heart's source, so they ran heart's test suite inside
the sandbox. Heart's tests construct synthetic episodes and denial strings. Both
leaked into production records.

**Journal.** Episode `abce2c5d`'s inbox held 18 events belonging to
`20260913-031007-3cafc213` and `20260913-031007-3e0a8365`, `task_id: "scope"`,
`source: "heart"` — test fixtures, written into the real inbox as if they were
work.

**Scope detector.** `heart/episode.py:175-181 _DENIAL_SIGNS` scans the agent log
for refusal text. Round-two `scope_refused_paths`:

```
740c585a: ["EACCES", "Read-only file system", "], [",
           "could not create cache path ...: [Errno 30] Read-only file system",
           "error: insufficient permission for adding an object to repository database\n",
           "implement: [Errno 30] Read-only file system",
           "permission denied writing src/app.py",
           "read-only file system", "src/app.py", "src/secrets/key.pem",
           "src_gen/x.py", "test_calc.py"]
```

`test_calc.py`, `src/secrets/key.pem`, `src/app.py` and `], [` are all literals
from `tests/test_heart.py` (see `tests/test_heart.py:48,73,102,131,133`). The
detector scraped pytest's own output about denials and reported them as denials.
All three episodes carry `scope_suspect: true` on this basis.

**Separately, a genuine false positive.** All three also report
`"could not create cache path ...: [Errno 30] Read-only file system"`. A
read-only cache mount is the sandbox working as designed, not the agent being
refused a path the spec allowed. (Which cache — uv, pip, npm — is unverified.)

Two distinct fixes: stop scanning verifier/test stdout for denial signs, and
exclude cache paths from scope attribution.

---

## Issue 4 — `art rewards` re-journals its backlog every run

Every run prints `art rewards: skipped 29 unscored episode(s)`.
`arteries/actionlog.py:209-255` scans the entire runs directory each invocation.
Episodes with `reward.total: null` are deliberately skipped rather than scored
zero (correct), but each skip appends a `reward.unscored` event *again* on every
run.

Today's journal: **344 `reward.unscored`** events out of ~700 total, from a
backlog of 29 episodes. Growth is O(runs × backlog) and the backlog never
shrinks, because an unscored episode can never become scored.

---

## Today's host journal, for reference

`~/.local/share/heart/events/20260913.ndjson`:

```
reward.unscored 344   role.finished 48   decision.retrieval.route 38
verify.round 36       decision.retrieval.gate 24   role.started 20
episode.started 16    diff.captured 16   episode.finished 16
turn.observed 12      assistant.response 12        memory.assistant.stored 12
memory.coverage.measured 12   memory.ephemeral.extracted 12
decision.memory.write_policy 12   memory.frame.built 12
decision.memory.read_policy 12    prompt.gate.decided 12
corpus.suggestion.cached 12       decision.retrieval.packet 12
decision.retrieval.select 11      memory.compile.completed 11
reward.episode 11     memory.evergreen.promoted 6  sandbox.denied 5
memory.compile.duplicates_rejected 4   review.findings 4   run.resumed 1
```

Each round-two episode contributed 14 host events.

---

## Reproduction

```bash
# the blocker
docker run --rm --network heart-egress python:3.13-slim \
  python3 -c "import socket; socket.gethostbyname('host.docker.internal')"
# → socket.gaierror: [Errno -3]

# issue 1
psql capillaries -c "select status, count(*) from arteries.episodes group by 1"

# issue 2
art journal drain          # → {"files": N, "lines": M, "stored": 0}

# issue 3
grep -n "test_calc.py\|src/secrets/key.pem" tests/test_heart.py
```

---

## Priority

1 and 2 make the memory story false rather than merely noisy: nothing sandboxed
reaches any tier, and no episode is ever recorded as finished. 3 corrupts the
scope signal that feeds reward. 4 is noise growth only.

The blocker should be a refusal before either, since it costs 250s per run to
discover and looks like a successful no-op.
