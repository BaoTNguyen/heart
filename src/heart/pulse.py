"""Read side of the event spine: live tail, cross-layer episode timelines,
golden-signal insights, and the symptom-level health check.

Division of labor per the SRE book: `health` is the symptom check (few simple
rules, exit code as the alert primitive), `episode <id>` is the cause
drill-down, `insights` is the dashboard. Percentiles, never averages.
"""
from __future__ import annotations

import datetime
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from .events import spool_dir


def _matches(e: dict, episode: str | None, task: str | None, source: str | None) -> bool:
    return (
        (not episode or e.get("episode_id") == episode)
        and (not task or e.get("task_id") == task)
        and (not source or e.get("source") == source)
    )


def load_events(
    episode: str | None = None, task: str | None = None, source: str | None = None
) -> list[dict]:
    events = []
    for path in sorted(spool_dir().glob("*.ndjson")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _matches(e, episode, task, source):
                events.append(e)
    events.sort(key=lambda e: e.get("ts", ""))
    return events


def render(e: dict, t0: str | None = None) -> str:
    ts = e.get("ts", "")
    if t0:
        try:
            dt = datetime.datetime.fromisoformat(ts) - datetime.datetime.fromisoformat(t0)
            when = f"+{dt.total_seconds():7.1f}s"
        except ValueError:
            when = ts
    else:
        when = ts[11:19]
    parts = [when, f"{e.get('source', '?'):<9}", e.get("kind", "?")]
    if e.get("role"):
        parts.append(f"role={e['role']}")
    if e.get("duration_ms") is not None:
        parts.append(f"{e['duration_ms']}ms")
    payload = e.get("payload") or {}
    parts += [f"{k}={_short(v)}" for k, v in payload.items()]
    return "  ".join(str(p) for p in parts)


def _short(v) -> str:
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


def tail(
    n: int = 20,
    episode: str | None = None,
    task: str | None = None,
    source: str | None = None,
    follow: bool = True,
) -> None:
    for e in load_events(episode, task, source)[-n:]:
        print(render(e))
    if not follow:
        return
    # follow the newest spool file; new days create new files, so re-resolve
    offsets: dict[Path, int] = {}
    try:
        while True:
            files = sorted(spool_dir().glob("*.ndjson"))
            for path in files[-2:]:  # current day + rollover window
                if path not in offsets:
                    offsets[path] = path.stat().st_size  # skip history, printed above
                    continue
                pos = offsets[path]
                size = path.stat().st_size
                if size > pos:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        f.seek(pos)
                        for line in f:
                            try:
                                e = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if _matches(e, episode, task, source):
                                print(render(e), flush=True)
                    offsets[path] = size
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


def episode_timeline(episode_id: str) -> list[str]:
    events = load_events(episode=episode_id)
    if not events:
        return [f"no events for episode {episode_id}"]
    t0 = events[0].get("ts")
    return [render(e, t0=t0) for e in events]


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def _cutoff_iso(hours: float) -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)
    ).isoformat()


def _window(hours: float) -> list[dict]:
    # all emitters write UTC isoformat, so lexicographic compare is safe
    cutoff = _cutoff_iso(hours)
    return [e for e in load_events() if e.get("ts", "") >= cutoff]


def _payload(e: dict) -> dict:
    return e.get("payload") or {}


def insights(hours: float = 24) -> list[str]:
    events = _window(hours)
    lines = [f"window: last {hours:g}h  events={len(events)}"]
    if not events:
        return lines

    started = {e["episode_id"]: e for e in events
               if e["kind"] == "episode.started" and e.get("episode_id")}
    finished = {e["episode_id"]: e for e in events
                if e["kind"] == "episode.finished" and e.get("episode_id")}
    crashed = [e for e in events if e["kind"] == "episode.failed"]
    turns = sum(1 for e in events if e["kind"] == "turn.observed")
    outcomes = Counter(_payload(e).get("outcome") for e in finished.values())
    lines.append(
        f"traffic: episodes={len(started)} finished={len(finished)} "
        f"crashed={len(crashed)} turns={turns}"
        + ("  outcomes: " + ", ".join(f"{k}={v}" for k, v in outcomes.most_common())
           if outcomes else "")
    )

    role_dur: dict[str, list[float]] = defaultdict(list)
    for e in events:
        if e["kind"] == "role.finished" and e.get("duration_ms") is not None:
            role_dur[e.get("role", "?")].append(e["duration_ms"] / 1000)
    for role, ds in sorted(role_dur.items()):
        lines.append(f"latency role={role}: p50={_pct(ds, .5):.1f}s p95={_pct(ds, .95):.1f}s n={len(ds)}")
    for label in ("pass", "fail"):
        ds = [e["duration_ms"] / 1000 for e in finished.values()
              if _payload(e).get("outcome") == label and e.get("duration_ms") is not None]
        if ds:
            lines.append(f"latency episode/{label}: p50={_pct(ds, .5):.1f}s p95={_pct(ds, .95):.1f}s n={len(ds)}")

    failures = Counter(e["kind"] for e in events if e["kind"].endswith(".failed"))
    if failures:
        lines.append("failures: " + ", ".join(f"{k}={v}" for k, v in failures.most_common()))

    rounds: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for e in events:
        if e["kind"] == "verify.round" and e.get("episode_id"):
            rounds[e["episode_id"]].append((_payload(e)["attempt"], _payload(e)["passed"]))
    attempted = [eid for eid, rs in rounds.items() if any(a == 0 and not p for a, p in rs)]
    rescued = [eid for eid in attempted if any(p for a, p in rounds[eid] if a > 0)]
    if rounds:
        lines.append(f"fix loop: first-verify-failed={len(attempted)} rescued={len(rescued)}")

    # routing scorecard: a cheap tier that keeps failing is the misroute
    # signature — the signal that the heuristic (or a future learned gate)
    # needs its thresholds moved
    routes = {e["episode_id"]: _payload(e).get("tier") for e in events
              if e["kind"] == "route.decided" and e.get("episode_id")}
    if routes:
        parts = []
        for tier in ("cheap", "standard", "strong"):
            outs = [_payload(finished[eid]).get("outcome")
                    for eid, t in routes.items() if t == tier and eid in finished]
            if outs:
                parts.append(f"{tier}={sum(o == 'pass' for o in outs)}/{len(outs)} pass")
        if parts:
            lines.append("routing: " + ", ".join(parts))

    gate = Counter(_payload(e).get("chosen") for e in events
                   if e["kind"] == "decision.retrieval.gate")
    if gate:
        lines.append("gate: " + ", ".join(f"{k}={v}" for k, v in gate.most_common()))

    calib = Counter(
        f"{_payload(e)['review_verdict']}/{_payload(e).get('outcome')}"
        for e in finished.values() if _payload(e).get("review_verdict")
    )
    if calib:
        lines.append("review calibration (verdict/outcome): "
                     + ", ".join(f"{k}={v}" for k, v in calib.most_common()))

    stores = Counter(_payload(e).get("store") for e in events if _payload(e).get("store"))
    degraded = stores.get("jsonl", 0) + stores.get("lost", 0)
    if degraded:
        lines.append(f"DEGRADED store: jsonl={stores.get('jsonl', 0)} "
                     f"lost={stores.get('lost', 0)} of {sum(stores.values())} — check Postgres")
    return lines


def health(hours: float = 24, zombie_minutes: float = 10) -> tuple[list[str], int]:
    """Few, simple, reliable symptom rules. Exit code is the alert primitive."""
    events = _window(hours)
    if not events:
        return [f"OK  no events in last {hours:g}h (stack idle)"], 0
    warns: list[str] = []

    done = {e.get("episode_id") for e in events
            if e["kind"] in ("episode.finished", "episode.failed")}
    stale = _cutoff_iso(zombie_minutes / 60)
    zombies = [eid for eid, e in
               {e["episode_id"]: e for e in events
                if e["kind"] == "episode.started" and e.get("episode_id")}.items()
               if eid not in done and e.get("ts", "") < stale]
    if zombies:
        warns.append(f"WARN  {len(zombies)} episode(s) started >{zombie_minutes:g}m ago, never finished: "
                     + ", ".join(zombies[:3]))

    failures = Counter(e["kind"] for e in events if e["kind"].endswith(".failed"))
    if failures:
        warns.append("WARN  failure events: "
                     + ", ".join(f"{k}={v}" for k, v in failures.most_common(5)))

    degraded = sum(1 for e in events if _payload(e).get("store") in ("jsonl", "lost"))
    if degraded:
        warns.append(f"WARN  {degraded} ledger write(s) fell back from Postgres — check the DB")

    if warns:
        return warns, 1
    return [f"OK  {len(events)} events, no zombies, no failures, no degradation"], 0
