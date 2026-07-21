"""The event spine: append-only NDJSON spool written as things happen.

Every layer appends here — heart lifecycle, arteries runlog/ledger tee, marrow
training — one JSON object per line, correlated by episode_id/task_id. Postgres
(arteries) remains the queryable archive; the spool is the live view.

Spool: $HEART_SPOOL_DIR or ~/.local/share/heart/events/YYYYMMDD.ndjson
Emission must never break the work it observes: emit() swallows everything.
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path


def spool_dir() -> Path:
    return Path(
        os.environ.get("HEART_SPOOL_DIR", str(Path.home() / ".local" / "share" / "heart" / "events"))
    )


def emit(
    source: str,
    kind: str,
    *,
    episode_id: str | None = None,
    task_id: str | None = None,
    role: str | None = None,
    duration_ms: int | None = None,
    **payload,
) -> None:
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        event: dict = {"ts": now.isoformat(), "source": source, "kind": kind}
        # env fallback lets subprocesses (agents, arteries hooks) stamp the
        # episode without plumbing arguments through
        episode_id = episode_id or os.environ.get("ARTERIES_EPISODE_ID")
        task_id = task_id or os.environ.get("ARTERIES_TASK_ID")
        for key, value in (
            ("episode_id", episode_id), ("task_id", task_id),
            ("role", role), ("duration_ms", duration_ms),
        ):
            if value is not None:
                event[key] = value
        # goal lineage: plexus sets these env vars around its dispatch; every
        # event heart emits picks them up unless the call site already set
        # them explicitly (additive payload fields, SPINE.md-legal)
        for key, env_var in (("goal_id", "PLEXUS_GOAL_ID"), ("feature_id", "PLEXUS_FEATURE_ID")):
            if key not in payload:
                value = os.environ.get(env_var)
                if value:
                    payload[key] = value
        if payload:
            event["payload"] = payload
        d = spool_dir()
        d.mkdir(parents=True, exist_ok=True)
        # ponytail: one small append per event is atomic enough on Linux;
        # revisit with a buffered writer only if hook latency ever shows it
        with open(d / now.strftime("%Y%m%d.ndjson"), "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass  # observability must never take down the observed
