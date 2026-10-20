"""The event spine: append-only NDJSON journal written as things happen.

Every layer appends here — heart lifecycle, arteries runlog/ledger tee, marrow
training — one JSON object per line, correlated by episode_id/task_id. Postgres
(arteries) remains the queryable archive; the journal is the live view.

Journal: $EVENT_JOURNAL_DIR or ~/.vascular/spool/events/YYYYMMDD.ndjson
Emission must never break the work it observes: emit() swallows everything.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import pulse_emit, vascular_paths


def journal_dir() -> Path:
    return vascular_paths.journal_dir()


def artifact(data: bytes | Path) -> str | None:
    """Store `data` under <journal_dir()>/artifacts/<sha256>; return the digest
    or None. A Path is streamed, never read whole. Never raises."""
    try:
        return pulse_emit.artifact(data, journal_dir())
    except Exception:
        return None


def emit(
    source: str,
    kind: str,
    *,
    episode_id: str | None = None,
    task_id: str | None = None,
    role: str | None = None,
    duration_ms: int | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    trace_id: str | None = None,
    origin: str | None = None,
    **payload,
) -> None:
    try:
        # env fallback lets subprocesses (agents, arteries hooks) stamp the
        # episode without plumbing arguments through
        episode_id = episode_id or os.environ.get("ARTERIES_EPISODE_ID")
        task_id = task_id or os.environ.get("ARTERIES_TASK_ID")
        # goal lineage: plexus sets these env vars around its dispatch; every
        # event heart emits picks them up unless the call site already set
        # them explicitly (additive payload fields, SPINE.md-legal)
        for key, env_var in (("goal_id", "PLEXUS_GOAL_ID"), ("feature_id", "PLEXUS_FEATURE_ID")):
            if key not in payload:
                value = os.environ.get(env_var)
                if value:
                    payload[key] = value
        # ts/source/kind and the envelope fields come from pulse_emit
        pulse_emit.emit(
            source, kind, journal_dir(), vascular_paths.path("state", "pulse", "contexts"),
            episode_id=episode_id, task_id=task_id, role=role, duration_ms=duration_ms,
            # None defers to pulse_emit's $PULSE_TRACE_ID/$PULSE_PARENT_SPAN fallback
            span_id=span_id, parent_span_id=parent_span_id, trace_id=trace_id,
            # operator|agent|external, passed through unchecked; None drops the key
            origin=origin,
            # top-level copies are what pulse queries by; None (never "") drops the key
            goal_id=payload.get("goal_id") or None, feature_id=payload.get("feature_id") or None,
            payload=payload or None,
        )
    except Exception:
        pass  # observability must never take down the observed
