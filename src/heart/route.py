"""Capability routing: pick the model + effort for a task, subject to hard
constraints, from declared skills corrected by measured outcomes.

This is the richer sibling of router.py (which maps a task to a single tier).
Here a *model manifest* gives each model per-skill scores, a context window, and
a difficulty ceiling; a task declares (or the classifier infers) the skills it
needs, its difficulty, and its context size; `route` filters to the models that
can actually run it and picks the best capability match, biased toward cheaper
models and corrected by a measured reward sidecar so declared scores can't drift
unchecked.

Capabilities are authored ORDINALLY, not as scores — a general `tier` sets a
baseline competence (and, by default, the difficulty ceiling), and `skills`
optionally notes the few a model is notably strong/weak at. No hand-written
floats: consensus gives you "frontier, great at planning, weak at vision," not
"planning 0.9." The declared part only *orders* models before data exists; the
real numbers come from the measured reward sidecar, which corrects the prior.

Config (~/.config/heart/models.json), additive to tiers/profiles/pricing:

    {"models": {
        "claude":     {"agent": "claude",   "tier": "frontier",
                       "skills": {"planning": "strong", "vision": "weak"},
                       "context": 200000, "cost": 3.0},
        "gpt":        {"agent": "api:gpt",   "tier": "frontier",
                       "skills": {"coding": "strong"}, "context": 400000, "cost": 2.0},
        "local-qwen": {"agent": "api:local", "tier": "small",
                       "skills": {"coding": "capable"}, "context": 32000, "cost": 0.1}}}

`tier` -> baseline competence + difficulty ceiling (override the ceiling with an
explicit `max_difficulty`). `skills` values are strong|capable|weak; an
unlisted skill inherits the tier baseline. Legacy numeric skill scores still
parse. Measured stats live in a SEPARATE machine-written sidecar so the
aggregator never clobbers hand-authored edits.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

from .events import emit

# The shared skill vocabulary. Both a model's manifest scores and a task's
# required skills draw from this; anything off-list is ignored. Extend by editing.
SKILLS = (
    "coding", "frontend", "backend", "planning", "debug",
    "refactor", "data", "docs", "vision", "infra",
)
EFFORTS = ("low", "medium", "high")

_DIFF_RANK = {"trivial": 0, "easy": 1, "unknown": 2, "medium": 2, "hard": 3, "expert": 4}

# General competence tier -> baseline skill prior + default difficulty ceiling.
# These are the only place a tier-word becomes a coarse number, and that number
# is just a rank seed the measured reward overrides — not a claim of precision.
_TIER_BASELINE = {"frontier": 0.6, "mid": 0.45, "small": 0.3}
_TIER_CEILING = {"frontier": "hard", "mid": "medium", "small": "easy"}
# Per-skill ordinal notes override the baseline for that skill.
_LEVEL = {"strong": 0.85, "capable": 0.6, "weak": 0.35}

# shrinkage: measured mean fully takes over from the declared prior at ~K samples
_BLEND_K = 8


def _config_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "heart" / "models.json"


def _stats_path() -> Path:
    return Path(os.environ.get("HEART_ROUTE_STATS",
                               str(Path.home() / ".local" / "share" / "heart" / "route_stats.json")))


def drank(difficulty: str) -> int:
    return _DIFF_RANK.get(difficulty, 2)


# --- manifest -------------------------------------------------------------

def load_manifest(path: str | Path | None = None) -> dict:
    """The model manifest, normalized. Falls back to synthesizing one from the
    legacy `tiers` map so a config that predates `models` still routes (uniform
    skills per tier, difficulty ceiling by tier), and returns {} if neither
    exists — callers treat an empty manifest as 'routing unavailable'."""
    p = Path(path) if path else _config_path()
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("models"):
        out = {}
        for name, m in data["models"].items():
            tier = m.get("tier", "mid")
            out[name] = {
                "agent": m.get("agent", name),
                "skills": _skill_scores(m.get("skills")),
                "baseline": _TIER_BASELINE.get(tier, 0.45),
                "context": int(m.get("context", 1_000_000)),
                "max_difficulty": m.get("max_difficulty") or _TIER_CEILING.get(tier, "medium"),
                "cost": float(m.get("cost", 1.0)),
            }
        return out
    # legacy fallback: one synthetic model per tier
    tiers = data.get("tiers") or {}
    tier_score = {"cheap": 0.5, "standard": 0.7, "strong": 0.9}
    tier_ceiling = {"cheap": "easy", "standard": "medium", "strong": "expert"}
    tier_cost = {"cheap": 0.1, "standard": 1.0, "strong": 3.0}
    out = {}
    for tier, agent in tiers.items():
        out[tier] = {
            "agent": agent,
            "skills": {},
            "baseline": tier_score.get(tier, 0.6),
            "context": 1_000_000,
            "max_difficulty": tier_ceiling.get(tier, "hard"),
            "cost": tier_cost.get(tier, 1.0),
        }
    return out


def _skill_scores(raw) -> dict:
    """Normalize a model's skill notes to {skill: coarse prior}. Accepts a dict of
    ordinal levels (strong|capable|weak), a bare list (treated as 'strong'), or
    legacy numeric scores. Off-vocabulary skills are dropped."""
    if isinstance(raw, list):
        return {s: _LEVEL["strong"] for s in raw if s in SKILLS}
    out = {}
    for s, v in (raw or {}).items():
        if s not in SKILLS:
            continue
        out[s] = float(v) if isinstance(v, (int, float)) else _LEVEL.get(str(v).lower(), _LEVEL["capable"])
    return out


# --- measured feedback loop ----------------------------------------------

def aggregate(events: list[dict]) -> dict:
    """Roll episode outcomes into per-(model, skill, difficulty) reward stats.

    Reads `episode.finished` events that carry agent + reward + the task's skills
    and difficulty. Multi-skill tasks credit every listed skill equally — noisy
    per task, but the confound washes out in aggregate (documented v1 choice).
    Keyed by difficulty too, so a model fed only hard tasks isn't unfairly
    compared to one fed easy ones.
    """
    acc: dict = {}
    for e in events:
        if e.get("kind") != "episode.finished":
            continue
        p = e.get("payload") or {}
        agent, reward = p.get("agent"), p.get("reward")
        skills, difficulty = p.get("skills") or [], p.get("difficulty", "unknown")
        if not agent or reward is None or not skills:
            continue
        for s in skills:
            cell = acc.setdefault(agent, {}).setdefault(f"{s}|{difficulty}", {"n": 0, "sum": 0.0})
            cell["n"] += 1
            cell["sum"] += float(reward)
    return {
        agent: {key: {"n": c["n"], "mean": round(c["sum"] / c["n"], 4)}
                for key, c in cells.items()}
        for agent, cells in acc.items()
    }


def refresh_stats() -> dict:
    """Rebuild the sidecar from the spool. Best-effort; returns the stats."""
    try:
        from .pulse import load_events
        stats = aggregate(load_events())
    except Exception:
        return {}
    path = _stats_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats))
    except OSError:
        pass
    return stats


def load_stats() -> dict:
    try:
        return json.loads(_stats_path().read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _blend(declared: float, cell: dict | None, k: int = _BLEND_K) -> float:
    """Declared score corrected by measured evidence. n=0 -> pure prior; the
    measured mean takes over as evidence accrues (weight n/(n+k))."""
    if not cell or not cell.get("n"):
        return declared
    n, mean = cell["n"], cell["mean"]
    w = n / (n + k)
    return (1 - w) * declared + w * mean


# --- classification (heuristic; the LLM seam is the caller/planner) --------

_SKILL_WORDS = {
    "frontend": ("css", "react", "component", "ui", "button", "layout", "html", "tailwind"),
    "backend": ("endpoint", "api", "server", "route", "handler", "database", "sql", "query"),
    "planning": ("design", "architect", "plan", "decompose", "approach", "strategy"),
    "debug": ("bug", "fix", "error", "crash", "traceback", "failing", "reproduce"),
    "refactor": ("refactor", "rename", "restructure", "extract", "migrate", "clean up"),
    "data": ("dataframe", "pipeline", "etl", "csv", "parquet", "aggregate", "dataset"),
    "docs": ("readme", "docstring", "document", "comment", "changelog"),
    "vision": ("image", "screenshot", "diagram", "photo", "visual", "ocr"),
    "infra": ("docker", "ci", "deploy", "kubernetes", "terraform", "workflow", "pipeline yaml"),
    "coding": ("implement", "function", "class", "write", "add", "test"),
}
_HARD_WORDS = ("concurren", "thread", "race", "deadlock", "protocol", "security",
               "performance", "distributed", "architect", "rewrite")


def classify(task) -> tuple[list[str], str, int]:
    """(skills, difficulty, min_context) for a task. Honors whatever the task
    already declares (from a planner); infers the rest from the prompt. A blunt
    keyword heuristic — good enough as the prior, and the measured loop corrects
    what it gets wrong."""
    text = task.prompt.lower()
    skills = [s for s in task.skills if s in SKILLS] if task.skills else [
        s for s, words in _SKILL_WORDS.items() if any(w in text for w in words)]
    if not skills:
        skills = ["coding"]

    if task.difficulty in _DIFF_RANK and task.difficulty not in ("unknown",):
        difficulty = task.difficulty
    else:
        words = len(task.prompt.split())
        hard = any(w in text for w in _HARD_WORDS)
        difficulty = "hard" if hard else "medium" if words > 60 else "easy"

    # rough context estimate: prompt size in tokens (~chars/4); a real one would
    # add the files the task will read, which the caller knows better than we do.
    min_context = task.min_context or max(2000, len(task.prompt) // 4)
    return skills, difficulty, min_context


# --- the decision ---------------------------------------------------------

@dataclass
class RouteDecision:
    agent: str
    effort: str
    skills: list[str]
    difficulty: str
    reason: str
    candidates: list[dict] = field(default_factory=list)


# Effort is effectively medium-by-default, high for genuinely hard work — low is
# valid but rarely the right call, so nothing maps to it automatically.
_DIFFICULTY_EFFORT = {"trivial": "medium", "easy": "medium", "unknown": "medium",
                      "medium": "medium", "hard": "high", "expert": "high"}


def route(
    task,
    manifest: dict | None = None,
    stats: dict | None = None,
    available=None,
    explore: float = 0.0,
    rng: random.Random | None = None,
) -> RouteDecision:
    """Pick (agent, effort) for a task.

    Hard filters first (a match to a model that can't run the task is useless):
    context window >= what the task touches, difficulty ceiling >= the task's
    difficulty, and availability if a probe is given. Then score the survivors by
    mean blended skill match, breaking ties toward the cheaper model — don't send
    easy work to the expensive one. With probability `explore`, pick a viable but
    under-measured candidate instead, so new/updated models earn the data that
    keeps their declared scores honest.
    """
    manifest = manifest if manifest is not None else load_manifest()
    stats = stats if stats is not None else load_stats()
    rng = rng or random
    skills, difficulty, min_context = classify(task)
    effort = task.effort if task.effort in EFFORTS else _DIFFICULTY_EFFORT.get(difficulty, "medium")

    if not manifest:
        # no manifest: fall back to whatever the task/agent already was
        return RouteDecision(agent=getattr(task, "agent", "claude") or "claude",
                             effort=effort, skills=skills, difficulty=difficulty,
                             reason="no model manifest; routing unavailable")

    scored: list[dict] = []
    filtered: list[str] = []
    for name, m in manifest.items():
        if m["context"] < min_context:
            filtered.append(f"{name}:context<{min_context}")
            continue
        if drank(m["max_difficulty"]) < drank(difficulty):
            filtered.append(f"{name}:below {difficulty}")
            continue
        if available is not None and not available(m["agent"]):
            filtered.append(f"{name}:unavailable")
            continue
        cells = stats.get(m["agent"]) or stats.get(name) or {}
        # unlisted skills fall back to the model's tier baseline, not zero — a
        # frontier model is generally competent even where you didn't annotate it
        per_skill = [_blend(m["skills"].get(s, m.get("baseline", 0.45)),
                            cells.get(f"{s}|{difficulty}")) for s in skills]
        score = sum(per_skill) / len(per_skill)
        measured_n = sum(cells.get(f"{s}|{difficulty}", {}).get("n", 0) for s in skills)
        scored.append({"name": name, "agent": m["agent"], "score": round(score, 4),
                       "cost": m["cost"], "n": measured_n})

    if not scored:
        # every model filtered out — take the highest-ceiling model as a last
        # resort rather than failing the task outright
        best = max(manifest.items(), key=lambda kv: drank(kv[1]["max_difficulty"]))
        return RouteDecision(agent=best[1]["agent"], effort="high", skills=skills,
                             difficulty=difficulty,
                             reason=f"no model cleared constraints ({'; '.join(filtered)}); "
                                    f"forced highest-ceiling model")

    # exploration: occasionally give a viable low-evidence candidate the traffic
    if explore and rng.random() < explore:
        least = min(scored, key=lambda c: c["n"])
        if least["n"] < _BLEND_K:
            pick, why = least, f"exploration (n={least['n']})"
        else:
            pick, why = _best(scored), "capability match"
    else:
        pick, why = _best(scored), "capability match"

    decision = RouteDecision(
        agent=pick["agent"], effort=effort, skills=skills, difficulty=difficulty,
        reason=f"{why}: {pick['name']} score={pick['score']} for {skills}@{difficulty}",
        candidates=sorted(scored, key=lambda c: -c["score"]),
    )
    try:
        emit("heart", "route.decided", task_id=getattr(task, "task_id", None),
             agent=decision.agent, effort=effort, skills=skills, difficulty=difficulty,
             reason=decision.reason, candidates=decision.candidates,
             filtered=filtered)
    except Exception:
        pass
    return decision


def _best(scored: list[dict]) -> dict:
    # highest score; ties -> cheapest (don't overspend when models are equal)
    return min(scored, key=lambda c: (-c["score"], c["cost"]))
