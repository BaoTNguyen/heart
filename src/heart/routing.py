"""Capability routing, in two stages: what a task demands, then which model serves it.

    demand(task) -> Demand     what this work needs: a tier, skills, difficulty, context
    pick(demand)  -> Choice    which model in the manifest serves that demand best

There was a second router. `router.py` mapped a task to one of `cheap|standard|strong`
and looked that word up in a `tiers` map, while this module scored a real manifest —
two files a letter apart, each with a `classify()` returning something different, and
the one that fired for plain `--agent auto` was the one that could not see the manifest.
One vocabulary now: a tier is `small|mid|frontier`, a task demands one and a model is
declared in one.

Capabilities are authored ORDINALLY, not as scores — a general `tier` sets a
baseline competence (and, by default, the difficulty ceiling), and `skills`
optionally notes the few a model is notably strong/weak at. No hand-written
floats: consensus gives you "frontier, great at planning, weak at vision," not
"planning 0.9." The declared part only *orders* models before data exists; the
real numbers come from the measured reward sidecar, which corrects the prior.

Config (~/.config/heart/models.json):

    {"models": {
        "claude":     {"agent": "claude",   "tier": "frontier",
                       "skills": {"planning": "strong", "vision": "weak"},
                       "context": 200000, "cost": 3.0},
        "gpt":        {"agent": "api:gpt",   "tier": "frontier",
                       "skills": {"coding": "strong"}, "context": 400000, "cost": 2.0},
        "local-qwen": {"agent": "api:local", "tier": "small",
                       "skills": {"coding": "capable"}, "context": 32000, "cost": 0.1}},
     "defaults": {"small": "local-qwen", "mid": "gpt", "frontier": "claude"}}

`models` is the source of truth for what exists. `defaults` names the model you
want when something asks for a tier by name rather than by task — a role that
declares `"tier": "small"`, or an escalation asking for `frontier`. A default is
a preference, not a classification: naming a frontier model as the `mid` default
is allowed and means "when mid-tier work shows up, use this".

`tier` -> baseline competence + difficulty ceiling (override the ceiling with an
explicit `max_difficulty`). `skills` values are strong|capable|weak; an
unlisted skill inherits the tier baseline. Legacy numeric skill scores still
parse. Measured stats live in a SEPARATE machine-written sidecar so the
aggregator never clobbers hand-authored edits.

The retired `cheap|standard|strong` spelling still reads, from a `tiers` map and
from HEART_TIER_CHEAP/STANDARD/STRONG, so a config written before this keeps
routing. It maps cheap->small, standard->mid, strong->frontier.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

from .agents_api import load_models_json, models_json_path
from .events import emit
from .pulse import load_events

# The shared skill vocabulary. Both a model's manifest scores and a task's
# required skills draw from this; anything off-list is ignored. Extend by editing.
SKILLS = (
    "coding", "frontend", "backend", "planning", "debug",
    "refactor", "data", "docs", "vision", "infra",
)
EFFORTS = ("low", "medium", "high")

#: Capability classes, weakest first. The one tier vocabulary: a task demands
#: one of these, a model declares one, `defaults` names one model per tier.
TIERS = ("small", "mid", "frontier")

#: The retired spelling, still read from config and the environment.
_LEGACY_TIERS = {"cheap": "small", "standard": "mid", "strong": "frontier"}
_LEGACY_OF = {new: old for old, new in _LEGACY_TIERS.items()}

_DIFF_RANK = {"trivial": 0, "easy": 1, "unknown": 2, "medium": 2, "hard": 3, "expert": 4}

#: Stage one's whole answer: how hard the work is -> what class of model it needs.
#: An explicit difficulty from a planner is honoured; `demand` infers one when the
#: task does not declare it.
DEMAND_TIER = {"trivial": "small", "easy": "small", "unknown": "mid",
               "medium": "mid", "hard": "frontier", "expert": "frontier"}

# General competence tier -> baseline skill prior + default difficulty ceiling.
# These are the only place a tier-word becomes a coarse number, and that number
# is just a rank seed the measured reward overrides — not a claim of precision.
_TIER_BASELINE = {"frontier": 0.6, "mid": 0.45, "small": 0.3}
_TIER_CEILING = {"frontier": "hard", "mid": "medium", "small": "easy"}
# Per-skill ordinal notes override the baseline for that skill.
_LEVEL = {"strong": 0.85, "capable": 0.6, "weak": 0.35}

# shrinkage: measured mean fully takes over from the declared prior at ~K samples
_BLEND_K = 8


def _stats_path() -> Path:
    return Path(os.environ.get("HEART_ROUTE_STATS",
                               str(Path.home() / ".local" / "share" / "heart" / "route_stats.json")))


def drank(difficulty: str) -> int:
    return _DIFF_RANK.get(difficulty, 2)


def tier_of(difficulty: str) -> str:
    """Difficulty -> the tier that work demands. Stage one, on its own, because
    a caller with a difficulty and no TaskSpec still deserves the answer."""
    return DEMAND_TIER.get(difficulty, "mid")


# --- manifest -------------------------------------------------------------

def load_manifest(path: str | Path | None = None) -> dict:
    """The model manifest, normalized. Falls back to synthesizing one from the
    legacy `tiers` map so a config that predates `models` still routes (uniform
    skills per tier, difficulty ceiling by tier), and returns {} if neither
    exists — callers treat an empty manifest as 'routing unavailable'."""
    p = Path(path) if path else models_json_path()
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
                "tier": tier,
                "skills": _skill_scores(m.get("skills")),
                "baseline": _TIER_BASELINE.get(tier, 0.45),
                "context": int(m.get("context", 1_000_000)),
                "max_difficulty": m.get("max_difficulty") or _TIER_CEILING.get(tier, "medium"),
                "cost": float(m.get("cost", 1.0)),
            }
        return out
    # legacy fallback: one synthetic model per tier, under the current spelling
    out = {}
    for legacy, agent in (data.get("tiers") or {}).items():
        tier = _LEGACY_TIERS.get(legacy, legacy)
        out[tier] = {
            "agent": agent,
            "tier": tier,
            "skills": {},
            "baseline": _TIER_BASELINE.get(tier, 0.6),
            "context": 1_000_000,
            # a hand-named per-tier default is a statement of intent, so let it
            # take work up to its class rather than capping it at the ceiling a
            # real manifest entry would get
            "max_difficulty": {"small": "easy", "mid": "medium"}.get(tier, "expert"),
            "cost": {"small": 0.1, "mid": 1.0}.get(tier, 3.0),
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


# --- naming a model -------------------------------------------------------

def defaults() -> dict:
    """{tier: name-or-agent-string}, the model to use when something asks for a
    tier by name. Reads `defaults`, then the retired `tiers` map."""
    data = load_models_json()
    out = {t: n for t, n in (data.get("defaults") or {}).items() if t in TIERS}
    for legacy, name in (data.get("tiers") or {}).items():
        out.setdefault(_LEGACY_TIERS.get(legacy, legacy), name)
    return out


def resolve(name: str, default: str | None = None) -> str:
    """A configured name -> an agent string heart's runner can execute.

    Four kinds of name, and they must not overlap:
      "auto"             routing decides per task; returned unchanged, because
                         the decision needs a task and this function has none
      a tier             `defaults[tier]`, else the cheapest model declared in
                         that tier, else `default`
      a manifest model   that model's `agent`
      anything else      verbatim, so `claude:opus5` still works unregistered

    A manifest model named after a tier is refused rather than guessed at: one
    of the two readings would silently win, and which one is not something the
    config author can see.
    """
    if name == "auto":
        return name
    manifest = load_manifest()
    # against the hand-authored `models` keys, not the manifest: a legacy
    # `tiers`-only config synthesizes entries named after the tiers on purpose
    collision = sorted(set(load_models_json().get("models") or {}) & set(TIERS))
    if collision:
        raise ValueError(
            f"model {collision[0]!r} in models.json is named after a tier "
            f"({', '.join(TIERS)}); rename the model")
    if name in TIERS or name in _LEGACY_TIERS:
        tier = _LEGACY_TIERS.get(name, name)
        for var in (tier, _LEGACY_OF.get(tier, tier)):
            env = os.environ.get(f"HEART_TIER_{var.upper()}")
            if env:
                return resolve(env, default)
        want = defaults().get(tier)
        if want:
            return manifest[want]["agent"] if want in manifest else want
        in_tier = [m for m in manifest.values() if m.get("tier") == tier]
        if in_tier:
            return min(in_tier, key=lambda m: m["cost"])["agent"]
        if default:
            return default
        raise ValueError(
            f"nothing serves tier {tier!r}: add it to `defaults` in "
            f"{models_json_path()}, or set HEART_TIER_{tier.upper()}")
    if name in manifest:
        return manifest[name]["agent"]
    return name


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
    """Rebuild the sidecar from the journal. Best-effort; returns the stats."""
    try:
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


# --- stage one: what the task demands (heuristic; the LLM seam is the planner) --

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

# Effort is effectively medium-by-default, high for genuinely hard work — low is
# valid but rarely the right call, so nothing maps to it automatically.
_DIFFICULTY_EFFORT = {"trivial": "medium", "easy": "medium", "unknown": "medium",
                      "medium": "medium", "hard": "high", "expert": "high"}


@dataclass
class Demand:
    """What a task needs, before any model is considered."""
    tier: str
    skills: list[str]
    difficulty: str
    effort: str
    min_context: int
    task_id: str | None = None
    declared: bool = False  # the planner said how hard this is; we did not guess


def demand(task) -> Demand:
    """Stage one. Honors whatever the task already declares (from a planner);
    infers the rest from the prompt. A blunt keyword heuristic — good enough as
    the prior, and the measured loop corrects what it gets wrong."""
    text = task.prompt.lower()
    skills = [s for s in task.skills if s in SKILLS] if task.skills else [
        s for s, words in _SKILL_WORDS.items() if any(w in text for w in words)]
    if not skills:
        skills = ["coding"]

    declared = task.difficulty in _DIFF_RANK and task.difficulty != "unknown"
    if declared:
        difficulty = task.difficulty
    else:
        words = len(task.prompt.split())
        hard = any(w in text for w in _HARD_WORDS)
        difficulty = "hard" if hard else "medium" if words > 60 else "easy"

    # rough context estimate: prompt size in tokens (~chars/4); a real one would
    # add the files the task will read, which the caller knows better than we do.
    min_context = task.min_context or max(2000, len(task.prompt) // 4)
    return Demand(
        tier=tier_of(difficulty), skills=skills, difficulty=difficulty,
        effort=task.effort if task.effort in EFFORTS
        else _DIFFICULTY_EFFORT.get(difficulty, "medium"),
        min_context=min_context, task_id=getattr(task, "task_id", None),
        declared=declared)


# --- stage two: which model serves it -------------------------------------

@dataclass
class Choice:
    agent: str
    effort: str
    tier: str            # the tier demanded, which is what a scorecard groups by
    skills: list[str]
    difficulty: str
    reason: str
    name: str = ""       # the manifest entry chosen, when there was one
    candidates: list[dict] = field(default_factory=list)


def pick(
    want: Demand,
    manifest: dict | None = None,
    stats: dict | None = None,
    available=None,
    explore: float = 0.0,
    rng: random.Random | None = None,
    default: str | None = None,
    episode_id: str | None = None,
) -> Choice:
    """Stage two: the model that serves `want`.

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
    skills, difficulty = want.skills, want.difficulty

    def decided(choice: Choice, filtered: list[str] = ()) -> Choice:
        """Every decision is recorded, including the ones made without a
        manifest — a routed episode with no route.decided reads as unrouted."""
        try:
            emit("heart", "route.decided", task_id=want.task_id, episode_id=episode_id,
                 agent=choice.agent, model=choice.name, effort=choice.effort,
                 tier=choice.tier, declared=want.declared, skills=choice.skills,
                 difficulty=choice.difficulty, reason=choice.reason,
                 candidates=choice.candidates, filtered=list(filtered))
        except Exception:
            pass
        return choice

    if not manifest:
        # no manifest: the tier default is the best answer available, and the
        # caller's own agent is the answer when there isn't even one of those
        return decided(Choice(
            agent=resolve(want.tier, default=default or "claude"),
            effort=want.effort, tier=want.tier, skills=skills,
            difficulty=difficulty, reason="no model manifest; tier default"))

    scored: list[dict] = []
    filtered: list[str] = []
    for name, m in manifest.items():
        if m["context"] < want.min_context:
            filtered.append(f"{name}:context<{want.min_context}")
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
        return decided(Choice(
            agent=best[1]["agent"], effort="high", tier=want.tier,
            skills=skills, difficulty=difficulty, name=best[0],
            reason=f"no model cleared constraints ({'; '.join(filtered)}); "
                   f"forced highest-ceiling model"), filtered)

    # exploration: occasionally give a viable low-evidence candidate the traffic
    if explore and rng.random() < explore:
        least = min(scored, key=lambda c: c["n"])
        if least["n"] < _BLEND_K:
            chosen, why = least, f"exploration (n={least['n']})"
        else:
            chosen, why = _best(scored), "capability match"
    else:
        chosen, why = _best(scored), "capability match"

    return decided(Choice(
        agent=chosen["agent"], effort=want.effort, tier=want.tier, skills=skills,
        difficulty=difficulty, name=chosen["name"],
        reason=f"{why}: {chosen['name']} score={chosen['score']} for {skills}@{difficulty}",
        candidates=sorted(scored, key=lambda c: -c["score"]),
    ), filtered)


def _best(scored: list[dict]) -> dict:
    # highest score; ties -> cheapest (don't overspend when models are equal)
    return min(scored, key=lambda c: (-c["score"], c["cost"]))
