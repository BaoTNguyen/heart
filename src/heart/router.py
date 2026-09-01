"""Model routing: cheap models for routine tasks, strong models for hard ones.

Activated with --agent auto. Tiers resolve to agent strings from
~/.config/heart/models.json:

    {"tiers": {"cheap": "api:qwen", "standard": "claude", "strong": "api:opus"}}

Env override per tier: HEART_TIER_CHEAP / HEART_TIER_STANDARD / HEART_TIER_STRONG.
An explicit --agent (anything but "auto") bypasses routing entirely.

Tiers rank capability, not price or vendor: a tier may resolve to a local
server, a metered API, or a subscription CLI seat. Under metered pricing
routing saves dollars; under subscriptions it preserves usage-window quota;
local tiers double as RL training traffic. The env override is the manual
valve when a subscription window runs hot.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

TIERS = ("cheap", "standard", "strong")
# ponytail: keyword heuristic, not a learned classifier — the decision ledger
# records every routing call, so a trained gate can replace this later
HARD_WORDS = ("refactor", "architect", "design", "concurren", "thread", "race",
              "migrat", "protocol", "security", "performance", "deadlock",
              "rewrite", "debug", "investigate")
EASY_WORDS = ("typo", "rename", "comment", "docstring", "format", "bump",
              "readme", "changelog", "lint", "whitespace", "version")
DIFFICULTY_TIER = {"easy": "cheap", "trivial": "cheap", "medium": "standard",
                   "hard": "strong", "expert": "strong"}


def classify(task) -> tuple[str, dict]:
    """Complexity heuristic -> (tier, signals). Explicit task.difficulty wins."""
    if task.difficulty in DIFFICULTY_TIER:
        return DIFFICULTY_TIER[task.difficulty], {"reason": "task.difficulty",
                                                  "difficulty": task.difficulty}
    text = task.prompt.lower()
    words = len(task.prompt.split())
    hard = [w for w in HARD_WORDS if w in text]
    easy = [w for w in EASY_WORDS if w in text]
    score = (2 if words > 150 else 1 if words > 50 else 0) + 2 * bool(hard)
    if not hard:
        score -= 2 * bool(easy)
    if len(task.public_verifiers) + len(task.hidden_verifiers) > 2:
        score += 1
    if 0 < len(task.allowed_paths) <= 2:  # narrow scope = small blast radius
        score -= 1
    tier = "strong" if score >= 3 else "cheap" if score <= 0 else "standard"
    return tier, {"reason": "heuristic", "score": score, "words": words,
                  "hard_hits": hard, "easy_hits": easy}


#: Models trusted to review. Ordered, and the order only breaks ties -- what
#: matters is that a reviewer is never the family that wrote the code, because
#: a model reviewing its own output brings the same blind spots to finding the
#: bug that it brought to writing it.
#:
#: Override in models.json with a "review_models" list; the rotation below
#: works for any number of entries, so adding or changing one needs no code.
DEFAULT_REVIEW_MODELS = ("claude:opus", "codex:sol")


def review_pool() -> list[str]:
    env = os.environ.get("HEART_REVIEW_MODELS")
    if env:
        return [a.strip() for a in env.split(",") if a.strip()]
    path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "heart" / "models.json"
    try:
        configured = json.loads(path.read_text()).get("review_models")
    except (OSError, json.JSONDecodeError):
        configured = None
    return list(configured) if configured else list(DEFAULT_REVIEW_MODELS)


def review_agent(coding_agent: str) -> str:
    """A reviewer that is not the model that wrote the code.

    Matched on the family -- the part before the colon -- not the exact agent
    string. `claude:opus` reviewing `claude:sonnet` would be a different model
    and the same training lineage, which is most of what independent review is
    supposed to rule out.

    A coder outside the pool (a local model, a subscription seat) still gets a
    reviewer: the first entry. And if every pool entry shares the coder's
    family, the first entry again -- a same-family reviewer is worth more than
    none, and refusing here would fail an episode over a config choice.
    """
    family = coding_agent.partition(":")[0]
    if family == "shell":
        # `shell` runs the prompt as bash; it is the harness, not a model, and
        # has no lineage to rotate away from. Rotating anyway pointed every
        # role-pipeline test at whatever real CLI sat first in the pool -- the
        # toy suite spent real tokens locally and failed on CI, where no such
        # binary exists.
        return coding_agent
    pool = review_pool()
    if not pool:
        return coding_agent
    return next((a for a in pool if a.partition(":")[0] != family), pool[0])


def resolve(tier: str, default: str | None = None) -> str:
    """Tier -> agent string. Falls back to `default` when the tier isn't configured."""
    env = os.environ.get(f"HEART_TIER_{tier.upper()}")
    if env:
        return env
    path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "heart" / "models.json"
    try:
        tiers = json.loads(path.read_text()).get("tiers", {})
    except (OSError, json.JSONDecodeError):
        tiers = {}
    if tier in tiers:
        return tiers[tier]
    if default:
        return default
    raise ValueError(
        f"no agent configured for tier {tier!r}: set HEART_TIER_{tier.upper()} "
        f'or add {{"tiers": {{"{tier}": "<agent>"}}}} to {path}'
    )
