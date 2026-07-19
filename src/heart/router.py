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
