"""Reward calculation. Weights follow rl_harness_heart_handoff.md's no-hidden-tests
formula; components that can't be computed yet are dropped and weights renormalize.

ponytail: regression_safety and retrieval_memory_usefulness are omitted until a
regression verifier and the arteries decision ledger exist.
"""
from __future__ import annotations

WEIGHTS = {
    "public_tests": 0.45,
    "lint_typecheck": 0.15,
    "diff_quality": 0.15,
    "efficiency": 0.10,
}
# doc's with-hidden-tests formula: hidden tests dominate when they exist
WEIGHTS_HIDDEN = {
    "hidden_tests": 0.45,
    "public_tests": 0.25,
    "lint_typecheck": 0.10,
    "diff_quality": 0.10,
    "efficiency": 0.05,
}

LINT_NAMES = {"lint", "typecheck", "format", "mypy", "ruff", "eslint", "tsc", "biome"}


def diff_changed_lines(diff_text: str) -> int:
    return sum(
        1
        for line in diff_text.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )


def compute(
    verifier_results: dict[str, dict],
    diff_text: str,
    duration_s: float,
    timeout_s: int,
    hidden_results: dict[str, dict] | None = None,
) -> dict:
    components: dict[str, float] = {}

    tests = [r for n, r in verifier_results.items() if n.lower() not in LINT_NAMES]
    lints = [r for n, r in verifier_results.items() if n.lower() in LINT_NAMES]
    if tests:
        components["public_tests"] = sum(r["passed"] for r in tests) / len(tests)
    if lints:
        components["lint_typecheck"] = sum(r["passed"] for r in lints) / len(lints)
    if hidden_results:
        components["hidden_tests"] = sum(
            r["passed"] for r in hidden_results.values()
        ) / len(hidden_results)

    # ponytail: diff quality = size heuristic (<=50 changed lines is full credit,
    # 0 at 500). Upgrade to a diff-review model once scored episodes exist.
    changed = diff_changed_lines(diff_text)
    components["diff_quality"] = 1.0 if changed <= 50 else max(0.0, 1.0 - (changed - 50) / 450)

    components["efficiency"] = max(0.0, 1.0 - duration_s / timeout_s) if timeout_s else 0.0

    weights = WEIGHTS_HIDDEN if hidden_results else WEIGHTS
    active = {k: weights[k] for k in components if k in weights}
    total_w = sum(active.values())
    total = sum(components[k] * w for k, w in active.items()) / total_w if total_w else 0.0
    return {"total": round(total, 4), "components": {k: round(v, 4) for k, v in components.items()}}
