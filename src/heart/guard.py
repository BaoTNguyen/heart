"""Guardrails: secret scanning over an episode's diff. STACK_READINESS.md
doesn't cover this directly but it's the same "don't ship a silent failure"
spirit as path_violations — a hit zeroes reward exactly like a path violation.
"""
from __future__ import annotations

import re

# Rules run only over ADDED lines ("+...", not "+++" file headers). Never
# capture/return the actual secret value — hits report the rule name and a
# truncated line snippet only.
_RULES: list[tuple[str, re.Pattern]] = [
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("github_token", re.compile(r"ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}")),
    ("slack_token", re.compile(r"xox[bpars]-[A-Za-z0-9-]{10,}")),
    (
        "generic_secret_assignment",
        re.compile(
            r"(?i)(api_?key|secret|token|password)\s*[:=]\s*[\"'][A-Za-z0-9+/_-]{20,}[\"']"
        ),
    ),
]


def scan_secrets(diff_text: str) -> list[str]:
    hits: list[str] = []
    for line in diff_text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added = line[1:]
        for rule_name, pattern in _RULES:
            if pattern.search(added):
                snippet = added.strip()[:60]
                hits.append(f"{rule_name}: {snippet}")
    return hits
