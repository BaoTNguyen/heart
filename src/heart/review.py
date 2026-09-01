"""Review as findings, not as a word.

The old shape was three turns wearing three names for two jobs: `review` judged
the diff and emitted APPROVE/REJECT, `review-fix` edited, `review2` judged again
from scratch. It failed in production for reasons that were structural, not
cosmetic:

  * One binary verdict carried four findings at three severities. The reviewer
    had to invent "Blocker:" and "not blockers" itself, because nothing asked.
  * The fix turn was handed `review.log[-1500:]`. In the run that motivated this,
    the review was 3471 chars and the blocker sat at char 13 -- the fixer never
    saw the thing it was supposed to fix, and the re-review rejected on the
    identical finding.
  * The verdict was `findall(APPROVE|REJECT)[-1]`, so the last occurrence of
    either word anywhere -- including inside a quoted diff -- decided it.
  * An APPROVE threw away everything the reviewer noticed on the way past.

So the stages are three different jobs now, each named for what it reads:

  assess   reads the diff            -> findings
  resolve  reads the findings        -> a disposition per finding
  confirm  reads findings + claims   -> which claims hold, plus what the fix broke

The verdict is derived (any blocker -> reject) rather than announced, which is
what lets the loop terminate on a checkable condition instead of on a word.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: Ordered worst-first. Only `blocker` blocks; the rest are recorded so that an
#: approval stops discarding what the reviewer noticed.
SEVERITIES = ("blocker", "concern", "note")


@dataclass
class Finding:
    severity: str
    claim: str
    file: str = ""
    line: int = 0
    evidence: str = ""

    def render(self, index: int) -> str:
        where = f"{self.file}:{self.line}" if self.file else "(no file given)"
        out = f"[{index}] {self.severity.upper()} {where}\n    {self.claim}"
        return out + (f"\n    evidence: {self.evidence}" if self.evidence else "")


ASSESS_PROMPT = (
    "Run `git diff` and review all changes for the task below: correctness, "
    "unintended edits, missing tests. Tests added by the pipeline's test role are "
    "expected and in scope.\n\n"
    "Reply with ONLY a JSON object listing what you found. Severity is "
    "`blocker` (this must not ship), `concern` (worth fixing, not fatal), or "
    "`note` (worth recording). An empty list means the change is sound.\n"
    '{"findings": [{"severity": "blocker", "file": "path/to/file.py", '
    '"line": 41, "claim": "<what is wrong, one sentence>", '
    '"evidence": "<how you know -- a command you ran, a line you read>"}]}\n\n'
    "Task: {prompt}"
)

RESOLVE_PROMPT = (
    "A reviewer found the following in the current changes:\n\n{findings}\n\n"
    "Address every BLOCKER. Fix concerns where cheap. Do not weaken or delete "
    "tests, and do not revert the task's work to make a finding go away.\n\n"
    "Then reply with ONLY a JSON array saying what you did with each one, by its "
    "index:\n"
    '[{"id": 0, "status": "fixed", "how": "<what you changed>"}, '
    '{"id": 1, "status": "wontfix", "how": "<why it should not be fixed here>"}]\n'
    "status is `fixed`, `wontfix`, or `disagree`.\n\n"
    "Original task: {prompt}"
)

CONFIRM_PROMPT = (
    "You reviewed these changes and found:\n\n{findings}\n\n"
    "Someone then worked on them and claims:\n\n{dispositions}\n\n"
    "Run `git diff` and do two things. First, check each claim: is that finding "
    "actually resolved? Second, look at what the fix itself changed and say "
    "whether it introduced anything new.\n\n"
    "Reply with ONLY the same JSON object as before -- the findings that still "
    "stand, plus any new ones. Drop the ones genuinely resolved. An empty list "
    "means it is now sound.\n"
    '{"findings": [{"severity": "blocker", "file": "...", "line": 0, '
    '"claim": "...", "evidence": "..."}]}\n\n'
    "Task: {prompt}"
)


def _fill(template: str, **values: str) -> str:
    """Substitute {name} placeholders without str.format().

    A review prompt is full of literal JSON braces -- it is asking for JSON --
    and format() reads those as fields and raises KeyError on the first one.
    Doubling them works only for templates written here; a --roles file supplies
    its own wording and cannot be expected to escape anything.
    """
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def _json_objects(raw: str):
    """Every JSON value in `raw` that actually decodes, outermost first.

    Same lesson as the decomposer's parser: a reviewer quotes code in fenced
    blocks and writes prose full of braces, so a single span guess (first brace
    to last, or the first ``` fence) finds nothing. Skipping past each successful
    decode keeps a nested array from being mistaken for the whole reply.
    """
    decoder = json.JSONDecoder()
    i = 0
    while i < len(raw):
        if raw[i] not in "{[":
            i += 1
            continue
        try:
            value, end = decoder.raw_decode(raw, i)
        except ValueError:
            i += 1
            continue
        i = end
        yield value


def parse_findings(raw: str) -> list[Finding] | None:
    """Findings from a reviewer's log, or None when it emitted no usable JSON.

    None is not "nothing wrong" -- it is "this reviewer did not answer the
    question", and the caller falls back to the legacy APPROVE/REJECT read
    rather than treating silence as approval.
    """
    for value in _json_objects(raw):
        items = value.get("findings") if isinstance(value, dict) else None
        if items is None:
            continue
        if not isinstance(items, list):
            continue
        out = []
        for d in items:
            if not isinstance(d, dict) or not d.get("claim"):
                continue
            severity = str(d.get("severity", "")).lower()
            out.append(Finding(
                severity=severity if severity in SEVERITIES else "concern",
                claim=str(d["claim"]),
                file=str(d.get("file") or ""),
                line=int(d["line"]) if str(d.get("line", "")).isdigit() else 0,
                evidence=str(d.get("evidence") or "")))
        return out
    return None


def parse_dispositions(raw: str) -> list[dict]:
    """What the fix turn claims it did, by finding index. Best effort: an
    unparseable answer means the confirm stage checks the diff with no claims to
    check against, which is strictly the old behaviour and no worse."""
    for value in _json_objects(raw):
        items = value if isinstance(value, list) else None
        if items is None:
            continue
        out = [d for d in items if isinstance(d, dict) and "id" in d]
        if out:
            return out
    return []


def verdict_from(findings: list[Finding]) -> str:
    """Derived, never announced. A reviewer cannot write a rejection and then
    approve, and the word APPROVE inside a quoted diff decides nothing."""
    return "reject" if any(f.severity == "blocker" for f in findings) else "approve"


def render_findings(findings: list[Finding]) -> str:
    return "\n".join(f.render(i) for i, f in enumerate(findings)) or "(none)"


def render_dispositions(dispositions: list[dict], findings: list[Finding]) -> str:
    lines = []
    for d in dispositions:
        try:
            claim = findings[int(d["id"])].claim[:80]
        except (ValueError, IndexError, KeyError, TypeError):
            claim = "(unknown finding)"
        lines.append(f"[{d.get('id')}] {d.get('status', '?')}: {d.get('how', '')}"
                     f"\n    was: {claim}")
    return "\n".join(lines) or "(no claims made)"


@dataclass
class ReviewResult:
    verdict: str | None = None
    findings: list[Finding] = field(default_factory=list)
    dispositions: list[dict] = field(default_factory=list)
    rounds: int = 0
    fell_back: bool = False   # the reviewer emitted no JSON; legacy read was used


def phase(task_prompt: str, *, assess, resolve, verify, legacy_verdict,
          rounds: int = 1, assess_prompt: str = ASSESS_PROMPT) -> ReviewResult:
    """assess/resolve are `(name, prompt) -> Path` returning the turn's log.
    `verify` re-runs the verifiers after a fix. `legacy_verdict(Path) -> str|None`
    is the pre-findings reader, used only when a reviewer emits no JSON.

    `rounds` is how many times a rejection may be acted on. The default of 1
    costs exactly what the old assess/fix/re-assess flow cost: one extra edit
    turn and one extra judgment per rejection, and nothing when it approves.

    `assess_prompt` comes from the review role, so a --roles file that predates
    findings keeps its own wording -- it just will not emit JSON, and the whole
    phase degrades to a single legacy APPROVE/REJECT read. Opting in is a matter
    of asking for findings.
    """
    result = ReviewResult()
    findings: list[Finding] = []
    dispositions: list[dict] = []

    for n in range(1, rounds + 2):
        prompt = (_fill(assess_prompt, prompt=task_prompt) if n == 1 else
                  _fill(CONFIRM_PROMPT, prompt=task_prompt,
                        findings=render_findings(findings),
                        dispositions=render_dispositions(dispositions, findings)))
        log = assess(f"review.{n}" if n == 1 else f"review-confirm.{n}", prompt)
        parsed = parse_findings(log.read_text(errors="replace"))
        if parsed is None:
            result.fell_back = True
            result.verdict = legacy_verdict(log) or result.verdict
            result.rounds = n
            return result
        findings = parsed
        result.findings, result.rounds = findings, n
        result.verdict = verdict_from(findings)
        if result.verdict != "reject" or n > rounds:
            return result
        # The whole finding list, not a tail of a log: the truncation this
        # replaces cut the blocker off the front and the fixer never saw it.
        fix_log = resolve(f"review-fix.{n}",
                          _fill(RESOLVE_PROMPT, prompt=task_prompt,
                                findings=render_findings(findings)))
        dispositions = parse_dispositions(fix_log.read_text(errors="replace"))
        result.dispositions = dispositions
        verify()
    return result
