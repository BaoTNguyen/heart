---
status: accepted
date: 2026-10-28
approved-by: operator
drafted-by: operator
retire-evidence: a change in threat or requirement; replay is not valid evidence (no adversarial tasks)
switch: none
---
# A sandboxed verifier gets HEART_PULSE and no credential

## Context and Problem Statement

A verifier judges the agent's work, so it must never hold a credential the
agent could reach through it. Container verifiers got only `ARTERIES_*` and
trace ids. Suites from before B11 read journal files and need `HEART_PULSE`:
they passed in an unsandboxed verifier and failed in a container, and plexus
replay escalated twice on that difference (heart #27).

## Decision Drivers

* The grader must not differ from an unsandboxed one in ways that change verdicts
* Nothing that authenticates may reach the grader

## Considered Options

* Pass `HEART_PULSE` when set, and nothing else from the host env
* Keep the env as it was: replay escalates on old suites' own tests

Other options were not recorded at the time.

## Decision Outcome

Chosen option: "pass `HEART_PULSE` only", because it chooses how journals are
read and authenticates nothing.

### Consequences

* Good, because sandboxed and unsandboxed verifiers judge old suites the same.
* Bad, because every later addition to the verifier env needs this same argument.

### Confirmation

`src/heart/sandbox.py::verifier_profile_for`. Proven by
`tests/test_sandbox.py::test_a_verifier_is_never_handed_a_credential` and
`::test_a_verifier_reads_journals_the_way_an_unsandboxed_one_would`.
