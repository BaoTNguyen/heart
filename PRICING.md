# The rate card

`~/.config/heart/models.json` is where every dollar figure in this stack comes
from. heart prices episodes against it; plexus prices interactive turns against
it through `heart.runner.model_pricing()`. There is deliberately one copy — a
second card in the control plane would drift, and a wrong cost is still a
plausible number, so nothing ever alerts.

## When an unknown model appears

A model the card has never seen still gets billed, at the provider-wide
fallback rate. That is a guess. It is usually close and occasionally wrong by
an order of magnitude, and nothing in the output says which.

**Whenever `heart models check` reports an unpriced model, resolve it before
trusting any cost figure that includes it.** The procedure below is written for
an agent to execute; a human runs the same four steps.

### 1. Find out what is unpriced

```bash
heart models check                 # last 30 days of observed traffic
heart models check --hours 168     # last week
```

This reads the spine, so it only ever names models the fleet actually ran. You
never track a vendor's catalogue — the fleet tells you what it is being billed
for.

### 2. Read the rate off the vendor's own page

| Provider | Page |
|---|---|
| Claude / Anthropic | <https://platform.claude.com/docs/en/docs/about-claude/pricing> |
| OpenAI / Codex | <https://openai.com/api/pricing/> |

Fetch the page. Find the row for the **exact model id** the spine reported
(`claude-opus-5`, not "Opus"). Take base input and output USD per million
tokens.

Rules that matter more than they look:

- **Exact id only.** `claude-opus-5` and `claude-opus-4-8` happen to share a
  rate today; `claude-opus-4-1` is 3x both. Never carry a rate across a version
  because the names look alike.
- **If the page does not list the model, stop.** Record nothing. Report that it
  is unlisted and leave the fallback in place. A guessed rate that looks
  official is worse than a gap that is labelled — the gap gets fixed.
- **Ignore discount tiers** unless you are certain they apply: batch (-50%),
  volume, and negotiated rates are not the list price. Enter list price, then
  override in the workspace if you have a negotiated deal.
- **Watch for premium modes.** Claude fast mode bills Opus at 2x
  ($10/$50). The stack does not currently record whether a turn used it, so
  fast-mode turns under-report. Do not fold the premium into the base rate to
  compensate — that misprices every normal turn instead.

### 3. Record it, with provenance

```bash
heart models set-price claude-opus-5 \
  --input 5 --output 25 \
  --source https://platform.claude.com/docs/en/docs/about-claude/pricing
```

`--source` is required and the call fails without it. A rate nobody can
re-check is one people stop questioning, which is how a stale number survives a
price change. `--verified` defaults to today and is what ages the card.

### 4. Confirm it took

```bash
heart models check
```

The model should have moved out of the unpriced list.

## Scheduled changes

Vendors announce price changes ahead of the date. Enter the future row when you
read it, not on the day:

```bash
heart models set-price claude-sonnet-5 --input 3 --output 15 \
  --source https://platform.claude.com/docs/en/docs/about-claude/pricing \
  --effective-from 2026-09-01
```

Rows accumulate per model; `model_pricing()` picks the latest one whose
`effective_from` has arrived. The change lands on its own. `heart models check`
lists anything landing within 30 days.

## Local models

A local model server costs nothing to run, and this is detected rather than
configured: `_price()` checks `agents_api.is_local_endpoint()` and returns a
hard `0.0`. Do not add a `0.0` rate for a local model to make it look priced —
that turns a detected fact into a hand-maintained one.

## Staleness

`heart models check` flags any rate on a model in active use that has not been
re-verified in 90 days, and any rate with no `verified` date at all. An undated
rate can never look stale, so it gets trusted indefinitely — re-record it with
`set-price` rather than leaving it.

## Precedence

For an interactive turn, plexus resolves rates most-specific first:

1. a per-model override in that workspace's `accounting.pricing[provider].models`
   — a negotiated rate has to beat the published one
2. this card, keyed by model id
3. the provider-wide rate in the workspace config

A model in none of those is counted in `gaps.unpriced` rather than billed at
zero.

## What is deliberately not automated

There is no job that scrapes a price feed and writes this file unattended.
Billing numbers enter the card only through a step where someone read the
vendor's page. A third-party aggregator drifting silently would produce exactly
the failure this card exists to prevent, and it would look correct while doing
it.
