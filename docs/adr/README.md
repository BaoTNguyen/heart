# Decisions

Why heart is shaped the way it is, one file per decision in trimmed MADR 4.
Format and rules: [vascular/docs/adr](https://github.com/BaoTNguyen/vascular/tree/main/docs/adr).
The "In force" table is generated from each record's front matter: run
`python3 docs/adr/index.py` after adding or superseding one. CI fails when the
table is stale.

## In force

<!-- adr-index:start -->
| Decision | Date |
|---|---|
| [A sandboxed verifier gets HEART_PULSE and no credential](2026-10-28-verifier-env.md) | 2026-10-28 |
<!-- adr-index:end -->

## Stack decisions that bind heart

| Decision | Date |
|---|---|
| [GitHub enforces PR plus CI on main](https://github.com/BaoTNguyen/vascular/blob/main/docs/adr/2026-10-30-protect-main.md) | 2026-10-30 |
