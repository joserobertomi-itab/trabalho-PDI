---
name: grill-with-docs
description: >-
  Lazily update CONTEXT.md glossary and docs/adr when domain terms or architectural
  decisions are resolved in conversation. Use when new vocabulary appears, ADR
  conflicts arise, or glossary gaps block clear issues/PRs.
---

# Grill with docs (lazy domain docs)

## When to use

- A new domain term is needed repeatedly → add to `CONTEXT.md` glossary.
- An implementation contradicts an ADR → note conflict or draft new ADR.
- Do **not** bulk-create docs upfront; update when terms/decisions are real.

## CONTEXT.md edits

- Short definitions; use project's Portuguese class names where they match folders.
- English prose in committed files.

## ADRs

- Location: `docs/adr/NNNN-title.md`
- Next number: check existing files.
- Format: context, decision, consequences.
- Not a substitute for specs (no spec-driven development).

## Reference

`docs/agents/domain.md`
