# Agent harness

This folder defines how AI assistants should work in **pdiseg** — roles, workflows, and constraints.

## Start here

1. Read root **`AGENTS.md`** (policy + skill index).
2. Read **`CONTEXT.md`** (domain glossary).
3. Read **`.agents/constraints.md`** (hard limits).
4. Pick a **workflow** under `workflows/` for the task at hand.

## Layout

```
.agents/
├── manifest.yaml       # machine-readable index
├── constraints.md      # non-negotiable project limits
├── roles/              # how to behave (engineer, reviewer, debugger)
└── workflows/          # step-by-step task playbooks
```

## Cursor integration

| Layer | Location | Purpose |
|-------|----------|---------|
| Rules | `.cursor/rules/*.mdc` | Always-on or file-scoped guardrails |
| Skills | `.cursor/skills/*/SKILL.md` | Deep workflows invoked by name |
| Docs | `docs/agents/` | Issue tracker, triage, harness reference |

Invoke a skill explicitly: e.g. “use the pdiseg-pipeline skill”.

## What we do **not** use

- **Spec-driven development** — no `specs/`, no “implement from PRD only” gate. Use GitHub issues + `CONTEXT.md` + ADRs + code.
- **OCR / trained models** for segmentation (academic PDI constraint).

## Verify harness

```sh
make agent-check
```
