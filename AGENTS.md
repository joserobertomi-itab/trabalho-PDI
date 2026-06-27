# Agent harness — pdiseg

Entry point for AI assistants (Cursor, Claude Code, etc.). **No spec-driven development** — work from issues, `CONTEXT.md`, ADRs, and code.

## Language

- Committed artifacts: **English**
- User chat: **Brazilian Portuguese**

## Quick start

1. Read `CONTEXT.md` (domain glossary).
2. Read `.agents/constraints.md` (hard limits).
3. Pick a workflow from `.agents/workflows/` or a skill from `.cursor/skills/`.
4. Before done: skill **run-verification** or `make check`.

## Harness layout

| Path | Purpose |
|------|---------|
| `.agents/manifest.yaml` | Index of roles, workflows, skills |
| `.agents/roles/` | engineer, reviewer, debugger |
| `.agents/workflows/` | Task playbooks |
| `.cursor/rules/` | Cursor rules (`.mdc`) |
| `.cursor/skills/` | Invokable skills (`SKILL.md`) |
| `docs/agents/` | Issue tracker, triage, harness doc |

Full guide: `docs/agents/harness.md`.  
Source layout: `docs/src/ARCHITECTURE.md`.

## Skills (invoke by name)

| Skill | Use when |
|-------|----------|
| `pdiseg-pipeline` | Changing detection / scoring / config |
| `pdiseg-debug-notebook` | Visual debug, masks, scores |
| `pdiseg-calibrate-review` | Overlays, boxes.json, review UI |
| `pdiseg-docker` | Compose, Makefile docker targets |
| `grill-with-docs` | Update CONTEXT / ADRs lazily |
| `triage-issue` | GitHub issues + labels via `gh` |
| `run-verification` | `make check`, tests, smoke |

## Issue tracker

GitHub Issues via `gh` — see `docs/agents/issue-tracker.md`.

## Triage labels

`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix` — see `docs/agents/triage-labels.md`.

## Domain docs

Root `CONTEXT.md` + `docs/adr/` — see `docs/agents/domain.md`.

## Verify harness

```sh
make agent-check
```


<claude-mem-context>
# Memory Context

# [trabalho-PDI] recent context, 2026-06-27 11:01am GMT-3

No previous sessions found.
</claude-mem-context>