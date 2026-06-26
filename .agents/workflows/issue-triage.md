# Workflow: Issue triage

Use when creating or grooming GitHub issues (see `docs/agents/issue-tracker.md`).

## Labels

| Label | When |
|-------|------|
| `needs-triage` | New issue, not classified |
| `needs-info` | Waiting on reporter |
| `ready-for-agent` | Clear scope, agent can implement |
| `ready-for-human` | Needs human judgment / eval |
| `wontfix` | Out of scope |

## Issue body template

```markdown
## Problem
(what is wrong)

## Expected
(crop covers name label / fewer FPs / etc.)

## Repro
class + filename or steps

## Notes
(optional screenshots, metrics)
```

## Agent pickup

Only start implementation on `ready-for-agent` unless user explicitly overrides in chat.

Use skill **triage-issue** for label operations via `gh`.
