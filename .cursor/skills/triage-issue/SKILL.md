---
name: triage-issue
description: >-
  Create, label, and comment on GitHub issues for pdiseg using gh CLI and project
  triage labels. Use when filing bugs, grooming backlog, or applying needs-triage,
  ready-for-agent, ready-for-human labels.
---

# Triage issues

## Labels

See `docs/agents/triage-labels.md`.

## Commands

```sh
gh issue create --title "..." --body "$(cat <<'EOF'
...
EOF
)"

gh issue edit <n> --add-label ready-for-agent
gh issue view <n> --comments
```

## Workflow

`.agents/workflows/issue-triage.md`

## Issue tracker conventions

`docs/agents/issue-tracker.md`

Artifacts (title/body) in **English**.
