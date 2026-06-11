## Language policy

- All committed artifacts are written in English: CLAUDE.md/AGENTS.md, CONTEXT.md,
  ADRs, docs/, specs, PRDs, issue titles/bodies, commit messages, code comments.
- Day-to-day conversation with the user is in Brazilian Portuguese.
- When a skill produces an artifact (to-issues, to-prd, triage, ADRs, etc.),
  write the artifact in English even if the prompt was in Portuguese.

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The repo uses the default triage label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This repo uses a single-context domain docs layout: root `CONTEXT.md` plus root `docs/adr/`. See `docs/agents/domain.md`.
