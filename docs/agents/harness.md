# Agent harness

How AI tools should operate in this repository.

## Philosophy

- **Issue- and code-driven** — GitHub issues, `CONTEXT.md`, ADRs, existing modules.
- **Not spec-driven** — no mandatory `specs/` or PRD-before-code workflow.
- **Classical PDI** — academic constraints in `.agents/constraints.md`.

## Structure

```
.agents/
├── manifest.yaml
├── constraints.md
├── roles/          engineer | reviewer | debugger
└── workflows/      pipeline-change | debug | docker | triage | verify

.cursor/
├── rules/*.mdc     always-on + file-scoped guardrails
└── skills/*/SKILL.md

docs/agents/        issue tracker, triage, domain consumption
AGENTS.md           human + agent entry point
```

## Typical flows

| Task | Start with |
|------|------------|
| Fix detection | workflow `pipeline-change` + skill `pdiseg-pipeline` |
| Debug one frame | workflow `debug-segmentation` + skill `pdiseg-debug-notebook` |
| Docker delivery | workflow `docker-and-make` + skill `pdiseg-docker` |
| New glossary term | skill `grill-with-docs` |
| Close out work | skill `run-verification` |

## Cursor rules (auto-loaded)

- `language-policy` — EN artifacts, PT chat
- `no-ml-no-ocr` — classical CV only
- `no-spec-driven` — no spec gates
- `pdiseg-pipeline` — when editing `src/pdiseg/**/*.py`
- `python-tests` — ruff/mypy/pytest
- `docker-make` — compose profile syntax

## Maintenance

When adding a skill:

1. Create `.cursor/skills/<name>/SKILL.md` with YAML frontmatter.
2. Register in `.agents/manifest.yaml`.
3. Link from `AGENTS.md` if user-facing.
4. Run `make agent-check`.

## Related

- [domain.md](./domain.md)
- [issue-tracker.md](./issue-tracker.md)
- [triage-labels.md](./triage-labels.md)
