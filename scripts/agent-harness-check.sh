#!/usr/bin/env bash
# Validates agent harness files exist (CI-friendly).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

required=(
  AGENTS.md
  CLAUDE.md
  .agents/manifest.yaml
  .agents/constraints.md
  .agents/README.md
  docs/agents/harness.md
)

skills=(
  pdiseg-pipeline
  pdiseg-debug-notebook
  pdiseg-calibrate-review
  pdiseg-docker
  grill-with-docs
  triage-issue
  run-verification
)

rules=(
  language-policy.mdc
  no-ml-no-ocr.mdc
  no-spec-driven.mdc
  pdiseg-pipeline.mdc
  python-tests.mdc
  docker-make.mdc
)

workflows=(
  pipeline-change.md
  debug-segmentation.md
  docker-and-make.md
  issue-triage.md
  verify-before-done.md
)

roles=(
  engineer.md
  reviewer.md
  debugger.md
)

missing=0
check() {
  if [[ ! -e "$1" ]]; then
    echo "MISSING: $1" >&2
    missing=1
  fi
}

for f in "${required[@]}"; do check "$f"; done
for s in "${skills[@]}"; do check ".cursor/skills/$s/SKILL.md"; done
for r in "${rules[@]}"; do check ".cursor/rules/$r"; done
for w in "${workflows[@]}"; do check ".agents/workflows/$w"; done
for ro in "${roles[@]}"; do check ".agents/roles/$ro"; done

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

echo "agent harness: OK (${#skills[@]} skills, ${#rules[@]} rules, ${#workflows[@]} workflows)"
