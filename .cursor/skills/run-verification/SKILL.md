---
name: run-verification
description: >-
  Run pdiseg verification gate (make check, targeted pytest, docker-smoke) before
  completing a task. Use when finishing implementation, before PR, or when user
  asks if the project still passes CI.
---

# Run verification

## Default gate

```sh
make check
```

## Pipeline changes

```sh
uv run pytest tests/test_run.py tests/test_calibrate.py tests/test_detector_validation.py -q
```

## Docker changes

```sh
make docker-smoke
```

## Harness integrity

```sh
make agent-check
```

## Done report

Follow `.agents/workflows/verify-before-done.md` — summarize in Portuguese for the user.
