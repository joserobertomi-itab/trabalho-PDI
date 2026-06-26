# Workflow: Verify before done

Run before marking a task complete.

## Required

```sh
make check
```

## If pipeline touched

```sh
uv run pytest tests/test_run.py tests/test_calibrate.py tests/test_detector_validation.py -q
```

## If Docker touched

```sh
make docker-smoke
```

## If docs-only

- English artifacts
- Links in `AGENTS.md` / `README.md` still valid

## Report to user (PT-BR)

- What changed (1–3 bullets)
- Commands run + pass/fail
- Honest limitations
- Metrics if pipeline (crops, empty frames) when measured
