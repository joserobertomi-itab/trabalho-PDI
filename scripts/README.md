# Scripts

| Script | Purpose |
|--------|---------|
| `docker-entrypoint.sh` | Container entry: optional mount prep, drop privileges with `gosu` |
| `docker-export-artifacts.sh` | Copy named Docker volumes (`pdiseg-result`, etc.) to the host |
| `docker-smoke.sh` | End-to-end Compose test with a synthetic frame |
| `agent-harness-check.sh` | Verify `.agents/`, `.cursor/skills/`, and rules are present |

Run smoke test: `make docker-smoke`  
Run harness check: `make agent-check`
