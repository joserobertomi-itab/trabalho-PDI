## Docker Compose

### Entrega (arquivos em `./result`)

```sh
mkdir -p result calibration
cp .env.example .env
docker compose up --build
```

### Volumes nomeados (pipeline + review no mesmo volume)

No `.env`:

```env
OUT=pdiseg-result
CALIB=pdiseg-calibration
```

```sh
docker compose up --build
docker compose --profile tools run --rm calibrate
docker compose --profile tools up review
bash scripts/docker-export-artifacts.sh
```

Útil quando o review no Docker reclama de permissão em bind mount.

| `.env` | Uso |
|--------|-----|
| `OUT=./result` | Entrega — professor vê a pasta no host |
| `OUT=pdiseg-result` | Dev — mesmo volume entre serviços |

### Serviços

- `pipeline` — segmenta, grava em `/data/output`
- `calibrate` — `boxes.json`, `stats.csv`, overlays (profile `tools`)
- `review` — http://localhost:8765/ (profile `tools`, só leitura)

O review roda como usuário `pdiseg`, sem `chown`. Depois de mudar o código: `docker compose build review`.

No Linux, se precisar:

```env
DOCKER_UID=1000
DOCKER_GID=1000
```

### Limites (notebook)

No `.env`: `DOCKER_CPUS=1.0`, `DOCKER_MEMORY=1536m`, `OMP_NUM_THREADS=1`. Ajuste se o PC aguentar mais.

### Make

`make docker-up`, `make docker-calibrate`, `make docker-review`, `make docker-export`, `make docker-smoke`.

### Dataset novo (avaliação complementar)

Mesma estrutura de pastas, troca o conteúdo de `dataset/`, roda de novo. Sem mudar código.
