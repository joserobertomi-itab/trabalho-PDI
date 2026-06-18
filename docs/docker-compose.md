## Docker Compose

### Entrega

```sh
mkdir -p result calibration
cp .env.example .env
docker compose up --build
```

### Volumes nomeados (pipeline + review)

No `.env`:

```env
OUT=pdiseg-result
CALIB=pdiseg-calibration
```

```sh
docker compose up --build
docker compose --profile tools run --rm calibrate
docker compose --profile tools up review
make docker-export
```

| `OUT` | Uso |
|-------|-----|
| `./result` | Entrega — professor vê pasta no host |
| `pdiseg-result` | Mesmo volume entre pipeline e review |

### Serviços

| Serviço | Comando | Função |
|---------|---------|--------|
| `pipeline` | `docker compose up` | Segmenta → `/data/output` |
| `calibrate` | `docker compose --profile tools run --rm calibrate` | `boxes.json`, `stats.csv` |
| `review` | `docker compose --profile tools up review` | http://localhost:8765/ |

### Make

```sh
make docker-up
make docker-calibrate
make docker-review
make docker-export
make docker-smoke
```

Variáveis: `DATA`, `OUT`, `CALIB`, `LIMIT`, `PORT` — ver README.

### Dataset novo (avaliação complementar)

Mesma estrutura de pastas em `dataset/`, roda de novo sem mudar código.
