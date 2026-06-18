# Segmentação de embalagens de aves (PDI — IFG, Trabalho Prático 1)

Localiza e recorta a região da embalagem nas imagens do dataset. Só segmentação — sem classificação nem OCR.

Enunciado: [requirements.md](./requirements.md) · Glossário: [CONTEXT.md](./CONTEXT.md)

---

## 1. Entrada e saída

```
dataset/                          result/
├── Peito_Congelado/              ├── Peito_Congelado/
│   ├── img001.jpg                │   ├── img001_segmentada_1.png
│   └── ...                       │   └── ...
└── ...                           └── ...
```

- O nome da pasta (`Peito_Congelado`, `Moela`, etc.) só organiza a saída — o algoritmo não usa isso.
- Base em `data/Train_and_Validation/`: 18 classes × 50 imagens (1280×720, escala de cinza).
- Arquivos `*.jpgZone.Identifier` são ignorados.

---

## 2. Pré-requisitos

- [uv](https://docs.astral.sh/uv/) (recomendado) ou Python 3.12+
- [Docker](https://docs.docker.com/) + Docker Compose (para entrega via container)

Restrições do trabalho: só técnicas da Parte 1 do curso (limiarização, morfologia, filtros, histogramas, etc.). Sem IA.

---

## 3. Instalação

```sh
make setup
```

Equivalente: `uv sync --extra dev`

---

## 4. Rodar a segmentação (local)

```sh
make run
```

Gera `result/<Classe>/<arquivo>_segmentada_<N>.png`.

Com outro caminho de entrada/saída:

```sh
make run DATA=dataset OUT=result
```

---

## 5. Calibrar (overlays + boxes.json)

```sh
make calibrate
```

Gera `calibration/` com overlays de amostra, `boxes.json` e `stats.csv`.

Para `boxes.json` em **todos** os frames:

```sh
make calibrate LIMIT=9999
```

---

## 6. Ver resultados no navegador (review)

```sh
make review
```

Abre http://127.0.0.1:8765/ — mostra source, overlay e crops. Não roda o detector de novo.

Outra porta:

```sh
make review PORT=9000
```

---

## 7. Rodar com Docker (entrega)

```sh
mkdir -p result calibration
cp .env.example .env
docker compose up --build
```

Ou via Make:

```sh
make docker-up
```

Dataset com o nome do enunciado:

```sh
DATA=./dataset docker compose up --build
```

Ferramentas extras (profile `tools`):

```sh
make docker-calibrate
make docker-review
```

Detalhes: [docs/docker-compose.md](./docs/docker-compose.md).

Se o review no Docker der erro de permissão, no `.env`:

```env
OUT=pdiseg-result
CALIB=pdiseg-calibration
```

Depois: `make docker-export`

---

## 8. Referência do Make

Rode `make` ou `make help` para listar tudo.

### Variáveis

| Variável | Padrão | Uso |
|----------|--------|-----|
| `DATA` | `data/Train_and_Validation` | Pasta de entrada |
| `OUT` | `result` | Saída da segmentação |
| `CALIB` | `calibration` | Saída da calibração |
| `LIMIT` | `3` | Overlays por classe no calibrate |
| `PORT` | `8765` | Porta do review |

### Comandos

| Comando | O que faz |
|---------|-----------|
| `make run` | segmenta → `result/` |
| `make calibrate` | gera `calibration/` |
| `make review` | viewer web |
| `make docker-up` | pipeline no Docker |
| `make docker-calibrate` | calibrate no Docker |
| `make docker-review` | review no Docker |
| `make docker-export` | copia volume nomeado → `./result` |
| `make test` | pytest |
| `make lint` | ruff check |
| `make format` | ruff format |
| `make typecheck` | mypy |
| `make check` | lint + mypy + testes |
| `make clean` | apaga `result/`, `calibration/`, caches |

### CLIs (sem Make)

```sh
uv run pdiseg [INPUT] [OUTPUT]
uv run pdiseg-calibrate [INPUT] [OUTPUT_DIR] [--per-class-limit N]
uv run pdiseg-review --dataset DATA --calibration CALIB --result OUT [--port 8765]
```

---

## 9. Avaliação

- 60% — desempenho na base fornecida
- 40% — imagens não vistas (código congelado)

---

## 10. Entrega

Colab no Moodle **ou** Docker Compose.

Enviar link do Colab para `alessandro.rodrigues@ifg.edu.br`.
