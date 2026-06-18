# Segmentação de embalagens de aves (PDI — IFG, Trabalho Prático 1)

Localiza e recorta a região da embalagem nas imagens do dataset. Só segmentação — sem classificação nem OCR.

Enunciado original: [requirements.md](./requirements.md). Termos do domínio: [CONTEXT.md](./CONTEXT.md).

## Entrada e saída

```
dataset/                          result/
├── Peito_Congelado/              ├── Peito_Congelado/
│   ├── img001.jpg                │   ├── img001_segmentada_1.png
│   └── ...                       │   └── ...
└── ...                           └── ...
```

O nome da pasta (`Peito_Congelado`, `Moela`, etc.) só organiza a saída. O algoritmo não usa isso.

A base em `data/Train_and_Validation/` tem 18 classes × 50 imagens (1280×720, escala de cinza). Arquivos `*.jpgZone.Identifier` são ignorados.

## Restrições

Só técnicas da Parte 1 do curso: limiarização, morfologia, filtros, histogramas, etc. Sem IA / segmentação automática de biblioteca.

## Rodar local

Precisa de [uv](https://docs.astral.sh/uv/) (ou Python 3.12+).

```sh
make setup
make run          # gera result/
make calibrate    # gera calibration/ (overlays, boxes.json, stats.csv)
make review       # http://127.0.0.1:8765/
```

Paths customizados:

```sh
make run DATA=dataset OUT=result
make calibrate LIMIT=9999    # boxes.json em todos os frames
```

CLIs:

```sh
uv run pdiseg [INPUT] [OUTPUT]
uv run pdiseg-calibrate [INPUT] [OUTPUT_DIR] [--per-class-limit N]
uv run pdiseg-review --dataset DATA --calibration CALIB --result OUT
```

O review viewer só mostra o que já foi gerado — não roda o detector de novo.

## Docker

Entrega via Compose (`docker compose up --build`). Detalhes em [docs/docker-compose.md](./docs/docker-compose.md).

```sh
mkdir -p result calibration
cp .env.example .env
docker compose up --build
```

Para o dataset com o nome do enunciado:

```sh
DATA=./dataset docker compose up --build
```

Ferramentas extras:

```sh
docker compose --profile tools run --rm calibrate
docker compose --profile tools up review
```

Se o review no Docker der erro de permissão, use volumes nomeados no `.env`:

```env
OUT=pdiseg-result
CALIB=pdiseg-calibration
```

Depois: `bash scripts/docker-export-artifacts.sh` para copiar para `./result`.

## Desenvolvimento

```sh
make check    # lint + mypy + testes
```

CI no GitHub Actions: lint, mypy, pytest, build Docker.

## Avaliação

60% base fornecida, 40% imagens não vistas (código congelado).

## Entrega

Colab no Moodle **ou** Docker Compose. Enviar link do Colab para `alessandro.rodrigues@ifg.edu.br`.
