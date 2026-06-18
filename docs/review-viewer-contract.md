# Review viewer

Ferramenta web para olhar source, overlay e crops. Não executa o detector.

## Pastas

| Pasta | Obrigatório | Conteúdo |
|-------|-------------|----------|
| `dataset/` | sim | Imagens originais |
| `calibration/` | sim | `boxes.json`, `stats.csv` |
| `result/` | não | `*_segmentada_N.png` |

Gerar calibration:

```sh
make calibrate
```

Gerar result:

```sh
make run
```

## `boxes.json`

Chave = caminho relativo ao dataset, ex. `Peito_Congelado/img001.jpg`:

```json
{
  "Peito_Congelado/img001.jpg": {
    "candidates": [[x, y, w, h]],
    "kept": [[x, y, w, h]],
    "labels": [[x, y, w, h]]
  }
}
```

Coordenadas `[x, y, largura, altura]` na imagem original.

## Subir

```sh
make review
```

ou

```sh
uv run pdiseg-review --dataset data/Train_and_Validation --calibration calibration --result result
```

Abrir http://127.0.0.1:8765/

## Sem algum arquivo

- Sem entrada no `boxes.json` → mostra source e crops do disco, sem overlay.
- Sem PNG em `result/` → renderiza crop a partir da caixa verde se tiver metadata.
- Sem source → lista o frame com aviso.
