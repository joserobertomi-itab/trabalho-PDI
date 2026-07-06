"""Build the T2 technical report PDF (matplotlib PdfPages, no extra deps).

Report text is Brazilian Portuguese on purpose: it is the graded course
deliverable, not repo documentation.

Inputs : templates/, result/recognition.csv, calibration/recognition_sweep.csv
Output : docs/report/t2_report.pdf

Usage: uv run python scripts/build-t2-report.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from pdiseg.io.dataset import load_image
from pdiseg.recognition.classify import UNKNOWN
from pdiseg.recognition.config import RecognitionConfig

A4 = (8.27, 11.69)


def _text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.92, title, fontsize=16, fontweight="bold", va="top")
    fig.text(0.08, 0.86, body, fontsize=10, va="top", wrap=True, family="serif", linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)


def _method_body(cfg: RecognitionConfig, images: int, classes: int) -> str:
    return (
        "Objetivo. Identificar automaticamente o produto presente em cada segmento gerado\n"
        "pelo Trabalho 1, comparando descritores locais do segmento com um template por\n"
        "classe (recorte de regiao discriminativa da embalagem), sem tecnicas de IA/ML.\n"
        "\n"
        "Metodo.\n"
        f"  1. Deteccao de pontos de interesse e extracao de descritores {cfg.descriptor.upper()}\n"
        "     (implementacao scikit-image; sem OpenCV) no template e em cada segmento.\n"
        "  2. Casamento por forca bruta com cross-check e razao de Lowe\n"
        f"     (max_ratio = {cfg.max_ratio}).\n"
        "  3. Pontuacao = fracao dos pontos do template com correspondencia valida no\n"
        "     segmento (metrica de calibracao exigida pelo enunciado).\n"
        "  4. Por imagem-fonte, agrega-se o melhor score de cada template entre os seus\n"
        "     segmentos; vence o template de maior score somente se ultrapassar o limiar\n"
        f"     min_match_frac = {cfg.min_match_frac}; caso contrario a resposta e '{UNKNOWN}'.\n"
        "     Essa recusa explicita e o mecanismo de reducao de falsos positivos.\n"
        "\n"
        "Dados.\n"
        f"  {classes} classes de bandeja, {images} imagens avaliadas; templates recortados da\n"
        "  regiao do rotulo detectada pelo T1 (1 por classe, diretorio templates/).\n"
        "\n"
        "Reproducao.\n"
        "  make run          # segmentacao T1\n"
        "  make templates    # bootstrap dos templates\n"
        "  make recognize    # reconhecimento + varredura de calibracao\n"
        "  uv run python scripts/build-t2-report.py\n"
    )


def _template_page(pdf: PdfPages, templates_root: Path) -> None:
    paths = sorted(templates_root.glob("*.png"))
    cols = 3
    rows = -(-len(paths) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=A4)
    fig.suptitle("Templates (1 por classe)", fontsize=14, fontweight="bold")
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, path in zip(np.ravel(axes), paths, strict=False):
        ax.imshow(load_image(path), cmap="gray")
        ax.set_title(path.stem.split("_", 1)[-1].replace("_", " "), fontsize=5)
    pdf.savefig(fig)
    plt.close(fig)


def _sweep_page(pdf: PdfPages, sweep_csv: Path, chosen: float) -> None:
    with sweep_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    t = [float(r["min_match_frac"]) for r in rows]
    acc = [float(r["accuracy"]) for r in rows]
    fpr = [float(r["false_positive_rate"]) for r in rows]
    unk = [int(r["unknown"]) / int(r["images"]) for r in rows]

    fig, ax = plt.subplots(figsize=(A4[0], A4[1] / 2.2))
    ax.plot(t, acc, label="acuracia", lw=2)
    ax.plot(t, fpr, label="taxa de falsos positivos", lw=2)
    ax.plot(t, unk, label="fracao 'unknown'", lw=1.5, ls="--")
    ax.axvline(chosen, color="k", lw=1, ls=":", label=f"limiar escolhido = {chosen}")
    ax.set_xlabel("min_match_frac (fracao de correspondencias validas do template)")
    ax.set_ylabel("proporcao")
    ax.set_title("Calibracao do limiar de decisao")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _load_predictions(predictions_csv: Path) -> list[dict[str, str]]:
    with predictions_csv.open() as handle:
        return list(csv.DictReader(handle))


def _confusion_page(pdf: PdfPages, predictions: list[dict[str, str]]) -> None:
    classes = sorted({r["class_name"] for r in predictions})
    labels = [*classes, UNKNOWN]
    index = {name: i for i, name in enumerate(labels)}
    matrix = np.zeros((len(classes), len(labels)), dtype=int)
    for r in predictions:
        matrix[index[r["class_name"]], index[r["predicted"]]] += 1

    fig, ax = plt.subplots(figsize=A4)
    ax.imshow(matrix, cmap="Blues")
    short = [name.split("_")[0] for name in labels]
    ax.set_xticks(range(len(labels)), short, rotation=90, fontsize=6)
    ax.set_yticks(range(len(classes)), [n.split("_")[0] for n in classes], fontsize=6)
    ax.set_xlabel("predito")
    ax.set_ylabel("classe verdadeira")
    ax.set_title("Matriz de confusao (por imagem-fonte)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=5, color=color)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _results_page(pdf: PdfPages, predictions: list[dict[str, str]], cfg: RecognitionConfig) -> None:
    scores_correct = [float(r["score"]) for r in predictions if r["correct"] == "1"]
    scores_fp = [
        float(r["score"])
        for r in predictions
        if r["correct"] == "0" and r["predicted"] != UNKNOWN
    ]
    counts = Counter(r["predicted"] == UNKNOWN for r in predictions)
    total = len(predictions)
    correct = len(scores_correct)
    fp = len(scores_fp)
    unknown = counts[True]

    fig = plt.figure(figsize=A4)
    ax = fig.add_axes((0.1, 0.55, 0.8, 0.35))
    bins = np.linspace(0, 1, 41)
    ax.hist(scores_correct, bins=bins, alpha=0.7, label="acertos")
    if scores_fp:
        ax.hist(scores_fp, bins=bins, alpha=0.7, label="falsos positivos")
    ax.axvline(cfg.min_match_frac, color="k", ls=":", lw=1)
    ax.set_xlabel("melhor score (fracao de matches validos)")
    ax.set_ylabel("imagens")
    ax.set_title("Distribuicao dos scores emitidos")
    ax.legend()

    body = (
        f"Total de imagens-fonte avaliadas : {total}\n"
        f"Classificacoes corretas          : {correct}  ({correct / total:.1%})\n"
        f"Falsos positivos                 : {fp}  ({fp / total:.1%})\n"
        f"Respostas '{UNKNOWN}'               : {unknown}  ({unknown / total:.1%})\n"
        "\n"
        "Limitacoes e discussao.\n"
        "  - A decisao usa exclusivamente descritores locais (exigencia do enunciado);\n"
        "    embalagens de classes distintas com o mesmo layout de rotulo tendem a se\n"
        "    confundir e sao o principal residuo de falsos positivos.\n"
        "  - Segmentos do T1 sem o rotulo visivel produzem poucos pontos de interesse e\n"
        "    caem em 'unknown' - comportamento desejado (recusa em vez de chute).\n"
        "  - O limiar foi escolhido na varredura priorizando a menor taxa de falsos\n"
        "    positivos com perda minima de acuracia, conforme o enunciado.\n"
    )
    fig.text(0.1, 0.45, "Resultados", fontsize=14, fontweight="bold", va="top")
    fig.text(0.1, 0.40, body, fontsize=10, va="top", family="monospace", linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", default="templates")
    parser.add_argument("--predictions", default="result/recognition.csv")
    parser.add_argument("--sweep", default="calibration/recognition_sweep.csv")
    parser.add_argument("--out", default="docs/report/t2_report.pdf")
    args = parser.parse_args(argv)

    cfg = RecognitionConfig()
    predictions = _load_predictions(Path(args.predictions))
    classes = len({r["class_name"] for r in predictions})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        _text_page(
            pdf,
            "Trabalho Pratico 2 - Reconhecimento de Produtos por\n"
            "Correspondencia de Caracteristicas Locais",
            _method_body(cfg, len(predictions), classes),
        )
        _template_page(pdf, Path(args.templates))
        _sweep_page(pdf, Path(args.sweep), cfg.min_match_frac)
        _confusion_page(pdf, predictions)
        _results_page(pdf, predictions, cfg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
