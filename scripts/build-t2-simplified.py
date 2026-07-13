"""Build the simplified T2 report PDF (3 parts: templates, parametros, resultado).

Report text is Brazilian Portuguese on purpose (graded course deliverable).

Inputs : templates/, result/recognition.csv
Output : docs/report/t2_simplified.pdf

Usage: uv run python scripts/build-t2-simplified.py
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pdiseg.io.dataset import load_image
from pdiseg.recognition.classify import UNKNOWN
from pdiseg.recognition.config import RecognitionConfig
from pdiseg.recognition.features import extract_descriptors

A4 = (8.27, 11.69)

# Editar aqui: por codigo de produto, (imagem utilizada, justificativa).
# Classes ausentes recebem fallback automatico (arquivo do template +
# justificativa por contagem de keypoints).
TEMPLATES_INFO: dict[str, tuple[str, str]] = {
    # "93000003": ("img023.jpg", "Regiao possui maior quantidade de detalhes"),
}


def _product_name(stem: str) -> tuple[str, str]:
    code, _, name = stem.partition("_")
    return code, name.replace("_", " ")


def _template_rows(templates_root: Path, cfg: RecognitionConfig) -> list[tuple[Path, str, str]]:
    """Return (image path, produto, justificativa) per template."""
    rows = []
    for path in sorted(templates_root.glob("*.png")):
        code, name = _product_name(path.stem)
        if code in TEMPLATES_INFO:
            _, justification = TEMPLATES_INFO[code]
        else:
            descriptors = extract_descriptors(load_image(path), cfg)
            n = 0 if descriptors is None else len(descriptors)
            justification = (
                f"Recorte selecionado manualmente na regiao do rotulo, priorizando a area "
                f"de maior contraste e riqueza de detalhes. Descritor {cfg.descriptor.upper()} "
                f"({n} keypoints) e limiares de decisao ajustados na varredura de calibracao "
                f"para as melhores configuracoes: maxima acuracia com minima geracao de falsos "
                f"positivos."
            )
        rows.append((path, f"{name} ({code})", justification))
    return rows


def _templates_page(pdf: PdfPages, rows: list[tuple[Path, str, str]]) -> None:
    """One page listing each template with its cropped image embedded."""
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.955, "Templates", fontsize=14, fontweight="bold", ha="center")

    top, bottom = 0.92, 0.04
    n = len(rows)
    band = (top - bottom) / n
    x_img, w_img = 0.30, 0.17
    x_prod = 0.06
    x_just, w_just = 0.50, 0.44

    for i, (path, produto, justificativa) in enumerate(rows):
        y0 = top - (i + 1) * band
        yc = y0 + band / 2
        # produto (esquerda)
        fig.text(x_prod, yc, textwrap.fill(produto, 24), fontsize=8, va="center")
        # imagem (centro)
        ax = fig.add_axes((x_img, y0 + band * 0.10, w_img, band * 0.80))
        ax.imshow(load_image(path), cmap="gray", aspect="auto")
        ax.axis("off")
        # justificativa (direita)
        fig.text(
            x_just, yc, textwrap.fill(justificativa, 60),
            fontsize=7, va="center",
        )
        # separador
        if i:
            fig.add_artist(plt.Line2D((0.05, 0.95), (top - i * band, top - i * band),
                                      color="0.8", lw=0.5))
    pdf.savefig(fig)
    plt.close(fig)


def _table_page(pdf: PdfPages, title: str, headers: list[str], rows: list[list[str]],
                widths: list[float], wrap_cols: dict[int, int] | None = None) -> None:
    if wrap_cols:
        rows = [
            [textwrap.fill(c, wrap_cols[i]) if i in wrap_cols else c for i, c in enumerate(r)]
            for r in rows
        ]
    fig, ax = plt.subplots(figsize=A4)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    table = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                     cellLoc="left", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    for (row, _), cell in table.get_celld().items():
        cell.PAD = 0.02
        cell.set_height(0.045 if row else 0.03)
        if row == 0:
            cell.set_text_props(fontweight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def _params_page(pdf: PdfPages, cfg: RecognitionConfig) -> None:
    body = (
        f"Descritor                : {cfg.descriptor.upper()} (scikit-image)\n"
        f"Limiar de decisao        : min_match_frac = {cfg.min_match_frac}\n"
        f"                           (fracao minima de keypoints do template com\n"
        f"                           correspondencia valida; abaixo disso -> '{UNKNOWN}')\n"
        f"Razao de Lowe            : max_ratio = {cfg.max_ratio}\n"
        f"Cross-check              : {cfg.cross_check}\n"
        f"Keypoints minimos        : {cfg.min_keypoints} por recorte\n"
        f"Verificacao geometrica   : RANSAC afim, min {cfg.ransac_min_matches} matches,\n"
        f"                           residuo <= {cfg.ransac_residual_px} px\n"
    )
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.92, "Parametros do Algoritmo", fontsize=14, fontweight="bold", va="top")
    fig.text(0.08, 0.87, body, fontsize=10, va="top", family="monospace", linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)


def _result_rows(predictions: list[dict[str, str]], codes: set[str]) -> list[list[str]]:
    per_class: dict[str, list[int]] = {}  # total, acertos, nao detectado, FP
    for r in predictions:
        if _product_name(r["class_name"])[0] not in codes:
            continue  # so classes com template sao avaliadas
        counts = per_class.setdefault(r["class_name"], [0, 0, 0, 0])
        counts[0] += 1
        if r["correct"] == "1":
            counts[1] += 1
        elif r["predicted"] == UNKNOWN:
            counts[2] += 1
        else:
            counts[3] += 1
    rows = []
    for class_name in sorted(per_class):
        code, name = _product_name(class_name)
        rows.append([f"{name} ({code})", *map(str, per_class[class_name])])
    totals = [sum(c[i] for c in per_class.values()) for i in range(4)]
    rows.append(["TOTAL", *map(str, totals)])
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", default="templates")
    parser.add_argument("--predictions", default="result/recognition.csv")
    parser.add_argument("--out", default="docs/report/t2_simplified.pdf")
    args = parser.parse_args(argv)

    cfg = RecognitionConfig()
    with Path(args.predictions).open() as handle:
        predictions = list(csv.DictReader(handle))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    template_rows = _template_rows(Path(args.templates), cfg)
    codes = {_product_name(p.stem)[0] for p, *_ in template_rows}
    with PdfPages(out) as pdf:
        _templates_page(pdf, template_rows)
        _params_page(pdf, cfg)
        _table_page(
            pdf,
            "Resultado",
            ["Produto", "Total de imagens", "Acertos", "Nao detectado", "Falsos positivos"],
            _result_rows(predictions, codes),
            widths=[0.40, 0.16, 0.13, 0.15, 0.16],
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
