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


def _template_rows(templates_root: Path, cfg: RecognitionConfig) -> list[list[str]]:
    rows = []
    for path in sorted(templates_root.glob("*.png")):
        code, name = _product_name(path.stem)
        if code in TEMPLATES_INFO:
            source, justification = TEMPLATES_INFO[code]
        else:
            descriptors = extract_descriptors(load_image(path), cfg)
            n = 0 if descriptors is None else len(descriptors)
            source = path.name
            justification = (
                f"Recorte automatico do rotulo (detector T1) com {n} keypoints "
                f"{cfg.descriptor.upper()} — frame mais rico em detalhes entre os testados"
            )
        rows.append([f"{name}\n({code})", source, justification])
    return rows


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


def _result_rows(predictions: list[dict[str, str]]) -> list[list[str]]:
    per_class: dict[str, list[int]] = {}  # total, acertos, nao detectado, FP
    for r in predictions:
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
    with PdfPages(out) as pdf:
        _table_page(
            pdf,
            "Templates",
            ["Produto", "Imagem utilizada", "Justificativa"],
            _template_rows(Path(args.templates), cfg),
            widths=[0.28, 0.30, 0.42],
            wrap_cols={1: 38, 2: 52},
        )
        _params_page(pdf, cfg)
        _table_page(
            pdf,
            "Resultado",
            ["Produto", "Total de imagens", "Acertos", "Nao detectado", "Falsos positivos"],
            _result_rows(predictions),
            widths=[0.40, 0.16, 0.13, 0.15, 0.16],
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
