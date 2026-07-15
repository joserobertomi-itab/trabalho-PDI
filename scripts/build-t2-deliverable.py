"""Build the graded T2 deliverable PDF (cover + intro + templates + params + results).

Dense A4 packing: related sections share pages; tables fill their axes (no internal gaps).
Text is Brazilian Portuguese on purpose (graded course deliverable).

Inputs : templates/, result/recognition.csv, calibration/recognition_sweep.csv
Output : docs/report/Relatorio_T2_SIFT.pdf

Usage: uv run python scripts/build-t2-deliverable.py
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
ML = 0.06  # left margin
MW = 0.88  # content width

AUTHORS = "JOSÉ ROBERTO, FELIPE VERSIANE"
REPO_URL = "https://github.com/joserobertomi-itab/trabalho-PDI"
YEAR = "2026"

TEMPLATES_INFO: dict[str, tuple[str, str]] = {
    "93000005": (
        "93000005_Meio_das_Asas_Congelado.png",
        "Rótulo nítido, bom contraste — keypoints SIFT consistentes.",
    ),
    "93000006": (
        "93000006_Coxinhas_das_Asas_Congelado.png",
        "Rótulo nítido, bom contraste — keypoints SIFT consistentes.",
    ),
    "93000025": (
        "93000025_File_de_Peito_Congelado.png",
        "Rótulo nítido, bom contraste — keypoints SIFT consistentes.",
    ),
    "93000064": (
        "93000064_Coracao.png",
        "Rótulo nítido, bom contraste — keypoints SIFT consistentes.",
    ),
    "93000068": (
        "93000068_Moela_Congelada.png",
        "Rótulo nítido, bom contraste — keypoints SIFT consistentes.",
    ),
    "93000096": (
        "93000096_File_de_Coxas_e_Sobre_Coxas_com_Pele_Congelado.png",
        "Rótulo nítido, bom contraste — keypoints SIFT consistentes.",
    ),
    "93000106": (
        "93000106_Coxas_e_Sobrecoxas_Congelado.png",
        "Rótulo nítido, bom contraste — keypoints SIFT consistentes.",
    ),
}


def _product_name(stem: str) -> tuple[str, str]:
    code, _, name = stem.partition("_")
    return code, name.replace("_", " ")


def _wrap(text: str, width: int = 102) -> str:
    return "\n".join(
        textwrap.fill(line, width) if line.strip() else "" for line in text.split("\n")
    )


def _text(
    fig: plt.Figure,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float = 9.0,
    fontweight: str = "normal",
    family: str = "serif",
    linespacing: float = 1.25,
    width: int = 102,
) -> float:
    wrapped = _wrap(text, width)
    fig.text(
        x, y, wrapped, fontsize=fontsize, fontweight=fontweight, family=family,
        va="top", ha="left", linespacing=linespacing,
    )
    n_lines = wrapped.count("\n") + 1
    line_h = (fontsize * linespacing) / (A4[1] * 72)
    return y - n_lines * line_h - 0.008


def _title(fig: plt.Figure, y: float, title: str) -> float:
    return _text(fig, ML, y, title, fontsize=11, fontweight="bold", family="sans-serif", width=70)


def _table(
    fig: plt.Figure,
    y_top: float,
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[float],
    *,
    fontsize: float = 7.5,
    row_h_fig: float = 0.024,
    highlight_row: int | None = None,
) -> float:
    """Draw a table that fully fills its axes; return y just below it."""
    n_rows = len(rows) + 1  # + header
    h = n_rows * row_h_fig
    ax = fig.add_axes((ML, y_top - h, MW, h))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.0)
    cell_h = 1.0 / n_rows
    for (r, _), cell in table.get_celld().items():
        cell.PAD = 0.008
        cell.set_height(cell_h)
        cell.set_edgecolor("#bbbbbb")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dedede")
        elif highlight_row is not None and r == highlight_row + 1:
            cell.set_facecolor("#c6efce")
        elif rows and r == len(rows) and str(rows[-1][0]).upper().startswith("TOTAL"):
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#ececec")
    return y_top - h - 0.012


def _cover_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.72, "INSTITUTO FEDERAL DE EDUCAÇÃO, CIÊNCIA E TECNOLOGIA DE GOIÁS",
             fontsize=10.5, fontweight="bold", ha="center")
    fig.text(0.5, 0.685, "BACHARELADO EM CIÊNCIA DA COMPUTAÇÃO",
             fontsize=10.5, fontweight="bold", ha="center")
    fig.text(0.5, 0.58, AUTHORS, fontsize=12, fontweight="bold", ha="center")
    fig.text(
        0.5, 0.48,
        "RECONHECIMENTO DE PRODUTOS POR CORRESPONDÊNCIA DE\nCARACTERÍSTICAS LOCAIS",
        fontsize=13, fontweight="bold", ha="center", linespacing=1.3,
    )
    fig.text(0.5, 0.40, "RELATÓRIO TÉCNICO — TRABALHO PRÁTICO 2 (SIFT)",
             fontsize=11, fontweight="bold", ha="center")
    fig.text(0.5, 0.28, f"ANÁPOLIS, {YEAR}", fontsize=11, fontweight="bold", ha="center")
    pdf.savefig(fig)
    plt.close(fig)


def _intro_templates_page(
    pdf: PdfPages,
    n_classes: int,
    class_names: list[str],
    rows: list[tuple[Path, str, str, str]],
) -> None:
    """Intro + source + full template table on one packed page."""
    names = ", ".join(class_names[:-1]) + f" e {class_names[-1]}" if len(class_names) > 1 else class_names[0]
    fig = plt.figure(figsize=A4)
    y = 0.97
    y = _title(fig, y, "1. INTRODUÇÃO")
    y = _text(
        fig, ML, y,
        "Segunda etapa do trabalho prático: reconhecer produtos em embalagens de bandeja a "
        "partir dos segmentos do T1, sem IA/ML. Usa-se SIFT (descritores locais), robusto a "
        "escala, rotação, iluminação e perspectiva. "
        f"{n_classes} classes ({names}): template = recorte do rótulo; vence o maior placar "
        f"se ≥ limiar calibrado; senão '{UNKNOWN}'.",
        fontsize=8.6,
    )
    y = _title(fig, y, "1.2 CÓDIGO-FONTE E TEMPLATES")
    y = _text(
        fig, ML, y,
        f"Código: {REPO_URL}\n"
        "Local: make setup && make run && make recognize && make report\n"
        "Docker: make docker-tools  |  DATA=./dataset OUT=./resultado make docker-up\n"
        "Templates: templates/*.png  ·  Entrada T1: result/<Classe>/*_segmented_*.png",
        fontsize=7.8, family="monospace", linespacing=1.22,
    )
    y = _title(fig, y, "2. TABELA DOS TEMPLATES")
    y = _text(
        fig, ML, y,
        "Uma fotografia nítida por classe; recorte da região do rótulo com o nome do produto.",
        fontsize=8.2,
    )
    top, bottom = y - 0.002, 0.025
    n = len(rows)
    band = (top - bottom) / max(n, 1)
    for i, (path, produto, arquivo, justificativa) in enumerate(rows):
        y0 = top - (i + 1) * band
        yc = y0 + band / 2
        fig.text(0.055, yc + band * 0.12, textwrap.fill(produto, 26), fontsize=7.0,
                 va="center", fontweight="bold")
        fig.text(0.055, yc - band * 0.22, textwrap.fill(arquivo, 30), fontsize=5.2,
                 va="center", color="0.35")
        ax = fig.add_axes((0.33, y0 + band * 0.04, 0.24, band * 0.92))
        ax.imshow(load_image(path), cmap="gray", aspect="auto")
        ax.axis("off")
        fig.text(0.60, yc, textwrap.fill(justificativa, 40), fontsize=7.2, va="center")
        if i:
            fig.add_artist(plt.Line2D((0.05, 0.95), (top - i * band,) * 2, color="0.82", lw=0.4))
    pdf.savefig(fig)
    plt.close(fig)


def _template_rows(templates_root: Path, cfg: RecognitionConfig) -> list[tuple[Path, str, str, str]]:
    rows = []
    for path in sorted(templates_root.glob("*.png")):
        code, name = _product_name(path.stem)
        if code in TEMPLATES_INFO:
            arquivo, justification = TEMPLATES_INFO[code]
        else:
            descriptors = extract_descriptors(load_image(path), cfg)
            n = 0 if descriptors is None else len(descriptors)
            arquivo, justification = path.name, f"Recorte do rótulo (SIFT, {n} keypoints)."
        rows.append((path, f"{name} ({code})", arquivo, justification))
    return rows


def _params_results_page(
    pdf: PdfPages,
    cfg: RecognitionConfig,
    sweep_csv: Path | None,
    result_rows: list[list[str]],
    total: int,
    correct: int,
    unknown: int,
    fp: int,
) -> None:
    """Params + calibration + results + conclusion packed on one page."""
    acc = 100.0 * correct / total if total else 0.0
    fig = plt.figure(figsize=A4)
    y = 0.97
    y = _title(fig, y, "3. PARÂMETROS DO ALGORITMO")
    y = _text(
        fig, ML, y,
        f"Descritor {cfg.descriptor.upper()} (scikit-image). Casamento força-bruta com "
        f"cross-check={cfg.cross_check} e razão de Lowe (max_ratio={cfg.max_ratio}). "
        f"RANSAC afim (≥{cfg.ransac_min_matches} matches, residual ≤{cfg.ransac_residual_px} px). "
        f"Placar = fração de keypoints do template com match válido; classificação só se "
        f"≥ min_match_frac={cfg.min_match_frac}.",
        fontsize=8.4,
    )
    param_rows = [
        ["Descritor", f"{cfg.descriptor.upper()} (scikit-image)"],
        ["Razão de Lowe (max_ratio)", str(cfg.max_ratio)],
        ["Cross-check", str(cfg.cross_check)],
        ["Keypoints mínimos / recorte", str(cfg.min_keypoints)],
        ["RANSAC (min matches / residual)", f"{cfg.ransac_min_matches} / {cfg.ransac_residual_px} px"],
        ["Score", "fração de keypoints do template com match válido"],
        ["Limiar de decisão (min_match_frac)", str(cfg.min_match_frac)],
    ]
    y = _table(
        fig, y, ["Parâmetro", "Valor"], param_rows, [0.55, 0.45],
        fontsize=7.2, row_h_fig=0.022,
    )

    y = _title(fig, y, "3.1 HISTÓRICO DE CALIBRAÇÃO")
    y = _text(
        fig, ML, y,
        "Único parâmetro calibrado: min_match_frac. Placares calculados uma vez; varredura "
        "do limiar priorizando confiabilidade (menos FP).",
        fontsize=8.2,
    )

    if sweep_csv is not None and sweep_csv.is_file():
        with sweep_csv.open() as handle:
            raw = list(csv.DictReader(handle))
        rows_f = [r for r in raw if 0.0 < float(r["min_match_frac"]) <= 0.10]
        if not rows_f:
            rows_f = raw[:10]
        table_rows: list[list[str]] = []
        highlight = None
        for i, r in enumerate(rows_f):
            frac = float(r["min_match_frac"])
            table_rows.append([
                f"{frac:.2f}", r["correct"], r["unknown"], r["false_positives"],
                f"{100 * float(r['accuracy']):.1f}",
            ])
            if abs(frac - cfg.min_match_frac) < 1e-9:
                highlight = i
        y = _table(
            fig, y,
            ["Limiar", "Acertos", "Não det.", "FP", "Acurácia %"],
            table_rows, [0.14, 0.18, 0.20, 0.18, 0.20],
            fontsize=6.6, row_h_fig=0.0175, highlight_row=highlight,
        )
        y = _text(
            fig, ML, y,
            f"Linha destacada = limiar escolhido ({cfg.min_match_frac}). "
            f"Método: argmax do placar; abaixo do limiar → '{UNKNOWN}'.",
            fontsize=7.2,
        )

    y = _title(fig, y, "4. RESULTADOS")
    y = _text(
        fig, ML, y,
        f"Com min_match_frac = {cfg.min_match_frac}, avaliadas {total} imagens das classes "
        "com template:",
        fontsize=8.2,
    )
    y = _table(
        fig, y,
        ["Produto", "Total", "Acertos", "Não det.", "FP"],
        result_rows, [0.48, 0.12, 0.14, 0.14, 0.10],
        fontsize=6.8, row_h_fig=0.019,
    )
    y = _text(
        fig, ML, y,
        f"Acurácia global: {acc:.1f}%  ·  Acertos: {correct}  ·  "
        f"Não detectados: {unknown}  ·  Falsos positivos: {fp}",
        fontsize=8.6, fontweight="bold", family="sans-serif",
    )
    y = _title(fig, y, "5. CONCLUSÃO")
    _text(
        fig, ML, y,
        "Classificação viável com visão clássica (SIFT + Lowe + RANSAC), sem treinamento "
        "supervisionado, com um template por classe. Limiares baixos aumentam cobertura e FP; "
        f"min_match_frac = {cfg.min_match_frac} equilibrou confiabilidade e cobertura "
        f"({acc:.1f}%). Entrega Docker (make docker-tools) reproduz T1 → T2 → PDFs sem alterar "
        "o código congelado.",
        fontsize=8.4,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _result_rows(
    predictions: list[dict[str, str]], codes: set[str],
) -> tuple[list[list[str]], int, int, int, int]:
    per_class: dict[str, list[int]] = {}
    for r in predictions:
        if _product_name(r["class_name"])[0] not in codes:
            continue
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
    return rows, totals[0], totals[1], totals[2], totals[3]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", default="templates")
    parser.add_argument("--predictions", default="result/recognition.csv")
    parser.add_argument("--sweep", default="calibration/recognition_sweep.csv")
    parser.add_argument("--out", default="docs/report/Relatorio_T2_SIFT.pdf")
    args = parser.parse_args(argv)

    cfg = RecognitionConfig()
    with Path(args.predictions).open() as handle:
        predictions = list(csv.DictReader(handle))

    templates_root = Path(args.templates)
    template_rows = _template_rows(templates_root, cfg)
    codes = {_product_name(p.stem)[0] for p, *_ in template_rows}
    class_labels = [_product_name(p.stem)[1] for p, *_ in template_rows]
    result_rows, total, correct, unknown, fp = _result_rows(predictions, codes)
    acc = 100.0 * correct / total if total else 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sweep = Path(args.sweep)
    with PdfPages(out) as pdf:
        _cover_page(pdf)
        _intro_templates_page(pdf, len(class_labels), class_labels, template_rows)
        _params_results_page(
            pdf, cfg, sweep if sweep.is_file() else None,
            result_rows, total, correct, unknown, fp,
        )
    print(f"wrote {out} ({acc:.1f}% accuracy, {total} images)")


if __name__ == "__main__":
    main()
