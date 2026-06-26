#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Segmentação do rótulo principal usando técnicas integradas:
- Texto claro sobre fundo escuro (diferença de Gaussianas)
- Corpo escuro sólido (limiar adaptativo)
- Bimodalidade (escuro + claro adjacentes)
- Filtros geométricos e posicionais
- Ajuste automático de parâmetros
Uso:
    python segmenta_rotulo.py -c imagem.jpg
    python segmenta_rotulo.py -c diretorio/
    python segmenta_rotulo.py -c all
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import numpy as np
import cv2
from skimage import exposure, measure, segmentation, filters, morphology
from skimage.morphology import remove_small_objects

# =============================================================================
# PARÂMETROS INICIAIS (baseados no código de inspiração)
# =============================================================================

INPUT_DIR  = "."
OUTPUT_DIR = "resultado"

class Config:
    def __init__(self):
        # Detecção de texto claro sobre fundo escuro
        self.sigma = 45          # raio do blur do fundo
        self.cT = 24             # diferença mínima texto-fundo
        self.pbg = 66            # percentil de fundo escuro
        self.bold = 3            # abertura para descartar texto fino
        self.dil = 11            # dilatação para agrupar letras

        # Corpo escuro sólido (gate anti-textura)
        self.body_block_div = 6
        self.body_C = 18

        # Filtros geométricos
        self.min_bold = 300      # mínimo de pixels de texto grosso
        self.min_a = 0.004       # área mínima como fração da imagem
        self.max_a = 0.06        # área máxima como fração da imagem
        self.max_dim = 0.60      # dimensão máxima como fração da imagem
        self.max_ar = 4.5        # razão de aspecto máxima
        self.max_bg = 118        # fundo local escuro
        self.min_body = 0.35     # sobreposição mínima com corpo escuro

        # Bimodalidade (para verificar escuro+claro)
        self.bimodal_thresh = 0.30  # proporção mínima da classe menos frequente

        # Posicionamento
        self.lateral_margin = 60   # descartar regiões perto das bordas laterais

        # Expansão final
        self.expand = 15

        # Seleção final
        self.max_crops = 2
        self.rel_score_min = 0.60

    def adjust_for_attempt(self, attempt):
        if attempt == 0:
            return
        elif attempt == 1:
            self.sigma = 35
            self.cT = 18
            self.pbg = 70
            self.bold = 2
            self.dil = 9
            self.min_bold = 200
            self.min_a = 0.003
            self.max_a = 0.08
            self.max_bg = 130
            self.min_body = 0.30
            self.lateral_margin = 40
            self.bimodal_thresh = 0.25
            self.rel_score_min = 0.50
        elif attempt == 2:
            self.sigma = 30
            self.cT = 15
            self.pbg = 75
            self.bold = 2
            self.dil = 7
            self.min_bold = 150
            self.min_a = 0.002
            self.max_a = 0.10
            self.max_bg = 140
            self.min_body = 0.25
            self.lateral_margin = 30
            self.bimodal_thresh = 0.20
            self.rel_score_min = 0.40
        return self

# =============================================================================
# FUNÇÕES AUXILIARES (adaptadas do código de inspiração)
# =============================================================================

def dark_body_mask(gray, P):
    """Máscara do corpo escuro sólido do rótulo (limiar adaptativo + morfologia)."""
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    bs = (min(H, W) // P['body_block_div']) | 1
    d = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY_INV, bs, P['body_C'])
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=2)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return d

def analyze_bimodality(roi):
    """Analisa bimodalidade (escuro+claro) em uma ROI."""
    if roi.size == 0:
        return 0, 0, 0, 0
    try:
        thresh = filters.threshold_otsu(roi)
    except:
        thresh = np.mean(roi)
    escuros = roi[roi <= thresh]
    claros = roi[roi > thresh]
    if len(escuros) == 0 or len(claros) == 0:
        return 0, thresh, 0, 0
    prop_escura = len(escuros) / len(roi)
    prop_clara = 1 - prop_escura
    if prop_escura < 0.3 or prop_clara < 0.3:
        return 0, thresh, prop_escura, 0
    contraste = np.mean(claros) - np.mean(escuros)
    desvio = abs(0.5 - prop_escura)
    score = contraste * (1 - desvio)
    return score, thresh, prop_escura, contraste

def bbox_distance(bbox1, bbox2):
    """Distância mínima entre dois bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    dx = max(0, x2 - (x1 + w1), x1 - (x2 + w2))
    dy = max(0, y2 - (y1 + h1), y1 - (y2 + h2))
    return np.hypot(dx, dy)

def nms_boxes(boxes, scores, overlap_thresh=0.35):
    """Non-Maximum Suppression para caixas sobrepostas."""
    if not boxes:
        return [], []
    indices = np.argsort(scores)[::-1]
    keep_boxes = []
    keep_scores = []
    for i in indices:
        x1, y1, w1, h1 = boxes[i]
        overlapping = False
        for k in range(len(keep_boxes)):
            x2, y2, w2, h2 = keep_boxes[k]
            ix = max(x1, x2)
            iy = max(y1, y2)
            ax = min(x1 + w1, x2 + w2)
            ay = min(y1 + h1, y2 + h2)
            inter = max(0, ax - ix) * max(0, ay - iy)
            area1 = w1 * h1
            area2 = w2 * h2
            if inter > overlap_thresh * min(area1, area2):
                overlapping = True
                break
        if not overlapping:
            keep_boxes.append(boxes[i])
            keep_scores.append(scores[i])
    return keep_boxes, keep_scores

# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def processar_imagem_com_ajuste(caminho_imagem, debug=False):
    """Processa a imagem com múltiplas tentativas de parâmetros."""
    img_color = cv2.imread(caminho_imagem)
    if img_color is None:
        raise ValueError(f"Não foi possível ler: {caminho_imagem}")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape
    A = H * W

    # Configurações para tentativas
    configs = [Config(), Config(), Config()]
    for i, cfg in enumerate(configs):
        cfg.adjust_for_attempt(i)

    best_boxes = []
    best_scores = []

    for attempt, cfg in enumerate(configs):
        P = cfg.__dict__
        if debug:
            print(f"\nTentativa {attempt+1}: sigma={P['sigma']}, cT={P['cT']}, min_body={P['min_body']}")

        # 1. Pré-processamento: apagar overlay de FPS (se houver)
        work = img_gray.copy()
        # Remove possível overlay no canto superior esquerdo
        fps_h = int(0.11 * H)
        fps_w = int(0.30 * W)
        if fps_h > 0 and fps_w > 0:
            work[:fps_h, :fps_w] = int(np.median(img_gray))

        # 2. Detecção de texto claro sobre fundo local escuro
        g = work.astype(np.float32)
        bg = cv2.GaussianBlur(g, (0, 0), P['sigma'])
        dark_bg = bg < np.percentile(bg, P['pbg'])
        text = (((g - bg) > P['cT']) & dark_bg).astype(np.uint8) * 255

        # 3. Abertura para descartar texto fino (código de barras, descrições)
        bold = cv2.morphologyEx(text, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                          (P['bold'], P['bold'])))

        # 4. Agrupar letras em rótulo (dilatação + fechamento)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P['dil'], P['dil']))
        grp = cv2.dilate(bold, se)
        grp = cv2.morphologyEx(grp, cv2.MORPH_CLOSE, se)

        # 5. Corpo escuro sólido (gate anti-textura)
        body = dark_body_mask(work, P)

        # 6. Componentes conectados
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(grp, 8)
        candidates = []  # (x, y, w, h, score, bimodal_score)

        for i in range(1, n):
            x, y, w, h, area = stats[i]
            comp = (lbl == i)
            bold_count = int(bold[comp].sum() // 255)

            # Filtros geométricos
            if bold_count < P['min_bold']:
                continue
            if area < P['min_a'] * A or area > P['max_a'] * A:
                continue
            if h > P['max_dim'] * H or w > P['max_dim'] * W:
                continue
            ar = max(w, h) / max(1, min(w, h))
            if ar > P['max_ar']:
                continue

            # Fundo local escuro
            if bg[y:y+h, x:x+w].mean() > P['max_bg']:
                continue

            # Sobreposição com corpo escuro
            body_ov = (body[y:y+h, x:x+w] > 0).mean()
            if body_ov < P['min_body']:
                continue

            # Posição: descartar se estiver perto das bordas laterais
            if x < P['lateral_margin'] or x + w > W - P['lateral_margin']:
                continue

            # Análise de bimodalidade na ROI expandida
            exp = 20
            rx1 = max(0, x - exp)
            ry1 = max(0, y - exp)
            rx2 = min(W, x + w + exp)
            ry2 = min(H, y + h + exp)
            roi = work[ry1:ry2, rx1:rx2]
            bim_score, _, prop_escura, contraste = analyze_bimodality(roi)

            # Score final: combina área, bold_count, body_ov, bim_score
            score = (bold_count * body_ov * (contraste + 1) * (1 + bim_score)) / (area + 1)
            candidates.append((x, y, w, h, score, bim_score))

        if candidates:
            # NMS nas caixas
            boxes = [(c[0], c[1], c[2], c[3]) for c in candidates]
            scores = [c[4] for c in candidates]
            keep_boxes, keep_scores = nms_boxes(boxes, scores, overlap_thresh=0.35)

            # Seleção final: top N por score
            if keep_boxes:
                keep_scores = np.array(keep_scores)
                top_idx = np.argsort(keep_scores)[::-1]
                top_n = min(P['max_crops'], len(keep_boxes))
                rel_thresh = P['rel_score_min'] * keep_scores[top_idx[0]]
                final_boxes = []
                final_scores = []
                for idx in top_idx[:top_n]:
                    if keep_scores[idx] >= rel_thresh:
                        final_boxes.append(keep_boxes[idx])
                        final_scores.append(keep_scores[idx])
                if final_boxes and (len(final_boxes) > len(best_boxes) or
                    (len(final_boxes) == len(best_boxes) and np.mean(final_scores) > np.mean(best_scores))):
                    best_boxes = final_boxes
                    best_scores = final_scores
                    if debug:
                        print(f"  Encontrados {len(final_boxes)} candidatos com scores: {final_scores}")

        # Se encontrou pelo menos 1 candidato, pode parar (já ajustamos parâmetros)
        if best_boxes:
            break

    # 7. Fallback: se não encontrou, usar o maior componente escuro
    if not best_boxes:
        if debug:
            print("Fallback: maior componente escuro")
        _, bin_fb = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bin_fb = cv2.morphologyEx(bin_fb, cv2.MORPH_CLOSE, kernel)
        bin_fb = segmentation.clear_border(bin_fb, buffer_size=30)
        labeled_fb = measure.label(bin_fb, connectivity=2)
        regions_fb = measure.regionprops(labeled_fb)
        melhor = None
        maior_area = 0
        for r in regions_fb:
            minr, minc, maxr, maxc = r.bbox
            area = r.area
            if area > maior_area and area > 500:
                maior_area = area
                melhor = (minc, minr, maxc - minc, maxr - minr)  # x, y, w, h
        if melhor:
            x, y, w, h = melhor
            exp = 20
            x = max(0, x - exp)
            y = max(0, y - exp)
            w = min(W - x, w + 2*exp)
            h = min(H - y, h + 2*exp)
            best_boxes = [(x, y, w, h)]
            best_scores = [1.0]

    # Expansão final dos boxes para incluir região clara adjacente
    expanded_boxes = []
    for x, y, w, h in best_boxes:
        exp = 15
        x = max(0, x - exp)
        y = max(0, y - exp)
        w = min(W - x, w + 2*exp)
        h = min(H - y, h + 2*exp)
        expanded_boxes.append((x, y, w, h))

    # Retorna a caixa com maior score (ou a primeira)
    if expanded_boxes:
        # Se tiver várias, pega a de maior área (ou a de maior score)
        if best_scores:
            best_idx = np.argmax(best_scores)
            x, y, w, h = expanded_boxes[best_idx]
        else:
            x, y, w, h = expanded_boxes[0]
        return (y, x, y + h, x + w)  # ymin, xmin, ymax, xmax
    return None

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO EM MASSA
# =============================================================================

def processar_imagem_e_salvar(caminho_entrada, caminho_saida, debug=False):
    bbox = processar_imagem_com_ajuste(caminho_entrada, debug=debug)
    img_color = cv2.imread(caminho_entrada)
    if img_color is None:
        return False

    if bbox is not None:
        ymin, xmin, ymax, xmax = bbox
        cv2.rectangle(img_color, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        print(f"  Rótulo detectado em: ({xmin},{ymin}) -> ({xmax},{ymax})")
    else:
        print("  Nenhum rótulo encontrado.")

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    cv2.imwrite(caminho_saida, img_color)
    return True

def processar_diretorio(dir_entrada, dir_saida, debug=False):
    extensoes = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif')
    arquivos = []
    for ext in extensoes:
        arquivos.extend(glob.glob(os.path.join(dir_entrada, '**', ext), recursive=True))
    if not arquivos:
        print(f"Nenhuma imagem encontrada em {dir_entrada}")
        return

    print(f"Processando {len(arquivos)} imagens em {dir_entrada}...")
    for arq in arquivos:
        rel_path = os.path.relpath(arq, dir_entrada)
        saida = os.path.join(dir_saida, rel_path)
        print(f"  Processando {arq} -> {saida}")
        try:
            processar_imagem_e_salvar(arq, saida, debug=debug)
        except Exception as e:
            print(f"    Erro: {e}")

# =============================================================================
# INTERFACE DE LINHA DE COMANDO
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Segmentação do rótulo principal usando técnicas integradas.")
    parser.add_argument("-c", "--comando", required=True,
                        help="Caminho para uma imagem, diretório ou 'all' para processar todo o dataset.")
    parser.add_argument("-o", "--output", default=OUTPUT_DIR,
                        help="Diretório de saída (padrão: 'resultado')")
    parser.add_argument("--debug", action="store_true",
                        help="Exibe informações de depuração")
    args = parser.parse_args()

    entrada = args.comando
    saida_base = args.output

    if entrada.lower() == 'all':
        dataset_dir = 'dataset'
        if not os.path.isdir(dataset_dir):
            print(f"Diretório 'dataset' não encontrado no caminho atual.")
            sys.exit(1)
        processar_diretorio(dataset_dir, saida_base, debug=args.debug)
    elif os.path.isdir(entrada):
        processar_diretorio(entrada, saida_base, debug=args.debug)
    elif os.path.isfile(entrada):
        saida = os.path.join(saida_base, os.path.basename(entrada))
        os.makedirs(saida_base, exist_ok=True)
        print(f"Processando {entrada} -> {saida}")
        processar_imagem_e_salvar(entrada, saida, debug=args.debug)
    else:
        print(f"Erro: '{entrada}' não é um arquivo ou diretório válido.")
        sys.exit(1)

    print("Processamento concluído.")

if __name__ == "__main__":
    main()