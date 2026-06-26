#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main label segmentation using integrated techniques:
- Light text on dark background (Gaussian difference)
- Solid dark body (adaptive threshold)
- Bimodality (adjacent dark + light)
- Geometric and positional filters
- Automatic parameter adjustment
Usage:
    python reference_segment_label.py -c image.jpg
    python reference_segment_label.py -c directory/
    python reference_segment_label.py -c all
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
# INITIAL PARAMETERS (inspiration baseline)
# =============================================================================

INPUT_DIR  = "."
OUTPUT_DIR = "output"

class Config:
    def __init__(self):
        # Light-on-dark text detection
        self.sigma = 45          # background blur radius
        self.cT = 24             # min text-background difference
        self.pbg = 66            # dark background percentile
        self.bold = 3            # opening to discard fine text
        self.dil = 11            # dilation to group letters

        # Solid dark body (anti-texture gate)
        self.body_block_div = 6
        self.body_C = 18

        # Geometric filters
        self.min_bold = 300      # min bold text pixels
        self.min_a = 0.004       # min area as image fraction
        self.max_a = 0.06        # max area as image fraction
        self.max_dim = 0.60      # max dimension as image fraction
        self.max_ar = 4.5        # max aspect ratio
        self.max_bg = 118        # local dark background
        self.min_body = 0.35     # min overlap with dark body

        # Bimodality (dark+light check)
        self.bimodal_thresh = 0.30  # min minority class fraction

        # Positioning
        self.lateral_margin = 60   # discard regions near side borders

        # Final expansion
        self.expand = 15

        # Final selection
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
# HELPERS (adapted from inspiration code)
# =============================================================================

def dark_body_mask(gray, P):
    """Solid dark label body mask (adaptive threshold + morphology)."""
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
    """Analyze dark+light bimodality in an ROI."""
    if roi.size == 0:
        return 0, 0, 0, 0
    try:
        thresh = filters.threshold_otsu(roi)
    except:
        thresh = np.mean(roi)
    dark_pixels = roi[roi <= thresh]
    light_pixels = roi[roi > thresh]
    if len(dark_pixels) == 0 or len(light_pixels) == 0:
        return 0, thresh, 0, 0
    dark_fraction = len(dark_pixels) / len(roi)
    light_fraction = 1 - dark_fraction
    if dark_fraction < 0.3 or light_fraction < 0.3:
        return 0, thresh, dark_fraction, 0
    contrast = np.mean(light_pixels) - np.mean(dark_pixels)
    balance_dev = abs(0.5 - dark_fraction)
    score = contrast * (1 - balance_dev)
    return score, thresh, dark_fraction, contrast

def bbox_distance(bbox1, bbox2):
    """Minimum distance between two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    dx = max(0, x2 - (x1 + w1), x1 - (x2 + w2))
    dy = max(0, y2 - (y1 + h1), y1 - (y2 + h2))
    return np.hypot(dx, dy)

def nms_boxes(boxes, scores, overlap_thresh=0.35):
    """Non-maximum suppression for overlapping boxes."""
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
# MAIN PIPELINE
# =============================================================================

def process_image_with_retries(image_path, debug=False):
    """Process image with multiple parameter attempts."""
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise ValueError(f"Could not read: {image_path}")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape
    A = H * W

    # Attempt configurations
    configs = [Config(), Config(), Config()]
    for i, cfg in enumerate(configs):
        cfg.adjust_for_attempt(i)

    best_boxes = []
    best_scores = []

    for attempt, cfg in enumerate(configs):
        P = cfg.__dict__
        if debug:
            print(f"\nAttempt {attempt+1}: sigma={P['sigma']}, cT={P['cT']}, min_body={P['min_body']}")

        # 1. Preprocess: mask FPS overlay (if present)
        work = img_gray.copy()
        # Remove possible overlay in the top-left corner
        fps_h = int(0.11 * H)
        fps_w = int(0.30 * W)
        if fps_h > 0 and fps_w > 0:
            work[:fps_h, :fps_w] = int(np.median(img_gray))

        # 2. Detecção de texto claro sobre local dark background
        g = work.astype(np.float32)
        bg = cv2.GaussianBlur(g, (0, 0), P['sigma'])
        dark_bg = bg < np.percentile(bg, P['pbg'])
        text = (((g - bg) > P['cT']) & dark_bg).astype(np.uint8) * 255

        # 3. Opening to discard fine text (barcodes, descriptions)
        bold = cv2.morphologyEx(text, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                          (P['bold'], P['bold'])))

        # 4. Group letters into label (dilation + closing)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P['dil'], P['dil']))
        grp = cv2.dilate(bold, se)
        grp = cv2.morphologyEx(grp, cv2.MORPH_CLOSE, se)

        # 5. Solid dark body (anti-texture gate)
        body = dark_body_mask(work, P)

        # 6. Connected components
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(grp, 8)
        candidates = []  # (x, y, w, h, score, bimodal_score)

        for i in range(1, n):
            x, y, w, h, area = stats[i]
            comp = (lbl == i)
            bold_count = int(bold[comp].sum() // 255)

            # Geometric filters
            if bold_count < P['min_bold']:
                continue
            if area < P['min_a'] * A or area > P['max_a'] * A:
                continue
            if h > P['max_dim'] * H or w > P['max_dim'] * W:
                continue
            ar = max(w, h) / max(1, min(w, h))
            if ar > P['max_ar']:
                continue

            # Local dark background
            if bg[y:y+h, x:x+w].mean() > P['max_bg']:
                continue

            # Overlap with dark body
            body_ov = (body[y:y+h, x:x+w] > 0).mean()
            if body_ov < P['min_body']:
                continue

            # Position: discard if near side borders
            if x < P['lateral_margin'] or x + w > W - P['lateral_margin']:
                continue

            # Bimodality analysis on expanded ROI
            exp = 20
            rx1 = max(0, x - exp)
            ry1 = max(0, y - exp)
            rx2 = min(W, x + w + exp)
            ry2 = min(H, y + h + exp)
            roi = work[ry1:ry2, rx1:rx2]
            bim_score, _, dark_fraction, contrast = analyze_bimodality(roi)

            # Final score: combines area, bold_count, body_ov, bim_score
            score = (bold_count * body_ov * (contrast + 1) * (1 + bim_score)) / (area + 1)
            candidates.append((x, y, w, h, score, bim_score))

        if candidates:
            # NMS on boxes
            boxes = [(c[0], c[1], c[2], c[3]) for c in candidates]
            scores = [c[4] for c in candidates]
            keep_boxes, keep_scores = nms_boxes(boxes, scores, overlap_thresh=0.35)

            # Final selection: top N por score
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
                        print(f"  Found {len(final_boxes)} candidates with scores: {final_scores}")

        # Se encontrou pelo menos 1 candidato, pode parar (já ajustamos parâmetros)
        if best_boxes:
            break

    # 7. Fallback: if none found, use largest dark component
    if not best_boxes:
        if debug:
            print("Fallback: largest dark component")
        _, bin_fb = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bin_fb = cv2.morphologyEx(bin_fb, cv2.MORPH_CLOSE, kernel)
        bin_fb = segmentation.clear_border(bin_fb, buffer_size=30)
        labeled_fb = measure.label(bin_fb, connectivity=2)
        regions_fb = measure.regionprops(labeled_fb)
        best_region = None
        largest_area = 0
        for r in regions_fb:
            minr, minc, maxr, maxc = r.bbox
            area = r.area
            if area > largest_area and area > 500:
                largest_area = area
                best_region = (minc, minr, maxc - minc, maxr - minr)  # x, y, w, h
        if best_region:
            x, y, w, h = best_region
            exp = 20
            x = max(0, x - exp)
            y = max(0, y - exp)
            w = min(W - x, w + 2*exp)
            h = min(H - y, h + 2*exp)
            best_boxes = [(x, y, w, h)]
            best_scores = [1.0]

    # Final box expansion to include adjacent light region
    expanded_boxes = []
    for x, y, w, h in best_boxes:
        exp = 15
        x = max(0, x - exp)
        y = max(0, y - exp)
        w = min(W - x, w + 2*exp)
        h = min(H - y, h + 2*exp)
        expanded_boxes.append((x, y, w, h))

    # Return highest-score box (or the first)
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
# BATCH PROCESSING
# =============================================================================

def process_and_save_image(input_path, output_path, debug=False):
    bbox = process_image_with_retries(input_path, debug=debug)
    img_color = cv2.imread(input_path)
    if img_color is None:
        return False

    if bbox is not None:
        ymin, xmin, ymax, xmax = bbox
        cv2.rectangle(img_color, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        print(f"  Label detected at: ({xmin},{ymin}) -> ({xmax},{ymax})")
    else:
        print("  No label found.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_color)
    return True

def process_directory(input_dir, output_dir, debug=False):
    extensoes = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif')
    files = []
    for ext in extensoes:
        files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    if not files:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(files)} images in {input_dir}...")
    for arq in files:
        rel_path = os.path.relpath(arq, input_dir)
        output_path_single = os.path.join(output_dir, rel_path)
        print(f"  Processing {arq} -> {output_path_single}")
        try:
            process_and_save_image(arq, output_path_single, debug=debug)
        except Exception as e:
            print(f"    Error: {e}")

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Main label segmentation using integrated techniques.")
    parser.add_argument("-c", "--input", required=True,
                        help="Path to an image, directory, or 'all' to process the full dataset.")
    parser.add_argument("-o", "--output", default=OUTPUT_DIR,
                        help="Output directory (default: 'output')")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug information")
    args = parser.parse_args()

    input_arg = args.input
    output_base = args.output

    if input_arg.lower() == 'all':
        dataset_dir = 'dataset'
        if not os.path.isdir(dataset_dir):
            print(f"Directory 'dataset' not found in the current path.")
            sys.exit(1)
        process_directory(dataset_dir, output_base, debug=args.debug)
    elif os.path.isdir(input_arg):
        process_directory(input_arg, output_base, debug=args.debug)
    elif os.path.isfile(input_arg):
        output_path_single = os.path.join(output_base, os.path.basename(input_arg))
        os.makedirs(output_base, exist_ok=True)
        print(f"Processing {input_arg} -> {output_path_single}")
        process_and_save_image(input_arg, output_path_single, debug=args.debug)
    else:
        print(f"Error: '{input_arg}' is not a valid file or directory.")
        sys.exit(1)

    print("Processing complete.")

if __name__ == "__main__":
    main()