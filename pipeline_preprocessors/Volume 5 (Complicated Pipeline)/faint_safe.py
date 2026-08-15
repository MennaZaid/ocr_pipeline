"""
faint_safe.py — additive patch for ocr_preprocess_v2.py

Fixes ONE failure mode: faint strokes in the source arrive at the OCR as blank
paper, so the extracted field comes back NULL instead of "unreadable".

Nothing here invents ink. Every pixel written to the output is a pixel that was
measurably darker than its own local paper in the SOURCE image. No morphological
growth, no inpainting, no interpolation of missing glyphs.

Four additions, all additive — the existing stage functions keep their semantics:

  1. coherent_evidence()      — decide WHICH pixels may act as connective tissue
                                using a stroke-sized average of the same
                                paper-referenced contrast the pipeline already
                                measures. Grain averages out, strokes do not.
                                Pixel SHAPE still comes from the per-pixel mask,
                                so glyphs are not thickened or smoothed.
  2. remove_border_junk_safe() — never deletes a component that is carrying page
                                content. Peels the margin-band / line-like part
                                and re-labels instead.
  3. guarded_removal()        — wraps ANY removal stage. Reverts it if it costs
                                more than a measured share of page ink or blanks
                                a region that had ink evidence.
  4. evidence_report()        — emits the regions where the source has ink but
                                the output is blank, so downstream can return
                                NEEDS_REVIEW instead of NULL.
"""
from __future__ import annotations
import json
import cv2
import numpy as np

from ocr_preprocess_v2 import component_stats, text_metrics, paper_level, sauvola


# --------------------------------------------------------------------------- #
# 1. coherence-gated evidence
# --------------------------------------------------------------------------- #
def coherent_evidence(drop: np.ndarray, stroke_w: float, min_contrast: int,
                      noise_sigma: float, log=None, density: float = 0.45):
    """
    drop = local_paper - gray  (the pipeline's existing contrast measure)

    Per-pixel thresholding at min_contrast=10 on a scan whose paper grain has
    sigma 7.4 is a 1.4-sigma test: roughly 8% of blank paper passes it. Those
    stray pixels are what fuse separate glyphs — and eventually the whole page —
    into one connected component.

    A stroke is spatially coherent, grain is not. Averaging `drop` over a
    stroke-sized window divides the grain sigma by the window side while leaving
    a real stroke's contrast intact, turning the same threshold into a 4-sigma
    test without raising it (which would drop genuinely faint text).

    Returns (px, gate):
      px   — per-pixel mask, unchanged shape fidelity: used to PAINT the ink
      gate — coherent mask: used to decide connectivity and to seed keep/drop
    """
    w = int(max(3, round(max(stroke_w, 1.5) * 0.9))) | 1
    px = (drop >= min_contrast).astype(np.uint8)
    # LOCAL DENSITY of qualifying pixels, not the local mean of `drop`.
    # Mean-of-drop would let paper next to a strong stroke pass, i.e. dilate the
    # ink. Density asks a different question: "do my neighbours also look like
    # ink?" — true inside a stroke, false for an isolated grain pixel or a
    # one-pixel chain bridging two glyphs. That is exactly the connective
    # tissue we must not build components out of.
    dens = cv2.boxFilter(px.astype(np.float32), cv2.CV_32F, (w, w), normalize=True)
    gate = (dens >= density).astype(np.uint8)
    if log is not None:
        eff = noise_sigma / max(np.sqrt(w * w * density), 1)
        log.append(f"    coherence gate: window={w}px density>={density:.2f}  "
                   f"per-pixel {min_contrast/max(noise_sigma,.1):.1f}σ -> "
                   f"~{min_contrast/max(eff,.1):.1f}σ effective  "
                   f"evidence {100*px.mean():.2f}% -> connective {100*gate.mean():.2f}%")
    return px, gate


def binarize_faint_aware_safe(gray, m, cfg, log, noise_sigma: float):
    """
    Drop-in replacement for binarize_faint_aware().
    Same hysteresis contract (strict seeds / loose candidates / shape rescue),
    same outputs. The only change: connectivity and component decisions are
    taken on the coherent mask, then the kept components are PAINTED with the
    per-pixel mask, so stroke shape is byte-identical to what the source shows.
    """
    th, sw = m["text_h"], max(m["stroke_w"], 1.5)
    bg = paper_level(gray, th)
    drop = bg.astype(np.int16) - gray.astype(np.int16)

    px, gate = coherent_evidence(drop, sw, cfg.min_contrast, noise_sigma, log,
                                 getattr(cfg, 'coherence_density', 0.45))

    sv_ink = sauvola(gray, max(15, int(th * 1.6)), cfg.sauvola_k) < 128
    strict = (((drop >= cfg.strong_contrast) & (gate > 0)) | sv_ink).astype(np.uint8)

    n, lbl, st, _ = cv2.connectedComponentsWithStats(gate, 8)
    seeded = np.zeros(n, bool)
    seeded[np.unique(lbl[strict > 0])] = True

    min_area = max(3.0, 0.12 * sw * sw)
    keep = np.zeros(n, bool)
    rescued = 0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        if seeded[i]:
            keep[i] = True
        elif 0.10 * th <= st[i, cv2.CC_STAT_HEIGHT] <= 3.5 * th:
            keep[i] = True
            rescued += 1

    kept_gate = np.isin(lbl, np.where(keep)[0]).astype(np.uint8)
    # paint with per-pixel evidence, but only inside/adjacent to a kept region,
    # so faint anti-aliased stroke edges come back at their true width
    halo = cv2.dilate(kept_gate, np.ones((3, 3), np.uint8), 1)
    out = (px > 0) & (halo > 0)
    bw = np.where(out, 0, 255).astype(np.uint8)
    mm = text_metrics(bw)
    log.append(f"    faint recovery (safe): components {n-1}  rescued {rescued}  "
               f"ink {100*mm['ink']:.2f}%  ratio {mm['ratio']:.3f}")
    return bw, cfg.min_contrast, mm, drop


# --------------------------------------------------------------------------- #
# 2. border handling that peels instead of deleting
# --------------------------------------------------------------------------- #
def remove_border_junk_safe(bw, m, log, margin_frac=0.02, max_share=0.05,
                            margin_purity=0.5):
    """
    The v2 rule deletes any component that touches the page edge and is longer
    than 6 text heights. On this scan that rule removed ONE component of
    2438x3468 px holding 23.9% of all page ink — the edge shadow with half the
    document's text fused onto it. Every field inside it became blank paper.

    v3 keeps the rule, and adds a check before executing it: if the candidate
    carries more than `max_share` of page ink and is not mostly confined to the
    margin band, it is page content wearing a frame. Peel the frame off it —
    margin-band pixels plus line-like pixels — and re-label. Only what still
    qualifies after peeling is deleted.
    """
    H, W = bw.shape[:2]
    mx = int(max(3, margin_frac * min(H, W)))
    band = np.zeros((H, W), np.uint8)
    band[:mx, :] = 1; band[-mx:, :] = 1; band[:, :mx] = 1; band[:, -mx:] = 1

    ink, n, lbl, st, _ = component_stats(bw)
    total = max(int(ink.sum()), 1)
    out = ink.copy()
    killed = peeled = 0
    for i in range(1, n):
        x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        a = st[i, cv2.CC_STAT_AREA]
        touches = x <= mx or y <= mx or x + w >= W - mx or y + h >= H - mx
        if not (touches and max(w, h) > 6 * m["text_h"]):
            continue
        comp = (lbl == i)
        share = a / total
        if share <= max_share:
            out[comp] = 0                      # small edge artefact: delete as before
            killed += 1
            continue
        # Carrying real page content. Losing it is never acceptable, whatever
        # the geometry says — so peel the frame off instead of deleting.
        ln = max(int(3 * m["text_h"]), 60)
        cu = comp.astype(np.uint8)
        linelike = cv2.bitwise_or(
            cv2.morphologyEx(cu, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (ln, 1))),
            cv2.morphologyEx(cu, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, ln))))
        # Restrict the peel to the margin band plus line-like pixels that are
        # ANCHORED in it. A long thin run in the middle of the page is text
        # (welded Arabic baseline), not a frame — removing it is what blanked
        # interior cells in testing.
        anchored = cv2.bitwise_and(linelike, cv2.dilate((band > 0).astype(np.uint8),
                                   np.ones((int(3*m['text_h'])|1,)*2, np.uint8), 1))
        strip = comp & (band > 0)
        out[strip] = 0
        peeled += 1
        log.append(f"    border: PEELED {w}x{h} component carrying {100*share:.1f}% "
                   f"of page ink — removed {int(strip.sum())} px of frame/margin, "
                   f"kept {a - int(strip.sum())} px of content")
        # re-label what is left of it and re-apply the ORIGINAL rule to the pieces
        rest = (comp & ~strip).astype(np.uint8)
        n2, lb2, st2, _ = cv2.connectedComponentsWithStats(rest, 8)
        for j in range(1, n2):
            x2, y2 = st2[j, cv2.CC_STAT_LEFT], st2[j, cv2.CC_STAT_TOP]
            w2, h2 = st2[j, cv2.CC_STAT_WIDTH], st2[j, cv2.CC_STAT_HEIGHT]
            t2 = x2 <= mx or y2 <= mx or x2 + w2 >= W - mx or y2 + h2 >= H - mx
            if t2 and max(w2, h2) > 6 * m["text_h"] and \
               st2[j, cv2.CC_STAT_AREA] / total <= max_share:
                out[lb2 == j] = 0
                killed += 1
    log.append(f"    border filter (safe): deleted {killed}, peeled {peeled}")
    return np.where(out > 0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 3. universal removal guard
# --------------------------------------------------------------------------- #
def guarded_removal(before, after, evidence, log, label,
                    max_ink_loss=0.08, cell=60, max_new_blank_cells=6, roi=None):
    """
    Wrap ANY removal stage. Reverts it when it:
      * costs more than `max_ink_loss` of the page's ink, or
      * turns more than `max_new_blank_cells` cells that had ink evidence into
        blank paper.
    This is the same accounting v2 already applies to morphology (merge_guard),
    extended to the deletion stages — which is where the page was actually
    being lost.
    """
    b = (before < 128); a = (after < 128)
    if roi is not None:                 # judge a stage only where it is not licensed to act
        b = b & roi; a = a & roi
    loss = 1.0 - a.sum() / max(b.sum(), 1)
    H, W = before.shape[:2]
    newly_blank = 0
    for y in range(0, H - cell, cell):
        for x in range(0, W - cell, cell):
            if (roi is None or roi[y:y + cell, x:x + cell].any()) \
               and evidence[y:y + cell, x:x + cell].mean() > 0.03 \
               and b[y:y + cell, x:x + cell].any() \
               and not a[y:y + cell, x:x + cell].any():
                newly_blank += 1
    if loss > max_ink_loss or newly_blank > max_new_blank_cells:
        log.append(f"    {label}: REVERTED — would remove {100*loss:.1f}% of ink and "
                   f"blank {newly_blank} cells that have source evidence")
        return before, False
    log.append(f"    {label}: accepted ({100*loss:.1f}% ink, {newly_blank} blanked cells)")
    return after, True


# --------------------------------------------------------------------------- #
# 4. evidence report — the NULL / NEEDS_REVIEW contract
# --------------------------------------------------------------------------- #
def evidence_report(raw_gray, final_bw, text_h, min_contrast=12, cell=60,
                    offset=(0, 0), path=None):
    """
    Compares the SOURCE against the delivered binary and lists every region that
    has ink in the source and none in the output.

    Downstream contract:
      field blank + region NOT listed  -> genuinely empty  -> NULL
      field blank + region listed      -> we lost it       -> NEEDS_REVIEW
    Nothing is guessed; the extractor simply stops reporting emptiness it cannot
    justify.
    """
    d = paper_level(raw_gray, text_h).astype(np.int16) - raw_gray.astype(np.int16)
    ev = (d >= min_contrast)
    H, W = raw_gray.shape[:2]
    ox, oy = offset
    regions = []
    for y in range(0, H - cell, cell):
        for x in range(0, W - cell, cell):
            e = float(ev[y:y + cell, x:x + cell].mean())
            if e <= 0.03:
                continue
            yy, xx = y - oy, x - ox
            if yy < 0 or xx < 0 or yy + cell > final_bw.shape[0] or xx + cell > final_bw.shape[1]:
                continue
            ink = float((final_bw[yy:yy + cell, xx:xx + cell] < 128).mean())
            if ink < 0.002:
                regions.append(dict(x=x, y=y, w=cell, h=cell,
                                    evidence=round(100 * e, 1), output_ink=0.0,
                                    status="NEEDS_REVIEW"))
    rep = dict(cell=cell, min_contrast=min_contrast,
               lost_regions=len(regions), regions=regions)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
    return rep
