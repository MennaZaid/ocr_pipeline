#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_safe.py — run ocr_preprocess_v2.py with the faint-safe patch applied.

No edits to ocr_preprocess_v2.py. This module imports it, replaces three
functions at runtime, then hands control to its own main(). Every flag you
already use still works:

    python run_safe.py scan.pdf --outdir out --save-stages
    python run_safe.py scan.pdf --calibrate
    python run_safe.py folder_of_scans/ --outdir out --ocr

Extra flag:
    --legacy      run the original pipeline unpatched (for A/B)

Extra output per page:
    <name>_evidence.json   regions with ink in the source and none in the output.
                           Your extractor must return NEEDS_REVIEW for those
                           boxes, never NULL.
"""
import sys
import cv2
import numpy as np

import ocr_preprocess_v2 as v2
import faint_safe as fs

LEGACY = "--legacy" in sys.argv
if LEGACY:
    sys.argv.remove("--legacy")

_S = {"drop": None, "sigma": 8.0}


def _report(ev, bw, crop, final, cell=60, path=None):
    """
    Cell audit in the frame the evidence map and the cleaned binary share
    (post-deskew, pre-crop), then shifted into FINAL image coordinates so the
    boxes line up with what the OCR actually receives.
    """
    import json
    ox, oy = crop
    H, W = bw.shape[:2]
    fh, fw = (final.shape[:2] if final is not None else (H, W))
    ink = bw < 128
    regions = []
    for y in range(0, H - cell, cell):
        for x in range(0, W - cell, cell):
            if ev[y:y + cell, x:x + cell].mean() <= 0.03:
                continue
            if ink[y:y + cell, x:x + cell].any():
                continue
            fx, fy = x - ox, y - oy
            if 0 <= fx <= fw - cell and 0 <= fy <= fh - cell:
                regions.append(dict(x=fx, y=fy, w=cell, h=cell, status="NEEDS_REVIEW"))
    rep = dict(frame="final_black_on_white", cell=cell,
               lost_regions=len(regions), regions=regions)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
    return rep


def _patch():
    # ---- capture the measured noise sigma ---------------------------------
    _cal = v2.calibrate

    def calibrate(gray):
        c = _cal(gray)
        _S["sigma"] = c["noise_sigma"]
        return c
    v2.calibrate = calibrate

    # ---- coherence-gated binarization -------------------------------------
    def binarize_faint_aware(gray, m, cfg, log):
        bw, k, mm, drop = fs.binarize_faint_aware_safe(gray, m, cfg, log, _S["sigma"])
        _S["drop"] = drop
        return bw, k, mm
    v2.binarize_faint_aware = binarize_faint_aware

    # ---- keep the evidence map aligned through deskew ----------------------
    _rot = v2.rotate_keep

    def rotate_keep(img, angle, border=255):
        out = _rot(img, angle, border)
        d = _S.get("drop")
        if d is not None and img.shape[:2] == d.shape[:2] and abs(angle) > 1e-6:
            _S["drop"] = _rot(d.astype(np.float32), angle, 0).astype(np.int16)
        return out
    v2.rotate_keep = rotate_keep

    # ---- guarded removal stages -------------------------------------------
    _clean = v2.clean_binary

    def clean_binary(bw, m, cfg, log):
        d = _S.get("drop")
        if d is None or d.shape[:2] != bw.shape[:2]:
            log.append("  noise filtering: (unguarded — no aligned evidence map)")
            return _clean(bw, m, cfg, log)
        ev = (d >= 12)
        H, W = bw.shape[:2]
        mx = int(max(3, 0.02 * min(H, W)))
        interior = np.zeros((H, W), bool)
        interior[mx:-mx, mx:-mx] = True
        cap = getattr(cfg, "guard_ink_loss", 0.08)
        log.append("  noise filtering (guarded):")
        for fn, tag in ((v2.remove_speckle, "speckle"),
                        (v2.filter_isolated, "isolation"),
                        (v2.remove_streaks, "streaks")):
            prev = bw
            bw, _ = fs.guarded_removal(prev, fn(prev, m, log), ev, log, tag, max_ink_loss=cap)
        if cfg.remove_border_junk:
            prev = bw
            bw, _ = fs.guarded_removal(prev, fs.remove_border_junk_safe(prev, m, log),
                                       ev, log, "border", max_ink_loss=cap, roi=interior)
        _S["ev"] = ev
        _S["post_clean"] = bw
        return bw
    v2.clean_binary = clean_binary

    # ---- record the crop offset so report boxes are in FINAL image coords --
    _crop = v2.crop_to_content

    def crop_to_content(bw, pad=12):
        out = _crop(bw, pad)
        ink = cv2.morphologyEx((bw < 128).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ys, xs = np.where(ink > 0)
        _S["crop"] = (max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0)) if len(xs) >= 50 else (0, 0)
        return out
    v2.crop_to_content = crop_to_content

    # ---- evidence report per page -----------------------------------------
    _proc = v2.process

    def process(name, bgr, cfg, has_text=False):
        _S["drop"] = None
        res = _proc(name, bgr, cfg, has_text)
        ev, bw = _S.get("ev"), _S.get("post_clean")
        if ev is not None and bw is not None and ev.shape[:2] == bw.shape[:2]:
            rep = _report(ev, bw, _S.get("crop", (0, 0)), res.final,
                          path=f"{name}_evidence.json")
            res.meta["needs_review_regions"] = rep["lost_regions"]
            res.log.append(f"  evidence report: {rep['lost_regions']} region(s) have ink in the "
                           f"source and none in the output -> NEEDS_REVIEW, not NULL")
        return res
    v2.process = process


if not LEGACY:
    _patch()
    print("[run_safe] faint-safe patch active"
          " (coherence gate, peeling border, removal guards, evidence report)")
else:
    print("[run_safe] --legacy: running the original pipeline unpatched")

v2.main()
