
# -*- coding: utf-8 -*-
"""
ocr_preprocess_v2.py — measurement-driven document pre-processing for Arabic OCR.

WHAT CHANGED vs v1 (and why)
----------------------------
v1 gated its stages on a source LABEL ("digital" vs "scan") and used pixel
constants tuned on one test image. On a 2220x3005 scanned court judgement that
produced strokes 50% too thick (stroke/height = 0.138 vs a healthy 0.08-0.10),
11.7% ink coverage, merged letters, and 79% of components left as noise.

v2 measures the page first and switches each stage on a measured property:

  * calibrate()          -> text height, stroke width, background noise sigma,
                            illumination range, ink/paper separation
  * every stage is enabled by one of those numbers, not by a label
  * binarization runs a FEEDBACK LOOP: it raises Sauvola's k until the measured
    stroke/height ratio lands in the healthy band. This is the fix for merged
    Arabic letters -- thinner strokes reopen the counters of ه ع ق م and the
    gaps between adjacent letters.
  * speckle removal is sized from the measured stroke width, plus a text-band
    test, instead of a fixed 28 px isolation radius that fails on dense pages
  * skew is estimated on a TEXT-ONLY mask (page frames, stamps and signatures
    are excluded) instead of on everything dark
  * stroke repair only runs when strokes are measurably too THIN
  * every morphological step is guarded: if it merges more than `merge_guard`
    of the components, it is reverted

Run  --calibrate  on your own documents first; it prints the measurements and
the decisions without writing anything.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXT = {".pdf"}

# Healthy band for stroke_width / text_height in printed Arabic & Latin.
RATIO_LO, RATIO_HI = 0.070, 0.115


# --------------------------------------------------------------------------- #
@dataclass
class Config:
    dpi: int = 300
    target_min_side: int = 1600
    max_upscale: float = 3.0
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    boost: float = 0.45          # 0 = never darken faint text, 1 = maximum recovery
    sauvola_k: float = 0.22      # STARTING k; the loop raises it if strokes are fat
    sauvola_k_max: float = 0.45
    deskew_limit: float = 6.0
    deskew_delta: float = 0.2
    merge_guard: float = 0.12    # revert a morph step if >12% of components merge
    remove_lines: bool = False
    drop_graphics: bool = False  # strip stamps / frames / signature scrawl
    dewarp: bool = False
    invert_if_dark: bool = True


@dataclass
class Result:
    name: str
    stages: dict = field(default_factory=dict)
    final: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def imread_unicode(path) -> np.ndarray | None:
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img) -> bool:
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if ok:
        buf.tofile(str(path))
    return ok


def pdf_pages(path: Path, dpi: int):
    try:
        import fitz
    except ImportError:
        sys.exit("PDF support needs PyMuPDF:  pip install pymupdf")
    out, doc = [], None
    doc = fitz.open(str(path))
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    for i, page in enumerate(doc):
        has_text = len(page.get_text().strip()) > 40
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR)
        out.append((f"{path.stem}_p{i+1:03d}", img, has_text))
    doc.close()
    return out


def collect_inputs(path: Path, cfg: Config):
    items = []
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT | PDF_EXT) \
        if path.is_dir() else [path]
    for f in files:
        e = f.suffix.lower()
        if e in PDF_EXT:
            items.extend(pdf_pages(f, cfg.dpi))
        elif e in IMAGE_EXT:
            img = imread_unicode(f)
            if img is None:
                print(f"  [skip] unreadable: {f}")
            else:
                items.append((f.stem, img, False))
    return items


# --------------------------------------------------------------------------- #
# CALIBRATION — everything downstream is driven by these numbers
# --------------------------------------------------------------------------- #
def component_stats(bw: np.ndarray):
    """bw: text=0, paper=255."""
    ink = (bw < 128).astype(np.uint8)
    n, lbl, st, cent = cv2.connectedComponentsWithStats(ink, 8)
    return ink, n, lbl, st, cent


def text_metrics(bw: np.ndarray) -> dict:
    """
    Robust text height + stroke width.
    Height uses the 75th percentile of LETTER-BODY components: in Arabic most
    components are dots/tashkeel, so the median under-estimates badly.
    Stroke width comes from the distance transform (2x median of the interior
    distance) — the single number that tells you if binarization fattened
    the glyphs.
    """
    H, W = bw.shape[:2]
    ink, n, lbl, st, _ = component_stats(bw)
    if n <= 2:
        return dict(text_h=max(12.0, H / 60), stroke_w=2.0, ratio=0.09, n_comp=0, ink=0.0)
    h = st[1:, cv2.CC_STAT_HEIGHT]
    w = st[1:, cv2.CC_STAT_WIDTH]
    a = st[1:, cv2.CC_STAT_AREA]
    sel = (h > 3) & (h < 0.08 * H) & (w < 0.5 * W) & (a > 12)
    text_h = float(np.percentile(h[sel], 75)) if sel.sum() > 20 else max(12.0, H / 60)

    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    d = dt[dt > 0]
    stroke_w = 2.0 * float(np.median(d)) if d.size else 2.0
    return dict(text_h=text_h, stroke_w=stroke_w, ratio=stroke_w / max(text_h, 1.0),
                n_comp=int(n - 1), ink=float(ink.mean()))


def calibrate(gray: np.ndarray) -> dict:
    """
    Physical measurements of the page. Each one switches a stage on or off.
    NOTE: v1's classifier asked "is the background near-white?" — true for every
    paper document, which is why a scanned court page was treated as a screenshot.
    Background NOISE SIGMA is the honest discriminator: a real scan carries
    sensor/paper grain (sigma > 3); a screenshot or vector PDF render does not.
    """
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = gray[gray > t]
    inkpx = gray[gray <= t]
    if paper.size < 100 or inkpx.size < 100:
        paper, inkpx = gray, gray

    # noise sigma measured on the paper only, after removing the low-frequency
    # illumination surface (otherwise a shadow inflates it)
    blur = cv2.GaussianBlur(gray, (0, 0), 9)
    resid = gray.astype(np.float32) - blur.astype(np.float32)
    mask = gray > t
    noise_sigma = float(resid[mask].std()) if mask.sum() > 500 else 0.0

    # illumination range: how much the paper brightness drifts across the page
    small = cv2.resize(gray, (48, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    illum_range = float(np.percentile(small, 95) - np.percentile(small, 25))

    ink_med = float(np.median(inkpx))
    paper_med = float(np.median(paper))
    separation = paper_med - ink_med

    return dict(otsu=float(t), ink_med=ink_med, paper_med=paper_med,
                separation=separation, noise_sigma=noise_sigma,
                illum_range=illum_range,
                is_bilevel=len(np.unique(gray)) <= 4)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def to_gray(b): return cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b


def maybe_invert(g):
    return cv2.bitwise_not(g) if np.median(g) < 110 else g


def upscale(gray, min_side, max_scale):
    h, w = gray.shape[:2]
    s = min(min_side / min(h, w), max_scale)
    if s <= 1.05:
        return gray, 1.0
    return cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), s


def flatten_background(gray, text_h):
    """Kernel sized from TEXT HEIGHT, not a constant — a 31px kernel that works
    at 1200px wide erases body text at 3000px wide."""
    k = int(max(15, text_h * 2.5)) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.GaussianBlur(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, ker), (0, 0), k / 3.0)
    return cv2.divide(gray, bg, scale=255).astype(np.uint8)


def denoise(gray, sigma):
    d = int(np.clip(sigma / 2, 1, 4)) * 2 + 1
    return cv2.bilateralFilter(gray, d, 35, 35)


def darken_faint_text(gray, cal, boost):
    """
    v1 bug: black point = 3rd percentile of the WHOLE image. On a page with 10%
    ink that lands deep inside real ink, so the stretch dragged every
    anti-aliased edge pixel to black and fattened every stroke.

    v2 anchors on Otsu: the black point is taken from the ink distribution, the
    white point from the paper distribution, and the amount of pull is scaled by
    how poor the separation actually is. A high-contrast page is left alone.
    """
    sep = cal["separation"]
    need = float(np.clip((110.0 - sep) / 110.0, 0.0, 1.0))
    if need < 0.05:
        return gray, 0.0                      # already high contrast: do nothing

    g = gray.astype(np.float32)
    inkpx = g[g <= cal["otsu"]]
    paper = g[g > cal["otsu"]]
    lo = float(np.percentile(inkpx, 15 + 35 * boost * need))
    hi = float(np.percentile(paper, 35))
    if hi - lo < 12:
        hi = lo + 12
    g = np.clip((g - lo) * (255.0 / (hi - lo)), 0, 255)

    gamma = 1.0 + 0.35 * boost * need         # far gentler than v1's flat 1.25
    inv = 255.0 * ((255.0 - g) / 255.0) ** (1.0 / gamma)
    return (255.0 - inv).astype(np.uint8), need


def apply_clahe(g, clip, grid):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(g)


def sauvola(gray, window, k, R=128.0):
    """Local adaptive threshold via box filters. text=0, paper=255."""
    window = max(15, int(window)) | 1
    g = gray.astype(np.float32)
    m = cv2.boxFilter(g, cv2.CV_32F, (window, window), normalize=True,
                      borderType=cv2.BORDER_REPLICATE)
    m2 = cv2.boxFilter(g * g, cv2.CV_32F, (window, window), normalize=True,
                       borderType=cv2.BORDER_REPLICATE)
    s = np.sqrt(np.maximum(m2 - m * m, 0))
    return np.where(g > m * (1.0 + k * (s / R - 1.0)), 255, 0).astype(np.uint8)


def binarize_to_target(gray, text_h, cfg, log):
    """
    THE FIX FOR MERGED LETTERS.
    Binarize, measure stroke/height, and raise k until the ratio enters the
    healthy band. Higher k = stricter threshold = thinner strokes = reopened
    letter counters. Falls back to the thinnest result if none qualifies.
    """
    window = max(15, int(text_h * 1.6))
    best, best_bw = None, None
    k = cfg.sauvola_k
    while k <= cfg.sauvola_k_max + 1e-9:
        bw = sauvola(gray, window, k)
        m = text_metrics(bw)
        log.append(f"    sauvola k={k:.2f} -> ratio={m['ratio']:.3f} ink={100*m['ink']:.1f}%")
        if m["ink"] < 0.005:                       # over-thresholded, text vanishing
            break
        if RATIO_LO <= m["ratio"] <= RATIO_HI:
            return bw, k, m
        if best is None or m["ratio"] < best["ratio"]:
            best, best_bw, best_k = m, bw, k
        if m["ratio"] < RATIO_LO:
            break
        k += 0.06
    return best_bw, best_k, best


def rotate_keep(img, angle, border=255):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    # INTER_NEAREST keeps a binary image binary. Cubic interpolation would create
    # grey ringing that the next global threshold turns into extra ink.
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def text_only_mask(bw, text_h):
    """
    Drop page frames, stamps, signatures and rules so they cannot dominate the
    skew estimate. v1 returned skew=0.0 on a visibly tilted page because the
    thick black border pinned the projection profile.
    """
    ink, n, lbl, st, _ = component_stats(bw)
    keep = np.zeros(n, bool)
    H, W = bw.shape[:2]
    for i in range(1, n):
        h, w = st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_WIDTH]
        if h > 3.5 * text_h or w > 0.55 * W or h > 0.2 * H:
            continue
        if st[i, cv2.CC_STAT_AREA] < 6:
            continue
        keep[i] = True
    return np.isin(lbl, np.where(keep)[0]).astype(np.uint8) * 255


def estimate_skew(text_mask, limit, delta):
    small = cv2.resize(text_mask, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
    best_a, best_s = 0.0, -1.0
    for a in np.arange(-limit, limit + delta, delta):
        rot = rotate_keep(small, float(a), border=0)
        p = rot.sum(axis=1, dtype=np.float64)
        s = float(((p[1:] - p[:-1]) ** 2).sum())
        if s > best_s:
            best_s, best_a = s, float(a)
    return best_a


def text_bands(bw, text_h):
    """Rows that belong to a text line — used to reject noise sitting in margins."""
    ink = (bw < 128).astype(np.uint8)
    prof = ink.sum(axis=1).astype(np.float32)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, int(text_h) | 1), 0).ravel()
    thr = max(prof.max() * 0.04, 1.0)
    return prof > thr


def remove_speckle(bw, m, log):
    """
    v1 kept a small blob if a big blob sat within 28 px. On a dense page that
    is always true, so nothing was removed (79% of components survived as noise).

    v2 uses two size-relative rules:
      * a genuine Arabic dot has area ~ stroke_w^2. Anything below ~35% of that
        cannot be a dot, whatever its neighbours — delete it.
      * a blob smaller than a dot that also sits outside every text band is
        margin noise — delete it.
    """
    sw = max(m["stroke_w"], 1.5)
    dot_area = sw * sw
    hard = max(2, int(0.35 * dot_area))
    soft = max(hard + 1, int(1.1 * dot_area))

    ink, n, lbl, st, cent = component_stats(bw)
    band = text_bands(bw, m["text_h"])
    out = ink.copy()
    killed = 0
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a > soft:
            continue
        cy = int(np.clip(cent[i][1], 0, len(band) - 1))
        if a <= hard or not band[cy]:
            out[lbl == i] = 0
            killed += 1
    log.append(f"    speckle: removed {killed}/{n-1} components "
               f"(hard<{hard}px, soft<{soft}px outside text bands)")
    return np.where(out > 0, 0, 255).astype(np.uint8)


def guarded_morph(bw, op, kernel, m_before, cfg, log, label):
    """Apply a morphological op; revert it if it merged too many components."""
    ink = 255 - bw
    res = cv2.morphologyEx(ink, op, kernel, iterations=1)
    out = 255 - res
    m_after = text_metrics(out)
    drop = 1.0 - m_after["n_comp"] / max(m_before["n_comp"], 1)
    if drop > cfg.merge_guard:
        log.append(f"    {label}: REVERTED ({100*drop:.0f}% of components merged)")
        return bw, m_before
    log.append(f"    {label}: applied ({100*drop:.0f}% merge)")
    return out, m_after


def repair_strokes_if_thin(bw, m, cfg, log):
    """Only close gaps when strokes are measurably TOO THIN. On the court
    document strokes were already 4 px — closing there is what welded the
    letters together."""
    if m["ratio"] >= RATIO_LO:
        log.append(f"    stroke repair: skipped (ratio {m['ratio']:.3f} already >= {RATIO_LO})")
        return bw, m
    return guarded_morph(bw, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8),
                         m, cfg, log, "stroke repair")


def remove_ruling_lines(bw, text_h, log):
    ink = 255 - bw
    h, w = bw.shape[:2]
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (int(max(30, text_h * 8)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(max(30, text_h * 8))))
    lines = cv2.add(cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk),
                    cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk))
    log.append(f"    ruling lines: {100*lines.mean()/255:.2f}% of page removed")
    return 255 - cv2.subtract(ink, lines)


def drop_graphics(bw, text_h, log):
    """Remove stamps, seals, frames and signature scrawl: components far larger
    than text in both dimensions."""
    ink, n, lbl, st, _ = component_stats(bw)
    out = ink.copy()
    killed = 0
    for i in range(1, n):
        h, w = st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_WIDTH]
        if h > 3.5 * text_h and w > 3.5 * text_h:
            out[lbl == i] = 0
            killed += 1
    log.append(f"    graphics: removed {killed} large components (stamps/frames)")
    return np.where(out > 0, 0, 255).astype(np.uint8)


def crop_to_content(bw, pad=12):
    ink = cv2.morphologyEx((bw < 128).astype(np.uint8), cv2.MORPH_OPEN,
                           np.ones((3, 3), np.uint8))
    ys, xs = np.where(ink > 0)
    if len(xs) < 50:
        return bw
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, bw.shape[1] - 1)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, bw.shape[0] - 1)
    return bw[y0:y1 + 1, x0:x1 + 1]


def auto_canny(gray, sigma=0.33):
    v = float(np.median(gray))
    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0),
                     int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v)))


def segment(bw, text_h):
    ink = 255 - bw
    th = max(int(text_h), 10)
    lk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, int(th * 2.4)), max(2, th // 5)))
    cnts, _ = cv2.findContours(cv2.dilate(ink, lk, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = [b for b in map(cv2.boundingRect, cnts) if b[2] > th and b[3] > th * 0.5]
    bk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, th * 3), max(4, int(th * 1.2))))
    cnts, _ = cv2.findContours(cv2.dilate(ink, bk, 2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = [b for b in map(cv2.boundingRect, cnts) if b[2] > th * 2 and b[3] > th]
    vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in blocks:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 120, 0), 3)
    for (x, y, w, h) in lines:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
    lines.sort(key=lambda b: (b[1] // max(th, 1), -b[0]))   # RTL reading order
    return vis, lines, blocks


# --------------------------------------------------------------------------- #
def process(name, bgr, cfg, has_text=False) -> Result:
    res = Result(name=name)
    S, L = res.stages, res.log

    S["01_original"] = bgr
    gray = to_gray(bgr)
    if cfg.invert_if_dark:
        gray = maybe_invert(gray)
    S["02_grayscale"] = gray

    cal = calibrate(gray)
    L.append(f"  calibration: separation={cal['separation']:.0f}  "
             f"noise_sigma={cal['noise_sigma']:.1f}  illum_range={cal['illum_range']:.0f}  "
             f"bilevel={cal['is_bilevel']}")

    # provisional metrics from a cheap Otsu pass, to size every kernel below
    _, otsu_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m0 = text_metrics(otsu_bw)
    L.append(f"  measured: text_h={m0['text_h']:.0f}px  stroke_w={m0['stroke_w']:.1f}px  "
             f"ratio={m0['ratio']:.3f}  ink={100*m0['ink']:.1f}%")
    res.meta["text_h"], res.meta["stroke_w0"], res.meta["ratio0"] = \
        round(m0["text_h"], 1), round(m0["stroke_w"], 2), round(m0["ratio"], 3)

    gray, scale = upscale(gray, cfg.target_min_side, cfg.max_upscale)
    if scale > 1.0:
        m0["text_h"] *= scale
        L.append(f"  upscaled x{scale:.2f}")
    S["03_upscaled"] = gray

    # --- stage decisions, each from a measurement ---------------------------
    if cal["illum_range"] > 22 and not cal["is_bilevel"]:
        gray = flatten_background(gray, m0["text_h"])
        S["04_bg_flattened"] = gray
        L.append(f"  bg flatten: ON (illum_range {cal['illum_range']:.0f} > 22)")
    else:
        L.append(f"  bg flatten: off (illum_range {cal['illum_range']:.0f})")

    if cal["noise_sigma"] > 4.0 and not cal["is_bilevel"]:
        gray = denoise(gray, cal["noise_sigma"])
        S["05_denoised"] = gray
        L.append(f"  denoise: ON (noise_sigma {cal['noise_sigma']:.1f} > 4)")
    else:
        L.append(f"  denoise: off (noise_sigma {cal['noise_sigma']:.1f})")

    if cal["separation"] < 110 and not cal["is_bilevel"]:
        gray = apply_clahe(gray, cfg.clahe_clip, cfg.clahe_grid)
        S["06_clahe"] = gray
        L.append(f"  clahe: ON (separation {cal['separation']:.0f} < 110)")
    else:
        L.append(f"  clahe: off (separation {cal['separation']:.0f})")

    cal2 = calibrate(gray)
    gray, need = darken_faint_text(gray, cal2, cfg.boost)
    S["07_faint_text_darkened"] = gray
    L.append(f"  faint-text boost: strength {need:.2f} "
             f"({'applied' if need > 0.05 else 'skipped, contrast already good'})")

    # NOTE: no unsharp before binarization. In v1 it added a dark halo that
    # Sauvola then converted into extra stroke width.
    S["08_edges_canny"] = auto_canny(gray)     # geometry/debug only, never OCR input

    bw, k_used, m = binarize_to_target(gray, m0["text_h"], cfg, L)
    S["09_binarized"] = bw
    res.meta["sauvola_k"] = round(k_used, 2)
    res.meta["ratio_after_binarize"] = round(m["ratio"], 3)
    L.append(f"  binarize: k={k_used:.2f}  ratio={m['ratio']:.3f}  ink={100*m['ink']:.1f}%")

    tmask = text_only_mask(bw, m["text_h"])
    S["10_text_only_mask"] = tmask
    angle = estimate_skew(tmask, cfg.deskew_limit, cfg.deskew_delta)
    res.meta["skew_deg"] = round(angle, 2)
    if abs(angle) > 0.15:
        bw = rotate_keep(bw, angle, 255)
    S["11_deskewed"] = bw
    L.append(f"  skew: {angle:+.2f} deg (measured on text-only mask)")

    bw = remove_speckle(bw, m, L)
    S["12_speckle_removed"] = bw

    if cfg.drop_graphics:
        bw = drop_graphics(bw, m["text_h"], L)
        S["13_graphics_removed"] = bw
    if cfg.remove_lines:
        bw = remove_ruling_lines(bw, m["text_h"], L)
        S["14_lines_removed"] = bw

    m = text_metrics(bw)
    bw, m = repair_strokes_if_thin(bw, m, cfg, L)
    S["15_stroke_repair"] = bw

    bw = crop_to_content(bw)
    S["16_cropped"] = bw

    mf = text_metrics(bw)
    vis, lines, blocks = segment(bw, mf["text_h"])
    S["17_segmentation"] = vis
    res.meta.update(lines=len(lines), blocks=len(blocks),
                    final_ratio=round(mf["ratio"], 3),
                    final_ink=round(100 * mf["ink"], 2),
                    final_comps=mf["n_comp"])
    res.final = bw
    S["18_FINAL_for_OCR"] = bw
    L.append(f"  FINAL: ratio={mf['ratio']:.3f} ink={100*mf['ink']:.1f}% "
             f"components={mf['n_comp']} lines={len(lines)}")
    return res


# --------------------------------------------------------------------------- #
def contact_sheet(res, out_path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = list(res.stages.items())
    cols = 4
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 5.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (t, im) in zip(axes, items):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im,
                  cmap=None if im.ndim == 3 else "gray")
        ax.set_title(t.replace("_", " "), fontsize=10)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    mt = res.meta
    fig.suptitle(f"{res.name}  |  k={mt.get('sauvola_k')}  "
                 f"stroke/height {mt.get('ratio0')} -> {mt.get('final_ratio')}  "
                 f"ink {mt.get('final_ink')}%  skew {mt.get('skew_deg')}deg", fontsize=13)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Measurement-driven OCR pre-processing (Arabic)")
    ap.add_argument("input")
    ap.add_argument("--outdir", default="preprocessed")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--calibrate", action="store_true",
                    help="measure and print decisions only, write nothing")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-stages", action="store_true")
    ap.add_argument("--boost", type=float, default=0.45,
                    help="faint-text recovery 0..1; lower if strokes come out fat")
    ap.add_argument("--sauvola-k", type=float, default=0.22, help="starting k")
    ap.add_argument("--sauvola-k-max", type=float, default=0.45)
    ap.add_argument("--remove-lines", action="store_true")
    ap.add_argument("--drop-graphics", action="store_true",
                    help="strip stamps, seals, frames, signatures")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--lang", default="ara+eng")
    a = ap.parse_args()

    cfg = Config(dpi=a.dpi, boost=a.boost, sauvola_k=a.sauvola_k,
                 sauvola_k_max=a.sauvola_k_max, remove_lines=a.remove_lines,
                 drop_graphics=a.drop_graphics)

    src = Path(a.input)
    if not src.exists():
        sys.exit(f"not found: {src}")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for name, img, has_text in collect_inputs(src, cfg) or []:
        print(f"\n== {name}  {img.shape[1]}x{img.shape[0]}")
        if has_text:
            print("   note: PDF page already has a text layer — extract it instead of OCR.")
        res = process(name, img, cfg, has_text)
        print("\n".join(res.log))
        if a.calibrate:
            continue
        imwrite_unicode(out / f"{name}_final.png", res.final)
        if a.save_stages:
            sd = out / f"{name}_stages"
            sd.mkdir(exist_ok=True)
            for kk, vv in res.stages.items():
                imwrite_unicode(sd / f"{kk}.png", vv)
        contact_sheet(res, out / f"{name}_stages.png", a.show)
        if a.ocr:
            try:
                import pytesseract
                txt = pytesseract.image_to_string(
                    res.final, config=f"--oem 1 --psm 6 -l {a.lang} -c preserve_interword_spaces=1")
                (out / f"{name}.txt").write_text(txt, encoding="utf-8")
                print(f"   ocr chars: {len(txt.strip())}")
            except ImportError:
                print("   [pytesseract not installed]")
    if not a.calibrate:
        print(f"\ndone -> {out.resolve()}")


if __name__ == "__main__":
    main()


