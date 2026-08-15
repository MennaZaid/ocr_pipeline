# -*- coding: utf-8 -*-
"""
preprocessors/ain_light.py — step 2a (AIN path): deskew + crop only.

No binarization, no denoising, no morphology, ever. AIN appeared to read a
heavily preprocessed (volume 5) page WORSE than the same page untouched, so
this path deliberately does the minimum: straighten the page and crop to
content, using the same measurement approach as the Qwen-side volumes, but
applied to the original grey/colour pixels rather than a binarized copy.

A temporary Otsu binarization is used INTERNALLY only, to measure skew angle
and content bounds. It is never written out and never touches the pixels
delivered to AIN.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SHARED_MODULE_DIR  # noqa: E402
sys.path.insert(0, str(SHARED_MODULE_DIR))

from ocr_preprocess_v2 import (  # noqa: E402
    to_gray, maybe_invert, upscale, maximize_contrast,
    text_metrics, text_only_mask, estimate_skew,
)


@dataclass
class AinConfig:
    keep_color: bool = True
    deskew_limit: float = 6.0
    deskew_delta: float = 0.2
    crop_pad: int = 16
    target_min_side: int = 1024
    max_upscale: float = 2.0
    light_contrast: bool = False   # opt-in; OFF by default, see module docstring
    invert_if_dark: bool = True


@dataclass
class AinResult:
    final: np.ndarray
    meta: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


def rotate_continuous(img: np.ndarray, angle: float) -> np.ndarray:
    """Cubic interpolation, paper-coloured border. Never applied to a binary
    image — that's what the volume scripts' own rotate_keep() is for."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    border = 255 if img.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _measure_bw(gray: np.ndarray) -> np.ndarray:
    """Scaffold-only binarization for skew/crop measurement. Never delivered."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _content_bounds(bw: np.ndarray, pad: int):
    ink = cv2.morphologyEx((bw < 128).astype(np.uint8), cv2.MORPH_OPEN,
                           np.ones((3, 3), np.uint8))
    ys, xs = np.where(ink > 0)
    if len(xs) < 50:
        return None
    H, W = bw.shape[:2]
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, W - 1)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, H - 1)
    return x0, x1, y0, y1


def preprocess_for_ain(bgr: np.ndarray, cfg: AinConfig | None = None) -> AinResult:
    cfg = cfg or AinConfig()
    L: list[str] = []

    gray0 = to_gray(bgr)
    if cfg.invert_if_dark:
        gray0 = maybe_invert(gray0)

    out = bgr if cfg.keep_color else gray0
    out, scale = upscale(out, cfg.target_min_side, cfg.max_upscale)
    if scale > 1.0:
        L.append(f"upscaled x{scale:.2f}")
    gray = to_gray(out) if out.ndim == 3 else out

    bw = _measure_bw(gray)
    m = text_metrics(bw)
    L.append(f"measured text_h={m['text_h']:.0f}px stroke_w={m['stroke_w']:.1f}px (scaffold only)")

    tmask = text_only_mask(bw, m["text_h"])
    angle = estimate_skew(tmask, cfg.deskew_limit, cfg.deskew_delta)
    meta = {"skew_deg": round(angle, 2)}
    if abs(angle) > 0.15:
        out = rotate_continuous(out, angle)
        gray = rotate_continuous(gray, angle)
        L.append(f"deskew {angle:+.2f} deg")
    else:
        L.append(f"deskew {angle:+.2f} deg (below threshold, skipped)")

    bw2 = _measure_bw(gray)
    bounds = _content_bounds(bw2, cfg.crop_pad)
    if bounds:
        x0, x1, y0, y1 = bounds
        out = out[y0:y1 + 1, x0:x1 + 1]
        L.append(f"cropped to {x1 - x0}x{y1 - y0}px (pad={cfg.crop_pad})")
    else:
        L.append("crop skipped (not enough content detected)")

    if cfg.light_contrast:
        g = to_gray(out) if out.ndim == 3 else out
        lo, hi = np.percentile(g, 0.5), np.percentile(g, 99.5)
        if hi - lo >= 10:
            if out.ndim == 3:
                out = np.clip((out.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                              0, 255).astype(np.uint8)
            else:
                out = maximize_contrast(g, L)
            L.append(f"light contrast applied [{lo:.0f},{hi:.0f}]->[0,255]")
        else:
            L.append("light contrast skipped (already high contrast)")

    return AinResult(final=out, meta=meta, log=L)
