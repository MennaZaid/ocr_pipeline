# -*- coding: utf-8 -*-
"""
pipeline_selector.py — step 2b (Qwen path only): decide which volume
(1/2/3/5) a given page should go through, based on measured image quality.

Pure functions only: given a grayscale page, return a routing decision.
Running the chosen volume script is a separate concern, handled in
run_pipeline.py, so this module can be unit-tested or recalibrated without
touching subprocess/IO code.

Thresholds below are UNCHANGED from the version already validated in earlier
conversation — they were not re-derived here since doing so without labeled
outcomes would just swap one unvalidated guess for another. run_pipeline.py
logs every routing decision to routing_report.jsonl specifically so these can
be checked against real documents and tuned deliberately.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import VOLUME_SCRIPTS


def estimate_quality(gray: np.ndarray) -> dict[str, float]:
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper_mask = gray > otsu
    ink_mask = ~paper_mask

    blur = cv2.GaussianBlur(gray, (0, 0), 9)
    resid = gray.astype(np.float32) - blur.astype(np.float32)
    noise_sigma = float(resid[paper_mask].std()) if paper_mask.sum() > 500 else 0.0

    small = cv2.resize(gray, (48, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    illum_range = float(np.percentile(small, 95) - np.percentile(small, 25))

    paper_med = float(np.median(gray[paper_mask])) if paper_mask.any() else float(np.median(gray))
    ink_med = float(np.median(gray[ink_mask])) if ink_mask.any() else float(np.median(gray))
    separation = paper_med - ink_med

    drop = cv2.GaussianBlur(gray, (0, 0), 3).astype(np.int16) - gray.astype(np.int16)
    faint_ratio = float((drop >= 6).mean())
    very_faint_ratio = float(((drop >= 4) & (drop < 10)).mean())

    edges = cv2.Canny(gray, 60, 160)
    edge_density = float((edges > 0).mean())
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_islands = cv2.morphologyEx((edges > 0).astype(np.uint8), cv2.MORPH_OPEN, thin_kernel)
    fragmented_edge_ratio = float((edge_islands > 0).mean())

    return {
        "noise_sigma": noise_sigma,
        "illum_range": illum_range,
        "separation": separation,
        "faint_ratio": faint_ratio,
        "very_faint_ratio": very_faint_ratio,
        "edge_density": edge_density,
        "fragmented_edge_ratio": fragmented_edge_ratio,
    }


def choose_pipeline(metrics: dict[str, float]) -> tuple[str, str]:
    """Returns (volume_key, human-readable reason)."""
    noise_sigma = metrics["noise_sigma"]
    illum_range = metrics["illum_range"]
    separation = metrics["separation"]
    faint_ratio = metrics["faint_ratio"]
    very_faint_ratio = metrics["very_faint_ratio"]
    fragmented_edge_ratio = metrics["fragmented_edge_ratio"]

    structural_damage = (
        very_faint_ratio > 0.07
        or fragmented_edge_ratio > 0.065
        or illum_range > 34
    )
    severe_core_risk = (noise_sigma > 5.8) or (separation < 72)
    stable_geometry = (
        fragmented_edge_ratio < 0.012
        and very_faint_ratio < 0.020
        and illum_range < 16
    )

    if (
        (noise_sigma > 6.8 and faint_ratio > 0.042 and separation < 90)
        or (noise_sigma > 7.2 and separation < 95)
    ):
        return "volume5", "Detected noisy low-contrast page with likely stroke loss risk."
    if structural_damage and (severe_core_risk or not stable_geometry):
        return "volume5", "Detected severe structural degradation (faint/broken strokes or non-uniformity)."
    if noise_sigma > 4.2 or separation < 95 or illum_range > 20:
        return "volume3", "Detected medium-quality page; balanced pipeline to avoid over-processing."
    if separation >= 125 and noise_sigma <= 2.0 and illum_range <= 12:
        return "volume1", "Detected very clean/high-contrast page."
    if separation >= 100 and noise_sigma <= 3.6 and illum_range <= 22:
        return "volume2", "Detected good quality page with mild artifacts."
    return "volume3", "Detected moderate-quality page; balanced pipeline."


def volume_script_path(volume_key: str):
    path = VOLUME_SCRIPTS[volume_key]
    if not path.exists():
        raise FileNotFoundError(
            f"Volume script for '{volume_key}' not found at {path}. "
            f"Check VOLUME_SCRIPTS in config.py against your actual folder layout."
        )
    return path
