# -*- coding: utf-8 -*-
"""
pdf_to_images.py — step 1: turn a PDF into per-page images.

This is the only place PyMuPDF (fitz) is touched. Every other module in this
repo receives images, not PDFs, so this is the single seam if the rendering
approach ever needs to change.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXT = {".pdf"}


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def pdf_to_page_images(pdf_path: Path, dpi: int = 300):
    """
    Returns [(page_number, bgr_image, has_native_text), ...], page_number
    starting at 1.

    has_native_text flags PDF pages that already carry a real text layer
    (get_text() > 40 chars) — those are born-digital and should be routed to
    direct text extraction instead of OCR, upstream of this repo's two paths.
    """
    try:
        import fitz  # type: ignore
    except ImportError as e:
        raise RuntimeError("PDF input needs PyMuPDF: pip install pymupdf") from e

    out = []
    doc = fitz.open(str(pdf_path))
    try:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        for i, page in enumerate(doc):
            has_text = len(page.get_text().strip()) > 40
            pix = page.get_pixmap(matrix=mat, alpha=False)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR)
            out.append((i + 1, img, has_text))
    finally:
        doc.close()
    return out


def load_as_pages(path: Path, dpi: int = 300):
    """
    Uniform entry point: PDFs are split into pages, a single image file is
    treated as one page. Downstream code (run_pipeline.py) never needs to
    know which kind of input it started from.
    """
    ext = path.suffix.lower()
    if ext in PDF_EXT:
        return pdf_to_page_images(path, dpi)
    if ext in IMAGE_EXT:
        img = imread_unicode(path)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        return [(1, img, False)]
    raise ValueError(f"Unsupported file: {path}")
