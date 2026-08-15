#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py — manual model selection, one path per run:

    python run_pipeline.py --input document.pdf --output out --model ain
    python run_pipeline.py --input document.pdf --output out --model omni
    python run_pipeline.py --input document.pdf --output out --model qwen3.8
    python run_pipeline.py --input document.pdf --output out --model all   # all three

Three paths, dispatched by --model:
  ain     -> ain_light preproc (deskew+crop only) -> models/ain_client.py  (local, transformers)
  omni    -> pipeline_selector -> vol N (1/2/3/5)  -> models/omni_client.py (local, transformers)
  qwen3.8 -> ain_light preproc (deskew+crop only)  -> models/qwen38_client.py (HTTP, self-hosted server — see config.QWEN38_BASE_URL)

The old Qwen2-VL-only "qwen" path has been retired — omni now owns the
volume 1/2/3/5 preprocessing pipeline that path used to run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from config import DEFAULT_DPI, DEFAULT_LANG, PROMPTS
from pdf_to_images import load_as_pages, imwrite_unicode, IMAGE_EXT, PDF_EXT
from pipeline_selector import estimate_quality, choose_pipeline, volume_script_path
from preprocessors.ain_light import preprocess_for_ain, AinConfig
from extraction_utils import parse_extraction_output

TXT_EXT = {".txt"}
MODEL_CHOICES = ["ain", "omni", "qwen3.8", "all"]


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr


def safe_name(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)


def expand_inputs(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.is_file() and path.suffix.lower() in TXT_EXT:
        base = path.parent
        out: list[Path] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            out.extend(expand_inputs(p if p.is_absolute() else (base / p).resolve()))
        return out
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT | PDF_EXT)
    if path.is_file() and path.suffix.lower() in IMAGE_EXT | PDF_EXT:
        return [path]
    raise ValueError(f"Unsupported input type: {path.suffix or '(no extension)'}")


def run_volume_script(volume_key: str, src_image: Path, workdir: Path, args) -> Path | None:
    script = volume_script_path(volume_key)
    cmd = [sys.executable, str(script), str(src_image), "--outdir", str(workdir),
          "--dpi", str(args.dpi), "--lang", args.lang]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONUTF8"] = "1"
    rc = subprocess.run(cmd, check=False, env=env).returncode
    if rc != 0:
        return None
    candidates = sorted(workdir.glob(f"{src_image.stem}*.png"))
    return candidates[-1] if candidates else None


# --------------------------------------------------------------------------- #
# One handler per model. Each returns (fields, raw_text, note_or_None).
# --------------------------------------------------------------------------- #
def run_ain_path(bgr, src_path, page_work_dir, page_id, args):
    result = preprocess_for_ain(bgr, AinConfig())
    imwrite_unicode(page_work_dir / f"{page_id}_ain_preprocessed.png", result.final)
    from models.ain_client import run_ain
    text = run_ain(result.final, args.prompt or PROMPTS["ain"])
    return parse_extraction_output(text), text, None


def run_omni_path(bgr, src_path, page_work_dir, page_id, args):
    """Volume 1/2/3/5 preprocessing -> Qwen2.5-Omni. This is the pipeline
    that used to be wired to Qwen2-VL under the retired "qwen" path — only
    the model at the end changed."""
    gray = to_gray(bgr)
    volume_key, reason = choose_pipeline(estimate_quality(gray))
    img_path = run_volume_script(volume_key, src_path, page_work_dir, args)
    if img_path is None:
        return [], "", f"{volume_key} preprocessing failed"
    from models.omni_client import run_omni
    text = run_omni(img_path, args.prompt or PROMPTS["omni"])
    return parse_extraction_output(text), text, f"volume={volume_key} ({reason})"


def run_qwen38_path(bgr, src_path, page_work_dir, page_id, args):
    result = preprocess_for_ain(bgr, AinConfig())
    imwrite_unicode(page_work_dir / f"{page_id}_qwen38_preprocessed.png", result.final)
    from models.qwen38_client import run_qwen38
    text = run_qwen38(result.final, args.prompt or PROMPTS["qwen3.8"])
    return parse_extraction_output(text), text, None


HANDLERS = {
    "ain": run_ain_path,
    "omni": run_omni_path,
    "qwen3.8": run_qwen38_path,
}


def main():
    ap = argparse.ArgumentParser(description="Run ONE chosen model path (or --model all for all three)")
    ap.add_argument("--input", required=True, help="pdf, image, folder, or txt manifest")
    ap.add_argument("--output", default="pipeline_out")
    ap.add_argument("--model", required=True, choices=MODEL_CHOICES)
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument("--lang", default=DEFAULT_LANG)
    ap.add_argument("--prompt", default=None, help="override the default prompt for the chosen model")
    args = ap.parse_args()

    models_to_run = list(HANDLERS.keys()) if args.model == "all" else [args.model]

    src = Path(args.input).resolve()
    files = expand_inputs(src)
    if not files:
        sys.exit("No supported files found.")

    out_base = Path(args.output).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    diagnostics = []
    failures = 0

    for doc in files:
        doc_stem = safe_name(doc)
        work_dir = out_base / f"{doc_stem}_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        text_dirs = {}
        for m in models_to_run:
            d = out_base / f"{doc_stem}_text_{m.replace('.', '')}"
            d.mkdir(parents=True, exist_ok=True)
            text_dirs[m] = d

        try:
            pages = load_as_pages(doc, args.dpi)
        except Exception as e:
            print(f"[skip] {doc}: {e}")
            failures += 1
            continue

        for page_num, bgr, has_native_text in pages:
            page_id = f"{doc_stem}_p{page_num:03d}"
            page_work_dir = work_dir / page_id
            page_work_dir.mkdir(parents=True, exist_ok=True)
            src_path = page_work_dir / f"{page_id}_source.png"
            imwrite_unicode(src_path, bgr)

            row = {"doc": doc.name, "page": page_num, "page_id": page_id,
                  "has_native_text": has_native_text}

            for m in models_to_run:
                try:
                    fields, text, note = HANDLERS[m](bgr, src_path, page_work_dir, page_id, args)
                except Exception as e:
                    fields, text, note = [], "", f"error: {e}"
                    failures += 1
                (text_dirs[m] / f"{page_id}.json").write_text(
                    json.dumps({"page_id": page_id, "path": m, "raw_text": text,
                              "fields": fields, "note": note}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
                row[f"{m}_output"] = fields
                if note:
                    row[f"{m}_note"] = note

            diagnostics.append(row)
            print(f"{page_id}: {', '.join(models_to_run)}")

    report_path = out_base / "routing_report.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        for r in diagnostics:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nreport -> {report_path}")

    if failures:
        sys.exit(f"{failures} page/document operation(s) failed.")


if __name__ == "__main__":
    main()