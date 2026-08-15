#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py — the orchestrator. Implements exactly this flow:

    input: pdf or image
       |
       v (if pdf)
    pdf_to_images.py  -->  per-page images
       |
       v  loop: each page image gets copied down BOTH paths independently
       |
       +-------------------------------+-------------------------------+
       |                                                                |
  PATH 1 (AIN)                                                   PATH 2 (Qwen)
  preprocessors/ain_light.py                              pipeline_selector.py
  (deskew + crop, no binarization)                         -> chooses volume 1/2/3/5
       |                                                                |
  models/ain_client.py  (STUB — fill in)                external volume script (subprocess)
       |                                                                |
       v                                                                v
  <doc>_text_ain/  (one result file per page)          models/qwen_client.py (STUB — fill in)
                                                                          |
                                                                          v
                                                        <doc>_text_qwen/  (one result file per page)

Both paths start from the SAME source page image, written once per page, and
run independently — path 2 never consumes path 1's output or vice versa.

Model calls (models/ain_client.py, models/qwen_client.py) are currently empty
stubs. This script calls them and, if they raise NotImplementedError, just
notes that in the report and moves on — preprocessing still runs end to end
without them, so you can validate paths 1 and 2 before wiring up inference.
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


# --------------------------------------------------------------------------- #
# PATH 2 plumbing: the volume scripts are external (auto_pipeline_selector's
# routing logic lives in pipeline_selector.py; the scripts themselves are
# run here as subprocesses, same as before).
# --------------------------------------------------------------------------- #
def run_volume_script(volume_key: str, src_image: Path, workdir: Path, args) -> Path | None:
    script = volume_script_path(volume_key)
    cmd = [
        sys.executable, str(script), str(src_image),
        "--outdir", str(workdir),
        "--dpi", str(args.dpi),
        "--lang", args.lang,
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    rc = subprocess.run(cmd, check=False, env=env).returncode
    if rc != 0:
        return None
    # volume scripts write "<stem>_final.png" (or similar) into workdir; adjust
    # this glob if your volume scripts' actual output naming differs.
    candidates = sorted(workdir.glob(f"{src_image.stem}*.png"))
    return candidates[-1] if candidates else None


def main():
    ap = argparse.ArgumentParser(description="PDF/image -> per-page loop -> fork into AIN path + Qwen path")
    ap.add_argument("input", help="pdf, image, folder, or txt manifest")
    ap.add_argument("--outdir", default="pipeline_out")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument("--lang", default=DEFAULT_LANG)
    ap.add_argument("--ain-prompt", default=PROMPTS["ain"])
    ap.add_argument("--qwen-prompt", default=PROMPTS["qwen"])
    ap.add_argument("--no-ain", action="store_true", help="skip path 1 entirely")
    ap.add_argument("--no-qwen", action="store_true", help="skip path 2 entirely")
    args = ap.parse_args()

    src = Path(args.input).resolve()
    files = expand_inputs(src)
    if not files:
        sys.exit("No supported files found.")

    out_base = Path(args.outdir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    diagnostics = []
    failures = 0

    for doc in files:
        doc_stem = safe_name(doc)
        work_dir = out_base / f"{doc_stem}_work"          # intermediate images live here
        ain_text_dir = out_base / f"{doc_stem}_text_ain"    # path 1 results
        qwen_text_dir = out_base / f"{doc_stem}_text_qwen"  # path 2 results
        work_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_ain:
            ain_text_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_qwen:
            qwen_text_dir.mkdir(parents=True, exist_ok=True)

        # --- input: pdf or image -> per-page images ---
        try:
            pages = load_as_pages(doc, args.dpi)
        except Exception as e:
            print(f"[skip] {doc}: {e}")
            failures += 1
            continue

        # --- loop: each page image goes down both paths ---
        for page_num, bgr, has_native_text in pages:
            page_id = f"{doc_stem}_p{page_num:03d}"
            page_work_dir = work_dir / page_id
            page_work_dir.mkdir(parents=True, exist_ok=True)

            src_path = page_work_dir / f"{page_id}_source.png"
            imwrite_unicode(src_path, bgr)

            row = {"doc": doc.name, "page": page_num, "page_id": page_id,
                  "has_native_text": has_native_text}
            if has_native_text:
                row["note"] = "PDF page has a native text layer — consider extracting directly instead of OCR."

            # ============================ PATH 1: AIN ============================
            if not args.no_ain:
                ain_result = preprocess_for_ain(bgr, AinConfig())
                ain_img_path = page_work_dir / f"{page_id}_ain_preprocessed.png"
                imwrite_unicode(ain_img_path, ain_result.final)
                row["ain_skew_deg"] = ain_result.meta.get("skew_deg")

                try:
                    from models.ain_client import run_ain
                    text = run_ain(ain_result.final, args.ain_prompt)
                    fields = parse_extraction_output(text)
                    (ain_text_dir / f"{page_id}.json").write_text(
                        json.dumps({"page_id": page_id, "path": "ain", "raw_text": text,
                                  "fields": fields}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
                    row["ain_output"] = fields
                except NotImplementedError:
                    row["ain_output"] = None
                    row["ain_note"] = "models/ain_client.py not implemented yet"

            # ============================ PATH 2: Qwen ============================
            if not args.no_qwen:
                gray = to_gray(bgr)
                metrics = estimate_quality(gray)
                volume_key, reason = choose_pipeline(metrics)
                row.update({"qwen_volume": volume_key, "qwen_route_reason": reason, **metrics})

                qwen_img_path = run_volume_script(volume_key, src_path, page_work_dir, args)
                if qwen_img_path is None:
                    failures += 1
                    print(f"[fail] {page_id}: {volume_key} preprocessing failed")
                else:
                    try:
                        from models.qwen_client import run_qwen
                        text = run_qwen(qwen_img_path, args.qwen_prompt)
                        fields = parse_extraction_output(text)
                        (qwen_text_dir / f"{page_id}.json").write_text(
                            json.dumps({"page_id": page_id, "path": "qwen", "volume": volume_key,
                                      "raw_text": text, "fields": fields},
                                      ensure_ascii=False, indent=1),
                            encoding="utf-8")
                        row["qwen_output"] = fields
                    except NotImplementedError:
                        row["qwen_output"] = None
                        row["qwen_note"] = "models/qwen_client.py not implemented yet"

            diagnostics.append(row)
            print(f"{page_id}: preprocessed"
                 f"{'  [ain]' if not args.no_ain else ''}"
                 f"{'  [qwen:' + row.get('qwen_volume', '?') + ']' if not args.no_qwen else ''}")

    report_path = out_base / "routing_report.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        for r in diagnostics:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nreport -> {report_path}")

    if failures:
        sys.exit(f"{failures} page/document operation(s) failed.")


if __name__ == "__main__":
    main()
