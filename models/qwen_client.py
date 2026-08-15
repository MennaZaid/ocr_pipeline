# -*- coding: utf-8 -*-
"""models/qwen_client.py — Qwen2-VL side of path 2 (volume 1/2/3/5 -> here)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import QWEN_MODEL_ID
from models.vlm_runner import run_vlm


def run_qwen(image, prompt: str, max_new_tokens: int = 256) -> str:
    return run_vlm(QWEN_MODEL_ID, image, prompt, max_new_tokens)
