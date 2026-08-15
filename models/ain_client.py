# -*- coding: utf-8 -*-
"""models/ain_client.py — AIN side of path 1 (ain_light preprocessing -> here)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import AIN_MODEL_ID
from models.vlm_runner import run_vlm


def run_ain(image, prompt: str, max_new_tokens: int = 256) -> str:
    return run_vlm(AIN_MODEL_ID, image, prompt, max_new_tokens)
