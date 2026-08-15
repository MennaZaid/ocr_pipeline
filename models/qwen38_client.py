# -*- coding: utf-8 -*-
"""
models/qwen38_client.py — path 4: Qwen3.8-27B, self-hosted.

Unlike ain/qwen/omni clients, this is NOT a local transformers.from_pretrained
model — Qwen3.8 has no direct-load quickstart in its own docs, only an
OpenAI-compatible API pattern. This calls that API. It works against:
  - a vLLM/SGLang/TokenSpeed server YOU run yourself (set QWEN38_BASE_URL), or
  - Qwen Cloud's hosted service, once that's live (same env vars, different URL).

Requires config.QWEN38_BASE_URL to be set — raises before making any request
if it isn't, so a missing server doesn't silently hang.
"""
from __future__ import annotations
import base64
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import QWEN38_MODEL_ID, QWEN38_BASE_URL, QWEN38_API_KEY

from PIL import Image


def _to_data_url(image) -> str:
    if not isinstance(image, Image.Image):
        import numpy as np, cv2
        if isinstance(image, np.ndarray):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
            image = Image.fromarray(rgb)
        else:
            image = Image.open(str(image)).convert("RGB")
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def run_qwen38(image, prompt: str, max_new_tokens: int = 256) -> str:
    if not QWEN38_BASE_URL:
        raise RuntimeError(
            "QWEN38_BASE_URL is not set. Start a vLLM/SGLang server for "
            "Qwen3.8-27B yourself and set QWEN38_BASE_URL (env var) to its "
            "OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
        )
    from openai import OpenAI
    client = OpenAI(base_url=QWEN38_BASE_URL, api_key=QWEN38_API_KEY)
    resp = client.chat.completions.create(
        model=QWEN38_MODEL_ID,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _to_data_url(image)}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=max_new_tokens,
        # Non-thinking mode per Qwen3.8's own recommended params, but with a
        # much lower temperature than their suggested 0.7 — their number is
        # tuned for open dialogue; this is deterministic field extraction,
        # where you want the same page to produce the same JSON every run.
        temperature=0.1,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content
