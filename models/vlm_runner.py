# -*- coding: utf-8 -*-
"""
models/vlm_runner.py — shared inference call for both AIN and Qwen2-VL.

Both models load via the same Qwen2VLForConditionalGeneration class (AIN is a
fine-tuned Qwen2-VL-7B), so one function here can serve both qwen_client.py
and ain_client.py instead of duplicating inference code.

TODO: implement.
"""
from __future__ import annotations


def run_vlm(model_id: str, image, prompt: str, max_new_tokens: int = 128) -> str:
    """
    model_id: local path or HF id of the model to load (AIN_MODEL_ID or
              QWEN_MODEL_ID from config.py).
    image:    a PIL.Image, a numpy BGR array, or a file path.
    prompt:   the instruction/question to run against the image.

    Returns the model's text output.
    """
    raise NotImplementedError("run_vlm: fill in model loading + inference")
