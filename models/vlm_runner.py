# -*- coding: utf-8 -*-
"""
models/vlm_runner.py — shared inference for AIN + Qwen2-VL.

Both load via Qwen2VLForConditionalGeneration (AIN is a fine-tuned
Qwen2-VL-7B), so one function serves qwen_client.py and ain_client.py.
Qwen2.5-Omni is a DIFFERENT model class (see models/omni_client.py) and
is not covered here.

Models are cached per model_id. On a single GPU, do not expect all three
models (qwen, ain, omni) to fit resident at once — see README "Kaggle /
memory notes".
"""
from __future__ import annotations

from PIL import Image

_MODEL_CACHE: dict[str, tuple] = {}


def _load(model_id: str):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    _MODEL_CACHE[model_id] = (model, processor)
    return model, processor


def unload(model_id: str) -> None:
    """Free VRAM before loading the next model — needed when running all
    three paths sequentially on one GPU (see omni_client.py, same pattern)."""
    import torch
    if model_id in _MODEL_CACHE:
        del _MODEL_CACHE[model_id]
        torch.cuda.empty_cache()


def run_vlm(model_id: str, image, prompt: str, max_new_tokens: int = 256) -> str:
    from qwen_vl_utils import process_vision_info
    model, processor = _load(model_id)

    if not isinstance(image, Image.Image):
        import numpy as np, cv2
        if isinstance(image, np.ndarray):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
            image = Image.fromarray(rgb)
        else:
            image = Image.open(str(image)).convert("RGB")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]
