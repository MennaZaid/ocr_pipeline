# -*- coding: utf-8 -*-
"""
models/omni_client.py — Qwen2.5-Omni-7B path, now wired to the volume
1/2/3/5 preprocessing pipeline (see run_pipeline.py's run_omni_path).

Different model class from vlm_runner.py's (Qwen2_5OmniForConditionalGeneration,
not Qwen2VLForConditionalGeneration) and a different vision utility
(qwen_omni_utils, not qwen_vl_utils) — cannot share vlm_runner.py, kept as
its own file instead of forcing a shared abstraction across two unrelated
model APIs.

Text-only: disable_talker() + return_audio=False, since this pipeline only
needs OCR/extraction text, not speech.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OMNI_MODEL_ID

from PIL import Image

_MODEL = None
_PROCESSOR = None


def _load():
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    _MODEL = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        OMNI_MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    _MODEL.disable_talker()
    _PROCESSOR = Qwen2_5OmniProcessor.from_pretrained(OMNI_MODEL_ID)
    # Workaround for a transformers/Omni compatibility bug: some versions
    # don't populate pad_token_id on the Talker subconfig, which generate()
    # reads from even with the talker disabled. Set it explicitly from the
    # tokenizer's own eos_token_id (standard fallback when no pad token
    # exists) rather than relying on the subconfig having it already.
    if _MODEL.generation_config.pad_token_id is None:
        _MODEL.generation_config.pad_token_id = _PROCESSOR.tokenizer.eos_token_id
    _MODEL_CACHE_READY = True
    return _MODEL, _PROCESSOR

def unload() -> None:
    global _MODEL, _PROCESSOR
    import torch
    _MODEL, _PROCESSOR = None, None
    torch.cuda.empty_cache()


def run_omni(image, prompt: str, max_new_tokens: int = 256) -> str:
    from qwen_omni_utils import process_mm_info
    model, processor = _load()

    if not isinstance(image, Image.Image):
        import numpy as np, cv2
        if isinstance(image, np.ndarray):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
            image = Image.fromarray(rgb)
        else:
            image = Image.open(str(image)).convert("RGB")

    conversation = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, return_audio=False)
    return processor.batch_decode(text_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]