# -*- coding: utf-8 -*-
"""
extraction_utils.py — turn a model's raw text output into the field list the
matcher needs.

Handles both prompt styles from config.py:
  - EXTRACTION_PROMPT            -> output is just the JSON array, parse directly.
  - EXTRACTION_PROMPT_TWO_STAGE  -> output is a short description, then
                                    RESULT_MARKER, then the JSON array.

Using one parser for both means switching which prompt you're testing doesn't
require touching run_pipeline.py or either model client — only this function
needs to know the two shapes exist.
"""
from __future__ import annotations

import json
import re

from config import RESULT_MARKER


def parse_extraction_output(raw_text: str) -> list[dict]:
    """
    Returns a list of {"full_name": ..., "national_id": ...} dicts.
    Returns [] (not an error) if nothing parseable is found — callers should
    treat an empty list from a non-empty raw_text as a parse failure worth
    logging, distinct from a genuine empty-page [] the model returned.
    """
    text = raw_text.strip()

    if RESULT_MARKER in text:
        text = text.split(RESULT_MARKER, 1)[1].strip()

    # strip markdown code fences if the model added them despite instructions
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    # if there's leading/trailing chatter around the array, isolate it
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    out = []
    for item in data:
        if isinstance(item, dict) and "full_name" in item and "national_id" in item:
            out.append({
                "full_name": item.get("full_name", ""),
                "national_id": item.get("national_id", ""),
                "needs_review": "؟" in str(item.get("full_name", "")) + str(item.get("national_id", "")),
            })
    return out
