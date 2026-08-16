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


def parse_extraction_output(raw_text: str) -> dict:
    """
    Returns {"description": str|None, "fields": list[dict]}.

    "description" is whatever text preceded RESULT_MARKER (the two-stage
    "what I see on this page" summary) — None if the model wasn't given a
    two-stage prompt, i.e. no marker was expected or found.
    "fields" is the same {"full_name", "national_id", "needs_review"} list
    as before. An empty fields list from non-empty raw_text is still worth
    logging as a possible parse failure, same as before.
    """
    text = raw_text.strip()
    description = None

    if RESULT_MARKER in text:
        description, text = text.split(RESULT_MARKER, 1)
        description = description.strip() or None
        text = text.strip()

    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"description": description, "fields": []}

    if not isinstance(data, list):
        return {"description": description, "fields": []}

    fields = []
    for item in data:
        if isinstance(item, dict) and "full_name" in item and "national_id" in item:
            fields.append({
                "full_name": item.get("full_name", ""),
                "national_id": item.get("national_id", ""),
                "needs_review": "؟" in str(item.get("full_name", "")) + str(item.get("national_id", "")),
            })
    return {"description": description, "fields": fields}
