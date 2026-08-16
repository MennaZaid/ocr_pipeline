# -*- coding: utf-8 -*-
"""
extraction_utils.py — turn a model's raw transcription into the field list
the downstream matcher/reviewer needs.

REVISION: vision models now output plain transcribed text, not JSON — see
config.py's prompts. This module's job changed from "parse JSON" to "search
transcribed text for name/national-id near their literal field labels."
This is a first-pass heuristic, not a validated extractor — tune NAME_LABELS
/ ID_LABELS in config.py and the regex below against real transcriptions.

raw_text (the full page transcription) is ALWAYS returned untouched
regardless of whether field extraction succeeds, so a human reviewer never
loses information even when the heuristic below misses.
"""
from __future__ import annotations

import re

from config import NAME_LABELS, ID_LABELS

# Egyptian national IDs are 14 digits. Accept ASCII or Arabic-Indic digits,
# and allow "؟" inside the run since the model marks unclear characters with
# it rather than guessing — a partially-unclear ID should still be found and
# flagged, not silently missed by the regex.
_DIGIT_RUN_RE = re.compile(r"[0-9\u0660-\u0669\uFF10-\uFF19؟]{8,14}")


def _find_after_label(text: str, labels: list[str], pattern: "re.Pattern|None" = None) -> str | None:
    """
    For each label, find its first occurrence and look at the text
    immediately following it on the same line (up to the next line break or
    a reasonable character limit). If `pattern` is given, return the first
    regex match in that window; otherwise return the trimmed text itself.
    """
    for label in labels:
        idx = text.find(label)
        if idx == -1:
            continue
        window = text[idx + len(label): idx + len(label) + 80]
        window = window.lstrip(" :\u200f\u200e-")
        if pattern is not None:
            m = pattern.search(window)
            if m:
                return m.group(0)
            continue
        line_end = window.find("\n")
        candidate = window if line_end == -1 else window[:line_end]
        candidate = candidate.strip()
        if candidate:
            return candidate
    return None


def extract_fields_from_transcription(raw_text: str) -> dict:
    """
    Returns {"full_name": str|None, "national_id": str|None,
             "needs_review": bool, "extraction_note": str|None}.

    needs_review is True if either field couldn't be found, OR if either
    found value contains "؟" (the model's own uncertainty marker).
    extraction_note explains WHY review is needed, for a human reviewer's
    benefit — e.g. "national_id not found near expected label" vs.
    "national_id contains unclear characters".
    """
    text = raw_text.strip()
    if not text:
        return {"full_name": None, "national_id": None,
                "needs_review": True, "extraction_note": "empty transcription"}

    name = _find_after_label(text, NAME_LABELS)
    national_id = _find_after_label(text, ID_LABELS, pattern=_DIGIT_RUN_RE)

    notes = []
    if name is None:
        notes.append("full_name not found near expected label")
    elif "؟" in name:
        notes.append("full_name contains unclear characters")

    if national_id is None:
        notes.append("national_id not found near expected label")
    elif "؟" in national_id:
        notes.append("national_id contains unclear characters")
    elif len(re.sub(r"[^\d]", "", national_id)) != 14:
        # found a digit run, but not 14 digits — flag rather than trust it,
        # per the earlier lesson: don't silently treat a case number or
        # birthdate as if it were the national id
        notes.append(f"digit run found but is {len(re.sub(r'[^\d]', '', national_id))} digits, not 14 — verify manually")

    return {
        "full_name": name,
        "national_id": national_id,
        "needs_review": bool(notes),
        "extraction_note": "; ".join(notes) if notes else None,
    }