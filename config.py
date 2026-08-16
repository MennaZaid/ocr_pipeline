# -*- coding: utf-8 -*-
"""
config.py — single place for paths, model ids, and prompts used across the repo.

Three active paths: ain, omni, qwen3.8.

REVISION: vision models no longer output JSON. Each path's prompt asks for a
plain transcription of the page — nothing else. A separate text-only pass in
extraction_utils.py (extract_fields_from_transcription) searches that
transcription for name/national-id using the literal field labels Egyptian
court documents print. This split exists because asking a vision model to
read faint/handwritten text AND classify defendant-vs-judge/lawyer AND
commit to rigid JSON, all in one generation, was producing confident-wrong
values (a birthdate or case number reported as if it were the national ID)
rather than honest transcription. Splitting "read" from "extract" removes
two of those three simultaneous jobs from the vision call.

ocr_preprocess_v2.py (the shared measurement/IO module that ain_light.py and
the volume scripts import from) must be importable, i.e. on PYTHONPATH or in
SHARED_MODULE_DIR below.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

SHARED_MODULE_DIR = REPO_ROOT / "pipeline_preprocessors" / "Volume 5 (Complicated Pipeline)"

VOLUME_SCRIPTS = {
    "volume1": REPO_ROOT / "pipeline_preprocessors" / "Volume 1" / "ocr_preprocess.py",
    "volume2": REPO_ROOT / "pipeline_preprocessors" / "Volume 2" / "ocr_preprocess.py",
    "volume3": REPO_ROOT / "pipeline_preprocessors" / "Volume 3" / "ocr_preprocess.py",
    "volume5": REPO_ROOT / "pipeline_preprocessors" / "Volume 5 (Complicated Pipeline)" / "run_safe.py",
}

DEFAULT_DPI = 300
DEFAULT_LANG = "ara+eng"

# --- model ids -------------------------------------------------------------
QWEN_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"   # kept only because vlm_runner.py's loader class is shared with AIN
AIN_MODEL_ID = "MBZUAI/AIN"
OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

QWEN38_MODEL_ID = "Qwen/Qwen3.8-27B"
QWEN38_BASE_URL = os.environ.get("QWEN38_BASE_URL", "")
QWEN38_API_KEY = os.environ.get("QWEN38_API_KEY", "not-needed")

# --- prompts: transcription only, no JSON, no role-filtering ---------------
# Each still differs slightly by what image the path actually receives.

AIN_PROMPT = """اقرأ هذه الصورة بعناية وانسخ كل النص المكتوب فيها، تماماً كما تراه، بدون تلخيص أو حذف أو إعادة ترتيب. إذا كان جزء من النص غير واضح، اكتب "؟" في مكانه بدلاً من التخمين. لا تفترض أي كلمة أو رقم غير مكتوب بوضوح.
"""

OMNI_PROMPT = """هذه صورة معالجة (أبيض وأسود) لصفحة من مستند قضائي بعد معالجة رقمية لتحسين وضوح النص. اقرأ كل النص المكتوب فيها بعناية وانسخه تماماً كما تراه، بدون تلخيص أو حذف أو إعادة ترتيب. إذا كان جزء من النص غير واضح رغم المعالجة، اكتب "؟" في مكانه بدلاً من التخمين. لا تفترض أي كلمة أو رقم غير مكتوب بوضوح.
"""

QWEN38_PROMPT = """هذه صورة لصفحة من مستند قضائي، تم فقط تصحيح ميلها وقصها دون أي تعديل آخر على الألوان أو وضوح النص. اقرأ كل النص المكتوب فيها بعناية وانسخه تماماً كما تراه، بدون تلخيص أو حذف أو إعادة ترتيب. إذا كان جزء من النص غير واضح، اكتب "؟" في مكانه بدلاً من التخمين. لا تفترض أي كلمة أو رقم غير مكتوب بوضوح.
"""

PROMPTS = {
    "ain": AIN_PROMPT,
    "omni": OMNI_PROMPT,
    "qwen3.8": QWEN38_PROMPT,
}

# --- second-stage field extraction (text-only, no model call) --------------
# Labels to search for in the transcription. TUNE THESE against real
# transcribed pages — these are a first guess, not validated.
NAME_LABELS = ["اسم المتهم", "المتهم", "الاسم"]
ID_LABELS = ["الرقم القومي", "رقم قومي", "رقم الهوية"]