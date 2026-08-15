# -*- coding: utf-8 -*-
"""
config.py — single place for paths and defaults used across the repo.

IMPORTANT: VOLUME_SCRIPTS below point at your existing volume 1/2/3/5
preprocessor scripts (ocr_preprocess.py per volume, run_safe.py for volume 5).
This repo does not duplicate their code — it calls them as subprocesses, the
same way auto_pipeline_selector.py did. Point these at wherever those scripts
actually live in your project; the paths below are placeholders matching the
layout implied by your earlier auto_pipeline_selector.py.

ocr_preprocess_v2.py (the shared measurement/IO module that ain_light.py and
the volume scripts import from — component_stats, text_metrics, pdf helpers,
etc.) must be importable, i.e. on PYTHONPATH or in SHARED_MODULE_DIR below.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Directory containing ocr_preprocess_v2.py (shared by ain_light.py and the
# volume scripts). Added to sys.path at runtime by run_pipeline.py.
SHARED_MODULE_DIR = REPO_ROOT

VOLUME_SCRIPTS = {
    "volume1": REPO_ROOT / "pipeline_preprocessors" / "Volume 1" / "ocr_preprocess.py",
    "volume2": REPO_ROOT / "pipeline_preprocessors" / "Volume 2" / "ocr_preprocess.py",
    "volume3": REPO_ROOT / "pipeline_preprocessors" / "Volume 3" / "ocr_preprocess.py",
    "volume5": REPO_ROOT / "pipeline_preprocessors" / "Volume 5 (Complicated Pipeline)" / "run_safe.py",
}
# ^ Verify these against your actual folder layout before running --run.

DEFAULT_DPI = 300
DEFAULT_LANG = "ara+eng"

# --- model ids -------------------------------------------------------------
# Both are loaded with Qwen2VLForConditionalGeneration — same class, same
# processor pattern. Point these at local/offline weight directories for the
# on-prem/no-cloud constraint; do NOT rely on them hitting the HF Hub at
# runtime in the CBE environment. Download and mirror the weights locally
# ahead of time, then set these to local paths.
QWEN_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
AIN_MODEL_ID = "MBZUAI/AIN"

# Kept deliberately simple: one language, one field pair per person, no
# nested boolean flags. A complex schema with conditional rules gives the
# model more chances to produce malformed JSON or hedge on format instead of
# content. Uncertainty is handled with a single inline marker ("؟") instead
# of separate flag fields — easy for the model to produce, easy for the
# matcher to detect (a "؟" anywhere in a field means NEEDS_REVIEW, no schema
# parsing required to find it).
EXTRACTION_PROMPT = """اقرأ هذه الصفحة من مستند قضائي، واستخرج بيانات كل شخص متهم فقط (وليس القضاة أو المحامين أو الموظفين).

أخرج النتيجة بصيغة JSON فقط، بدون أي نص آخر، بهذا الشكل بالضبط:
[{"full_name": "اسم المتهم كما هو مكتوب", "national_id": "الرقم القومي كما تراه"}]

مثال:
[{"full_name": "أحمد محمد علي", "national_id": "29001011234567"}]

إذا كان رقم أو حرف غير واضح في الصورة، اكتب "؟" مكانه بدلاً من التخمين.
إذا لم يوجد أي متهم في الصفحة، أخرج: []
"""

# Both models are pointed at the SAME prompt by default. This is a
# deliberate placeholder, not a claim that they need the same one — see the
# README section "Do AIN and Qwen need different prompts?" for how to find
# out. Once you have evidence either way, change one entry here without
# touching run_pipeline.py.
PROMPTS = {
    "qwen": EXTRACTION_PROMPT,
    "ain": EXTRACTION_PROMPT,
}

# --- alternative: read-then-extract ----------------------------------------
# Same task, but the model is asked to read the WHOLE page first before
# extracting anything. This can help on messy/multi-defendant pages, where a
# model that jumps straight to JSON sometimes locks onto the first name it
# sees and skips a second defendant further down, or extracts a digit it
# didn't actually attend to closely. The cost: roughly double the output
# length, so it needs a higher max_new_tokens, and the response has to be
# split before parsing — RESULT_MARKER is that split point, used by both the
# prompt (below) and extraction_utils.parse_extraction_output().
#
# This is NOT set as the default in PROMPTS above on purpose — whether it's
# worth the extra latency/cost is a question to answer empirically (see
# README "Do AIN and Qwen need different prompts?" — same A/B methodology
# applies here: same labeled pages, compare against EXTRACTION_PROMPT).
RESULT_MARKER = "النتيجة:"

EXTRACTION_PROMPT_TWO_STAGE = f"""اقرأ هذه الصفحة بالكامل أولاً بعناية.

الخطوة الأولى: اكتب وصفاً موجزاً (سطرين كحد أقصى) لما تراه — نوع المستند، عدد الأشخاص المتهمين المذكورين، وأي أجزاء غير واضحة أو تالفة.

الخطوة الثانية: بعد كلمة "{RESULT_MARKER}" اكتب فقط مصفوفة JSON بهذا الشكل، بدون أي نص آخر بعدها:
[{{"full_name": "اسم المتهم كما هو مكتوب", "national_id": "الرقم القومي كما تراه"}}]

مثال:
{RESULT_MARKER}
[{{"full_name": "أحمد محمد علي", "national_id": "29001011234567"}}]

لا تدرج القضاة أو المحامين أو الموظفين. إذا كان رقم أو حرف غير واضح اكتب "؟" مكانه بدلاً من التخمين.
إذا لم يوجد أي متهم في الصفحة، اكتب بعد "{RESULT_MARKER}": []
"""
