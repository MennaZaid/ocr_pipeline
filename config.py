# -*- coding: utf-8 -*-
"""
config.py — single place for paths, model ids, and prompts used across the repo.

Three active paths as of this revision: ain, omni, qwen3.8.
The original Qwen2-VL-only path ("qwen") has been retired in favor of Omni
taking over the volume 1/2/3/5 preprocessing pipeline. models/qwen_client.py
has been deleted; QWEN_MODEL_ID is kept below only because AIN's loader
(models/vlm_runner.py) shares the same Qwen2VLForConditionalGeneration class
AIN was fine-tuned from — it is not a separate runnable path anymore.

ocr_preprocess_v2.py (the shared measurement/IO module that ain_light.py and
the volume scripts import from — component_stats, text_metrics, pdf helpers,
etc.) must be importable, i.e. on PYTHONPATH or in SHARED_MODULE_DIR below.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Directory containing ocr_preprocess_v2.py (shared by ain_light.py and the
# volume scripts). Added to sys.path at runtime by run_pipeline.py.
SHARED_MODULE_DIR = REPO_ROOT / "pipeline_preprocessors" / "Volume 5 (Complicated Pipeline)"

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
# AIN loads via Qwen2VLForConditionalGeneration (it's a fine-tune of
# Qwen2-VL-7B) — see models/vlm_runner.py / models/ain_client.py.
QWEN_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"   # kept only because vlm_runner.py's loader class is shared; no standalone "qwen" path calls this anymore
AIN_MODEL_ID = "MBZUAI/AIN"

# Omni loads via a DIFFERENT class (Qwen2_5OmniForConditionalGeneration, see
# models/omni_client.py). Omni support has been in mainline transformers
# since 2025-04-14 — no special preview branch needed, see dependencies.txt.
OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

# Qwen3.8 is not a local transformers model — it's called over an
# OpenAI-compatible HTTP API against a server YOU run (vLLM/SGLang/
# TokenSpeed), on-prem, per the no-cloud constraint. Never point this at
# Qwen Cloud in this deployment.
QWEN38_MODEL_ID = "Qwen/Qwen3.8-27B"
QWEN38_BASE_URL = os.environ.get("QWEN38_BASE_URL", "")   # e.g. http://localhost:8000/v1 — empty = qwen38_client raises before any request, by design
QWEN38_API_KEY = os.environ.get("QWEN38_API_KEY", "not-needed")

# Point QWEN_MODEL_ID / AIN_MODEL_ID / OMNI_MODEL_ID at local/offline weight
# directories for the on-prem/no-cloud constraint — do NOT rely on hitting
# the HF Hub at runtime in the bank's environment. Download and mirror the
# weights locally ahead of time, then set these to local paths.

# --- prompts -----------------------------------------------------------
# Every model gets its OWN prompt — they differ in training background and
# in what kind of image they actually receive (see notes on each below).

# AIN: fine-tuned specifically on authentic Arabic documents/handwriting
# (CAMEL-Bench OCR & Document Understanding: 72.35 vs Qwen2-VL-7B's 42.73),
# so the prompt can trust it with faint/handwritten text rather than hedge.
# Always used in its two-stage form (see AIN_PROMPT_TWO_STAGE below) — AIN
# is the path most likely to see genuinely hard pages, and has no
# NULL-vs-NEEDS-REVIEW evidence signal the way volume 5 does.
AIN_PROMPT = """اقرأ هذه الصفحة من مستند قضائي بعناية، بما في ذلك أي خط يد أو نص باهت أو غير واضح. استخرج بيانات كل شخص متهم فقط (وليس القضاة أو المحامين أو الموظفين).

أخرج النتيجة بصيغة JSON فقط، بدون أي نص آخر، بهذا الشكل بالضبط:
[{"full_name": "اسم المتهم كما هو مكتوب", "national_id": "الرقم القومي كما تراه"}]

إذا كان رقم أو حرف غير واضح، اكتب "؟" مكانه بدلاً من التخمين.
"""

# Omni: receives the heavily preprocessed, binarized black-on-white output
# of volume 1/2/3/5 (it took over this role from the retired "qwen" path),
# so the prompt tells it plainly what kind of image it's looking at, and is
# explicit about role distinction since it has no Arabic-document-specific
# fine-tuning the way AIN does.
OMNI_PROMPT = """هذه صورة معالجة (أبيض وأسود) لصفحة من مستند قضائي بعد معالجة رقمية لتحسين وضوح النص. اقرأ النص بعناية واستخرج بيانات كل "متهم" فقط — الشخص الموجه إليه الاتهام في القضية. لا تدرج القضاة، المحامين، الشهود، أو الموظفين حتى لو ظهرت أسماؤهم بشكل بارز.

أخرج النتيجة بصيغة JSON فقط، بدون أي نص آخر، بهذا الشكل بالضبط:
[{"full_name": "اسم المتهم كما هو مكتوب", "national_id": "الرقم القومي كما تراه"}]

إذا كان رقم أو حرف غير واضح، اكتب "؟" مكانه بدلاً من التخمين.
"""

# Qwen3.8: a peer extractor alongside ain/omni (not an escalation/adjudicator
# — decided explicitly, see conversation history), kept non-thinking and
# deterministic in models/qwen38_client.py. Receives the SAME lightly
# preprocessed (deskew+crop only, still color, not binarized) image as AIN,
# via ain_light.py, but gets no assumption of Arabic-document specialization
# since it's a general-purpose model — the prompt says plainly what kind of
# image this is, like Omni's does for its own (differently) preprocessed input.
QWEN38_PROMPT = """هذه صورة لصفحة من مستند قضائي، تم فقط تصحيح ميلها وقصها دون أي تعديل آخر على الألوان أو وضوح النص. اقرأ النص بعناية، بما في ذلك أي أجزاء بخط اليد أو باهتة، واستخرج بيانات كل "متهم" فقط — الشخص الموجه إليه الاتهام في القضية. لا تدرج القضاة، المحامين، الشهود، أو الموظفين حتى لو ظهرت أسماؤهم بشكل بارز.

أخرج النتيجة بصيغة JSON فقط، بدون أي نص آخر، بهذا الشكل بالضبط:
[{"full_name": "اسم المتهم كما هو مكتوب", "national_id": "الرقم القومي كما تراه"}]

إذا كان رقم أو حرف غير واضح، اكتب "؟" مكانه بدلاً من التخمين.
إذا لم يوجد أي متهم في الصفحة، أخرج: []
"""

# --- two-stage: read-then-extract ------------------------------------------
# The model first writes a short plain-Arabic description of what it sees
# (document type, how many defendants, anything damaged/unclear/handwritten),
# THEN, after RESULT_MARKER, the JSON array. Forcing the description first
# tends to reduce role-confusion on messy multi-person pages (a model asked
# for JSON cold can lock onto the first name-shaped text it sees), and the
# description itself becomes a human-readable audit trail alongside the
# JSON — useful for a reviewer, and for a compliance decision like an
# account freeze. Costs roughly double the output length/latency, so it is
# NOT applied everywhere — see run_pipeline.py for per-path/per-volume routing.
RESULT_MARKER = "النتيجة:"

AIN_PROMPT_TWO_STAGE = f"""اقرأ هذه الصفحة بالكامل أولاً بعناية، بما في ذلك أي خط يد أو نص باهت أو غير واضح.

الخطوة الأولى: اكتب وصفاً موجزاً (سطرين كحد أقصى) لما تراه — نوع المستند، عدد الأشخاص المتهمين المذكورين، وأي أجزاء غير واضحة أو تالفة أو مكتوبة بخط اليد.

الخطوة الثانية: بعد كلمة "{RESULT_MARKER}" اكتب فقط مصفوفة JSON بهذا الشكل، بدون أي نص آخر بعدها:
[{{"full_name": "اسم المتهم كما هو مكتوب", "national_id": "الرقم القومي كما تراه"}}]

لا تدرج القضاة أو المحامين أو الموظفين. إذا كان رقم أو حرف غير واضح اكتب "؟" مكانه بدلاً من التخمين.
إذا لم يوجد أي متهم في الصفحة، اكتب بعد "{RESULT_MARKER}": []
"""

OMNI_PROMPT_TWO_STAGE = f"""هذه صورة معالجة (أبيض وأسود) لصفحة من مستند قضائي بعد معالجة رقمية لتحسين وضوح النص. اقرأها بالكامل أولاً بعناية.

الخطوة الأولى: اكتب وصفاً موجزاً (سطرين كحد أقصى) لما تراه — نوع المستند، عدد الأشخاص المتهمين المذكورين، وأي أجزاء غير واضحة رغم المعالجة.

الخطوة الثانية: بعد كلمة "{RESULT_MARKER}" اكتب فقط مصفوفة JSON بهذا الشكل، بدون أي نص آخر بعدها:
[{{"full_name": "اسم المتهم كما هو مكتوب", "national_id": "الرقم القومي كما تراه"}}]

لا تدرج القضاة، المحامين، الشهود، أو الموظفين حتى لو ظهرت أسماؤهم بشكل بارز. إذا كان رقم أو حرف غير واضح اكتب "؟" مكانه بدلاً من التخمين.
إذا لم يوجد أي متهم في الصفحة، اكتب بعد "{RESULT_MARKER}": []
"""

# Every prompt run_pipeline.py might select, single- and two-stage together —
# both variants need to exist here since routing picks between them at runtime
# (AIN: always two-stage. Omni: two-stage only for volume3/volume5 pages.
# Qwen3.8: always single-stage, peer extractor, no two-stage variant).
PROMPTS = {
    "ain": AIN_PROMPT,
    "ain_two_stage": AIN_PROMPT_TWO_STAGE,
    "omni": OMNI_PROMPT,
    "omni_two_stage": OMNI_PROMPT_TWO_STAGE,
    "qwen3.8": QWEN38_PROMPT,
}