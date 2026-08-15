# -*- coding: utf-8 -*-
"""
config.py — single place for paths and defaults used across the repo.

Three active paths as of this revision: ain, omni, qwen3.8.
The original Qwen2-VL-only path ("qwen") has been retired in favor of
Omni taking over the volume 1/2/3/5 preprocessing pipeline. models/qwen_client.py
and QWEN_MODEL_ID are left in the repo but unused by run_pipeline.py —
delete both if you don't want them kept around as dead code.

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
# AIN and the old "qwen" path both load with Qwen2VLForConditionalGeneration
# (AIN is a fine-tune of Qwen2-VL-7B) — kept for ain_client.py / vlm_runner.py.
QWEN_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"   # unused by run_pipeline.py now; kept for models/qwen_client.py if you keep that file
AIN_MODEL_ID = "MBZUAI/AIN"

# Omni loads via a DIFFERENT class (Qwen2_5OmniForConditionalGeneration, see
# models/omni_client.py) — this was previously missing from config.py, which
# is why the omni path crashed with ImportError.
OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

# Qwen3.8 is not a local transformers model — it's called over an
# OpenAI-compatible HTTP API against a server YOU run (vLLM/SGLang/etc.) or,
# once available, Qwen Cloud. Also previously missing — models/qwen38_client.py
# imports all three of these directly and will ImportError without them.
QWEN38_MODEL_ID = "Qwen/Qwen3.8-27B"
QWEN38_BASE_URL = os.environ.get("QWEN38_BASE_URL", "")   # e.g. http://localhost:8000/v1 — empty = qwen38_client raises before any request
QWEN38_API_KEY = os.environ.get("QWEN38_API_KEY", "not-needed")

# Point these at local/offline weight directories for the on-prem/no-cloud
# constraint — do NOT rely on hitting the HF Hub at runtime.
QWEN_MODEL_ID = QWEN_MODEL_ID
AIN_MODEL_ID = AIN_MODEL_ID
OMNI_MODEL_ID = OMNI_MODEL_ID

# --- prompts -----------------------------------------------------------
EXTRACTION_PROMPT = """اقرأ هذه الصفحة من مستند قضائي، واستخرج بيانات كل شخص متهم فقط (وليس القضاة أو المحامين أو الموظفين).

أخرج النتيجة بصيغة JSON فقط، بدون أي نص آخر، بهذا الشكل بالضبط:
[{"full_name": "اسم المتهم كما هو مكتوب", "national_id": "الرقم القومي كما تراه"}]

مثال:
[{"full_name": "أحمد محمد علي", "national_id": "29001011234567"}]

إذا كان رقم أو حرف غير واضح في الصورة، اكتب "؟" مكانه بدلاً من التخمين.
إذا لم يوجد أي متهم في الصفحة، أخرج: []
"""

AIN_PROMPT = """اقرأ هذه الصفحة من مستند قضائي بعناية، بما في ذلك أي خط يد أو نص باهت أو غير واضح. استخرج بيانات كل شخص متهم فقط (وليس القضاة أو المحامين أو الموظفين).

أخرج النتيجة بصيغة JSON فقط، بدون أي نص آخر، بهذا الشكل بالضبط:
[{"full_name": "اسم المتهم كما هو مكتوب", "national_id": "الرقم القومي كما تراه"}]

إذا كان رقم أو حرف غير واضح، اكتب "؟" مكانه بدلاً من التخمين.
"""

OMNI_PROMPT = """هذه صورة معالجة (أبيض وأسود) لصفحة من مستند قضائي بعد معالجة رقمية لتحسين وضوح النص. اقرأ النص بعناية واستخرج بيانات كل "متهم" فقط — الشخص الموجه إليه الاتهام في القضية. لا تدرج القضاة، المحامين، الشهود، أو الموظفين حتى لو ظهرت أسماؤهم بشكل بارز.

أخرج النتيجة بصيغة JSON فقط، بدون أي نص آخر، بهذا الشكل بالضبط:
[{"full_name": "اسم المتهم كما هو مكتوب", "national_id": "الرقم القومي كما تراه"}]

إذا كان رقم أو حرف غير واضح، اكتب "؟" مكانه بدلاً من التخمين.
"""

QWEN38_PROMPT = """هذه صورة لصفحة من مستند قضائي، تم فقط تصحيح ميلها وقصها دون أي تعديل آخر على الألوان أو وضوح النص. اقرأ النص بعناية، بما في ذلك أي أجزاء بخط اليد أو باهتة، واستخرج بيانات كل "متهم" فقط — الشخص الموجه إليه الاتهام في القضية. لا تدرج القضاة، المحامين، الشهود، أو الموظفين حتى لو ظهرت أسماؤهم بشكل بارز.

أخرج النتيجة بصيغة JSON فقط، بدون أي نص آخر، بهذا الشكل بالضبط:
[{"full_name": "اسم المتهم كما هو مكتوب", "national_id": "الرقم القومي كما تراه"}]

إذا كان رقم أو حرف غير واضح، اكتب "؟" مكانه بدلاً من التخمين.
إذا لم يوجد أي متهم في الصفحة، أخرج: []
"""
PROMPTS = {
    "ain": AIN_PROMPT,
    "omni": OMNI_PROMPT,
    "qwen3.8": QWEN38_PROMPT,
}

# --- alternative: read-then-extract ----------------------------------------
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