# Court-fraud OCR pipeline

Bank-internal tool: court documents naming fraud suspects come in as PDFs
(scanned, sometimes handwritten, damaged, or badly photographed). This
pipeline extracts each named defendant's full name and national ID so they
can be checked against the bank's client list and flagged before funds move.

**On-prem / no-cloud constraint**: every model in this pipeline runs on
infrastructure the bank controls. No path calls a hosted API on the public
internet — see the qwen3.8 section below for how that's enforced in code.

```
PDF -> pdf_to_images.py -> loop over pages
                                  |
              +-------------------+-------------------+
              |                   |                   |
          PATH: ain           PATH: omni          PATH: qwen3.8
   preprocessors/ain_light.py  pipeline_selector.py  preprocessors/ain_light.py
   (deskew + crop, NO           -> volume 1/2/3/5     (deskew + crop, NO
    binarization)                (external scripts)    binarization)
        |                          |                       |
  models/ain_client.py     models/omni_client.py    models/qwen38_client.py
        |                          |                       |
  {page_id}_ain.json      {page_id}_omni.json      {page_id}_qwen38.json
```

Each page is written to disk once as a source image; each of the three
paths reads from that same source image independently. **You choose ONE
path per run from the terminal** (`--model ain|omni|qwen3.8|all`) — there
is currently no automatic routing between paths and no automated
cross-path comparison. See "Not built yet" below.

## What changed this revision

The pipeline used to have four candidate paths: `ain`, `qwen` (plain
Qwen2-VL-7B, tied to the volume 1/2/3/5 preprocessors), `omni`
(Qwen2.5-Omni-7B), and `qwen3.8`. It's now **three paths**:

- The standalone `qwen` (Qwen2-VL-7B-only) path has been **retired**.
  `models/qwen_client.py` has been **deleted**.
- **`omni` now owns the volume 1/2/3/5 preprocessing pipeline** that `qwen`
  used to run through — same `pipeline_selector.py` routing logic, same
  volume scripts, just handed off to Omni at the model-call step instead of
  Qwen2-VL.
- `ain` and `qwen3.8` are unchanged in shape: both use the light
  `ain_light.py` preprocessing (deskew + crop only, no binarization).
- Each path now has its **own prompt** (previously all three shared one
  generic prompt) — see "Prompts" below.
- `dependencies.txt` replaces `requirements.txt`. The old special-case
  `transformers` preview-branch install for Omni is gone — Qwen2.5-Omni
  support has been in mainline `transformers` since 2025-04-14.

## Files

- `config.py` — paths to the volume 1/2/3/5 scripts, model ids for the
  three active paths, and every prompt (single-stage and two-stage
  variants). **Check `VOLUME_SCRIPTS` here first** if your folder layout
  differs from the delivered one.
- `pdf_to_images.py` — step 1. The only file that touches PyMuPDF/fitz.
- `preprocessors/ain_light.py` — light preprocessing shared by the `ain`
  and `qwen3.8` paths. Deskew + crop only — no binarization, no denoising,
  no morphology. See its module docstring for why (AIN specifically reads
  a heavily preprocessed page *worse* than an untouched one).
- `pipeline_selector.py` — `omni`'s routing logic (`estimate_quality`,
  `choose_pipeline`) into volume 1/2/3/5. Thresholds unvalidated against
  labeled documents — see `out/routing_report.jsonl` at runtime to check
  and eventually tune them.
- `pipeline_preprocessors/Volume 1|2|3|5/` — the four preprocessing tiers,
  easiest to hardest. **Volume 5 is deliberately not the default for
  everything** — it was tried on pages that didn't need that much
  preprocessing and measurably hurt the output on those pages. The tiered
  routing exists on purpose; don't collapse it to "always run volume 5."
- `models/vlm_runner.py` — shared inference for `ain` (both load via
  `Qwen2VLForConditionalGeneration`; AIN is a fine-tune of Qwen2-VL-7B).
- `models/ain_client.py` — thin wrapper around `vlm_runner.py`.
- `models/omni_client.py` — separate loader (`Qwen2_5OmniForConditionalGeneration`
  — a different model class, cannot share `vlm_runner.py`). Talker disabled,
  text-only output.
- `models/qwen38_client.py` — HTTP client for a **self-hosted**
  OpenAI-compatible server (vLLM/SGLang/TokenSpeed). Requires
  `QWEN38_BASE_URL` to be set to a server you control; raises immediately
  if it isn't, rather than hanging. Never points at Qwen Cloud in this
  deployment — see the on-prem constraint above.
- `extraction_utils.py` — parses a model's raw text into
  `{"description": str|None, "fields": [...]}`. `description` is the
  page-summary text from a two-stage prompt (`None` if the path used a
  single-stage prompt). Each field carries `needs_review: true` if it
  contains the model's own uncertainty marker, `؟`.
- `run_pipeline.py` — the orchestrator. Run this.

## Prompts

Every path has its own prompt now, reflecting what it actually receives
and what it's trained on — not one generic prompt shared across models:

| Path | Image it receives | Why the prompt differs |
|---|---|---|
| `ain` | Lightly preprocessed (deskew+crop, still color) | Fine-tuned on authentic Arabic documents/handwriting (CAMEL-Bench OCR: 72.35 vs Qwen2-VL-7B's 42.73) — prompt trusts it with faint/handwritten text rather than hedging |
| `omni` | Heavily preprocessed, binarized black-on-white (volume 1/2/3/5) | No Arabic-document-specific fine-tuning — prompt states plainly what kind of image this is and is explicit about role distinctions |
| `qwen3.8` | Lightly preprocessed (deskew+crop, still color) — same as `ain` | General-purpose model, no Arabic-document specialization assumed — prompt states plainly what kind of image this is, same spirit as `omni`'s |

**Two-stage prompting** (`config.PROMPTS["ain_two_stage"]` /
`["omni_two_stage"]`): the model writes a short plain-Arabic description of
what it sees (document type, number of defendants, anything
damaged/unclear/handwritten) *before* the JSON, split by `RESULT_MARKER`.
This tends to reduce role-confusion on messy multi-person pages (a model
asked for JSON cold can lock onto the first name-shaped text it finds), and
the description becomes a human-readable note alongside the JSON — useful
given this feeds an account-freeze decision that may get reviewed later.
Costs roughly double the output length/latency, so it's **not** applied
everywhere:

- `ain` → **always** two-stage (most likely path to see handwriting/damage;
  has no evidence-report signal the way volume 5 does)
- `omni` → two-stage **only** when `pipeline_selector.choose_pipeline`
  routed the page to `volume3` or `volume5`; volume1/volume2 pages stay
  single-stage for speed
- `qwen3.8` → **always** single-stage — it's a peer extractor alongside
  `ain`/`omni`, not a special-cased reasoning path (decided explicitly:
  thinking mode stays off, temperature stays low/deterministic in
  `qwen38_client.py`)

## Running it

```bash
# one path
python run_pipeline.py --input case.pdf --output out --model ain
python run_pipeline.py --input case.pdf --output out --model omni
python run_pipeline.py --input case.pdf --output out --model qwen3.8

# all three (still independent, still no automated comparison — see below)
python run_pipeline.py --input case.pdf --output out --model all
```

`qwen3.8` requires a self-hosted server first:
```bash
vllm serve Qwen/Qwen3.8-27B --port 8000
export QWEN38_BASE_URL=http://localhost:8000/v1
```
Confirm with infra/your supervisor that the box running this has adequate
GPU memory before depending on this path in production — Qwen3.8-27B is a
27B dense model and does not fit the same modest single-GPU budget the
`ain`/`omni` 7B-class paths were sized for. This is not yet confirmed.

Every page gets its own folder `out/<doc>_work/<page_id>/` containing the
source image and the path-specific preprocessed image.
`out/<doc>_text_<model>/<page_id>.json` holds that path's raw output, parsed
fields, and (if two-stage) the description text.
`out/routing_report.jsonl` logs every page: which volume `omni` used (if
run), plus every model's output for that page.

## Reading the output — the `؟` marker

Any field containing `؟` means the model could not read that character
confidently rather than guessing — `extraction_utils.py` surfaces this as
`needs_review: true` on that field. **There is currently no automated
pass/fail gate on this flag.** Treat it as a manual review signal until the
review workflow (see below) is finalized.

## Not built yet

- **No automated cross-path matcher/consensus step.** Path selection is
  manual (`--model`), and running `--model all` gives you three independent
  JSON outputs per page with no automated agreement scoring. When this gets
  built: national ID comparison must be **exact match only, never fuzzy**
  (a fuzzy/percentage score on a unique-key digit string risks matching the
  wrong person); `needs_review: true` on *either* path's output for a page
  should hard-block auto-pass regardless of any match score. **On hold**
  pending a design conversation with a bank supervisor.
- **Human review workflow.** There will be a human in the loop, but the
  process (who reviews, on what trigger, what they see) isn't designed yet.
  **On hold**, same conversation as above.
- **Qwen3.8's on-prem GPU sizing is unconfirmed.** Self-hosting is
  correctly wired in code (no cloud fallback), but whether the bank has
  hardware that can actually hold a 27B model hasn't been confirmed.
  **On hold**, pending supervisor conversation.
- **Volumes 1–3 have no NULL-vs-NEEDS-REVIEW evidence report** the way
  volume 5 does (`faint_safe.py`'s `evidence_report()`). A blank field from
  a volume 1/2/3 page currently can't be distinguished from a genuinely
  blank source field. Recommended fix (not yet applied): swap
  `remove_border_junk` for `remove_border_junk_safe` in volumes 1–3 (the
  unguarded version can delete a component carrying a large share of a
  page's real ink if it happens to touch the page edge), and call
  `evidence_report()` once at the end of each volume's `process()` as a
  diagnostic pass — no change to binarization itself.
- **AIN's OCR benchmark edge is general-domain**, not validated against
  this bank's actual documents (Egyptian court scans specifically; AIN's
  training data is only 35% authentic Arabic, the rest translated/synthetic).
  Recommend an empirical spot-check across all three paths on a small
  labeled batch of real case pages before trusting any one path's output
  over the others'.

## Hardware / install

See `dependencies.txt` for the full ordered install list, and
`how_to_run.txt` for step-by-step usage including the `qwen3.8` server
setup. Model weights (`AIN_MODEL_ID`, `OMNI_MODEL_ID`, `QWEN_MODEL_ID`)
should point at local/offline weight directories you've downloaded ahead of
time, not Hugging Face Hub ids — per the on-prem/no-cloud constraint,
inference should never need network access at runtime.
