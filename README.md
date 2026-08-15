# Court-fraud OCR pipeline

PDF -> per-page images -> loop -> each page forks into two independent paths:

```
PDF -> pdf_to_images.py -> loop over pages
                                  |
                +-----------------+-----------------+
                |                                   |
          PATH 1 (AIN)                        PATH 2 (Qwen)
    preprocessors/ain_light.py          pipeline_selector.py
    (deskew + crop, NO binarization)    -> volume 1/2/3/5 (external scripts)
                |                                   |
       models/ain_client.py                models/qwen_client.py
                |                                   |
        {page_id}_ain.json                  {page_id}_qwen.json
```

Both paths start from the **same** source page image, written once, and run
independently — path 2 never consumes path 1's output. Matching/consensus
between the two JSON outputs is the next piece to build, not included here.

## Files

- `config.py` — paths to your existing volume 1/2/3/5 scripts, model ids,
  defaults. **Check `VOLUME_SCRIPTS` here first** — it's a guess at your
  folder layout, not verified.
- `pdf_to_images.py` — step 1. The only file that touches PyMuPDF/fitz.
- `preprocessors/ain_light.py` — path 1's preprocessing. Deskew + crop only.
  No binarization, no denoising, no morphology — see module docstring for why.
- `pipeline_selector.py` — path 2's routing logic (`estimate_quality`,
  `choose_pipeline`). Pure functions, thresholds unchanged from the version
  already discussed — **not recalibrated**, see below.
- `models/vlm_runner.py` — one shared inference function for both models,
  since AIN and Qwen2-VL both load via `Qwen2VLForConditionalGeneration`.
- `models/qwen_client.py`, `models/ain_client.py` — thin wrappers around
  `vlm_runner` with each model's id and default prompt.
- `run_pipeline.py` — the orchestrator. Run this.

## Running it

```bash
# preprocessing only, no GPU/model weights needed — good for a first dry run
python run_pipeline.py case.pdf --outdir out --skip-inference

# full run, both paths, both models
python run_pipeline.py case.pdf --outdir out

# Qwen path only
python run_pipeline.py case.pdf --outdir out --no-ain

# AIN path only
python run_pipeline.py case.pdf --outdir out --no-qwen
```

Every page gets its own folder `out/<doc>_p<NNN>/` containing the source
image, both preprocessed images, and both model outputs as JSON.
`out/routing_report.jsonl` logs every page's measured quality metrics and
which volume it was routed to — use this to sanity-check and eventually tune
`pipeline_selector.py`'s thresholds against real documents.

## Things that still need your input before this runs end to end

1. **`VOLUME_SCRIPTS` in `config.py`** — point these at wherever your actual
   volume 1/2/3/5 scripts live.
2. **`SHARED_MODULE_DIR` in `config.py`** — must point at the folder
   containing `ocr_preprocess_v2.py`, since `ain_light.py` and the volume
   scripts both import shared measurement functions from it.
3. **Model weights** — for the on-prem/no-cloud constraint, `QWEN_MODEL_ID`
   and `AIN_MODEL_ID` in `config.py` should point at local weight directories
   you've downloaded ahead of time, not Hugging Face Hub ids, so inference
   never needs network access.
4. **`run_volume_script`'s output glob** in `run_pipeline.py` — it currently
   assumes the volume scripts write `<stem>*.png` into the given `--outdir`.
   Adjust if your scripts' actual output naming differs.
5. **Hardware** — Qwen2-VL-7B and AIN-7B are both 7B-parameter models. Running
   both per page on a single GPU (e.g. RTX 3060, 12GB) will likely need
   quantized weights or sequential (not concurrent) loading — `vlm_runner.py`
   caches each model once it's loaded, but doesn't address VRAM budget across
   both models being resident at once.

## Not built yet

- The matcher/consensus step that compares `{page_id}_ain.json` and
  `{page_id}_qwen.json` and decides auto-pass vs. human review.
- Threshold calibration for `pipeline_selector.py` against labeled documents.
