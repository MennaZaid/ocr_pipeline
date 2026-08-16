# \# Court-fraud OCR pipeline

# 

# Bank-internal tool: court documents naming fraud suspects come in as PDFs

# (scanned, sometimes handwritten, damaged, or badly photographed). This

# pipeline extracts each named defendant's full name and national ID so they

# can be checked against the bank's client list and flagged before funds move.

# 

# \*\*On-prem / no-cloud constraint\*\*: every model in this pipeline runs on

# infrastructure the bank controls. No path calls a hosted API on the public

# internet — see the qwen3.8 section below for how that's enforced in code.

# 

# ```

# PDF -> pdf\_to\_images.py -> loop over pages

# &#x20;                                 |

# &#x20;             +-------------------+-------------------+

# &#x20;             |                   |                   |

# &#x20;         PATH: ain           PATH: omni          PATH: qwen3.8

# &#x20;  preprocessors/ain\_light.py  pipeline\_selector.py  preprocessors/ain\_light.py

# &#x20;  (deskew + crop, NO           -> volume 1/2/3/5     (deskew + crop, NO

# &#x20;   binarization)                (external scripts)    binarization)

# &#x20;       |                          |                       |

# &#x20; models/ain\_client.py     models/omni\_client.py    models/qwen38\_client.py

# &#x20;       |                          |                       |

# &#x20; {page\_id}\_ain.json      {page\_id}\_omni.json      {page\_id}\_qwen38.json

# ```

# 

# Each page is written to disk once as a source image; each of the three

# paths reads from that same source image independently. \*\*You choose ONE

# path per run from the terminal\*\* (`--model ain|omni|qwen3.8|all`) — there

# is currently no automatic routing between paths and no automated

# cross-path comparison. See "Not built yet" below.

# 

# \## What changed this revision

# 

# The pipeline used to have four candidate paths: `ain`, `qwen` (plain

# Qwen2-VL-7B, tied to the volume 1/2/3/5 preprocessors), `omni`

# (Qwen2.5-Omni-7B), and `qwen3.8`. It's now \*\*three paths\*\*:

# 

# \- The standalone `qwen` (Qwen2-VL-7B-only) path has been \*\*retired\*\*.

# &#x20; `models/qwen\_client.py` has been \*\*deleted\*\*.

# \- \*\*`omni` now owns the volume 1/2/3/5 preprocessing pipeline\*\* that `qwen`

# &#x20; used to run through — same `pipeline\_selector.py` routing logic, same

# &#x20; volume scripts, just handed off to Omni at the model-call step instead of

# &#x20; Qwen2-VL.

# \- `ain` and `qwen3.8` are unchanged in shape: both use the light

# &#x20; `ain\_light.py` preprocessing (deskew + crop only, no binarization).

# \- Each path now has its \*\*own prompt\*\* (previously all three shared one

# &#x20; generic prompt) — see "Prompts" below.

# \- `dependencies.txt` replaces `requirements.txt`. The old special-case

# &#x20; `transformers` preview-branch install for Omni is gone — Qwen2.5-Omni

# &#x20; support has been in mainline `transformers` since 2025-04-14.

# 

# \## Files

# 

# \- `config.py` — paths to the volume 1/2/3/5 scripts, model ids for the

# &#x20; three active paths, and every prompt (single-stage and two-stage

# &#x20; variants). \*\*Check `VOLUME\_SCRIPTS` here first\*\* if your folder layout

# &#x20; differs from the delivered one.

# \- `pdf\_to\_images.py` — step 1. The only file that touches PyMuPDF/fitz.

# \- `preprocessors/ain\_light.py` — light preprocessing shared by the `ain`

# &#x20; and `qwen3.8` paths. Deskew + crop only — no binarization, no denoising,

# &#x20; no morphology. See its module docstring for why (AIN specifically reads

# &#x20; a heavily preprocessed page \*worse\* than an untouched one).

# \- `pipeline\_selector.py` — `omni`'s routing logic (`estimate\_quality`,

# &#x20; `choose\_pipeline`) into volume 1/2/3/5. Thresholds unvalidated against

# &#x20; labeled documents — see `out/routing\_report.jsonl` at runtime to check

# &#x20; and eventually tune them.

# \- `pipeline\_preprocessors/Volume 1|2|3|5/` — the four preprocessing tiers,

# &#x20; easiest to hardest. \*\*Volume 5 is deliberately not the default for

# &#x20; everything\*\* — it was tried on pages that didn't need that much

# &#x20; preprocessing and measurably hurt the output on those pages. The tiered

# &#x20; routing exists on purpose; don't collapse it to "always run volume 5."

# \- `models/vlm\_runner.py` — shared inference for `ain` (both load via

# &#x20; `Qwen2VLForConditionalGeneration`; AIN is a fine-tune of Qwen2-VL-7B).

# \- `models/ain\_client.py` — thin wrapper around `vlm\_runner.py`.

# \- `models/omni\_client.py` — separate loader (`Qwen2\_5OmniForConditionalGeneration`

# &#x20; — a different model class, cannot share `vlm\_runner.py`). Talker disabled,

# &#x20; text-only output.

# \- `models/qwen38\_client.py` — HTTP client for a \*\*self-hosted\*\*

# &#x20; OpenAI-compatible server (vLLM/SGLang/TokenSpeed). Requires

# &#x20; `QWEN38\_BASE\_URL` to be set to a server you control; raises immediately

# &#x20; if it isn't, rather than hanging. Never points at Qwen Cloud in this

# &#x20; deployment — see the on-prem constraint above.

# \- `extraction\_utils.py` — parses a model's raw text into

# &#x20; `{"description": str|None, "fields": \[...]}`. `description` is the

# &#x20; page-summary text from a two-stage prompt (`None` if the path used a

# &#x20; single-stage prompt). Each field carries `needs\_review: true` if it

# &#x20; contains the model's own uncertainty marker, `؟`.

# \- `run\_pipeline.py` — the orchestrator. Run this.

# 

# \## Prompts

# 

# Every path has its own prompt now, reflecting what it actually receives

# and what it's trained on — not one generic prompt shared across models:

# 

# | Path | Image it receives | Why the prompt differs |

# |---|---|---|

# | `ain` | Lightly preprocessed (deskew+crop, still color) | Fine-tuned on authentic Arabic documents/handwriting (CAMEL-Bench OCR: 72.35 vs Qwen2-VL-7B's 42.73) — prompt trusts it with faint/handwritten text rather than hedging |

# | `omni` | Heavily preprocessed, binarized black-on-white (volume 1/2/3/5) | No Arabic-document-specific fine-tuning — prompt states plainly what kind of image this is and is explicit about role distinctions |

# | `qwen3.8` | Lightly preprocessed (deskew+crop, still color) — same as `ain` | General-purpose model, no Arabic-document specialization assumed — prompt states plainly what kind of image this is, same spirit as `omni`'s |

# 

# \*\*Two-stage prompting\*\* (`config.PROMPTS\["ain\_two\_stage"]` /

# `\["omni\_two\_stage"]`): the model writes a short plain-Arabic description of

# what it sees (document type, number of defendants, anything

# damaged/unclear/handwritten) \*before\* the JSON, split by `RESULT\_MARKER`.

# This tends to reduce role-confusion on messy multi-person pages (a model

# asked for JSON cold can lock onto the first name-shaped text it finds), and

# the description becomes a human-readable note alongside the JSON — useful

# given this feeds an account-freeze decision that may get reviewed later.

# Costs roughly double the output length/latency, so it's \*\*not\*\* applied

# everywhere:

# 

# \- `ain` → \*\*always\*\* two-stage (most likely path to see handwriting/damage;

# &#x20; has no evidence-report signal the way volume 5 does)

# \- `omni` → two-stage \*\*only\*\* when `pipeline\_selector.choose\_pipeline`

# &#x20; routed the page to `volume3` or `volume5`; volume1/volume2 pages stay

# &#x20; single-stage for speed

# \- `qwen3.8` → \*\*always\*\* single-stage — it's a peer extractor alongside

# &#x20; `ain`/`omni`, not a special-cased reasoning path (decided explicitly:

# &#x20; thinking mode stays off, temperature stays low/deterministic in

# &#x20; `qwen38\_client.py`)

# 

# \## Running it

# 

# ```bash

# \# one path

# python run\_pipeline.py --input case.pdf --output out --model ain

# python run\_pipeline.py --input case.pdf --output out --model omni

# python run\_pipeline.py --input case.pdf --output out --model qwen3.8

# 

# \# all three (still independent, still no automated comparison — see below)

# python run\_pipeline.py --input case.pdf --output out --model all

# ```

# 

# `qwen3.8` requires a self-hosted server first:

# ```bash

# vllm serve Qwen/Qwen3.8-27B --port 8000

# export QWEN38\_BASE\_URL=http://localhost:8000/v1

# ```

# Confirm with infra/your supervisor that the box running this has adequate

# GPU memory before depending on this path in production — Qwen3.8-27B is a

# 27B dense model and does not fit the same modest single-GPU budget the

# `ain`/`omni` 7B-class paths were sized for. This is not yet confirmed.

# 

# Every page gets its own folder `out/<doc>\_work/<page\_id>/` containing the

# source image and the path-specific preprocessed image.

# `out/<doc>\_text\_<model>/<page\_id>.json` holds that path's raw output, parsed

# fields, and (if two-stage) the description text.

# `out/routing\_report.jsonl` logs every page: which volume `omni` used (if

# run), plus every model's output for that page.

# 

# \## Reading the output — the `؟` marker

# 

# Any field containing `؟` means the model could not read that character

# confidently rather than guessing — `extraction\_utils.py` surfaces this as

# `needs\_review: true` on that field. \*\*There is currently no automated

# pass/fail gate on this flag.\*\* Treat it as a manual review signal until the

# review workflow (see below) is finalized.

# 

# \## Not built yet

# 

# \- \*\*No automated cross-path matcher/consensus step.\*\* Path selection is

# &#x20; manual (`--model`), and running `--model all` gives you three independent

# &#x20; JSON outputs per page with no automated agreement scoring. When this gets

# &#x20; built: national ID comparison must be \*\*exact match only, never fuzzy\*\*

# &#x20; (a fuzzy/percentage score on a unique-key digit string risks matching the

# &#x20; wrong person); `needs\_review: true` on \*either\* path's output for a page

# &#x20; should hard-block auto-pass regardless of any match score. \*\*On hold\*\*

# &#x20; pending a design conversation with a bank supervisor.

# \- \*\*Human review workflow.\*\* There will be a human in the loop, but the

# &#x20; process (who reviews, on what trigger, what they see) isn't designed yet.

# &#x20; \*\*On hold\*\*, same conversation as above.

# \- \*\*Qwen3.8's on-prem GPU sizing is unconfirmed.\*\* Self-hosting is

# &#x20; correctly wired in code (no cloud fallback), but whether the bank has

# &#x20; hardware that can actually hold a 27B model hasn't been confirmed.

# &#x20; \*\*On hold\*\*, pending supervisor conversation.

# \- \*\*Volumes 1–3 have no NULL-vs-NEEDS-REVIEW evidence report\*\* the way

# &#x20; volume 5 does (`faint\_safe.py`'s `evidence\_report()`). A blank field from

# &#x20; a volume 1/2/3 page currently can't be distinguished from a genuinely

# &#x20; blank source field. Recommended fix (not yet applied): swap

# &#x20; `remove\_border\_junk` for `remove\_border\_junk\_safe` in volumes 1–3 (the

# &#x20; unguarded version can delete a component carrying a large share of a

# &#x20; page's real ink if it happens to touch the page edge), and call

# &#x20; `evidence\_report()` once at the end of each volume's `process()` as a

# &#x20; diagnostic pass — no change to binarization itself.

# \- \*\*AIN's OCR benchmark edge is general-domain\*\*, not validated against

# &#x20; this bank's actual documents (Egyptian court scans specifically; AIN's

# &#x20; training data is only 35% authentic Arabic, the rest translated/synthetic).

# &#x20; Recommend an empirical spot-check across all three paths on a small

# &#x20; labeled batch of real case pages before trusting any one path's output

# &#x20; over the others'.

# 

# \## Hardware / install

# 

# See `dependencies.txt` for the full ordered install list, and

# `how\_to\_run.txt` for step-by-step usage including the `qwen3.8` server

# setup. Model weights (`AIN\_MODEL\_ID`, `OMNI\_MODEL\_ID`, `QWEN\_MODEL\_ID`)

# should point at local/offline weight directories you've downloaded ahead of

# time, not Hugging Face Hub ids — per the on-prem/no-cloud constraint,

# inference should never need network access at runtime.

