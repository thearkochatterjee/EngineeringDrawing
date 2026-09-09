# Local engineering-drawing copilot

This workflow combines the primitive detector with structured facts and a
local Ollama model.  The language model does not read arbitrary files or
invent drawing data: it can only call a small set of read-only tools over
`drawing_facts.json`.

The evidence stays separate:

| Evidence | Meaning |
| --- | --- |
| Detector primitives | Geometry predicted from the image, with model confidence. |
| ParaCAD labels | Evaluation-only ground truth. It is never presented as information extracted from a new drawing. |
| OCR text | Unverified text candidates. An engineer must confirm units, tolerances, title-block fields, and callouts. |

Image distances are not physical dimensions.  The pipeline will report both
normalized and pixel-space values, but will call out that calibration or a
verified dimension is necessary to establish physical units.

## Streamlit workbench

The workbench defaults to five focused tabs: **Analyze**, **Copilot**,
**Review**, **Results**, and **Model**. It automatically finds the preferred
local Zarr archive and detector checkpoint, uses CUDA when PyTorch can use it,
selects a high-throughput but bounded worker count, enables OCR only when Tesseract is
installed, and keeps vision evidence to one 512-pixel image for an 8 GB GPU.

In **Analyze**, select one paged ParaCAD record/image-ID or upload a drawing,
then use **Analyze drawing**. Prediction, facts, review checks, overlay, and
a downloadable JSON package are created together under `workbench\runs`.

**Results** keeps job output, predictions, facts, evaluation metrics, labels,
and logs in one place. Full CLI operations, training, skills, and approved
knowledge controls remain available under **Model → Advanced operations,
training, skills, and knowledge**; long jobs are guarded and logged under
`workbench\jobs`. The Training panel lists total, train, validation, and test
sample counts for each local `.zarr` dataset before you start a job.

```powershell
.\setup.ps1 -Profile workbench
uv run streamlit run .\streamlit_app.py
```

Start it from this repository directory. The app writes user-created outputs
under `workbench\` by default; long GPU jobs require an explicit confirmation.

## CLI progress

The training, evaluation, preprocessing, Zarr-building, feature-extraction,
review-data, knowledge, overlay, and Copilot commands show Rich progress on
stderr by default. This keeps JSON/JSONL stdout safe to redirect. Add
`--no-progress` for plain CI/log output (`build_paracad_zarr.py` also retains
its older `--no-rich-progress` spelling).

## Engineering benchmark system

`benchmark_engineering.py` is the project regression harness. It keeps
deterministic facts-tool checks separate from local-LLM response checks, so a
model answer cannot hide an extraction or evidence-layer defect. Cases are
versioned JSONL and can cover geometry retrieval, dimensions, topology,
calculation safety, uncertainty, citations, engineering-review rules, latency,
and future feature/calculation tools. Engineer-reviewed cases are the source
of truth for pass criteria.

Create a starter suite from an existing analysis, then inspect and extend the
JSONL with reviewed scenarios:

```powershell
python .\benchmark_engineering.py init --facts .\workbench\runs\<run>\drawing_facts.json --output .\benchmarks\engineering_cases.jsonl
python .\benchmark_engineering.py validate --cases .\benchmarks\engineering_cases.jsonl
```

Run the deterministic layer first. It is fast, does not contact Ollama, and
is appropriate for routine regression testing:

```powershell
python .\benchmark_engineering.py run --cases .\benchmarks\engineering_cases.jsonl --mode tool --fail-on-failure
```

Run the complete grounded-copilot suite once the local model is available:

```powershell
python .\benchmark_engineering.py run --cases .\benchmarks\engineering_cases.jsonl --mode all --model qwen3.5:9b --fail-on-failure
```

Add held-out detector accuracy to the same report and prevent score drops
relative to an earlier report. Use conservative thresholds that have been
approved for the intended application:

```powershell
python .\benchmark_engineering.py run --cases .\benchmarks\engineering_cases.jsonl --mode tool --checkpoint .\checkpoints_full\best.pt --dataset .\ParaCAD_full_v3.zarr --detector-min-f1 0.20 --baseline .\benchmark_reports\engineering_benchmark_previous.json --max-regression 0.01 --fail-on-failure --device cuda
```

Every run writes JSON and Markdown reports under `benchmark_reports\` by
default. The **Model → Advanced operations → Advanced CLI runner** in
Streamlit also lists the engineering benchmark command.

## Run the pipeline on a drawing

In PowerShell, first predict primitives from the original image:

```powershell
$image = ".\data\ParaCAD\data\ParaCAD\dxfs_color1_pngs\1897-2_1_0.png"
python .\paracad_primitive_detr.py predict --checkpoint .\checkpoints\best.pt --image $image --output .\predicted_primitives.json --device cuda
```

Build the engineer-facing facts file.  OCR is optional: the command still
works if Tesseract is not installed, and records that OCR was skipped.

```powershell
python .\build_drawing_facts.py --image $image --prediction .\predicted_primitives.json --output .\drawing_facts.json
```

For a ParaCAD dataset drawing, add its JSONL split to compare prediction to
ground truth.  This is evaluation data only.

```powershell
python .\build_drawing_facts.py --image $image --prediction .\predicted_primitives.json --ground-truth-jsonl .\ParaCAD_processed_v2\train.jsonl --output .\drawing_facts.json
```

To add an image-space hole-clearance review threshold, use a normalized image
value only as a screening rule—it is not a physical manufacturing distance:

```powershell
python .\build_drawing_facts.py --image $image --prediction .\predicted_primitives.json --min-hole-edge-spacing 0.01 --output .\drawing_facts.json
```

## Query Qwen through Ollama

Start the Ollama service in the environment where Ollama is installed, then
confirm the exact downloaded model tag.  This script uses the HTTP service, so
Ollama does not need to be present on the Python process's PATH.

```powershell
python .\drawing_copilot.py --check-ollama
```

Ask one question with the model tag the previous command displays (the
requested default is `qwen3.5:9b`):

```powershell
python .\drawing_copilot.py --facts .\drawing_facts.json --model qwen3.5:9b --question "Summarize the detected geometry, then list the review risks."
```

Or start an interactive engineer-review session:

```powershell
python .\drawing_copilot.py --facts .\drawing_facts.json --model qwen3.5:9b --interactive
```

### Vision-assisted verification

The installed `qwen3.5:9b` model supports both tools and vision. Attach the
original drawing plus detector context when visual inspection would help. On
an 8 GB GPU, begin with one 512-pixel image and a 4K context; increase image
count only after confirming VRAM headroom.

```powershell
python .\drawing_copilot.py --facts .\drawing_facts.json --vision-crops auto --vision-max-images 1 --vision-max-side 512 --num-ctx 4096 --interactive
```

`--vision-crops auto` can attach the original image, a magenta detector overlay,
and a focused crop. The model must still use facts tools for feature IDs,
scores, dimensions, and measurements; vision is verification context rather
than a replacement for structured geometry.

### Approved standards and process retrieval

Put approved, revision-controlled text copies of standards, process notes, and
company drawing rules in a dedicated folder. Build a local inspectable index:

```powershell
python .\build_knowledge_index.py --source-dir .\approved_engineering_references --output .\engineering_knowledge.json
python .\drawing_copilot.py --facts .\drawing_facts.json --knowledge-index .\engineering_knowledge.json --question "What approved references apply to hole spacing?"
```

The index accepts `.txt`, `.md`, `.csv`, and `.json` files. Convert PDFs and
DOCX documents to reviewed text first. Responses cite `[K#]` chunks and must
still verify reference revision and applicability.

Examples of useful questions:

- `Which circles look like a repeated hole pattern, and which feature IDs support that?`
- `What predicted geometry has the lowest confidence and should be visually checked?`
- `Are there symmetry proposals? Explain their confidence and supporting features.`
- `List the ground-truth dimensions and constraints.` (dataset evaluation only)
- `What is the image-space center distance between P12 and P54?`

## What the copilot can do

The model can retrieve individual primitives, candidate holes, dimensions,
constraints, OCR text, title-block candidates, topology components, symmetry
proposals, repeated-hole groups, and deterministic review checks.  Its review
checks flag low-confidence primitives, very short lines, near-duplicate
detections, overlapping circle envelopes, and missing closed line profiles.
They are prompts for review, not automated design approval.

## Reusable engineering skills

The copilot also has a Hermes-style persistent skill library. A skill is a
versioned, inspectable JSON workflow that combines instructions with a fixed
sequence of the existing facts tools. It is deliberately not executable code:
skills cannot run commands, reach the network, open arbitrary files, alter a
drawing, or add a new source of evidence.

Four built-in skills are available: `drawing-intake-review`,
`hole-pattern-review`, `geometry-confidence-review`, and
`dataset-evaluation-review`.

```powershell
# Inspect all built-in and locally saved workflows.
python .\drawing_copilot.py --list-skills

# Run a workflow deterministically without asking the language model.
python .\drawing_copilot.py --facts .\drawing_facts.json --run-skill drawing-intake-review
```

In an interactive session, ask the model to *draft* a skill for a recurring
review workflow. It will show the fixed plan first. To explicitly permit the
model to save a reviewed draft in `engineering_skills\`, start it with:

```powershell
python .\drawing_copilot.py --facts .\drawing_facts.json --model qwen3.5:9b --interactive --allow-skill-write
```

Existing local skills cannot be overwritten unless you explicitly add
`--allow-skill-replace`. The commands use the project-local
`engineering_skills\` directory by default; use `--skills-dir` to choose a
different approved location.

For OCR, install both the `pytesseract` Python package and the Tesseract
executable, then rebuild the facts file.  OCR data is explicitly marked
unverified and is not linked to geometry automatically.

## Engineer feedback and training data

Capture decisions as append-only, auditable JSONL. Do not add unreviewed model
predictions to a training set.

```powershell
# Confirm or reject a detector feature after visual engineering review.
python .\drawing_feedback.py add --facts .\drawing_facts.json --feedback .\drawing_feedback.jsonl --status accepted --feature-id P87 --reviewer "initials" --note "Confirmed against drawing."

# Correct geometry or add a primitive the detector missed. Geometry uses normalized image coordinates.
python .\drawing_feedback.py add --facts .\drawing_facts.json --feedback .\drawing_feedback.jsonl --status corrected --feature-id P87 --primitive-type line --geometry '{"x1":0.10,"y1":0.20,"x2":0.60,"y2":0.20}' --reviewer "initials"

python .\drawing_feedback.py summary --facts .\drawing_facts.json --feedback .\drawing_feedback.jsonl
python .\drawing_feedback.py export-labels --facts .\drawing_facts.json --feedback .\drawing_feedback.jsonl --output .\reviewed_labels.jsonl
```

Export curated copilot behavior examples only after reviewing the feedback:

```powershell
python .\build_copilot_training_data.py --facts .\drawing_facts.json --feedback .\drawing_feedback.jsonl --output .\copilot_sft.jsonl
```

## Training commands

### Primitive detector

Train the vision detector on the complete ParaCAD Zarr dataset. The command is
an 8 GB starting point; lower `--batch-size` to `4` if CUDA runs out of memory.

```powershell
python .\paracad_primitive_detr.py train --dataset .\ParaCAD_full_v3.zarr --checkpoint-dir .\checkpoints_full --epochs 40 --batch-size 8 --num-workers 8 --image-size 512 --num-queries 160 --max-targets 160 --train-max-samples 0 --val-max-samples 0 --device cuda
```

This trains line/circle/arc recognition from images. Engineer-reviewed labels
exported above can be converted to a separate, compatible fine-tuning dataset.
Collect at least two independently reviewed drawings before doing this—the
builder enforces a separate validation split.

### High-throughput CUDA workstation

For the 32 GB RTX 5090 workstation, start with the following settings. They
use sixteen Zarr/Pillow workers while reserving CPU capacity for Windows and
the training process. `gpu-greedy` removes the per-drawing CUDA-to-CPU SciPy
matching round trip during training; validation remains exact, so its F1 and
geometry MAE can be compared with earlier runs.

```powershell
python .\paracad_primitive_detr.py train --dataset .\ParaCAD_full_v3.zarr --checkpoint-dir .\checkpoints_full --epochs 40 --batch-size 48 --num-workers 16 --matcher gpu-greedy --validation-matcher exact --train-max-samples 1000000 --val-max-samples 10000 --device cuda
```

Keep `--batch-size 48` initially: the current exact-matcher job already
reserves roughly 23 GB of the available VRAM. After the first epoch, increase
to `56` only if GPU memory remains below about 29 GB and the machine remains
responsive. Do not use `--compile` for this workflow unless a compatible
Triton installation is present.

### Resume or warm-start detector training

`last.pt` is written after each completed epoch. To continue that same run,
restore it into the same checkpoint folder. This preserves the model,
optimizer, learning-rate schedule, AMP scaler, completed epoch, and best F1;
`--epochs` remains the total desired epoch count.

```powershell
python .\paracad_primitive_detr.py train --dataset .\ParaCAD_full_v3.zarr --checkpoint-dir .\checkpoints_full --resume .\checkpoints_full\last.pt --epochs 40 --batch-size 48 --num-workers 16 --matcher gpu-greedy --validation-matcher exact --train-max-samples 1000000 --val-max-samples 10000 --device cuda
```

To start a *new* optimization run from prior model weights only, use
`--init-checkpoint` and a new output folder. This is a warm start: its
optimizer and learning-rate schedule intentionally reset.

```powershell
python .\paracad_primitive_detr.py train --dataset .\ParaCAD_full_v3.zarr --init-checkpoint .\checkpoints_full\best.pt --checkpoint-dir .\checkpoints_warmstart --epochs 40 --batch-size 48 --num-workers 16 --matcher gpu-greedy --validation-matcher exact --train-max-samples 1000000 --val-max-samples 10000 --device cuda
```

```powershell
# Repeat --labels for every reviewed_labels.jsonl export. Image paths in the
# exports are resolved relative to --image-root.
python .\build_feedback_zarr.py --labels .\reviewed_labels.jsonl --image-root . --output .\reviewed_feedback.zarr

# Fine-tune model weights from a compatible detector checkpoint. The optimizer
# is reset and the small feedback set must remain separate from the full dataset.
python .\paracad_primitive_detr.py train --dataset .\reviewed_feedback.zarr --init-checkpoint .\checkpoints\best.pt --checkpoint-dir .\checkpoints_feedback --epochs 10 --batch-size 4 --num-workers 2 --image-size 512 --num-queries 128 --max-targets 96 --train-max-samples 0 --val-max-samples 0 --device cuda
```

The `--init-checkpoint` model architecture must match `--d-model`,
`--num-queries`, and `--decoder-layers`. Keep the original full-data evaluation
set unchanged to measure whether feedback fine-tuning genuinely helps.

### Copilot behavior adapter

The Ollama `qwen3.5:9b` model is a quantized GGUF inference artifact and cannot
be directly fine-tuned by this repository. Use a smaller original
Transformers-format *text* checkpoint (a 3–4B instruct model is the practical
8 GB choice) to train the fact-grounded response behavior, while retaining
`qwen3.5:9b` for vision and tool-assisted review.

```powershell
python -m pip install -r .\requirements-copilot-training.txt
python .\train_copilot_lora.py --base-model Qwen/Qwen3-4B-Instruct-2507 --dataset .\copilot_sft.jsonl --output .\adapters\drawing-copilot-lora --epochs 3 --max-length 2048 --batch-size 1 --gradient-accumulation 16
```

The resulting adapter is not automatically an Ollama model. Merge/import it
using the base model's compatible Transformers/Ollama conversion workflow, and
evaluate it against held-out reviewed drawings before deployment.

## 8 GB VRAM guidance

`qwen3.5:9b` is now available through the local Ollama service and was smoke-
tested with the copilot's grounded tools at `--num-ctx 4096`. Run primitive
prediction first; then run the copilot as a separate process so the detector's
CUDA allocations are released. Keep the copilot at `--temperature 0` and the
conservative default `--num-ctx 4096` on an 8 GB GPU. Increase it only if
Ollama has sufficient VRAM headroom; reduce it further if Ollama reports
memory pressure. For vision requests, image tokens add memory and latency;
start with `--vision-max-images 1 --vision-max-side 512`.
