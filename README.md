# Engineering Drawing Copilot

Local tools for extracting lines, circles, and arcs from engineering drawings; producing reviewable structured facts and overlays; benchmarking detector and copilot behavior; and running a grounded local Ollama copilot.

The project uses **uv** for reproducible Python environments. Raw datasets, checkpoints, Ollama models, and generated workbench output are deliberately excluded from Git.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. It manages Python dependencies, but it does not install native applications or GPU drivers.

Windows PowerShell:

```powershell
.\setup.ps1 -Profile workbench
uv run streamlit run .\streamlit_app.py
```

macOS or Linux:

```bash
bash ./setup.sh --profile workbench
uv run streamlit run ./streamlit_app.py
```

The equivalent explicit commands are:

```powershell
uv sync --extra vision --extra workbench
uv run python .\project_setup.py doctor --profile workbench
```

`doctor` reports every available capability and gives the next action for anything unavailable. It never changes native software or downloads datasets.

## Installation profiles

| Profile | uv-managed packages | Intended use |
| --- | --- | --- |
| `core` | Facts, overlays, review checks, benchmarking | Lightweight non-ML utilities |
| `vision` | Core + PyTorch | Detector prediction, training, evaluation |
| `workbench` | Vision + Streamlit | Normal interactive operation |
| `data` | ModelScope SDK | Raw ParaCAD download only |
| `training` | Vision + Transformers/PEFT tooling | Detector and Copilot fine-tuning |
| `full` | All profiles + development tools | Maintainer workstation |

## ParaCAD data

Raw ParaCAD files are not committed to Git. The setup helper can retrieve the upstream dataset from [ModelScope's ParaCAD dataset page](https://www.modelscope.cn/datasets/yuwenbonnie/ParaCAD_dataset), but only after you review its terms and explicitly acknowledge them.

```powershell
uv sync --extra data --extra vision
uv run python .\project_setup.py data fetch `
  --accept-dataset-license `
  --build-zarr
```

This downloads raw data into `data\ParaCAD_download` and builds the derived training dataset at `ParaCAD_full_v3.zarr`, which the application detects automatically. The command has a 100 GiB free-space safety floor; adjust `--data-dir`, `--zarr-output`, or `--min-free-gb` only after confirming the upstream dataset size and your storage budget.

## Native prerequisites uv does not install

| Capability | Setup behavior |
| --- | --- |
| NVIDIA driver / CUDA runtime | `doctor` detects the GPU and whether PyTorch can use CUDA. |
| CUDA-specific PyTorch wheel | `doctor` reports CPU-only PyTorch; choose a compatible wheel from the PyTorch installer selector. |
| Tesseract | `doctor` reports that OCR will be skipped until its native executable is installed. |
| Ollama | `doctor` reports whether the local service and `qwen3.5:9b` model are available. |
| Git | Required for normal GitHub workflow, but not installed by this repository. |

After installing Ollama from its official installer, fetch the model yourself:

```powershell
ollama pull qwen3.5:9b
```

The bootstrap scripts intentionally do not install drivers, services, or models silently.

## Common commands

```powershell
uv run python .\paracad_primitive_detr.py predict --checkpoint .\checkpoints\best.pt --image .\drawing.png --output .\predicted_primitives.json --device auto

uv run python .\paracad_primitive_detr.py evaluate --dataset .\ParaCAD_full_v3.zarr --checkpoint .\checkpoints\best.pt --device auto

uv run python .\benchmark_engineering.py run --cases .\benchmarks\engineering_cases.jsonl --mode tool
```

See [ENGINEERING_COPILOT.md](ENGINEERING_COPILOT.md) for training, review, benchmark, and local-copilot workflows.

## Before publishing to GitHub

1. Choose and add an explicit project license; none has been selected yet.
2. Do not commit datasets, raw images, checkpoints, generated outputs, Ollama models, or credentials. `.gitignore` excludes those artifacts.
3. Pin the ModelScope dataset revision in release documentation after verifying the source version and license.
4. Publish benchmark cases and compact metrics reports, not raw restricted engineering inputs.
