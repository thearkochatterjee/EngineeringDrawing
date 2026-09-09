#!/usr/bin/env bash
# Cross-platform uv bootstrap. It installs only Python dependencies declared in
# pyproject.toml and reports native prerequisites through project_setup.py.
set -euo pipefail

profile="workbench"
with_data=0
build_zarr=0
accept_license=0
data_dir="data/ParaCAD_download"
zarr_output="ParaCAD_full_v3.zarr"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) profile="$2"; shift 2 ;;
    --with-data) with_data=1; shift ;;
    --build-zarr) build_zarr=1; shift ;;
    --accept-dataset-license) accept_license=1; shift ;;
    --data-dir) data_dir="$2"; shift 2 ;;
    --zarr-output) zarr_output="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: ./setup.sh [--profile core|vision|workbench|data|training|full] [--with-data --accept-dataset-license --build-zarr]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "[BLOCKED] uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/ and rerun ./setup.sh." >&2
  exit 2
fi

case "$profile" in
  core) extras=() ;;
  vision) extras=(vision) ;;
  workbench) extras=(vision workbench) ;;
  data) extras=(data) ;;
  training) extras=(vision copilot-training) ;;
  full) extras=(vision workbench data copilot-training) ;;
  *) echo "Invalid profile: $profile" >&2; exit 2 ;;
esac
if [[ "$with_data" == 1 && ! " ${extras[*]} " =~ " data " ]]; then extras+=(data); fi

sync=(sync)
for extra in "${extras[@]}"; do sync+=(--extra "$extra"); done
if [[ "$profile" == full ]]; then sync+=(--group dev); fi
uv "${sync[@]}"
uv run python ./project_setup.py doctor --profile "$profile" --data-dir "$data_dir" --zarr-output "$zarr_output"

if [[ "$with_data" == 1 ]]; then
  if [[ "$accept_license" != 1 ]]; then
    echo "[BLOCKED] Raw ParaCAD download was not started. Re-run with --accept-dataset-license after reviewing the upstream ModelScope dataset terms." >&2
    exit 2
  fi
  fetch=(run python ./project_setup.py data fetch --data-dir "$data_dir" --zarr-output "$zarr_output" --accept-dataset-license)
  if [[ "$build_zarr" == 1 ]]; then fetch+=(--build-zarr); fi
  uv "${fetch[@]}"
fi

echo "[NEXT] Start the workbench with: uv run streamlit run ./streamlit_app.py"
echo "[NOTICE] Native optional components (Ollama, Tesseract, NVIDIA driver/CUDA) are reported but never installed by this script."

