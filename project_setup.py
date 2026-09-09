#!/usr/bin/env python3
"""Setup, data acquisition, and native-capability diagnostics for this repo.

Run through uv after cloning:

    uv sync --extra vision --extra workbench
    uv run python project_setup.py doctor --profile workbench

    uv sync --extra data
    uv run python project_setup.py data fetch --accept-dataset-license

The script never installs native software, GPU drivers, OCR engines, or an
Ollama model.  ``doctor`` reports those prerequisites with direct next steps.
The data download is explicit and license-gated; no heavyweight download or
Zarr build occurs as a side effect of dependency installation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ID = "yuwenbonnie/ParaCAD_dataset"
DEFAULT_RAW_DATA = ROOT / "data" / "ParaCAD_download"
DEFAULT_ZARR_DATA = ROOT / "ParaCAD_full_v3.zarr"
DOWNLOAD_MANIFEST = ".engdraw_download.json"

PROFILES: dict[str, tuple[str, ...]] = {
    "core": (),
    "vision": ("vision",),
    "workbench": ("vision", "workbench"),
    "data": ("data",),
    "training": ("vision", "copilot-training"),
    "full": ("vision", "workbench", "data", "copilot-training"),
}
IMPORTS_BY_EXTRA: dict[str, tuple[tuple[str, str], ...]] = {
    "vision": (("torch", "PyTorch"),),
    "workbench": (("streamlit", "Streamlit"),),
    "data": (("modelscope", "ModelScope SDK"),),
    "copilot-training": (
        ("accelerate", "Accelerate"),
        ("bitsandbytes", "bitsandbytes"),
        ("datasets", "Datasets"),
        ("peft", "PEFT"),
        ("transformers", "Transformers"),
    ),
}


def _print(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def command_version(command: list[str], timeout: float = 5) -> str | None:
    """Return a short version line, or None when a native command is absent."""
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else "available"


def uv_version() -> str | None:
    return command_version(["uv", "--version"])


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def available_disk_gb(path: Path) -> float:
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024**3)


def profile_extras(profile: str) -> tuple[str, ...]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}. Choose one of {', '.join(PROFILES)}.")
    return PROFILES[profile]


def doctor(profile: str, data_dir: Path, zarr_path: Path, strict: bool, as_json: bool) -> int:
    extras = profile_extras(profile)
    findings: list[dict[str, Any]] = []

    def record(kind: str, name: str, state: str, message: str, required: bool = False) -> None:
        findings.append({"kind": kind, "name": name, "state": state, "message": message, "required": required})
        _print(state.upper(), f"{name}: {message}")

    python_ok = (3, 10) <= sys.version_info[:2] < (3, 13)
    record("runtime", "Python", "ok" if python_ok else "blocked", f"{sys.version.split()[0]} ({platform.platform()})", required=True)
    version = uv_version()
    record(
        "runtime",
        "uv",
        "ok" if version else "blocked",
        version or "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and rerun setup.",
        required=True,
    )
    record("runtime", "free disk", "ok", f"{available_disk_gb(ROOT):.1f} GiB available beside the repository")

    for extra in extras:
        for module, label in IMPORTS_BY_EXTRA[extra]:
            installed = module_available(module)
            record(
                "python",
                label,
                "ok" if installed else "missing",
                "installed via uv" if installed else f"Run the repository bootstrap with profile '{profile}' or: uv sync --extra {extra}",
                required=True,
            )

    torch_state = "not installed"
    if module_available("torch"):
        import torch  # type: ignore

        cuda = bool(torch.cuda.is_available())
        torch_state = f"{torch.__version__}; CUDA {'available' if cuda else 'unavailable'}"
        if cuda:
            torch_state += f"; {torch.cuda.get_device_name(0)}"
    record("native", "PyTorch acceleration", "ok" if "CUDA available" in torch_state else "notice", torch_state)
    nvidia = command_version(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=8)
    record(
        "native",
        "NVIDIA driver",
        "ok" if nvidia else "notice",
        nvidia or "Optional. Install a current NVIDIA driver and a CUDA-compatible PyTorch wheel for GPU acceleration.",
    )
    tesseract = command_version(["tesseract", "--version"])
    record(
        "native",
        "Tesseract OCR",
        "ok" if tesseract else "notice",
        tesseract or "Optional. Install Tesseract with your OS package manager to enable OCR; uv cannot install native OCR binaries.",
    )
    ollama = command_version(["ollama", "--version"])
    record(
        "native",
        "Ollama",
        "ok" if ollama else "notice",
        ollama or "Optional. Install Ollama from https://ollama.com/download to enable the local copilot; uv cannot install the Ollama service.",
    )
    if ollama:
        models = (subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10, check=False).stdout or "").strip()
        record("native", "Qwen 3.5 9B", "ok" if "qwen3.5:9b" in models else "notice", "installed" if "qwen3.5:9b" in models else "Run: ollama pull qwen3.5:9b")

    raw_present = data_dir.is_dir() and any(data_dir.iterdir())
    zarr_present = zarr_path.is_dir() and (zarr_path / "zarr.json").is_file()
    record(
        "data",
        "raw ParaCAD data",
        "ok" if raw_present else "notice",
        str(data_dir) if raw_present else "Not downloaded. Run: uv run --extra data python project_setup.py data fetch --accept-dataset-license",
    )
    record(
        "data",
        "ParaCAD Zarr dataset",
        "ok" if zarr_present else "notice",
        str(zarr_path) if zarr_present else "Not built. Add --build-zarr to the data fetch command, or point the app at an existing .zarr directory.",
    )

    if as_json:
        print(json.dumps({"profile": profile, "findings": findings}, indent=2))
    blocked = [item for item in findings if item["required"] and item["state"] in {"blocked", "missing"}]
    return 1 if strict and blocked else 0


def _modelscope_downloader() -> Any:
    try:
        from modelscope.hub.snapshot_download import dataset_snapshot_download  # type: ignore
    except ImportError:
        try:
            from modelscope import dataset_snapshot_download  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ModelScope is not installed. Run: uv sync --extra data") from exc
    return dataset_snapshot_download


def fetch_data(args: argparse.Namespace) -> int:
    destination = args.data_dir.resolve()
    if not args.accept_dataset_license:
        raise ValueError(
            "Downloading is license-gated. Review the ParaCAD dataset card, then rerun with --accept-dataset-license. "
            "Source: https://www.modelscope.cn/datasets/yuwenbonnie/ParaCAD_dataset"
        )
    if destination.exists() and any(destination.iterdir()) and not args.resume:
        raise FileExistsError(f"Destination is not empty: {destination}. Use --resume to let ModelScope continue an existing download.")
    free_gb = available_disk_gb(destination)
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GiB is free near {destination}; the configured safety floor is {args.min_free_gb:.1f} GiB. "
            "Choose a larger data directory or lower --min-free-gb only after confirming the dataset size."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _print("INFO", f"Downloading ModelScope dataset {args.dataset_id} to {destination}")
    downloader = _modelscope_downloader()
    kwargs: dict[str, Any] = {"dataset_id": args.dataset_id, "local_dir": str(destination)}
    if args.revision:
        kwargs["revision"] = args.revision
    downloaded = Path(downloader(**kwargs))
    manifest = {
        "dataset_id": args.dataset_id,
        "revision": args.revision or "default",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "local_dir": str(destination),
        "modelscope_returned_path": str(downloaded),
        "license_accepted_by_command_flag": True,
    }
    (destination / DOWNLOAD_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _print("DONE", f"Raw ParaCAD data is available at {destination}")
    if args.build_zarr:
        return build_zarr(args)
    _print("NEXT", "Build the training dataset with: uv run --extra vision python project_setup.py data build-zarr")
    return 0


def build_zarr(args: argparse.Namespace) -> int:
    input_dir = args.data_dir.resolve()
    output_dir = args.zarr_output.resolve()
    if not input_dir.is_dir() or not any(input_dir.iterdir()):
        raise FileNotFoundError(f"Raw ParaCAD data was not found at {input_dir}. Run the data fetch command first or pass --data-dir.")
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Zarr output already exists: {output_dir}. Use --overwrite only when you intend to replace it.")
    command = [
        sys.executable,
        str(ROOT / "build_paracad_zarr.py"),
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--variant-mode",
        "source",
        "--fallback-any-style",
        "--only-needed-images",
        "--archive-workers",
        str(args.archive_workers),
    ]
    if args.overwrite:
        command.append("--overwrite")
    _print("INFO", "Building Zarr dataset. This is intentionally a separate, potentially long-running step.")
    _print("INFO", " ".join(f'"{part}"' if " " in part else part for part in command))
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise RuntimeError(f"Zarr build failed with exit code {result.returncode}.")
    _print("DONE", f"Zarr dataset built at {output_dir}")
    return 0


def data_status(data_dir: Path, zarr_output: Path) -> int:
    manifest_path = data_dir / DOWNLOAD_MANIFEST
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {"warning": "download manifest is not valid JSON"}
    payload = {
        "raw_data_dir": str(data_dir.resolve()),
        "raw_data_present": data_dir.is_dir() and any(data_dir.iterdir()),
        "download_manifest": manifest,
        "zarr_output": str(zarr_output.resolve()),
        "zarr_present": zarr_output.is_dir() and (zarr_output / "zarr.json").is_file(),
        "free_disk_gb": round(available_disk_gb(data_dir), 2),
    }
    print(json.dumps(payload, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="uv-friendly setup, data download, and native-capability checks for Engineering Drawing Copilot.")
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Report uv-managed and native prerequisites without changing the computer.")
    check.add_argument("--profile", choices=sorted(PROFILES), default="workbench")
    check.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DATA)
    check.add_argument("--zarr-output", type=Path, default=DEFAULT_ZARR_DATA)
    check.add_argument("--strict", action="store_true", help="Return failure if a required uv-managed dependency is missing.")
    check.add_argument("--json", action="store_true", help="Also emit a JSON capability report.")
    data = commands.add_parser("data", help="Download raw ParaCAD data or build its local Zarr derivative.")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    fetch = data_commands.add_parser("fetch", help="Explicitly download ParaCAD via the ModelScope SDK.")
    fetch.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    fetch.add_argument("--revision", help="Optional ModelScope revision pinned by the project maintainer.")
    fetch.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DATA)
    fetch.add_argument("--zarr-output", type=Path, default=DEFAULT_ZARR_DATA)
    fetch.add_argument("--min-free-gb", type=float, default=100.0, help="Refuse a new download if less free space than this remains.")
    fetch.add_argument("--accept-dataset-license", action="store_true", help="Confirm that you reviewed and accept the upstream dataset terms.")
    fetch.add_argument("--resume", action="store_true", help="Allow ModelScope to continue an existing destination directory.")
    fetch.add_argument("--build-zarr", action="store_true", help="Build the local Zarr dataset after downloading the raw data.")
    fetch.add_argument("--archive-workers", type=int, default=max(1, (os.cpu_count() or 2) - 8))
    build = data_commands.add_parser("build-zarr", help="Convert an existing raw ParaCAD download to Zarr.")
    build.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DATA)
    build.add_argument("--zarr-output", type=Path, default=DEFAULT_ZARR_DATA)
    build.add_argument("--archive-workers", type=int, default=max(1, (os.cpu_count() or 2) - 8))
    build.add_argument("--overwrite", action="store_true", help="Replace an existing Zarr output after explicit confirmation.")
    status = data_commands.add_parser("status", help="Print local raw-data and Zarr availability without changing anything.")
    status.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DATA)
    status.add_argument("--zarr-output", type=Path, default=DEFAULT_ZARR_DATA)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "doctor":
            return doctor(args.profile, args.data_dir, args.zarr_output, args.strict, args.json)
        if args.data_command == "fetch":
            return fetch_data(args)
        if args.data_command == "build-zarr":
            return build_zarr(args)
        return data_status(args.data_dir, args.zarr_output)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as exc:
        _print("ERROR", str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

