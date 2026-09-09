"""Local Streamlit workbench for the engineering-drawing copilot.

Launch from this directory after installing requirements-streamlit.txt:
    python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import streamlit as st
from PIL import Image

from build_copilot_training_data import example_for
from drawing_copilot import FactTools, _system_prompt, ask_ollama
from drawing_facts import build_drawing_facts, write_facts
from drawing_feedback import append_feedback, export_labels, feedback_for_drawing, parse_geometry
from engineering_knowledge import KnowledgeIndex, build_index, write_index
from vision_evidence import build_vision_evidence, render_prediction_overlay


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workbench"
CLI_SCRIPTS = {
    "Primitive detector (train, predict, evaluate)": "paracad_primitive_detr.py",
    "ParaCAD archive to Zarr": "build_paracad_zarr.py",
    "Parallel ParaCAD preprocessing": "preprocess_paracad_parallel.py",
    "Single-process ParaCAD preprocessing": "preprocess_paracad.py",
    "Structured drawing-feature extraction": "extract_drawing_features.py",
    "Build drawing facts": "build_drawing_facts.py",
    "Render prediction/ground-truth overlay": "render_primitive_overlay.py",
    "Engineer feedback": "drawing_feedback.py",
    "Build reviewed-feedback Zarr": "build_feedback_zarr.py",
    "Build Copilot SFT data": "build_copilot_training_data.py",
    "Build knowledge index": "build_knowledge_index.py",
    "Grounded drawing Copilot": "drawing_copilot.py",
    "QLoRA Copilot training": "train_copilot_lora.py",
    "Engineering benchmark suite": "benchmark_engineering.py",
}
ARTIFACT_SUFFIXES = {".json", ".jsonl", ".log", ".txt"}


def workspace_path(name: str) -> Path:
    WORKSPACE.mkdir(exist_ok=True)
    return WORKSPACE / name


def jobs_path() -> Path:
    path = workspace_path("jobs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_path(value: str) -> Path:
    return Path(value.strip()).expanduser()


def first_existing(candidates: list[Path], kind: str) -> Path | None:
    for path in candidates:
        if (path.is_dir() if kind == "dir" else path.is_file()):
            return path
    return None


def automatic_dataset() -> Path | None:
    preferred = [ROOT / "ParaCAD_full_v3.zarr", ROOT / "ParaCAD_full.zarr", ROOT / "reviewed_feedback.zarr"]
    return first_existing(preferred + sorted(path for path in ROOT.glob("*.zarr") if path.is_dir()), "dir")


def automatic_checkpoint() -> Path | None:
    candidates = [
        ROOT / "checkpoints" / "best.pt",
        ROOT / "checkpoints_full" / "best.pt",
        ROOT / "checkpoints_feedback" / "best.pt",
        ROOT / "checkpoints" / "last.pt",
    ]
    return first_existing(candidates, "file")


def automatic_knowledge_index() -> Path | None:
    return first_existing([WORKSPACE / "engineering_knowledge.json", ROOT / "engineering_knowledge.json"], "file")


def ocr_available() -> bool:
    return shutil.which("tesseract") is not None


def automatic_workers() -> int:
    """Use the data-loader capacity of a workstation without consuming every core."""
    return min(16, max(2, (os.cpu_count() or 2) - 8))


@st.cache_data(show_spinner=False)
def automatic_training_batch_size() -> int:
    """Pick a safe high-throughput default from the installed GPU memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        memory_mb = int(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, IndexError, ValueError, subprocess.TimeoutExpired):
        return 16
    if memory_mb >= 24_000:
        # The primitive model's matching tensors and any concurrent desktop
        # applications also occupy VRAM, so leave practical headroom on a
        # nominal 32 GB card rather than starting at its theoretical maximum.
        return 48
    if memory_mb >= 12_000:
        return 24
    return 8


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def zarr_split_size(dataset_path: str, split: str) -> int:
    """Return a split size without loading its record IDs into memory."""
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise RuntimeError('Missing dependency "zarr". Install requirements-primitive-detr.txt.') from exc

    root = zarr.open_group(dataset_path, mode="r")
    if split == "all":
        return int(root["records/image_id"].shape[0])
    try:
        return int(root[f"splits/{split}"].shape[0])
    except KeyError as exc:
        raise ValueError(f"Split '{split}' was not found in {dataset_path}.") from exc


@st.cache_data(show_spinner=False)
def zarr_dataset_summary(dataset_path: str) -> dict[str, int | str | None]:
    """Read only Zarr metadata needed to size a training job."""
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise RuntimeError('Missing dependency "zarr". Install requirements-primitive-detr.txt.') from exc

    root = zarr.open_group(dataset_path, mode="r")
    try:
        total = int(root["records/image_id"].shape[0])
    except KeyError:
        total = int(root["records/split"].shape[0])
    counts: dict[str, int | str | None] = {"dataset": Path(dataset_path).name, "total": total}
    for split in ("train", "val", "test"):
        try:
            counts[split] = int(root[f"splits/{split}"].shape[0])
        except KeyError:
            counts[split] = None
    return counts


def local_training_datasets() -> list[Path]:
    paths = [path for path in ROOT.glob("*.zarr") if path.is_dir()]
    preferred = automatic_dataset()
    if preferred is not None and preferred not in paths:
        paths.append(preferred)
    return sorted(set(paths), key=lambda path: path.name.lower())


@st.cache_data(show_spinner=False)
def zarr_image_page(dataset_path: str, split: str, page: int, page_size: int) -> list[tuple[int, int]]:
    """Read just one small page of record/image identifiers for the picker."""
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise RuntimeError('Missing dependency "zarr". Install requirements-primitive-detr.txt.') from exc

    root = zarr.open_group(dataset_path, mode="r")
    start = page * page_size
    stop = start + page_size
    if split == "all":
        stop = min(stop, int(root["records/image_id"].shape[0]))
        record_ids = np.arange(start, stop, dtype=np.int64)
    else:
        record_ids = np.asarray(root[f"splits/{split}"][start:stop], dtype=np.int64)
    if not record_ids.size:
        return []
    image_ids = np.asarray(root["records/image_id"][record_ids], dtype=np.int64)
    return [(int(record_id), int(image_id)) for record_id, image_id in zip(record_ids, image_ids, strict=True)]


def image_suffix(image_bytes: bytes) -> str:
    """Determine a safe filename suffix while leaving the original bytes unchanged."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        return {
            "BMP": ".bmp",
            "GIF": ".gif",
            "JPEG": ".jpg",
            "PNG": ".png",
            "TIFF": ".tiff",
            "WEBP": ".webp",
        }.get(str(image.format).upper(), ".png")


def extract_zarr_images(dataset_path: Path, selections: list[tuple[int, int]], destination: Path) -> list[Path]:
    """Materialize selected Zarr image records as normal image files for the pipeline."""
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise RuntimeError('Missing dependency "zarr". Install requirements-primitive-detr.txt.') from exc

    root = zarr.open_group(str(dataset_path), mode="r")
    offsets = root["images/data_offsets"]
    data = root["images/data_bytes"]
    destination.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for record_id, expected_image_id in selections:
        image_id = int(root["records/image_id"][record_id])
        if image_id < 0:
            raise ValueError(f"Record {record_id:,} does not have a linked image.")
        if image_id != expected_image_id:
            raise ValueError(f"Record {record_id:,} changed while it was selected; refresh the dataset page and try again.")
        start, end = (int(value) for value in offsets[image_id : image_id + 2])
        image_bytes = np.asarray(data[start:end], dtype=np.uint8).tobytes()
        output = destination / f"paracad_record_{record_id:07d}_image_{image_id:07d}{image_suffix(image_bytes)}"
        output.write_bytes(image_bytes)
        files.append(output)
    return files


def facts_path_from_state() -> Path | None:
    raw = st.session_state.get("facts_path", "").strip()
    path = as_path(raw) if raw else None
    return path if path and path.is_file() else None


def load_facts_or_notice() -> tuple[dict[str, Any], Path] | None:
    path = facts_path_from_state()
    if path is None:
        st.info("Build or select a drawing facts JSON file in the Analysis tab first.")
        return None
    try:
        return read_json(path), path
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read facts: {exc}")
        return None


def run_process(command: list[str], label: str) -> bool:
    """Run a bounded local utility synchronously and show its captured output."""
    with st.status(label, expanded=True) as status:
        status.write(" ".join(command))
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if result.stdout:
            status.code(result.stdout, language="text")
        if result.stderr:
            status.code(result.stderr, language="text")
        if result.returncode:
            status.update(label=f"{label} failed", state="error")
            return False
        status.update(label=f"{label} complete", state="complete")
        return True


def command_display(command: list[str]) -> str:
    """Render the exact shell-free command in Windows-friendly form."""
    return subprocess.list2cmdline(command)


def start_background_job(label: str, command: list[str]) -> bool:
    """Start one local project command and persist its output to a workbench log."""
    previous = active_job()
    if previous and previous.get("return_code") is None:
        st.error(f"{previous['label']} is still running. Finish or stop it before starting another job.")
        return False
    started = datetime.now(UTC)
    log_path = jobs_path() / f"{started.strftime('%Y%m%dT%H%M%SZ')}_{label.lower().replace(' ', '_')}.log"
    try:
        handle = log_path.open("w", encoding="utf-8", buffering=1)
        handle.write(f"Command: {command_display(command)}\nWorking directory: {ROOT}\nStarted: {started.isoformat()}\n\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        try:
            handle.close()
        except UnboundLocalError:
            pass
        st.error(f"Could not start {label}: {exc}")
        return False
    st.session_state.active_job = {
        "label": label,
        "command": command,
        "log_path": str(log_path),
        "started_at": started.isoformat(),
        "process": process,
        "handle": handle,
        "return_code": None,
        "recorded": False,
    }
    return True


def active_job() -> dict[str, Any] | None:
    """Update and return the session's active local process, if one exists."""
    job = st.session_state.get("active_job")
    if not isinstance(job, dict):
        return None
    process = job.get("process")
    if process is None:
        return job
    return_code = process.poll()
    if return_code is not None and job.get("return_code") is None:
        job["return_code"] = int(return_code)
        handle = job.get("handle")
        if handle is not None and not handle.closed:
            handle.close()
        if not job.get("recorded"):
            history = st.session_state.setdefault("job_history", [])
            history.insert(
                0,
                {
                    "label": job["label"],
                    "command": job["command"],
                    "log_path": job["log_path"],
                    "started_at": job["started_at"],
                    "return_code": job["return_code"],
                },
            )
            st.session_state.job_history = history[:20]
            job["recorded"] = True
    return job


def log_tail(log_path: Path, max_bytes: int = 24_000) -> str:
    if not log_path.is_file():
        return "No output has been written yet."
    with log_path.open("rb") as source:
        source.seek(0, 2)
        source.seek(max(0, source.tell() - max_bytes))
        return source.read().decode("utf-8", errors="replace")


def training_progress_from_log(text: str) -> dict[str, Any] | None:
    """Extract safe, approximate training status from the CLI's plain progress log."""
    epoch_matches = list(re.finditer(r"\[EPOCH\s+(\d+)\s*/\s*(\d+)\]", text))
    if not epoch_matches:
        return None
    epoch_match = epoch_matches[-1]
    epoch, epoch_total = (int(value) for value in epoch_match.groups())
    current_text = text[epoch_match.end() :]
    size_match = re.search(r"train=([\d,]+),\s*val=([\d,]+)", text)
    batch_size_match = re.search(r"--batch-size\s+(\d+)", text)
    batch_matches = list(re.finditer(r"\bbatch\s+([\d,]+):\s+loss=([0-9.]+)", current_text))
    validation_matches = list(re.finditer(r"\[EVALUATE]\s+([\d,]+)\s*/\s*([\d,]+)\s+samples", current_text))
    details: dict[str, Any] = {"epoch": epoch, "epoch_total": epoch_total, "stage": "Starting epoch"}
    if validation_matches and (not batch_matches or validation_matches[-1].start() > batch_matches[-1].start()):
        completed, total = (int(value.replace(",", "")) for value in validation_matches[-1].groups())
        details.update({"stage": "Validation", "completed": completed, "total": total})
        return details
    if batch_matches:
        batch, loss = batch_matches[-1].groups()
        details.update({"stage": "Training", "completed": int(batch.replace(",", "")), "loss": float(loss)})
        if size_match and batch_size_match:
            train_samples = int(size_match.group(1).replace(",", ""))
            batch_size = int(batch_size_match.group(1))
            details["total"] = (train_samples + batch_size - 1) // batch_size
    return details


def stop_background_job(job: dict[str, Any]) -> tuple[bool, str]:
    """Stop an app-launched job and its DataLoader/process-worker descendants."""
    process = job.get("process")
    if process is None or process.poll() is not None:
        return False, "The job has already ended."
    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return False, str(exc)
    if result.returncode:
        return False, (result.stderr or result.stdout or "taskkill did not report a reason.").strip()
    job["stop_requested"] = True
    return True, "Stop requested for the training process and its worker tree."


def last_json_line(text: str) -> dict[str, Any] | None:
    """Recover single-line CLI result JSON (for example detector evaluation)."""
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def render_job_monitor(key_prefix: str) -> None:
    """Show the active operation, its tail output, and recent completed jobs."""
    job = active_job()
    if job is None:
        st.info("No local CLI operation is currently running.")
    else:
        log_path = as_path(str(job["log_path"]))
        return_code = job.get("return_code")
        if return_code is None:
            if job.get("stop_requested"):
                st.info(f"Stopping: {job['label']}")
                status = "stop requested"
            else:
                st.warning(f"Running: {job['label']}")
                status = "running"
        elif return_code == 0:
            st.success(f"Completed: {job['label']}")
            status = "complete"
        elif job.get("stop_requested"):
            st.info(f"Stopped: {job['label']}")
            status = "stopped"
        else:
            st.error(f"Failed: {job['label']} (exit code {return_code})")
            status = "failed"
        st.caption(f"Status: {status}  •  started {job['started_at']}")
        st.code(command_display(job["command"]), language="powershell")
        current_log = log_tail(log_path)
        training = training_progress_from_log(current_log)
        if training is not None:
            epoch_percent = int(100 * training["epoch"] / max(training["epoch_total"], 1))
            st.progress(epoch_percent, text=f"Epoch {training['epoch']} of {training['epoch_total']}  •  {training['stage']}")
            if training.get("total"):
                completed = int(training.get("completed", 0))
                total = int(training["total"])
                stage_percent = int(100 * min(completed, total) / max(total, 1))
                detail = f"{training['stage']}: {completed:,}/{total:,}"
                if training.get("loss") is not None:
                    detail += f"  •  loss {training['loss']:.4f}"
                st.progress(stage_percent, text=detail)
            elif training.get("loss") is not None:
                st.caption(f"Last reported loss: {training['loss']:.4f}")
        controls = st.columns(3)
        with controls[0]:
            if st.button("Refresh live progress", key=f"{key_prefix}_refresh_active_job"):
                st.rerun()
        with controls[1]:
            confirm_stop = st.checkbox("Confirm stop", key=f"{key_prefix}_confirm_stop", disabled=return_code is not None)
            if return_code is None and st.button("Stop active job", key=f"{key_prefix}_stop_active_job", disabled=not confirm_stop):
                stopped, message = stop_background_job(job)
                (st.warning if stopped else st.error)(message)
        with controls[2]:
            if log_path.is_file():
                st.download_button("Download log", data=log_path.read_bytes(), file_name=log_path.name, mime="text/plain", key=f"{key_prefix}_download_job_log")
        st.code(current_log, language="text")

    history = st.session_state.get("job_history", [])
    if history:
        st.subheader("Recent jobs")
        st.dataframe(
            [{key: item[key] for key in ("label", "started_at", "return_code", "log_path")} for item in history],
            use_container_width=True,
            hide_index=True,
        )


def ollama_models(url: str) -> list[str]:
    with urlopen(f"{url.rstrip('/')}/api/tags", timeout=10) as response:  # nosec B310 - user chooses a local Ollama URL.
        payload = json.loads(response.read().decode("utf-8"))
    return [str(model.get("name", "unknown")) for model in payload.get("models", [])]


def get_copilot(facts: dict[str, Any], facts_path: Path, settings: dict[str, Any]) -> tuple[FactTools, list[dict[str, Any]], list[dict[str, str]]]:
    """Create or reuse one chat/tools session for the current evidence/configuration."""
    identity = json.dumps({"facts": str(facts_path.resolve()), **settings}, sort_keys=True)
    if st.session_state.get("copilot_identity") != identity:
        knowledge = KnowledgeIndex.load(as_path(settings["knowledge_index"])) if settings["knowledge_index"] else None
        evidence: list[dict[str, str]] = []
        if settings["vision"]:
            image = as_path(settings["vision_image"]) if settings["vision_image"] else Path(str(facts["drawing"]["image_path"]))
            evidence = build_vision_evidence(image, facts, "auto", settings["vision_max_side"], settings["vision_max_images"])
        tools = FactTools(
            facts,
            skills_dir=as_path(settings["skills_dir"]),
            allow_skill_write=settings["allow_skill_write"],
            allow_skill_replace=settings["allow_skill_replace"],
            knowledge_index=knowledge,
            vision_evidence=evidence,
        )
        st.session_state.copilot_identity = identity
        st.session_state.copilot_tools = tools
        st.session_state.copilot_messages = [{"role": "system", "content": _system_prompt(facts, evidence)}]
        st.session_state.chat_display = []
        st.session_state.vision_evidence = evidence
    return st.session_state.copilot_tools, st.session_state.copilot_messages, st.session_state.get("vision_evidence", [])


def sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("Workspace")
        dataset = automatic_dataset()
        checkpoint = automatic_checkpoint()
        st.caption(f"Dataset: {dataset.name if dataset else 'not found'}")
        st.caption(f"Checkpoint: {checkpoint.name if checkpoint else 'not found'}")
        st.caption(f"OCR: {'available' if ocr_available() else 'not installed'}")
        facts_path = facts_path_from_state()
        st.caption(f"Active analysis: {facts_path.name if facts_path else 'none'}")
        model = "qwen3.5:9b"
        ollama_url = "http://127.0.0.1:11434"
        num_ctx = 4096
        timeout = 180.0
        vision = True
        vision_image = ""
        vision_max_images = 1
        vision_max_side = 512
        knowledge_index = str(automatic_knowledge_index() or "")
        skills_dir = "engineering_skills"
        allow_skill_write = False
        allow_skill_replace = False
        with st.expander("Advanced session settings"):
            model = st.text_input("Ollama model", value=model)
            ollama_url = st.text_input("Ollama URL", value=ollama_url)
            vision = st.checkbox("Attach visual evidence", value=vision)
            if vision:
                vision_max_side = st.select_slider("Vision image size", options=[512, 768, 1024], value=512)
            knowledge_index = st.text_input("Knowledge index", value=knowledge_index)
            skills_dir = st.text_input("Skills directory", value=skills_dir)
            allow_skill_write = st.checkbox("Allow saving reviewed skill drafts", value=False)
            allow_skill_replace = st.checkbox("Allow skill replacement", value=False, disabled=not allow_skill_write)
            if st.button("Check Ollama", use_container_width=True):
                try:
                    models = ollama_models(ollama_url)
                    if model in models:
                        st.success(f"{model} is available.")
                    else:
                        st.warning(f"{model} is not installed. Available: {', '.join(models)}")
                except OSError as exc:
                    st.error(f"Could not reach Ollama: {exc}")
    return {
        "model": model,
        "ollama_url": ollama_url,
        "num_ctx": int(num_ctx),
        "timeout": float(timeout),
        "vision": vision,
        "vision_image": vision_image,
        "vision_max_images": int(vision_max_images),
        "vision_max_side": int(vision_max_side),
        "knowledge_index": knowledge_index,
        "skills_dir": skills_dir,
        "allow_skill_write": allow_skill_write,
        "allow_skill_replace": allow_skill_replace,
    }


def analysis_tab() -> None:
    st.subheader("Analyze a drawing")
    st.caption("Run prediction, build source-labeled facts, inspect the overlay, and download the resulting JSON.")
    left, right = st.columns(2)
    with left:
        with st.expander("Select drawing(s) from a ParaCAD Zarr dataset", expanded=True):
            dataset_raw = st.text_input("ParaCAD Zarr dataset", value=str(ROOT / "ParaCAD_full_v3.zarr"))
            split, page_size = st.columns(2)
            with split:
                selected_split = st.selectbox("Dataset split", ["val", "train", "all"], help="Validation is selected by default because it is the evaluation split.")
            with page_size:
                selected_page_size = st.selectbox("Drawings per page", [25, 50, 100, 250], index=1)
            dataset_path = as_path(dataset_raw)
            try:
                if not dataset_path.is_dir():
                    raise ValueError("Choose an existing .zarr dataset directory.")
                total = zarr_split_size(str(dataset_path), selected_split)
                page_count = max(1, (total + selected_page_size - 1) // selected_page_size)
                current_page = int(st.session_state.get("dataset_page_number", 1))
                if current_page > page_count:
                    st.session_state.dataset_page_number = 1
                page_number = int(
                    st.number_input(
                        "Page",
                        min_value=1,
                        max_value=page_count,
                        value=int(st.session_state.get("dataset_page_number", 1)),
                        step=1,
                        key="dataset_page_number",
                    )
                )
                records = zarr_image_page(str(dataset_path), selected_split, page_number - 1, selected_page_size)
                labels = [f"Record {record_id:,}  |  image {image_id:,}" for record_id, image_id in records if image_id >= 0]
                choices = dict(zip(labels, ((record_id, image_id) for record_id, image_id in records if image_id >= 0), strict=True))
                selection_context = f"{dataset_path.resolve()}::{selected_split}::{page_number}::{selected_page_size}"
                if st.session_state.get("dataset_selection_context") != selection_context:
                    st.session_state.dataset_selection_context = selection_context
                    st.session_state.dataset_selected_labels = []
                selected_labels = st.multiselect(
                    "Dataset drawing(s)",
                    labels,
                    key="dataset_selected_labels",
                    placeholder="Search by record or image ID",
                    help="Select one or more drawings from this page. The archive stores IDs, not source filenames.",
                )
                st.caption(f"{total:,} drawings in this split. Showing {len(records):,} drawings on page {page_number:,} of {page_count:,}.")
                if st.button("Extract selected drawings", disabled=not selected_labels):
                    selected_records = [choices[label] for label in selected_labels]
                    extracted = extract_zarr_images(dataset_path, selected_records, workspace_path("dataset_images"))
                    st.session_state.dataset_image_paths = [str(path) for path in extracted]
                    st.session_state.analysis_image = str(extracted[0])
                    st.success(f"Extracted {len(extracted)} drawing(s) to {WORKSPACE / 'dataset_images'}.")
            except (ImportError, OSError, ValueError, KeyError) as exc:
                st.error(f"Could not list this dataset: {exc}")

        extracted_images = [path for path in st.session_state.get("dataset_image_paths", []) if as_path(path).is_file()]
        if extracted_images:
            active_dataset_image = st.selectbox(
                "Active extracted dataset drawing",
                extracted_images,
                format_func=lambda path: Path(path).name,
                help="This is the drawing used by detection and fact-building below.",
            )
            st.session_state.analysis_image = active_dataset_image
        image_path = st.text_input("Drawing image", value="")
        uploaded = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        if uploaded and st.button("Save uploaded image"):
            destination = workspace_path(f"upload_{uploaded.name}")
            destination.write_bytes(uploaded.getvalue())
            st.session_state.analysis_image = str(destination)
            st.success(f"Saved to {destination}")
        image_path = image_path.strip() or st.session_state.get("analysis_image", "")
        if image_path:
            st.caption(f"Using: {image_path}")
        checkpoint = st.text_input("Primitive detector checkpoint", value=str(ROOT / "checkpoints" / "best.pt"))
        prediction_path = st.text_input("Prediction JSON", value=str(workspace_path("predicted_primitives.json")))
        device = st.selectbox("Detector device", ["cuda", "auto", "cpu"], index=0)
        if st.button("Run primitive detection", type="primary"):
            if not as_path(image_path).is_file() or not as_path(checkpoint).is_file():
                st.error("Select an existing image and checkpoint.")
            else:
                command = [sys.executable, str(ROOT / "paracad_primitive_detr.py"), "predict", "--checkpoint", checkpoint, "--image", image_path, "--output", prediction_path, "--device", device, "--no-progress"]
                if run_process(command, "Primitive detection"):
                    st.success("Prediction JSON is ready.")
    with right:
        facts_output = st.text_input("Facts output", value=str(workspace_path("drawing_facts.json")))
        ground_truth = st.text_input("Optional ParaCAD ground-truth JSONL", value="")
        no_ocr = st.checkbox("Skip OCR", value=True)
        hole_spacing = st.number_input("Hole edge-spacing screening threshold (normalized)", min_value=0.0, value=0.0, step=0.001)
        if st.button("Build drawing facts"):
            if not as_path(image_path).is_file() or not as_path(prediction_path).is_file():
                st.error("Select an existing image and prediction JSON.")
            else:
                try:
                    facts = build_drawing_facts(
                        as_path(image_path),
                        as_path(prediction_path),
                        ground_truth_jsonl=as_path(ground_truth) if ground_truth.strip() else None,
                        use_ocr=not no_ocr,
                        min_hole_edge_spacing=hole_spacing or None,
                    )
                    write_facts(facts, as_path(facts_output))
                    st.session_state.facts_path = facts_output
                    st.success(f"Wrote {facts_output}")
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    st.error(str(exc))
    loaded = load_facts_or_notice()
    if not loaded:
        return
    facts, _ = loaded
    summary = facts["summary"]
    metrics = st.columns(4)
    metrics[0].metric("Predicted primitives", summary["predicted_primitive_count"])
    metrics[1].metric("Mean confidence", summary["mean_detector_confidence"])
    metrics[2].metric("Review findings", summary["rule_finding_count"])
    metrics[3].metric("Closed line profiles", summary["closed_line_profile_count"])
    image = as_path(str(facts["drawing"]["image_path"]))
    if image.is_file():
        original, overlay = st.columns(2)
        original.image(str(image), caption="Original drawing", use_container_width=True)
        try:
            overlay.image(render_prediction_overlay(image, facts), caption="Detector overlay (magenta)", use_container_width=True)
        except OSError as exc:
            st.warning(f"Could not render overlay: {exc}")
    st.subheader("Deterministic review checks")
    st.dataframe(facts.get("checks", {}).get("findings", []), use_container_width=True, hide_index=True)
    st.download_button("Download facts JSON", data=json.dumps(facts, indent=2), file_name="drawing_facts.json", mime="application/json")


def render_analysis_results() -> None:
    loaded = load_facts_or_notice()
    if not loaded:
        st.info("Choose a drawing and select Analyze drawing to create a complete result package.")
        return
    facts, facts_path = loaded
    summary = facts.get("summary", {})
    metrics = st.columns(4)
    metrics[0].metric("Detected features", summary.get("predicted_primitive_count", 0))
    metrics[1].metric("Average confidence", summary.get("mean_detector_confidence", "—"))
    metrics[2].metric("Review findings", summary.get("rule_finding_count", 0))
    metrics[3].metric("Closed profiles", summary.get("closed_line_profile_count", 0))
    image = as_path(str(facts.get("drawing", {}).get("image_path", "")))
    if image.is_file():
        original, overlay = st.columns(2)
        original.image(str(image), caption="Original drawing", use_container_width=True)
        try:
            overlay.image(render_prediction_overlay(image, facts), caption="Detected geometry", use_container_width=True)
        except OSError as exc:
            st.warning(f"Could not render the overlay: {exc}")
    findings = facts.get("checks", {}).get("findings", [])
    if findings:
        st.subheader("Items to review")
        st.dataframe(findings, use_container_width=True, hide_index=True)
    with st.expander("Detected feature details"):
        rows = []
        for feature in facts.get("features", {}).get("predicted_primitives", []):
            rows.append({key: value for key, value in feature.items() if key != "geometry"} | dict(feature.get("geometry") or {}))
        st.dataframe(rows, use_container_width=True, hide_index=True)
    st.download_button("Download analysis JSON", data=json.dumps(facts, indent=2), file_name=facts_path.name, mime="application/json")


def guided_analysis_tab() -> None:
    """Minimal, optimized path from a selected image to facts and overlay."""
    st.subheader("Analyze a drawing")
    st.caption("Choose one dataset drawing or upload a file. The workbench automatically uses the best local detector, CUDA when available, and a compact vision-ready facts package.")
    source = st.radio("Drawing source", ["ParaCAD dataset", "Upload image"], horizontal=True)
    image_path = ""
    uploaded = None
    if source == "ParaCAD dataset":
        dataset_path = automatic_dataset()
        if dataset_path is None:
            fallback = st.text_input("ParaCAD Zarr dataset folder")
            dataset_path = as_path(fallback) if fallback.strip() else None
        if dataset_path is None or not dataset_path.is_dir():
            st.error("No ParaCAD Zarr dataset was found. Select an existing dataset folder to continue.")
        else:
            st.caption(f"Using dataset: {dataset_path.name}")
            with st.expander("Choose a dataset drawing", expanded=True):
                split = st.selectbox("Split", ["val", "train", "test", "all"], index=0, key="guided_split")
                try:
                    total = zarr_split_size(str(dataset_path), split)
                    page_count = max(1, (total + 49) // 50)
                    current_page = int(st.session_state.get("guided_page", 1))
                    if current_page > page_count:
                        st.session_state.guided_page = 1
                    page = int(st.number_input("Page", min_value=1, max_value=page_count, value=int(st.session_state.get("guided_page", 1)), key="guided_page"))
                    records = zarr_image_page(str(dataset_path), split, page - 1, 50)
                    records = [record for record in records if record[1] >= 0]
                    st.caption(f"Showing page {page:,} of {page_count:,} ({total:,} drawings).")
                    if not records:
                        st.info("This split has no drawings on the selected page.")
                    else:
                        selection = st.selectbox(
                            "Drawing",
                            records,
                            format_func=lambda record: f"Record {record[0]:,}  |  image {record[1]:,}",
                            key="guided_record",
                        )
                        if st.button("Use this drawing", type="primary"):
                            extracted = extract_zarr_images(dataset_path, [selection], workspace_path("dataset_images"))
                            st.session_state.guided_image_path = str(extracted[0])
                            st.success(f"Loaded {extracted[0].name}")
                except (ImportError, OSError, ValueError, KeyError) as exc:
                    st.error(f"Could not read this dataset: {exc}")
        image_path = str(st.session_state.get("guided_image_path", ""))
    else:
        uploaded = st.file_uploader("Drawing image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="guided_upload")
        if uploaded is not None:
            st.caption(f"Ready to analyze: {uploaded.name}")
        image_path = ""

    if source == "ParaCAD dataset" and as_path(image_path).is_file():
        st.image(image_path, caption=Path(image_path).name, width=340)
    checkpoint = automatic_checkpoint()
    if checkpoint is None:
        st.error("No detector checkpoint was found. Place a trained best.pt in checkpoints or use the Advanced area.")
        return
    if st.button("Analyze drawing", type="primary", disabled=uploaded is None and not as_path(image_path).is_file()):
        if uploaded is not None:
            upload_name = f"upload_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{Path(uploaded.name).name}"
            destination = workspace_path("uploads") / upload_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(uploaded.getvalue())
            image_path = str(destination)
            st.session_state.guided_image_path = image_path
        source_image = as_path(image_path)
        run_dir = workspace_path("runs") / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{source_image.stem}"
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = run_dir / "predicted_primitives.json"
        facts_path = run_dir / "drawing_facts.json"
        command = [
            sys.executable,
            str(ROOT / "paracad_primitive_detr.py"),
            "predict",
            "--checkpoint",
            str(checkpoint),
            "--image",
            str(source_image),
            "--output",
            str(prediction_path),
            "--device",
            "auto",
            "--no-progress",
        ]
        if run_process(command, "Detecting drawing features"):
            try:
                facts = build_drawing_facts(source_image, prediction_path, use_ocr=ocr_available())
                write_facts(facts, facts_path)
                st.session_state.facts_path = str(facts_path)
                st.session_state.guided_last_run = str(run_dir)
                st.success("Analysis complete. Ask a question in Copilot or review the findings below.")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                st.error(f"Prediction completed, but facts could not be built: {exc}")
    st.divider()
    render_analysis_results()


def chat_tab(settings: dict[str, Any]) -> None:
    st.subheader("Grounded engineering copilot")
    loaded = load_facts_or_notice()
    if not loaded:
        return
    facts, facts_path = loaded
    try:
        tools, messages, evidence = get_copilot(facts, facts_path, settings)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not prepare copilot session: {exc}")
        return
    if evidence:
        st.caption("Vision evidence: " + ", ".join(item["id"] for item in evidence))
    if st.button("Clear chat"):
        st.session_state.pop("copilot_identity", None)
        st.rerun()
    for message in st.session_state.get("chat_display", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Ask about features, risk checks, dimensions, standards, or skills")
    if prompt:
        st.session_state.chat_display.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Qwen is reviewing grounded evidence..."):
                try:
                    answer = ask_ollama(
                        facts,
                        settings["model"],
                        settings["ollama_url"],
                        prompt,
                        0.0,
                        settings["num_ctx"],
                        settings["timeout"],
                        tools=tools,
                        messages=messages,
                        vision_evidence=evidence,
                    )
                except Exception as exc:  # Surface model/backend errors to the user without ending the app.
                    answer = f"**Error:** {exc}"
            st.markdown(answer)
        st.session_state.chat_display.append({"role": "assistant", "content": answer})


def feedback_tab() -> None:
    st.subheader("Engineer feedback")
    loaded = load_facts_or_notice()
    if not loaded:
        return
    facts, facts_path = loaded
    feedback_path = as_path(st.text_input("Feedback JSONL", value=str(workspace_path("drawing_feedback.jsonl"))))
    features = facts.get("features", {}).get("predicted_primitives", [])
    ids = [feature["feature_id"] for feature in features]
    with st.form("feedback_form"):
        status = st.selectbox("Decision", ["accepted", "rejected", "corrected", "missing"])
        feature_id = st.selectbox("Detected feature", ids, disabled=status == "missing") if ids else ""
        primitive_type = st.selectbox("Primitive type", ["line", "circle", "arc"])
        geometry = st.text_area("Corrected/missing geometry JSON", placeholder='{"x1": 0.1, "y1": 0.2, "x2": 0.6, "y2": 0.2}')
        reviewer = st.text_input("Reviewer", value="engineer")
        note = st.text_area("Review note")
        submitted = st.form_submit_button("Save review decision")
    if submitted:
        try:
            selected = next((item for item in features if item["feature_id"] == feature_id), None)
            if status == "missing":
                corrected = parse_geometry(geometry, primitive_type)
                selected_id = None
            elif selected is None:
                raise ValueError("Select a detected feature.")
            elif status == "corrected":
                primitive_type = selected["type"]
                corrected = parse_geometry(geometry, primitive_type)
                selected_id = feature_id
            else:
                corrected = None
                primitive_type = selected["type"]
                selected_id = feature_id
            append_feedback(
                feedback_path,
                {
                    "schema_version": "drawing-feedback/v1",
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "drawing_image_path": facts["drawing"]["image_path"],
                    "facts_path": str(facts_path),
                    "status": status,
                    "feature_id": selected_id,
                    "primitive_type": primitive_type,
                    "corrected_geometry": corrected,
                    "note": note,
                    "reviewer": reviewer,
                },
            )
            st.success("Review decision appended.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    events = feedback_for_drawing(feedback_path, facts["drawing"]["image_path"])
    st.dataframe(events, use_container_width=True, hide_index=True)
    first, second = st.columns(2)
    with first:
        if st.button("Export reviewed labels"):
            output = workspace_path("reviewed_labels.jsonl")
            labels = export_labels(facts, events, include_unreviewed=False)
            output.write_text(json.dumps(labels) + "\n", encoding="utf-8")
            st.success(f"Wrote {output}")
    with second:
        if st.button("Export copilot SFT example"):
            output = workspace_path("copilot_sft.jsonl")
            with output.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(example_for(facts, events)) + "\n")
            st.success(f"Appended reviewed SFT example to {output}")


def knowledge_tab() -> None:
    st.subheader("Approved engineering knowledge")
    source_dir = st.text_input("Approved text-reference folder", value="")
    output = st.text_input("Knowledge-index output", value=str(workspace_path("engineering_knowledge.json")))
    if st.button("Build or refresh knowledge index"):
        try:
            if not source_dir.strip():
                raise ValueError("Choose a dedicated approved-reference folder; the workspace root is not used implicitly.")
            index, warnings = build_index(as_path(source_dir))
            write_index(index, as_path(output))
            st.session_state.knowledge_index_path = output
            st.success(f"Indexed {index['chunk_count']} chunks.")
            for warning in warnings:
                st.warning(warning)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    index_path = st.text_input("Search index", value=st.session_state.get("knowledge_index_path", ""))
    query = st.text_input("Search approved references")
    if query and st.button("Search references"):
        try:
            st.json(KnowledgeIndex.load(as_path(index_path)).search(query))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.error(str(exc))


def skills_tab(settings: dict[str, Any]) -> None:
    st.subheader("Reusable skills")
    loaded = load_facts_or_notice()
    if not loaded:
        return
    facts, facts_path = loaded
    try:
        tools, _, _ = get_copilot(facts, facts_path, settings)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        return
    library = tools.list_skills()
    st.dataframe(library["skills"], use_container_width=True, hide_index=True)
    names = [skill["name"] for skill in library["skills"]]
    if names:
        selected = st.selectbox("Inspect or run", names)
        st.json(tools.get_skill(selected))
        raw_inputs = st.text_input("Skill inputs JSON", value="{}")
        if st.button("Run selected skill"):
            try:
                st.json(tools.run_skill(selected, json.loads(raw_inputs)))
            except json.JSONDecodeError as exc:
                st.error(str(exc))
    st.caption("Draft custom skills through chat, or paste a manifest below. Saving is enabled only by the sidebar permission checkbox.")
    draft_json = st.text_area("Custom skill draft JSON", placeholder='{"name":"drawing-triage","description":"...","instructions":"...","tool_plan":[{"tool":"check_rules","arguments":{}}]}')
    if st.button("Create skill draft from JSON") and draft_json:
        try:
            draft = tools.create_skill_draft(**json.loads(draft_json))
            st.session_state.skill_draft_id = draft.get("draft_id")
            st.json(draft)
        except (json.JSONDecodeError, TypeError) as exc:
            st.error(str(exc))
    draft_id = st.session_state.get("skill_draft_id")
    if draft_id and st.button(f"Save skill draft {draft_id}"):
        st.json(tools.save_skill(draft_id))


def discover_artifacts() -> list[Path]:
    """Return recent, readable outputs without recursively scanning the raw dataset."""
    files: dict[Path, Path] = {}
    if WORKSPACE.is_dir():
        for path in WORKSPACE.rglob("*"):
            if path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES:
                files[path.resolve()] = path
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES:
            files[path.resolve()] = path
    return sorted(files.values(), key=lambda path: path.stat().st_mtime, reverse=True)[:200]


def artifact_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def render_structured_artifact(path: Path, payload: Any) -> None:
    """Give common program outputs a useful engineering-oriented summary."""
    if not isinstance(payload, dict):
        st.json(payload)
        return
    if "drawing" in payload and "summary" in payload:
        summary = payload.get("summary", {})
        metrics = st.columns(4)
        metrics[0].metric("Predicted primitives", summary.get("predicted_primitive_count", 0))
        metrics[1].metric("Mean confidence", summary.get("mean_detector_confidence", "—"))
        metrics[2].metric("Review findings", summary.get("rule_finding_count", 0))
        metrics[3].metric("Ground-truth matches", summary.get("matched_prediction_count", "—"))
        st.dataframe(payload.get("checks", {}).get("findings", []), use_container_width=True, hide_index=True)
        if st.button("Use this facts file in the workbench", key=f"use_facts_{path}"):
            st.session_state.pending_facts_path = str(path)
            st.rerun()
    elif "primitives" in payload:
        primitives = payload.get("primitives", [])
        st.metric("Primitives", len(primitives) if isinstance(primitives, list) else 0)
        if isinstance(primitives, list):
            rows = []
            for item in primitives:
                if not isinstance(item, dict):
                    continue
                rows.append({key: value for key, value in item.items() if key != "geometry"} | dict(item.get("geometry") or {}))
            st.dataframe(rows, use_container_width=True, hide_index=True)
    elif "f1" in payload or "precision" in payload or "recall" in payload:
        keys = [key for key in ("test_samples", "precision", "recall", "f1", "matched_geometry_mae") if key in payload]
        metrics = st.columns(max(1, len(keys)))
        for column, key in zip(metrics, keys):
            column.metric(key.replace("_", " ").title(), payload[key])
    elif "stats" in payload:
        st.dataframe([payload.get("stats", {})], use_container_width=True, hide_index=True)
    st.json(payload)


def artifacts_tab() -> None:
    st.subheader("Outputs and analysis")
    st.caption("Inspect program outputs, logs, detector metrics, predictions, facts, and reviewed labels without leaving the workbench.")
    files = discover_artifacts()
    manual_path = st.text_input("Or inspect an output by path", value="")
    if manual_path.strip():
        selected = as_path(manual_path)
        if not selected.is_file():
            st.error("Choose an existing JSON, JSONL, text, or log file.")
            return
    elif files:
        selected = st.selectbox("Output artifact", files, format_func=artifact_label)
    else:
        st.info("No JSON, JSONL, text, or job-log artifacts were found in the workspace yet. Enter a path above to inspect another output.")
        return
    st.caption(f"{selected.stat().st_size:,} bytes  •  modified {datetime.fromtimestamp(selected.stat().st_mtime).isoformat(timespec='seconds')}")
    if selected.suffix.lower() == ".json":
        try:
            render_structured_artifact(selected, json.loads(selected.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read JSON: {exc}")
    else:
        text = log_tail(selected)
        result = last_json_line(text)
        if result is not None:
            st.caption("Structured result detected in this CLI output")
            render_structured_artifact(selected, result)
        st.code(text, language="text")
    st.download_button("Download selected artifact", data=selected.read_bytes(), file_name=selected.name)


def results_tab() -> None:
    st.subheader("Results")
    render_job_monitor("results")
    st.divider()
    artifacts_tab()


def model_tab(settings: dict[str, Any]) -> None:
    st.subheader("Model and data")
    dataset = automatic_dataset()
    checkpoint = automatic_checkpoint()
    first, second, third = st.columns(3)
    first.metric("Dataset", dataset.name if dataset else "Not found")
    second.metric("Checkpoint", checkpoint.name if checkpoint else "Not found")
    third.metric("Automatic workers", automatic_workers())
    st.caption("Inference uses `auto` device selection: CUDA when PyTorch can use it, otherwise CPU. Training automatically uses a GPU-native matcher and high-throughput worker defaults; evaluation remains exact for trustworthy metrics.")
    if st.button("Evaluate current model", type="primary", disabled=dataset is None or checkpoint is None):
        command = [
            sys.executable,
            str(ROOT / "paracad_primitive_detr.py"),
            "evaluate",
            "--dataset",
            str(dataset),
            "--checkpoint",
            str(checkpoint),
            "--batch-size",
            "32",
            "--num-workers",
            str(automatic_workers()),
            "--device",
            "auto",
            "--no-progress",
        ]
        if start_background_job("Detector evaluation", command):
            st.rerun()
    with st.expander("Advanced operations, training, skills, and knowledge"):
        st.caption("These controls expose the full project when you need to tune paths, run an alternate CLI command, retrain, or manage grounded knowledge.")
        advanced_operations, advanced_training, advanced_copilot = st.tabs(["Operations", "Training", "Copilot assets"])
        with advanced_operations:
            operations_tab()
        with advanced_training:
            training_tab()
        with advanced_copilot:
            knowledge_tab()
            st.divider()
            skills_tab(settings)


def operations_tab() -> None:
    st.subheader("Operations console")
    st.caption("Run one local project operation at a time. The app stores live output in `workbench\\jobs` and keeps a recent job history.")
    render_job_monitor("operations")
    st.divider()
    evaluate, features, advanced = st.tabs(["Evaluate detector", "Extract features", "Advanced CLI runner"])
    with evaluate:
        with st.form("evaluate_detector"):
            dataset = st.text_input("Evaluation dataset", value=str(ROOT / "ParaCAD_full_v3.zarr"))
            checkpoint = st.text_input("Evaluation checkpoint", value=str(ROOT / "checkpoints" / "best.pt"))
            first, second, third = st.columns(3)
            batch_size = first.number_input("Batch size", min_value=1, value=32)
            workers = second.number_input("Loader workers", min_value=0, value=automatic_workers())
            max_samples = third.number_input("Maximum samples (0 = all)", min_value=0, value=0)
            device = st.selectbox("Evaluation device", ["cuda", "auto", "cpu"])
            submitted = st.form_submit_button("Start detector evaluation", type="primary")
        if submitted:
            if not as_path(dataset).is_dir() or not as_path(checkpoint).is_file():
                st.error("Choose an existing Zarr dataset and checkpoint.")
            else:
                command = [
                    sys.executable,
                    str(ROOT / "paracad_primitive_detr.py"),
                    "evaluate",
                    "--dataset",
                    dataset,
                    "--checkpoint",
                    checkpoint,
                    "--batch-size",
                    str(batch_size),
                    "--num-workers",
                    str(workers),
                    "--max-samples",
                    str(max_samples),
                    "--device",
                    device,
                    "--no-progress",
                ]
                if start_background_job("Detector evaluation", command):
                    st.rerun()
    with features:
        with st.form("extract_features"):
            dataset = st.text_input("Feature dataset", value=str(ROOT / "ParaCAD_full_v3.zarr"))
            output = st.text_input("Feature JSONL output", value=str(workspace_path("drawing_features.jsonl")))
            first, second, third = st.columns(3)
            split = first.selectbox("Split", ["train", "val", "test", "all"], index=1)
            limit = second.number_input("Feature limit (0 = all)", min_value=0, value=100)
            batch_size = third.number_input("Feature batch size", min_value=1, value=512)
            include_primitives = st.checkbox("Include individual primitive geometry", value=False)
            submitted = st.form_submit_button("Start feature extraction", type="primary")
        if submitted:
            if not as_path(dataset).is_dir():
                st.error("Choose an existing Zarr dataset.")
            else:
                command = [
                    sys.executable,
                    str(ROOT / "extract_drawing_features.py"),
                    "--dataset",
                    dataset,
                    "--output",
                    output,
                    "--split",
                    split,
                    "--limit",
                    str(limit),
                    "--batch-size",
                    str(batch_size),
                    "--no-progress",
                ]
                if include_primitives:
                    command.append("--include-primitives")
                if start_background_job("Feature extraction", command):
                    st.rerun()
    with advanced:
        st.caption("Uses the selected project script directly—never a shell. Put one argument on each line so Windows paths with spaces remain unambiguous.")
        script_label = st.selectbox("Project CLI", list(CLI_SCRIPTS))
        arguments = st.text_area("Arguments (one per line)", value="--help", height=190)
        show_cli_progress = st.checkbox("Keep cursor-style Rich CLI progress in the job log", value=False)
        approved = st.checkbox("I reviewed the command and any output/overwrite paths", value=False)
        argument_values = [line.strip() for line in arguments.splitlines() if line.strip()]
        command = [sys.executable, str(ROOT / CLI_SCRIPTS[script_label]), *argument_values]
        if not show_cli_progress and "--no-progress" not in command:
            command.append("--no-progress")
        st.code(command_display(command), language="powershell")
        if st.button("Start selected CLI operation", type="primary", disabled=not approved):
            if start_background_job(script_label, command):
                st.rerun()


def training_tab() -> None:
    st.subheader("Training management")
    st.warning("Training can run for hours and consume GPU memory. This app starts one reviewed local job at a time; inspect its output in Operations or Outputs.")
    st.markdown("**Available training datasets**")
    summaries: list[dict[str, int | str | None]] = []
    errors: list[str] = []
    for dataset_path in local_training_datasets():
        try:
            summaries.append(zarr_dataset_summary(str(dataset_path)))
        except (ImportError, OSError, ValueError, KeyError) as exc:
            errors.append(f"{dataset_path.name}: {exc}")
    if summaries:
        st.dataframe(summaries, use_container_width=True, hide_index=True, column_config={"total": "Total samples", "train": "Train", "val": "Validation", "test": "Test"})
    else:
        st.info("No local .zarr training datasets were found.")
    for error in errors:
        st.warning(f"Could not read dataset counts — {error}")
    detector, feedback, lora = st.tabs(["Detector training", "Reviewed-feedback dataset", "Text Copilot QLoRA"])
    with detector:
        with st.form("detector_training"):
            dataset = st.text_input("Training Zarr dataset", value=str(ROOT / "ParaCAD_full_v3.zarr"))
            checkpoint_dir = st.text_input("Checkpoint output directory", value=str(ROOT / "checkpoints_full"))
            first, second, third = st.columns(3)
            epochs = first.number_input("Epochs", min_value=1, value=40)
            batch_size = second.number_input("Batch size", min_value=1, value=automatic_training_batch_size())
            workers = third.number_input("Loader workers", min_value=0, value=automatic_workers())
            first, second, third = st.columns(3)
            train_limit = first.number_input("Train samples (0 = all)", min_value=0, value=100_000)
            val_limit = second.number_input("Validation samples (0 = all)", min_value=0, value=10_000)
            device = third.selectbox("Training device", ["cuda", "auto", "cpu"])
            resume_last = st.checkbox("Resume from this folder's last checkpoint", value=False)
            confirmed = st.checkbox("I understand this starts a long-running detector training job.")
            submitted = st.form_submit_button("Start detector training", type="primary")
        if submitted:
            if not confirmed:
                st.error("Confirm the training launch first.")
            elif not as_path(dataset).is_dir():
                st.error("Choose an existing training Zarr dataset.")
            elif resume_last and not (as_path(checkpoint_dir) / "last.pt").is_file():
                st.error("No last.pt checkpoint exists in the selected checkpoint output folder.")
            else:
                command = [
                    sys.executable,
                    str(ROOT / "paracad_primitive_detr.py"),
                    "train",
                    "--dataset",
                    dataset,
                    "--checkpoint-dir",
                    checkpoint_dir,
                    "--epochs",
                    str(epochs),
                    "--batch-size",
                    str(batch_size),
                    "--num-workers",
                    str(workers),
                    "--matcher",
                    "gpu-greedy",
                    "--validation-matcher",
                    "exact",
                    "--train-max-samples",
                    str(train_limit),
                    "--val-max-samples",
                    str(val_limit),
                    "--device",
                    device,
                    "--no-progress",
                ]
                if resume_last:
                    command.extend(["--resume", str(as_path(checkpoint_dir) / "last.pt")])
                if start_background_job("Detector training", command):
                    st.rerun()
    with feedback:
        st.caption("First create a compact Zarr dataset from reviewed feedback. Then use Detector training with that Zarr and an `--init-checkpoint` through the Advanced CLI runner.")
        with st.form("feedback_dataset"):
            labels = st.text_input("Reviewed labels JSONL", value=str(workspace_path("reviewed_labels.jsonl")))
            image_root = st.text_input("Image root", value=str(ROOT))
            output = st.text_input("Reviewed-feedback Zarr output", value=str(ROOT / "reviewed_feedback.zarr"))
            val_ratio = st.slider("Validation ratio", min_value=0.05, max_value=0.45, value=0.20, step=0.05)
            confirmed = st.checkbox("I understand an existing output requires explicit overwrite in the Advanced runner.")
            submitted = st.form_submit_button("Build reviewed-feedback dataset", type="primary")
        if submitted:
            if not confirmed:
                st.error("Confirm the output review first.")
            elif not as_path(labels).is_file() or not as_path(image_root).is_dir():
                st.error("Choose an existing reviewed-label JSONL and image root.")
            else:
                command = [
                    sys.executable,
                    str(ROOT / "build_feedback_zarr.py"),
                    "--labels",
                    labels,
                    "--image-root",
                    image_root,
                    "--output",
                    output,
                    "--val-ratio",
                    str(val_ratio),
                    "--no-progress",
                ]
                if start_background_job("Build reviewed feedback Zarr", command):
                    st.rerun()
    with lora:
        st.caption("QLoRA trains a separate Transformers-format text model. It cannot fine-tune the Ollama GGUF directly; retain `qwen3.5:9b` for local vision/tool inference.")
        with st.form("lora_training"):
            base_model = st.text_input("Transformers base model or local path", value="Qwen/Qwen3-4B-Instruct-2507")
            dataset = st.text_input("Copilot SFT JSONL", value=str(workspace_path("copilot_sft.jsonl")))
            output = st.text_input("LoRA adapter output", value=str(ROOT / "adapters" / "drawing-copilot-lora"))
            first, second, third = st.columns(3)
            epochs = first.number_input("LoRA epochs", min_value=1.0, value=3.0, step=1.0)
            max_length = second.number_input("Maximum tokens", min_value=512, value=2048, step=256)
            accumulation = third.number_input("Gradient accumulation", min_value=1, value=16)
            confirmed = st.checkbox("I installed requirements-copilot-training.txt and understand this needs CUDA.")
            submitted = st.form_submit_button("Start QLoRA training", type="primary")
        if submitted:
            if not confirmed:
                st.error("Confirm the CUDA environment and training launch first.")
            elif not as_path(dataset).is_file():
                st.error("Choose an existing Copilot SFT JSONL dataset.")
            else:
                command = [
                    sys.executable,
                    str(ROOT / "train_copilot_lora.py"),
                    "--base-model",
                    base_model,
                    "--dataset",
                    dataset,
                    "--output",
                    output,
                    "--epochs",
                    str(epochs),
                    "--max-length",
                    str(max_length),
                    "--gradient-accumulation",
                    str(accumulation),
                    "--no-progress",
                ]
                if start_background_job("QLoRA training", command):
                    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Engineering Drawing Copilot", page_icon="📐", layout="wide")
    st.title("Engineering Drawing Copilot")
    st.caption("A focused local workbench for drawing analysis, engineering review, and grounded questions.")
    pending_facts = st.session_state.pop("pending_facts_path", None)
    if pending_facts:
        st.session_state.facts_path = pending_facts
    settings = sidebar()
    tabs = st.tabs(["Analyze", "Copilot", "Review", "Results", "Model"])
    with tabs[0]:
        guided_analysis_tab()
    with tabs[1]:
        chat_tab(settings)
    with tabs[2]:
        feedback_tab()
    with tabs[3]:
        results_tab()
    with tabs[4]:
        model_tab(settings)


if __name__ == "__main__":
    main()
