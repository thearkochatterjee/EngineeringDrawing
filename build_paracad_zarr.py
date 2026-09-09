#!/usr/bin/env python3
"""
Build an archive-aware Zarr dataset from ParaCAD-style engineering drawings.

The output is intended to be a long-lived, model-agnostic dataset artifact:

  - /images stores each encoded image once as raw PNG/JPEG bytes.
  - /records stores supervised examples that point into /images.
  - /primitives, /constraints, /dimensions, and /dimension_refs store flat
    numeric tables for detection, graph, and regression training.
  - JSON blobs preserve the exact structured labels and source metadata.

Examples:

  python build_paracad_zarr.py ^
    --input E:\\engdraw\\data\\ParaCAD\\data\\ParaCAD ^
    --output E:\\engdraw\\ParaCAD_full.zarr ^
    --variant-mode source

  python build_paracad_zarr.py ^
    --input E:\\engdraw\\data\\ParaCAD\\data\\ParaCAD ^
    --output E:\\engdraw\\ParaCAD_all_variants.zarr ^
    --variant-mode all

Install dependencies:

  python -m pip install zarr ijson

`ijson` is optional but strongly recommended for the multi-GB JSON files.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import struct
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import zarr  # type: ignore
except Exception as exc:  # pragma: no cover - dependency guard
    print(
        "[ERROR] Missing dependency 'zarr'. Install it with:\n"
        "  python -m pip install zarr ijson\n"
        f"Import error: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    import ijson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ijson = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    from rich.console import Console  # type: ignore
    from rich.progress import (  # type: ignore
        Progress,
        ProgressColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Console = None
    Progress = None
    ProgressColumn = object  # type: ignore
    Text = None  # type: ignore

try:
    from preprocess_paracad_parallel import (  # type: ignore
        add_dimension_associations,
        add_geometry_checks,
        extract_label_text,
        normalize_primitives,
        parse_assistant_output,
        primitive_id_from_ref,
    )
except Exception as exc:  # pragma: no cover - local script guard
    print(
        "[ERROR] Could not import parsing helpers from preprocess_paracad_parallel.py. "
        "Run this script from the repository root or keep both scripts together.\n"
        f"Import error: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
JSON_EXTS = {".json", ".jsonl"}

STYLE_TO_ID = {"unknown": 0, "color1": 1, "color2": 2, "white": 3}
ID_TO_STYLE = {v: k for k, v in STYLE_TO_ID.items()}

FORMAT_TO_ID = {"unknown": 0, "png": 1, "jpg": 2, "jpeg": 2, "webp": 3, "bmp": 4, "tif": 5, "tiff": 5}

PRIMITIVE_TO_ID = {"unknown": 0, "line": 1, "circle": 2, "arc": 3, "point": 4}
CONSTRAINT_TO_ID = {
    "unknown": 0,
    "Coincident": 1,
    "PointOnObject": 2,
    "Horizontal": 3,
    "Vertical": 4,
    "Parallel": 5,
    "Perpendicular": 6,
    "Tangent": 7,
    "Equal": 8,
}
DIMENSION_TO_ID = {"unknown": 0, "linear": 1, "diameter": 2, "radius": 3, "angular": 4}

SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
ID_TO_SPLIT = {v: k for k, v in SPLIT_TO_ID.items()}


def json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class AsciiBarColumn(ProgressColumn):  # type: ignore[misc, valid-type]
    def __init__(self, width: int = 28):
        super().__init__()
        self.width = width

    def render(self, task: Any) -> Any:
        if Text is None:
            return ""
        if task.total is None or task.total <= 0:
            return Text("[" + "." * self.width + "]")
        fraction = max(0.0, min(1.0, float(task.completed) / float(task.total)))
        filled = int(round(fraction * self.width))
        return Text("[" + "#" * filled + "-" * (self.width - filled) + "]")


class ProgressReporter:
    def __init__(
        self,
        enabled: bool = True,
        force_terminal: bool = True,
        console_width: int = 160,
    ):
        self.enabled = bool(enabled and Progress is not None)
        self.console = (
            Console(
                force_terminal=force_terminal,
                force_interactive=force_terminal,
                legacy_windows=False,
                color_system=None,
                no_color=True,
                emoji=False,
                highlight=False,
                width=console_width,
            )
            if self.enabled and Console is not None
            else None
        )
        self.progress = None
        if self.enabled:
            self.progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                AsciiBarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                refresh_per_second=2,
                transient=False,
            )

    def __enter__(self) -> "ProgressReporter":
        if self.progress is not None:
            self.progress.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.progress is not None:
            self.progress.stop()

    def add_task(self, description: str, total: Optional[float] = None) -> Optional[int]:
        if self.progress is None:
            return None
        return int(self.progress.add_task(description, total=total))

    def update(
        self,
        task_id: Optional[int],
        *,
        advance: Optional[float] = None,
        completed: Optional[float] = None,
        total: Optional[float] = None,
        description: Optional[str] = None,
    ) -> None:
        if self.progress is None or task_id is None:
            return
        kwargs: Dict[str, Any] = {}
        if advance is not None:
            kwargs["advance"] = advance
        if completed is not None:
            kwargs["completed"] = completed
        if total is not None:
            kwargs["total"] = total
        if description is not None:
            kwargs["description"] = description
        if kwargs:
            self.progress.update(task_id, **kwargs)

    def log(self, message: str, *, error: bool = False) -> None:
        if error:
            print(message, file=sys.stderr)
            return
        if self.console is not None:
            self.console.print(message)
        else:
            print(message)


class ProgressFile:
    def __init__(
        self,
        handle: Any,
        progress: ProgressReporter,
        task_ids: Sequence[Optional[int]],
    ):
        self.handle = handle
        self.progress = progress
        self.task_ids = list(task_ids)

    def read(self, size: int = -1) -> bytes:
        data = self.handle.read(size)
        if data:
            n = len(data)
            for task_id in self.task_ids:
                self.progress.update(task_id, advance=n)
        return data

    def readinto(self, buffer: Any) -> int:
        n = self.handle.readinto(buffer)
        if n:
            for task_id in self.task_ids:
                self.progress.update(task_id, advance=n)
        return n

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "ProgressFile":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.handle, name)


def infer_style(text: str | Path) -> str:
    s = str(text).replace("\\", "/").lower()
    if "color1" in s or "16_color1" in s:
        return "color1"
    if "color2" in s or "16_color2" in s:
        return "color2"
    if "white" in s or "sg6-16_white" in s:
        return "white"
    return "unknown"


def infer_image_format(name: str, data: Optional[bytes] = None) -> str:
    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix in FORMAT_TO_ID:
        return suffix
    if data:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if data.startswith(b"\xff\xd8"):
            return "jpg"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "webp"
        if data.startswith(b"BM"):
            return "bmp"
    return "unknown"


def needed_key_for_image(basename: str, style: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(bytes([STYLE_TO_ID.get(style, 0)]))
    h.update(b"\0")
    h.update(basename.lower().encode("utf-8", errors="ignore"))
    return int.from_bytes(h.digest(), "big")


def needed_key_for_basename(basename: str) -> int:
    digest = hashlib.blake2b(
        basename.lower().encode("utf-8", errors="ignore"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def image_is_needed(
    basename: str,
    style: str,
    needed_image_keys: Optional[set[int]],
    needed_basenames: Optional[set[int]],
) -> bool:
    if needed_image_keys is None and needed_basenames is None:
        return True
    if needed_basenames is not None and needed_key_for_basename(basename) in needed_basenames:
        return True
    if needed_image_keys is not None and needed_key_for_image(basename, style) in needed_image_keys:
        return True
    return False


def drawing_id_from_image_name(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return stem


def stable_split(drawing_id: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.blake2b(drawing_id.encode("utf-8", errors="ignore"), digest_size=8).digest()
    x = int.from_bytes(digest, "big") / float(1 << 64)
    if x < train_ratio:
        return "train"
    if x < train_ratio + val_ratio:
        return "val"
    return "test"


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def should_skip_dir(path: Path, output_path: Path, skip_names: set[str]) -> bool:
    name = path.name
    if name in skip_names:
        return True
    try:
        path.resolve().relative_to(output_path.resolve())
        return True
    except Exception:
        return False


def iter_files(root: Path, exts: set[str], output_path: Path, skip_names: set[str]) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames if not should_skip_dir(current / d, output_path, skip_names)
        ]
        for filename in filenames:
            p = current / filename
            if p.suffix.lower() in exts:
                yield p


def is_tar_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tar") or name.endswith(".tar.gz") or name.endswith(".tgz")


def discover_tar_archives(root: Path, output_path: Path, skip_names: set[str]) -> List[Path]:
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames if not should_skip_dir(current / d, output_path, skip_names)
        ]
        for filename in filenames:
            p = current / filename
            if is_tar_archive(p):
                out.append(p)
    return sorted(out)


def discover_split_archive_groups(root: Path, output_path: Path, skip_names: set[str]) -> List[List[Path]]:
    groups: Dict[str, List[Path]] = {}
    pattern = re.compile(r"^(?P<prefix>.+)_part_(?P<suffix>[a-z]{2})$", re.I)
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames if not should_skip_dir(current / d, output_path, skip_names)
        ]
        for filename in filenames:
            m = pattern.match(filename)
            if not m:
                continue
            p = current / filename
            groups.setdefault(str(current / m.group("prefix")), []).append(p)

    out: List[List[Path]] = []
    for paths in groups.values():
        parts = sorted(paths, key=lambda p: p.name.lower())
        if len(parts) >= 2:
            out.append(parts)
    return out


class ConcatenatedFile:
    """Minimal read-only file object for split tar.gz parts."""

    def __init__(self, paths: Sequence[Path]):
        self.paths = list(paths)
        self.index = 0
        self.handle: Optional[Any] = None

    def readable(self) -> bool:
        return True

    def _open_next(self) -> bool:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.index >= len(self.paths):
            return False
        self.handle = self.paths[self.index].open("rb")
        self.index += 1
        return True

    def read(self, size: int = -1) -> bytes:
        chunks: List[bytes] = []
        remaining = size

        while size < 0 or remaining > 0:
            if self.handle is None and not self._open_next():
                break

            assert self.handle is not None
            chunk = self.handle.read(-1 if size < 0 else remaining)
            if chunk:
                chunks.append(chunk)
                if size >= 0:
                    remaining -= len(chunk)
                continue

            if not self._open_next():
                break

        return b"".join(chunks)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def fast_image_size_bytes(data: bytes) -> Tuple[int, int]:
    try:
        head = data[:32]
        if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
            width, height = struct.unpack(">II", head[16:24])
            return int(width), int(height)

        if head[:2] == b"\xff\xd8":
            bio = io.BytesIO(data)
            bio.seek(2)
            while True:
                marker_start = bio.read(1)
                if not marker_start:
                    break
                if marker_start != b"\xff":
                    continue
                marker = bio.read(1)
                while marker == b"\xff":
                    marker = bio.read(1)
                if not marker:
                    break
                marker_int = marker[0]
                if marker_int in {0xD8, 0xD9}:
                    continue
                size_bytes = bio.read(2)
                if len(size_bytes) != 2:
                    break
                segment_size = struct.unpack(">H", size_bytes)[0]
                if segment_size < 2:
                    break
                if marker_int in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                    frame = bio.read(5)
                    if len(frame) == 5:
                        height, width = struct.unpack(">HH", frame[1:5])
                        return int(width), int(height)
                    break
                bio.seek(segment_size - 2, os.SEEK_CUR)

        if head.startswith(b"BM") and len(head) >= 26:
            width = struct.unpack("<I", head[18:22])[0]
            height = abs(struct.unpack("<i", head[22:26])[0])
            return int(width), int(height)

        if head.startswith((b"GIF87a", b"GIF89a")) and len(head) >= 10:
            width, height = struct.unpack("<HH", head[6:10])
            return int(width), int(height)
    except Exception:
        pass

    if Image is not None:
        try:
            with Image.open(io.BytesIO(data)) as img:
                return int(img.width), int(img.height)
        except Exception:
            pass

    return -1, -1


def iter_json_records_stream(
    path: Path,
    chunk_size: int = 1 << 20,
    progress: Optional[ProgressReporter] = None,
    file_task: Optional[int] = None,
    overall_task: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    reporter = progress or ProgressReporter(enabled=False)

    if path.suffix.lower() == ".jsonl":
        with path.open("rb") as f:
            for line in f:
                reporter.update(file_task, advance=len(line))
                reporter.update(overall_task, advance=len(line))
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
        return

    if ijson is not None:
        try:
            with path.open("rb") as raw:
                first = raw.read(1)
                raw.seek(0)
                f = ProgressFile(raw, reporter, [file_task, overall_task])
                if first == b"[":
                    for obj in ijson.items(f, "item"):
                        if isinstance(obj, dict):
                            yield obj
                    return
                obj = json.load(f)
                if isinstance(obj, dict):
                    yield obj
                return
        except Exception:
            pass

    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    buffer = ""
    eof = False

    def fill() -> None:
        nonlocal buffer, eof
        if eof:
            return
        chunk = handle.read(chunk_size)
        if not chunk:
            buffer += utf8.decode(b"", final=True)
            eof = True
        else:
            buffer += utf8.decode(chunk)

    with path.open("rb") as raw_handle:
        handle = ProgressFile(raw_handle, reporter, [file_task, overall_task])
        fill()
        buffer = buffer.lstrip()
        while not buffer and not eof:
            fill()
            buffer = buffer.lstrip()

        if not buffer:
            return

        if buffer[0] != "[":
            while not eof:
                fill()
            try:
                obj = json.loads(buffer)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                return
            return

        buffer = buffer[1:]

        while True:
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

            if not buffer:
                return
            if buffer[0] == "]":
                return
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue

            while True:
                try:
                    obj, end = decoder.raw_decode(buffer)
                    buffer = buffer[end:]
                    if isinstance(obj, dict):
                        yield obj
                    break
                except json.JSONDecodeError:
                    if eof:
                        return
                    fill()


def zarr_create_array(group: Any, name: str, shape: Tuple[int, ...], chunks: Tuple[int, ...], dtype: Any, fill_value: Any = 0) -> Any:
    kwargs = {
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "fill_value": fill_value,
        "overwrite": True,
    }
    if hasattr(group, "create_array"):
        try:
            return group.create_array(name, **kwargs)
        except TypeError:
            pass
    return group.create_dataset(name, **kwargs)


def append_array(arr: Any, values: np.ndarray) -> int:
    values = np.asarray(values, dtype=arr.dtype)
    if values.size == 0:
        return int(arr.shape[0])
    old = int(arr.shape[0])
    new_shape = (old + int(values.shape[0]),) + tuple(arr.shape[1:])
    arr.resize(new_shape)
    arr[old:new_shape[0]] = values
    return old


class BlobStore:
    def __init__(self, group: Any, prefix: str, data_chunk: int = 8 << 20, offset_chunk: int = 100_000):
        self.data = zarr_create_array(group, f"{prefix}_bytes", (0,), (data_chunk,), np.uint8, 0)
        self.offsets = zarr_create_array(group, f"{prefix}_offsets", (1,), (offset_chunk,), np.uint64, 0)
        self.offsets[0] = 0

    def append(self, data: bytes) -> Tuple[int, int]:
        start = int(self.offsets[int(self.offsets.shape[0]) - 1])
        if data:
            append_array(self.data, np.frombuffer(data, dtype=np.uint8))
        end = start + len(data)
        old = int(self.offsets.shape[0])
        self.offsets.resize((old + 1,))
        self.offsets[old] = end
        return start, end

    def append_many(self, blobs: Sequence[bytes]) -> None:
        if not blobs:
            return
        start = int(self.offsets[int(self.offsets.shape[0]) - 1])
        lengths = np.fromiter((len(blob) for blob in blobs), dtype=np.uint64, count=len(blobs))
        if int(lengths.sum()) > 0:
            joined = b"".join(blobs)
            append_array(self.data, np.frombuffer(joined, dtype=np.uint8))
        cumulative = start + np.cumsum(lengths, dtype=np.uint64)
        old = int(self.offsets.shape[0])
        self.offsets.resize((old + len(blobs),))
        self.offsets[old : old + len(blobs)] = cumulative


class ZarrWriter:
    def __init__(self, output: Path, overwrite: bool, zarr_format: int):
        mode = "w" if overwrite else "w-"
        try:
            self.root = zarr.open_group(str(output), mode=mode, zarr_format=zarr_format)
        except TypeError:
            self.root = zarr.open_group(str(output), mode=mode)

        self.images = self.root.require_group("images")
        self.records = self.root.require_group("records")
        self.primitives = self.root.require_group("primitives")
        self.constraints = self.root.require_group("constraints")
        self.dimensions = self.root.require_group("dimensions")
        self.dimension_refs = self.root.require_group("dimension_refs")
        self.splits = self.root.require_group("splits")

        self.image_bytes = BlobStore(self.images, "data")
        self.image_meta = BlobStore(self.images, "meta_json")
        self.image_width = zarr_create_array(self.images, "width", (0,), (100_000,), np.int32, -1)
        self.image_height = zarr_create_array(self.images, "height", (0,), (100_000,), np.int32, -1)
        self.image_style = zarr_create_array(self.images, "style", (0,), (100_000,), np.uint8, 0)
        self.image_format = zarr_create_array(self.images, "format", (0,), (100_000,), np.uint8, 0)
        self._image_data_buffer: List[bytes] = []
        self._image_meta_buffer: List[bytes] = []
        self._image_width_buffer: List[int] = []
        self._image_height_buffer: List[int] = []
        self._image_style_buffer: List[int] = []
        self._image_format_buffer: List[int] = []
        self._image_buffer_bytes = 0
        self._image_batch_count = 2048
        self._image_batch_bytes = 128 << 20

        self.record_label = BlobStore(self.records, "label_json")
        self.record_raw_label = BlobStore(self.records, "raw_label")
        self.record_meta = BlobStore(self.records, "meta_json")
        self.record_image_id = zarr_create_array(self.records, "image_id", (0,), (100_000,), np.int64, -1)
        self.record_width = zarr_create_array(self.records, "image_width", (0,), (100_000,), np.int32, -1)
        self.record_height = zarr_create_array(self.records, "image_height", (0,), (100_000,), np.int32, -1)
        self.record_split = zarr_create_array(self.records, "split", (0,), (100_000,), np.uint8, 0)
        self.record_style = zarr_create_array(self.records, "style", (0,), (100_000,), np.uint8, 0)
        self.record_primitive_offset = zarr_create_array(self.records, "primitive_offset", (0,), (100_000,), np.int64, 0)
        self.record_primitive_count = zarr_create_array(self.records, "primitive_count", (0,), (100_000,), np.int32, 0)
        self.record_constraint_offset = zarr_create_array(self.records, "constraint_offset", (0,), (100_000,), np.int64, 0)
        self.record_constraint_count = zarr_create_array(self.records, "constraint_count", (0,), (100_000,), np.int32, 0)
        self.record_dimension_offset = zarr_create_array(self.records, "dimension_offset", (0,), (100_000,), np.int64, 0)
        self.record_dimension_count = zarr_create_array(self.records, "dimension_count", (0,), (100_000,), np.int32, 0)
        self._record_label_buffer: List[bytes] = []
        self._record_raw_label_buffer: List[bytes] = []
        self._record_meta_buffer: List[bytes] = []
        self._record_image_id_buffer: List[int] = []
        self._record_width_buffer: List[int] = []
        self._record_height_buffer: List[int] = []
        self._record_split_buffer: List[int] = []
        self._record_style_buffer: List[int] = []
        self._record_primitive_offset_buffer: List[int] = []
        self._record_primitive_count_buffer: List[int] = []
        self._record_constraint_offset_buffer: List[int] = []
        self._record_constraint_count_buffer: List[int] = []
        self._record_dimension_offset_buffer: List[int] = []
        self._record_dimension_count_buffer: List[int] = []
        self._record_batch_count = 4096

        self.primitive_record = zarr_create_array(self.primitives, "record_index", (0,), (500_000,), np.int64, 0)
        self.primitive_local = zarr_create_array(self.primitives, "local_index", (0,), (500_000,), np.int32, 0)
        self.primitive_type = zarr_create_array(self.primitives, "type", (0,), (500_000,), np.uint8, 0)
        self.primitive_geom = zarr_create_array(self.primitives, "geometry", (0, 7), (100_000, 7), np.float32, np.nan)
        self.primitive_geom_norm = zarr_create_array(self.primitives, "geometry_norm", (0, 7), (100_000, 7), np.float32, np.nan)
        self.primitive_valid = zarr_create_array(self.primitives, "is_valid", (0,), (500_000,), np.int8, -1)
        self._primitive_record_buffer: List[int] = []
        self._primitive_local_buffer: List[int] = []
        self._primitive_type_buffer: List[int] = []
        self._primitive_geom_buffer: List[List[float]] = []
        self._primitive_geom_norm_buffer: List[List[float]] = []
        self._primitive_valid_buffer: List[int] = []

        self.constraint_record = zarr_create_array(self.constraints, "record_index", (0,), (500_000,), np.int64, 0)
        self.constraint_type = zarr_create_array(self.constraints, "type", (0,), (500_000,), np.uint8, 0)
        self.constraint_source = zarr_create_array(self.constraints, "source_local_index", (0,), (500_000,), np.int32, -1)
        self.constraint_target = zarr_create_array(self.constraints, "target_local_index", (0,), (500_000,), np.int32, -1)
        self.constraint_pt1 = zarr_create_array(self.constraints, "point_type1", (0,), (500_000,), np.int8, -1)
        self.constraint_pt2 = zarr_create_array(self.constraints, "point_type2", (0,), (500_000,), np.int8, -1)
        self._constraint_record_buffer: List[int] = []
        self._constraint_type_buffer: List[int] = []
        self._constraint_source_buffer: List[int] = []
        self._constraint_target_buffer: List[int] = []
        self._constraint_pt1_buffer: List[int] = []
        self._constraint_pt2_buffer: List[int] = []

        self.dimension_record = zarr_create_array(self.dimensions, "record_index", (0,), (200_000,), np.int64, 0)
        self.dimension_type = zarr_create_array(self.dimensions, "type", (0,), (200_000,), np.uint8, 0)
        self.dimension_value = zarr_create_array(self.dimensions, "value", (0,), (200_000,), np.float32, np.nan)
        self.dimension_computed = zarr_create_array(self.dimensions, "computed_from_geometry", (0,), (200_000,), np.float32, np.nan)
        self.dimension_ref_offset = zarr_create_array(self.dimensions, "ref_offset", (0,), (200_000,), np.int64, 0)
        self.dimension_ref_count_arr = zarr_create_array(self.dimensions, "ref_count", (0,), (200_000,), np.int32, 0)
        self._dimension_record_buffer: List[int] = []
        self._dimension_type_buffer: List[int] = []
        self._dimension_value_buffer: List[float] = []
        self._dimension_computed_buffer: List[float] = []
        self._dimension_ref_offset_buffer: List[int] = []
        self._dimension_ref_count_buffer: List[int] = []

        self.dimref_dimension = zarr_create_array(self.dimension_refs, "dimension_index", (0,), (500_000,), np.int64, 0)
        self.dimref_primitive = zarr_create_array(self.dimension_refs, "primitive_local_index", (0,), (500_000,), np.int32, -1)
        self.dimref_point_type = zarr_create_array(self.dimension_refs, "point_type", (0,), (500_000,), np.int8, -1)
        self._dimref_dimension_buffer: List[int] = []
        self._dimref_primitive_buffer: List[int] = []
        self._dimref_point_type_buffer: List[int] = []

        self.split_arrays = {
            name: zarr_create_array(self.splits, name, (0,), (100_000,), np.int64, 0)
            for name in ("train", "val", "test")
        }
        self._split_buffers: Dict[str, List[int]] = {name: [] for name in ("train", "val", "test")}

    @property
    def image_count(self) -> int:
        return int(self.image_width.shape[0]) + len(self._image_width_buffer)

    @property
    def record_count(self) -> int:
        return int(self.record_image_id.shape[0]) + len(self._record_image_id_buffer)

    @property
    def primitive_count(self) -> int:
        return int(self.primitive_record.shape[0]) + len(self._primitive_record_buffer)

    @property
    def constraint_count(self) -> int:
        return int(self.constraint_record.shape[0]) + len(self._constraint_record_buffer)

    @property
    def dimension_count(self) -> int:
        return int(self.dimension_record.shape[0]) + len(self._dimension_record_buffer)

    @property
    def dimension_ref_count(self) -> int:
        return int(self.dimref_dimension.shape[0]) + len(self._dimref_dimension_buffer)

    def append_image(self, data: bytes, width: int, height: int, style: str, image_format: str, meta: Dict[str, Any]) -> int:
        image_id = self.image_count
        meta = dict(meta)
        meta["image_id"] = image_id
        self._image_data_buffer.append(data)
        self._image_meta_buffer.append(json_dumps(meta))
        self._image_width_buffer.append(width)
        self._image_height_buffer.append(height)
        self._image_style_buffer.append(STYLE_TO_ID.get(style, 0))
        self._image_format_buffer.append(FORMAT_TO_ID.get(image_format, 0))
        self._image_buffer_bytes += len(data)
        if (
            len(self._image_width_buffer) >= self._image_batch_count
            or self._image_buffer_bytes >= self._image_batch_bytes
        ):
            self.flush_images()
        return image_id

    def flush_images(self) -> None:
        if not self._image_width_buffer:
            return
        self.image_bytes.append_many(self._image_data_buffer)
        self.image_meta.append_many(self._image_meta_buffer)
        append_array(self.image_width, np.asarray(self._image_width_buffer, dtype=np.int32))
        append_array(self.image_height, np.asarray(self._image_height_buffer, dtype=np.int32))
        append_array(self.image_style, np.asarray(self._image_style_buffer, dtype=np.uint8))
        append_array(self.image_format, np.asarray(self._image_format_buffer, dtype=np.uint8))
        self._image_data_buffer.clear()
        self._image_meta_buffer.clear()
        self._image_width_buffer.clear()
        self._image_height_buffer.clear()
        self._image_style_buffer.clear()
        self._image_format_buffer.clear()
        self._image_buffer_bytes = 0

    def append_record(
        self,
        image_id: int,
        width: int,
        height: int,
        style: str,
        split: str,
        label: Dict[str, Any],
        raw_label: str,
        meta: Dict[str, Any],
    ) -> int:
        record_id = self.record_count
        primitive_offset = self.primitive_count
        constraint_offset = self.constraint_count
        dimension_offset = self.dimension_count

        primitives = label.get("primitives", []) or []
        constraints = label.get("constraints", []) or []
        dimensions = label.get("dimensions", []) or []
        primitive_lookup = {p.get("id"): i for i, p in enumerate(primitives) if p.get("id") is not None}

        self._append_primitives(record_id, primitives, width, height)
        self._append_constraints(record_id, constraints, primitive_lookup)
        self._append_dimensions(record_id, dimensions, primitive_lookup)

        self._record_label_buffer.append(json_dumps(label))
        self._record_raw_label_buffer.append(raw_label.encode("utf-8", errors="replace"))
        self._record_meta_buffer.append(json_dumps(meta))
        self._record_image_id_buffer.append(image_id)
        self._record_width_buffer.append(width)
        self._record_height_buffer.append(height)
        self._record_split_buffer.append(SPLIT_TO_ID[split])
        self._record_style_buffer.append(STYLE_TO_ID.get(style, 0))
        self._record_primitive_offset_buffer.append(primitive_offset)
        self._record_primitive_count_buffer.append(len(primitives))
        self._record_constraint_offset_buffer.append(constraint_offset)
        self._record_constraint_count_buffer.append(len(constraints))
        self._record_dimension_offset_buffer.append(dimension_offset)
        self._record_dimension_count_buffer.append(len(dimensions))
        self._split_buffers[split].append(record_id)
        if len(self._record_image_id_buffer) >= self._record_batch_count:
            self.flush_records()
        return record_id

    def flush_records(self) -> None:
        for split_name, values in self._split_buffers.items():
            if values:
                append_array(self.split_arrays[split_name], np.asarray(values, dtype=np.int64))
                values.clear()

        if self._record_image_id_buffer:
            self.record_label.append_many(self._record_label_buffer)
            self.record_raw_label.append_many(self._record_raw_label_buffer)
            self.record_meta.append_many(self._record_meta_buffer)
            append_array(self.record_image_id, np.asarray(self._record_image_id_buffer, dtype=np.int64))
            append_array(self.record_width, np.asarray(self._record_width_buffer, dtype=np.int32))
            append_array(self.record_height, np.asarray(self._record_height_buffer, dtype=np.int32))
            append_array(self.record_split, np.asarray(self._record_split_buffer, dtype=np.uint8))
            append_array(self.record_style, np.asarray(self._record_style_buffer, dtype=np.uint8))
            append_array(self.record_primitive_offset, np.asarray(self._record_primitive_offset_buffer, dtype=np.int64))
            append_array(self.record_primitive_count, np.asarray(self._record_primitive_count_buffer, dtype=np.int32))
            append_array(self.record_constraint_offset, np.asarray(self._record_constraint_offset_buffer, dtype=np.int64))
            append_array(self.record_constraint_count, np.asarray(self._record_constraint_count_buffer, dtype=np.int32))
            append_array(self.record_dimension_offset, np.asarray(self._record_dimension_offset_buffer, dtype=np.int64))
            append_array(self.record_dimension_count, np.asarray(self._record_dimension_count_buffer, dtype=np.int32))
            self._record_label_buffer.clear()
            self._record_raw_label_buffer.clear()
            self._record_meta_buffer.clear()
            self._record_image_id_buffer.clear()
            self._record_width_buffer.clear()
            self._record_height_buffer.clear()
            self._record_split_buffer.clear()
            self._record_style_buffer.clear()
            self._record_primitive_offset_buffer.clear()
            self._record_primitive_count_buffer.clear()
            self._record_constraint_offset_buffer.clear()
            self._record_constraint_count_buffer.clear()
            self._record_dimension_offset_buffer.clear()
            self._record_dimension_count_buffer.clear()

        if self._primitive_record_buffer:
            append_array(self.primitive_record, np.asarray(self._primitive_record_buffer, dtype=np.int64))
            append_array(self.primitive_local, np.asarray(self._primitive_local_buffer, dtype=np.int32))
            append_array(self.primitive_type, np.asarray(self._primitive_type_buffer, dtype=np.uint8))
            append_array(self.primitive_geom, np.asarray(self._primitive_geom_buffer, dtype=np.float32))
            append_array(self.primitive_geom_norm, np.asarray(self._primitive_geom_norm_buffer, dtype=np.float32))
            append_array(self.primitive_valid, np.asarray(self._primitive_valid_buffer, dtype=np.int8))
            self._primitive_record_buffer.clear()
            self._primitive_local_buffer.clear()
            self._primitive_type_buffer.clear()
            self._primitive_geom_buffer.clear()
            self._primitive_geom_norm_buffer.clear()
            self._primitive_valid_buffer.clear()

        if self._constraint_record_buffer:
            append_array(self.constraint_record, np.asarray(self._constraint_record_buffer, dtype=np.int64))
            append_array(self.constraint_type, np.asarray(self._constraint_type_buffer, dtype=np.uint8))
            append_array(self.constraint_source, np.asarray(self._constraint_source_buffer, dtype=np.int32))
            append_array(self.constraint_target, np.asarray(self._constraint_target_buffer, dtype=np.int32))
            append_array(self.constraint_pt1, np.asarray(self._constraint_pt1_buffer, dtype=np.int8))
            append_array(self.constraint_pt2, np.asarray(self._constraint_pt2_buffer, dtype=np.int8))
            self._constraint_record_buffer.clear()
            self._constraint_type_buffer.clear()
            self._constraint_source_buffer.clear()
            self._constraint_target_buffer.clear()
            self._constraint_pt1_buffer.clear()
            self._constraint_pt2_buffer.clear()

        if self._dimension_record_buffer:
            append_array(self.dimension_record, np.asarray(self._dimension_record_buffer, dtype=np.int64))
            append_array(self.dimension_type, np.asarray(self._dimension_type_buffer, dtype=np.uint8))
            append_array(self.dimension_value, np.asarray(self._dimension_value_buffer, dtype=np.float32))
            append_array(self.dimension_computed, np.asarray(self._dimension_computed_buffer, dtype=np.float32))
            append_array(self.dimension_ref_offset, np.asarray(self._dimension_ref_offset_buffer, dtype=np.int64))
            append_array(self.dimension_ref_count_arr, np.asarray(self._dimension_ref_count_buffer, dtype=np.int32))
            self._dimension_record_buffer.clear()
            self._dimension_type_buffer.clear()
            self._dimension_value_buffer.clear()
            self._dimension_computed_buffer.clear()
            self._dimension_ref_offset_buffer.clear()
            self._dimension_ref_count_buffer.clear()

        if self._dimref_dimension_buffer:
            append_array(self.dimref_dimension, np.asarray(self._dimref_dimension_buffer, dtype=np.int64))
            append_array(self.dimref_primitive, np.asarray(self._dimref_primitive_buffer, dtype=np.int32))
            append_array(self.dimref_point_type, np.asarray(self._dimref_point_type_buffer, dtype=np.int8))
            self._dimref_dimension_buffer.clear()
            self._dimref_primitive_buffer.clear()
            self._dimref_point_type_buffer.clear()

    def _append_primitives(self, record_id: int, primitives: Sequence[Dict[str, Any]], width: int, height: int) -> None:
        if not primitives:
            return

        norm_primitives = normalize_primitives(list(primitives), width, height) if width > 0 and height > 0 else []
        rows = [primitive_to_geom(p) for p in primitives]
        norm_rows = [primitive_to_geom(p) for p in norm_primitives] if norm_primitives else [[math.nan] * 7 for _ in primitives]

        self._primitive_record_buffer.extend([record_id] * len(primitives))
        self._primitive_local_buffer.extend(range(len(primitives)))
        self._primitive_type_buffer.extend(
            PRIMITIVE_TO_ID.get(str(p.get("type", "unknown")), 0) for p in primitives
        )
        self._primitive_geom_buffer.extend(rows)
        self._primitive_geom_norm_buffer.extend(norm_rows)
        self._primitive_valid_buffer.extend(valid_to_int(p.get("is_valid")) for p in primitives)

    def _append_constraints(
        self,
        record_id: int,
        constraints: Sequence[Dict[str, Any]],
        primitive_lookup: Dict[str, int],
    ) -> None:
        if not constraints:
            return

        self._constraint_record_buffer.extend([record_id] * len(constraints))
        self._constraint_type_buffer.extend(
            CONSTRAINT_TO_ID.get(str(c.get("type", "unknown")), 0) for c in constraints
        )
        self._constraint_source_buffer.extend(primitive_lookup.get(c.get("source"), -1) for c in constraints)
        self._constraint_target_buffer.extend(primitive_lookup.get(c.get("target"), -1) for c in constraints)
        self._constraint_pt1_buffer.extend(point_type_to_int(c.get("pointType1")) for c in constraints)
        self._constraint_pt2_buffer.extend(point_type_to_int(c.get("pointType2")) for c in constraints)

    def _append_dimensions(
        self,
        record_id: int,
        dimensions: Sequence[Dict[str, Any]],
        primitive_lookup: Dict[str, int],
    ) -> None:
        if not dimensions:
            return

        for d in dimensions:
            dim_index = self.dimension_count
            refs = d.get("refs", []) or []
            self._dimension_record_buffer.append(record_id)
            self._dimension_type_buffer.append(DIMENSION_TO_ID.get(str(d.get("type", "unknown")).lower(), 0))
            self._dimension_value_buffer.append(float_or_nan(d.get("value")))
            self._dimension_computed_buffer.append(float_or_nan(d.get("computed_from_geometry")))
            self._dimension_ref_offset_buffer.append(self.dimension_ref_count)
            self._dimension_ref_count_buffer.append(len(refs))

            for ref in refs:
                primitive_id = primitive_id_from_ref(str(ref))
                point_type = -1
                if "." in str(ref):
                    try:
                        point_type = int(str(ref).split(".", 1)[1])
                    except Exception:
                        point_type = -1
                self._dimref_dimension_buffer.append(dim_index)
                self._dimref_primitive_buffer.append(primitive_lookup.get(primitive_id, -1) if primitive_id else -1)
                self._dimref_point_type_buffer.append(point_type)


def primitive_to_geom(p: Dict[str, Any]) -> List[float]:
    g = p.get("geometry", {}) or {}
    ptype = p.get("type")
    if ptype == "line":
        return [
            float_or_nan(g.get("x1")),
            float_or_nan(g.get("y1")),
            float_or_nan(g.get("x2")),
            float_or_nan(g.get("y2")),
            math.nan,
            math.nan,
            math.nan,
        ]
    if ptype == "circle":
        return [
            float_or_nan(g.get("cx")),
            float_or_nan(g.get("cy")),
            math.nan,
            math.nan,
            float_or_nan(g.get("r")),
            math.nan,
            math.nan,
        ]
    if ptype == "arc":
        return [
            float_or_nan(g.get("cx")),
            float_or_nan(g.get("cy")),
            math.nan,
            math.nan,
            float_or_nan(g.get("r")),
            float_or_nan(g.get("start_param")),
            float_or_nan(g.get("end_param")),
        ]
    return [math.nan] * 7


def valid_to_int(value: Any) -> int:
    if value is True:
        return 1
    if value is False:
        return 0
    return -1


def point_type_to_int(value: Any) -> int:
    if value is None:
        return -1
    try:
        return int(value)
    except Exception:
        return -1


def float_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


class ImageIndex:
    def __init__(self, path: Path):
        self.path = path
        for candidate in (
            path,
            path.with_name(path.name + "-wal"),
            path.with_name(path.name + "-shm"),
        ):
            try:
                candidate.unlink(missing_ok=True)
            except Exception:
                pass
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE images (
                basename TEXT NOT NULL,
                style INTEGER NOT NULL,
                image_id INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                image_format INTEGER NOT NULL,
                source TEXT NOT NULL,
                member TEXT,
                PRIMARY KEY (basename, style)
            )
            """
        )
        self.conn.execute("CREATE INDEX images_basename_idx ON images(basename)")
        self.conn.commit()

    def has(self, basename: str, style: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM images WHERE basename=? AND style=? LIMIT 1",
            (basename, STYLE_TO_ID.get(style, 0)),
        )
        return cur.fetchone() is not None

    def insert(
        self,
        basename: str,
        style: str,
        image_id: int,
        width: int,
        height: int,
        image_format: str,
        source: str,
        member: Optional[str],
    ) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO images
            (basename, style, image_id, width, height, image_format, source, member)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                basename,
                STYLE_TO_ID.get(style, 0),
                image_id,
                width,
                height,
                FORMAT_TO_ID.get(image_format, 0),
                source,
                member,
            ),
        )

    def find(self, basename: str, style: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT basename, style, image_id, width, height, image_format, source, member
            FROM images WHERE basename=? AND style=? LIMIT 1
            """,
            (basename, STYLE_TO_ID.get(style, 0)),
        )
        row = cur.fetchone()
        return row_to_image(row)

    def find_any(self, basename: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT basename, style, image_id, width, height, image_format, source, member
            FROM images WHERE basename=?
            ORDER BY CASE style WHEN 1 THEN 0 WHEN 2 THEN 1 WHEN 3 THEN 2 ELSE 3 END
            LIMIT 1
            """,
            (basename,),
        )
        row = cur.fetchone()
        return row_to_image(row)

    def find_all(self, basename: str) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT basename, style, image_id, width, height, image_format, source, member
            FROM images WHERE basename=?
            ORDER BY CASE style WHEN 1 THEN 0 WHEN 2 THEN 1 WHEN 3 THEN 2 ELSE 3 END
            """,
            (basename,),
        )
        return [row_to_image(row) for row in cur.fetchall() if row_to_image(row) is not None]

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def row_to_image(row: Optional[Tuple[Any, ...]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    basename, style, image_id, width, height, image_format, source, member = row
    return {
        "basename": basename,
        "style": ID_TO_STYLE.get(int(style), "unknown"),
        "image_id": int(image_id),
        "width": int(width),
        "height": int(height),
        "format": next((k for k, v in FORMAT_TO_ID.items() if v == int(image_format)), "unknown"),
        "source": source,
        "member": member,
    }


def index_image_bytes(
    writer: ZarrWriter,
    image_index: ImageIndex,
    data: bytes,
    basename: str,
    style: str,
    source: str,
    member: Optional[str],
    stats: Counter[str],
    progress: Optional[ProgressReporter] = None,
    indexed_task: Optional[int] = None,
) -> Optional[int]:
    if style == "unknown":
        style = infer_style(source if member is None else f"{source}/{member}")
    if image_index.has(basename, style):
        stats["duplicate_images_skipped"] += 1
        return None

    width, height = fast_image_size_bytes(data)
    image_format = infer_image_format(basename, data)
    meta = {
        "basename": basename,
        "style": style,
        "source": source,
        "member": member,
        "format": image_format,
        "width": width,
        "height": height,
    }
    image_id = writer.append_image(data, width, height, style, image_format, meta)
    image_index.insert(basename, style, image_id, width, height, image_format, source, member)
    stats["images_indexed"] += 1
    stats[f"images_{style}"] += 1
    if progress is not None:
        progress.update(indexed_task, advance=1)
    return image_id


def index_filesystem_images(
    image_files: Sequence[Path],
    input_root: Path,
    output_path: Path,
    skip_names: set[str],
    writer: ZarrWriter,
    image_index: ImageIndex,
    stats: Counter[str],
    limit_images: int,
    progress: ProgressReporter,
    files_task: Optional[int],
    indexed_task: Optional[int],
    overall_task: Optional[int],
    needed_image_keys: Optional[set[int]],
    needed_basenames: Optional[set[int]],
) -> None:
    for path in image_files:
        if limit_images and stats["images_indexed"] >= limit_images:
            return
        try:
            file_size = path.stat().st_size
        except Exception:
            file_size = 0
        basename = path.name
        style = infer_style(path)
        if not image_is_needed(basename, style, needed_image_keys, needed_basenames):
            stats["images_skipped_not_needed"] += 1
            progress.update(files_task, advance=1)
            progress.update(overall_task, advance=file_size)
            continue
        if image_index.has(basename, style):
            stats["duplicate_images_skipped"] += 1
            progress.update(files_task, advance=1)
            progress.update(overall_task, advance=file_size)
            continue
        try:
            data = path.read_bytes()
        except Exception:
            stats["image_file_read_errors"] += 1
            progress.update(files_task, advance=1)
            progress.update(overall_task, advance=file_size)
            continue
        index_image_bytes(
            writer,
            image_index,
            data,
            basename,
            style,
            safe_rel(path, input_root),
            None,
            stats,
            progress,
            indexed_task,
        )
        progress.update(files_task, advance=1)
        progress.update(overall_task, advance=len(data))
        if stats["images_indexed"] % 10_000 == 0:
            image_index.commit()
            progress.log(f"[IMAGES] indexed {stats['images_indexed']:,} images")


def index_tar_images(
    archive_path: Path,
    input_root: Path,
    writer: ZarrWriter,
    image_index: ImageIndex,
    stats: Counter[str],
    limit_images: int,
    progress: ProgressReporter,
    archive_task: Optional[int],
    indexed_task: Optional[int],
    overall_task: Optional[int],
    needed_image_keys: Optional[set[int]],
    needed_basenames: Optional[set[int]],
) -> None:
    source = safe_rel(archive_path, input_root)
    try:
        with archive_path.open("rb") as raw:
            reader = ProgressFile(raw, progress, [archive_task, overall_task])
            with tarfile.open(fileobj=reader, mode="r|*") as tf:
                for member in tf:
                    if limit_images and stats["images_indexed"] >= limit_images:
                        return
                    if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_EXTS:
                        continue
                    basename = Path(member.name).name
                    style = infer_style(member.name or source)
                    if not image_is_needed(basename, style, needed_image_keys, needed_basenames):
                        stats["images_skipped_not_needed"] += 1
                        continue
                    if image_index.has(basename, style):
                        stats["duplicate_images_skipped"] += 1
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        stats["archive_member_read_errors"] += 1
                        continue
                    data = fh.read()
                    index_image_bytes(
                        writer,
                        image_index,
                        data,
                        basename,
                        style,
                        source,
                        member.name,
                        stats,
                        progress,
                        indexed_task,
                    )
                    if stats["images_indexed"] % 10_000 == 0:
                        image_index.commit()
                        progress.log(f"[IMAGES] indexed {stats['images_indexed']:,} images")
    except Exception as exc:
        stats["archive_read_errors"] += 1
        progress.log(f"[WARN] Could not read archive {archive_path}: {exc}", error=True)


def index_split_tar_images(
    parts: Sequence[Path],
    input_root: Path,
    writer: ZarrWriter,
    image_index: ImageIndex,
    stats: Counter[str],
    limit_images: int,
    progress: ProgressReporter,
    archive_task: Optional[int],
    indexed_task: Optional[int],
    overall_task: Optional[int],
    needed_image_keys: Optional[set[int]],
    needed_basenames: Optional[set[int]],
) -> None:
    source = "+".join(safe_rel(p, input_root) for p in parts)
    concat = ConcatenatedFile(parts)
    try:
        reader = ProgressFile(concat, progress, [archive_task, overall_task])
        with tarfile.open(fileobj=reader, mode="r|gz") as tf:
            for member in tf:
                if limit_images and stats["images_indexed"] >= limit_images:
                    return
                if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_EXTS:
                    continue
                basename = Path(member.name).name
                style = infer_style(member.name or source)
                if not image_is_needed(basename, style, needed_image_keys, needed_basenames):
                    stats["images_skipped_not_needed"] += 1
                    continue
                if image_index.has(basename, style):
                    stats["duplicate_images_skipped"] += 1
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    stats["archive_member_read_errors"] += 1
                    continue
                data = fh.read()
                index_image_bytes(
                    writer,
                    image_index,
                    data,
                    basename,
                    style,
                    source,
                    member.name,
                    stats,
                    progress,
                    indexed_task,
                )
                if stats["images_indexed"] % 10_000 == 0:
                    image_index.commit()
                    progress.log(f"[IMAGES] indexed {stats['images_indexed']:,} images")
    except Exception as exc:
        stats["split_archive_read_errors"] += 1
        progress.log(f"[WARN] Could not read split archive {source}: {exc}", error=True)
    finally:
        concat.close()


def archive_reader_worker(
    parts: Sequence[Path],
    input_root: Path,
    progress: ProgressReporter,
    archive_task: Optional[int],
    overall_task: Optional[int],
    needed_image_keys: Optional[set[int]],
    needed_basenames: Optional[set[int]],
    out_queue: "Queue[Tuple[str, Any]]",
    stop_event: Event,
) -> None:
    source = archive_label_for_filter(parts, input_root)
    local_stats: Counter[str] = Counter()
    reader: Optional[ProgressFile] = None

    try:
        if len(parts) == 1:
            raw = parts[0].open("rb")
            reader = ProgressFile(raw, progress, [archive_task, overall_task])
            tar_mode = "r|*"
        else:
            concat = ConcatenatedFile(parts)
            reader = ProgressFile(concat, progress, [archive_task, overall_task])
            tar_mode = "r|gz"

        with tarfile.open(fileobj=reader, mode=tar_mode) as tf:
            for member in tf:
                if stop_event.is_set():
                    break
                if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_EXTS:
                    continue

                basename = Path(member.name).name
                style = infer_style(member.name or source)
                if not image_is_needed(basename, style, needed_image_keys, needed_basenames):
                    local_stats["images_skipped_not_needed"] += 1
                    continue

                fh = tf.extractfile(member)
                if fh is None:
                    local_stats["archive_member_read_errors"] += 1
                    continue
                data = fh.read()

                while not stop_event.is_set():
                    try:
                        out_queue.put(
                            ("image", (data, basename, style, source, member.name)),
                            timeout=0.5,
                        )
                        break
                    except Full:
                        continue
    except Exception as exc:
        local_stats["archive_read_errors"] += 1
        while True:
            try:
                out_queue.put(("error", f"[WARN] Could not read archive {source}: {exc}"), timeout=0.5)
                break
            except Full:
                if stop_event.is_set():
                    break
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
        while True:
            try:
                out_queue.put(("done", (source, local_stats)), timeout=0.5)
                break
            except Full:
                continue


def index_archive_jobs_parallel(
    archive_jobs: Sequence[Tuple[Sequence[Path], Optional[int]]],
    input_root: Path,
    writer: ZarrWriter,
    image_index: ImageIndex,
    stats: Counter[str],
    limit_images: int,
    progress: ProgressReporter,
    archive_jobs_task: Optional[int],
    indexed_task: Optional[int],
    overall_task: Optional[int],
    needed_image_keys: Optional[set[int]],
    needed_basenames: Optional[set[int]],
    workers: int,
    queue_size: int,
) -> None:
    if not archive_jobs:
        return

    stop_event = Event()
    out_queue: "Queue[Tuple[str, Any]]" = Queue(maxsize=max(1, queue_size))
    done_count = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for parts, archive_task in archive_jobs:
            executor.submit(
                archive_reader_worker,
                parts,
                input_root,
                progress,
                archive_task,
                overall_task,
                needed_image_keys,
                needed_basenames,
                out_queue,
                stop_event,
            )

        while done_count < len(archive_jobs):
            try:
                kind, payload = out_queue.get(timeout=0.5)
            except Empty:
                continue

            if kind == "image":
                if limit_images and stats["images_indexed"] >= limit_images:
                    stop_event.set()
                    continue
                data, basename, style, source, member = payload
                index_image_bytes(
                    writer,
                    image_index,
                    data,
                    basename,
                    style,
                    source,
                    member,
                    stats,
                    progress,
                    indexed_task,
                )
                if stats["images_indexed"] % 10_000 == 0:
                    image_index.commit()
                    progress.log(f"[IMAGES] indexed {stats['images_indexed']:,} images")
                if limit_images and stats["images_indexed"] >= limit_images:
                    stop_event.set()
            elif kind == "error":
                progress.log(str(payload), error=True)
            elif kind == "done":
                _source, local_stats = payload
                stats.update(local_stats)
                done_count += 1
                progress.update(archive_jobs_task, advance=1)


def parse_label_from_item(item: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str, str]:
    if isinstance(item.get("primitives"), list):
        label = {
            "primitives": item.get("primitives", []) or [],
            "constraints": item.get("constraints", []) or [],
            "dimensions": item.get("dimensions", []) or [],
        }
        return add_geometry_checks(label), "", "processed"

    raw_label = extract_label_text(item) or ""
    if not raw_label:
        return None, "", "missing_label"

    parsed = parse_assistant_output(raw_label)
    if not parsed.get("primitives"):
        return None, raw_label, "missing_primitives"

    parsed["dimensions"] = add_dimension_associations(parsed.get("dimensions", []) or [])
    parsed = add_geometry_checks(parsed)
    return parsed, raw_label, "ok"


def image_name_from_item(item: Dict[str, Any]) -> Optional[str]:
    value = item.get("image") or item.get("image_path") or item.get("file_name")
    if isinstance(value, str) and value:
        return value
    return None


def select_images_for_record(
    image_index: ImageIndex,
    basename: str,
    source_style: str,
    variant_mode: str,
    fallback_any_style: bool,
) -> List[Dict[str, Any]]:
    if variant_mode == "all":
        return image_index.find_all(basename)
    if variant_mode == "first":
        found = image_index.find_any(basename)
        return [found] if found else []

    found = image_index.find(basename, source_style) if source_style != "unknown" else None
    if found:
        return [found]
    if fallback_any_style:
        fallback = image_index.find_any(basename)
        return [fallback] if fallback else []
    return []


def write_records(
    json_files: Sequence[Path],
    input_root: Path,
    writer: ZarrWriter,
    image_index: ImageIndex,
    stats: Counter[str],
    args: argparse.Namespace,
    progress: ProgressReporter,
    json_files_task: Optional[int],
    records_task: Optional[int],
    overall_task: Optional[int],
) -> None:
    progress.log(f"[JSON] files: {len(json_files):,}")
    for json_file_i, json_path in enumerate(json_files, 1):
        source_json = safe_rel(json_path, input_root)
        source_style = infer_style(json_path)
        file_records = 0
        file_ok = 0
        try:
            json_size = json_path.stat().st_size
        except Exception:
            json_size = None
        json_bytes_task = progress.add_task(f"JSON bytes: {source_json}", total=json_size)

        for source_index, item in enumerate(
            iter_json_records_stream(
                json_path,
                args.json_chunk_size,
                progress=progress,
                file_task=json_bytes_task,
                overall_task=overall_task,
            )
        ):
            if args.limit_records and stats["records_written"] >= args.limit_records:
                return

            stats["raw_json_records"] += 1
            file_records += 1

            image_name = image_name_from_item(item)
            if not image_name:
                stats["missing_image_name"] += 1
                continue
            basename = Path(image_name).name
            drawing_id = drawing_id_from_image_name(basename)
            label, raw_label, status = parse_label_from_item(item)
            if label is None:
                stats[status] += 1
                continue

            image_hits = select_images_for_record(
                image_index=image_index,
                basename=basename,
                source_style=source_style,
                variant_mode=args.variant_mode,
                fallback_any_style=args.fallback_any_style,
            )
            if not image_hits and args.require_image:
                stats["missing_indexed_image"] += 1
                continue
            if not image_hits:
                image_hits = [
                    {
                        "image_id": -1,
                        "width": -1,
                        "height": -1,
                        "style": source_style,
                        "format": "unknown",
                        "source": "",
                        "member": None,
                    }
                ]

            split = stable_split(drawing_id, args.train_ratio, args.val_ratio)

            for hit in image_hits:
                meta = {
                    "record_id": writer.record_count,
                    "drawing_id": drawing_id,
                    "image_name": basename,
                    "source_image_name": image_name,
                    "source_json": source_json,
                    "source_index": source_index,
                    "source_style": source_style,
                    "style": hit["style"],
                    "split": split,
                    "image_id": hit["image_id"],
                    "image_source": hit.get("source"),
                    "image_member": hit.get("member"),
                }
                writer.append_record(
                    image_id=int(hit["image_id"]),
                    width=int(hit["width"]),
                    height=int(hit["height"]),
                    style=str(hit["style"]),
                    split=split,
                    label=label,
                    raw_label=raw_label,
                    meta=meta,
                )
                stats["records_written"] += 1
                stats[f"records_{split}"] += 1
                stats[f"records_{hit['style']}"] += 1
                file_ok += 1
                progress.update(records_task, advance=1)

            if stats["records_written"] % args.progress_every == 0 and stats["records_written"]:
                progress.log(
                    f"[RECORDS] {stats['records_written']:,} records | "
                    f"{writer.primitive_count:,} primitives | {writer.dimension_count:,} dimensions"
                )

        if json_size is not None:
            progress.update(json_bytes_task, completed=json_size)
        progress.update(json_files_task, advance=1)
        progress.log(
            f"[JSON] {json_file_i:,}/{len(json_files):,} {source_json}: "
            f"{file_ok:,} written from {file_records:,} raw"
        )


def write_schema_attrs(root: Any, args: argparse.Namespace, stats: Counter[str], elapsed: float) -> None:
    root.attrs.update(
        {
            "schema_name": "paracad_engineering_drawings",
            "schema_version": "1.0.0",
            "created_by": "build_paracad_zarr.py",
            "elapsed_seconds": round(elapsed, 3),
            "input_root": str(Path(args.input).resolve()),
            "variant_mode": args.variant_mode,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
            "styles": STYLE_TO_ID,
            "splits": SPLIT_TO_ID,
            "primitive_types": PRIMITIVE_TO_ID,
            "constraint_types": CONSTRAINT_TO_ID,
            "dimension_types": DIMENSION_TO_ID,
            "geometry_columns": [
                "x1_or_cx",
                "y1_or_cy",
                "x2",
                "y2",
                "radius",
                "start_param",
                "end_param",
            ],
            "notes": [
                "Images are stored as raw encoded image bytes in /images/data_bytes with /images/data_offsets.",
                "Records point to /images by /records/image_id.",
                "/records/label_json_* stores exact structured primitives, constraints, and dimensions.",
                "/primitives/geometry stores native label coordinates; /primitives/geometry_norm stores image-normalized coordinates when image size is known.",
                "Splits are assigned by drawing_id, so visual/style variants of the same drawing stay together.",
            ],
            "stats": dict(stats),
        }
    )


def discover_json_files_for_args(
    input_root: Path,
    output_path: Path,
    skip_names: set[str],
    args: argparse.Namespace,
) -> List[Path]:
    json_files = sorted(iter_files(input_root, JSON_EXTS, output_path, skip_names))
    if args.json_regex:
        pattern = re.compile(args.json_regex)
        json_files = [p for p in json_files if pattern.search(str(p))]
    if args.limit_json_files:
        json_files = json_files[: args.limit_json_files]
    return json_files


def safe_file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def sum_file_sizes(paths: Iterable[Path]) -> int:
    return sum(safe_file_size(path) for path in paths)


def archive_label_for_filter(paths: Sequence[Path], input_root: Path) -> str:
    return "+".join(safe_rel(path, input_root) for path in paths)


def archive_allowed(label: str, args: argparse.Namespace) -> bool:
    if args.include_archive_regex and not re.search(args.include_archive_regex, label):
        return False
    if args.exclude_archive_regex and re.search(args.exclude_archive_regex, label):
        return False
    return True


def collect_needed_images(
    json_files: Sequence[Path],
    input_root: Path,
    args: argparse.Namespace,
    progress: ProgressReporter,
    overall_task: Optional[int],
) -> Tuple[Optional[set[int]], Optional[set[int]], Counter[str]]:
    needed_image_keys: set[int] = set()
    needed_basenames: set[int] = set()
    stats: Counter[str] = Counter()

    use_basename_filter = args.variant_mode in {"all", "first"} or args.fallback_any_style
    files_task = progress.add_task("Needed-image JSON files", total=len(json_files))

    for i, json_path in enumerate(json_files, 1):
        source_json = safe_rel(json_path, input_root)
        source_style = infer_style(json_path)
        json_size = safe_file_size(json_path)
        file_task = progress.add_task(f"Needed-image scan: {source_json}", total=json_size or None)

        for item in iter_json_records_stream(
            json_path,
            args.json_chunk_size,
            progress=progress,
            file_task=file_task,
            overall_task=overall_task,
        ):
            stats["needed_scan_records"] += 1
            image_name = image_name_from_item(item)
            if not image_name:
                stats["needed_scan_missing_image_name"] += 1
                continue

            basename = Path(image_name).name
            if use_basename_filter:
                needed_basenames.add(needed_key_for_basename(basename))
            else:
                needed_image_keys.add(needed_key_for_image(basename, source_style))

        if json_size:
            progress.update(file_task, completed=json_size)
        progress.update(files_task, advance=1)
        progress.log(
            f"[NEEDED] {i:,}/{len(json_files):,} {source_json}: "
            f"{stats['needed_scan_records']:,} labels scanned"
        )

    if use_basename_filter:
        stats["needed_basenames"] = len(needed_basenames)
        return None, needed_basenames, stats

    stats["needed_image_keys"] = len(needed_image_keys)
    return needed_image_keys, None, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Zarr dataset from full ParaCAD raw JSON/image archives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="ParaCAD root containing JSON files, image folders, and archives.")
    parser.add_argument("--output", required=True, help="Output .zarr directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output Zarr.")
    parser.add_argument("--zarr-format", type=int, choices=[2, 3], default=2, help="Zarr storage format.")
    parser.add_argument("--variant-mode", choices=["source", "all", "first"], default="source", help="Which image variants to pair with each label.")
    parser.add_argument("--fallback-any-style", action="store_true", help="If source style image is missing, use any available style.")
    parser.add_argument("--require-image", action=argparse.BooleanOptionalAction, default=True, help="Skip labels without an indexed image.")
    parser.add_argument("--no-files", action="store_true", help="Do not index loose image files.")
    parser.add_argument("--no-archives", action="store_true", help="Do not stream image tar/tar.gz archives.")
    parser.add_argument("--no-split-archives", action="store_true", help="Do not stream split archives like SG6-16_white_part_aa/ab.")
    parser.add_argument("--only-needed-images", action="store_true", help="Pre-scan labels and only store images referenced by selected JSON labels.")
    parser.add_argument("--include-archive-regex", default="", help="Only process tar/split archive paths matching this regex.")
    parser.add_argument("--exclude-archive-regex", default="", help="Skip tar/split archive paths matching this regex.")
    parser.add_argument("--archive-workers", type=int, default=1, help="Parallel archive reader/decompressor workers; Zarr writes remain single-threaded.")
    parser.add_argument("--archive-queue-size", type=int, default=64, help="Bounded image queue size for --archive-workers.")
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--json-chunk-size", type=int, default=1 << 20, help="Chunk size for fallback streaming JSON parser.")
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--no-rich-progress", "--no-progress", dest="no_rich_progress", action="store_true", help="Use plain log lines instead of Rich progress bars.")
    parser.add_argument("--auto-rich-progress", action="store_true", help="Let Rich auto-detect terminal support instead of forcing live progress output.")
    parser.add_argument("--progress-width", type=int, default=160, help="Console width used for Rich progress rendering.")
    parser.add_argument("--limit-images", type=int, default=0, help="Debug limit for images indexed.")
    parser.add_argument("--limit-records", type=int, default=0, help="Debug limit for records written.")
    parser.add_argument("--limit-json-files", type=int, default=0, help="Debug limit for JSON files processed.")
    parser.add_argument("--json-regex", default="", help="Only process JSON files whose path matches this regex.")
    parser.add_argument("--keep-index", action="store_true", help="Keep the temporary SQLite image index beside the Zarr.")
    parser.add_argument(
        "--skip-dir",
        action="append",
        default=["ParaCAD_processed", "ParaCAD_processed_v2", ".git", "__pycache__"],
        help="Directory name to skip while scanning. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if args.train_ratio + args.val_ratio > 1.0:
        print("[ERROR] --train-ratio + --val-ratio must be <= 1.0", file=sys.stderr)
        return 2
    if not input_root.exists():
        print(f"[ERROR] input does not exist: {input_root}", file=sys.stderr)
        return 2
    if output_path.exists() and not args.overwrite:
        print(f"[ERROR] output exists; pass --overwrite to replace it: {output_path}", file=sys.stderr)
        return 2

    start = time.time()
    stats: Counter[str] = Counter()
    skip_names = set(args.skip_dir or [])
    skip_names.add(output_path.name)

    image_files = [] if args.no_files else sorted(iter_files(input_root, IMAGE_EXTS, output_path, skip_names))
    archives = [] if args.no_archives else discover_tar_archives(input_root, output_path, skip_names)
    split_groups = [] if args.no_split_archives else discover_split_archive_groups(input_root, output_path, skip_names)
    json_files = discover_json_files_for_args(input_root, output_path, skip_names, args)

    if args.include_archive_regex or args.exclude_archive_regex:
        archives = [
            archive
            for archive in archives
            if archive_allowed(archive_label_for_filter([archive], input_root), args)
        ]
        split_groups = [
            group
            for group in split_groups
            if archive_allowed(archive_label_for_filter(group, input_root), args)
        ]

    loose_image_bytes = sum_file_sizes(image_files)
    archive_bytes = sum_file_sizes(archives)
    split_archive_bytes = sum_file_sizes(path for group in split_groups for path in group)
    json_bytes = sum_file_sizes(json_files)
    needed_scan_bytes = json_bytes if args.only_needed_images else 0
    overall_total = loose_image_bytes + archive_bytes + split_archive_bytes + json_bytes + needed_scan_bytes

    writer = ZarrWriter(output_path, overwrite=args.overwrite, zarr_format=args.zarr_format)
    index_path = output_path.with_name(output_path.name + ".image_index.sqlite")
    image_index = ImageIndex(index_path)

    with ProgressReporter(
        enabled=not args.no_rich_progress,
        force_terminal=not args.auto_rich_progress,
        console_width=args.progress_width,
    ) as progress:
        progress.log(f"[INIT] input={input_root}")
        progress.log(f"[INIT] output={output_path}")
        progress.log(
            "[SCAN] "
            f"loose_images={len(image_files):,}, "
            f"tar_archives={len(archives):,}, "
            f"split_archive_groups={len(split_groups):,}, "
            f"json_files={len(json_files):,}"
        )

        overall_task = progress.add_task("Overall input read", total=overall_total or None)
        indexed_task = progress.add_task(
            "Images indexed",
            total=args.limit_images if args.limit_images else None,
        )
        json_files_task = progress.add_task("JSON files", total=len(json_files))
        records_task = progress.add_task(
            "Records written",
            total=args.limit_records if args.limit_records else None,
        )
        needed_image_keys: Optional[set[int]] = None
        needed_basenames: Optional[set[int]] = None

        if args.only_needed_images:
            progress.log("[NEEDED] scanning labels to identify referenced images")
            needed_image_keys, needed_basenames, needed_stats = collect_needed_images(
                json_files,
                input_root,
                args,
                progress,
                overall_task,
            )
            stats.update(needed_stats)
            if needed_image_keys is not None:
                progress.log(f"[NEEDED] style-specific image keys: {len(needed_image_keys):,}")
            if needed_basenames is not None:
                progress.log(f"[NEEDED] image basenames: {len(needed_basenames):,}")

        try:
            if not args.no_files:
                progress.log("[IMAGES] indexing loose image files")
                loose_task = progress.add_task("Loose image files", total=len(image_files))
                index_filesystem_images(
                    image_files,
                    input_root,
                    output_path,
                    skip_names,
                    writer,
                    image_index,
                    stats,
                    args.limit_images,
                    progress,
                    loose_task,
                    indexed_task,
                    overall_task,
                    needed_image_keys,
                    needed_basenames,
                )
                image_index.commit()

            archive_work_allowed = not args.limit_images or stats["images_indexed"] < args.limit_images
            if args.archive_workers > 1 and archive_work_allowed and (archives or split_groups):
                archive_jobs: List[Tuple[Sequence[Path], Optional[int]]] = []
                progress.log(
                    f"[IMAGES] archive jobs: {len(archives) + len(split_groups):,} "
                    f"with {args.archive_workers:,} worker(s)"
                )
                archive_jobs_task = progress.add_task(
                    "Archive jobs",
                    total=len(archives) + len(split_groups),
                )
                for archive in archives:
                    archive_label = safe_rel(archive, input_root)
                    progress.log(f"[IMAGES] archive queued: {archive_label}")
                    archive_task = progress.add_task(
                        f"Archive bytes: {archive.name}",
                        total=safe_file_size(archive) or None,
                    )
                    archive_jobs.append(([archive], archive_task))
                for parts in split_groups:
                    label = archive_label_for_filter(parts, input_root)
                    progress.log(f"[IMAGES] split archive queued: {label}")
                    split_task = progress.add_task(
                        f"Split archive bytes: {parts[0].name}",
                        total=sum_file_sizes(parts) or None,
                    )
                    archive_jobs.append((parts, split_task))

                index_archive_jobs_parallel(
                    archive_jobs,
                    input_root,
                    writer,
                    image_index,
                    stats,
                    args.limit_images,
                    progress,
                    archive_jobs_task,
                    indexed_task,
                    overall_task,
                    needed_image_keys,
                    needed_basenames,
                    args.archive_workers,
                    args.archive_queue_size,
                )
                image_index.commit()
            else:
                if not args.no_archives and archive_work_allowed:
                    progress.log(f"[IMAGES] tar archives: {len(archives):,}")
                    archive_files_task = progress.add_task("Tar archive files", total=len(archives))
                    for i, archive in enumerate(archives, 1):
                        if args.limit_images and stats["images_indexed"] >= args.limit_images:
                            break
                        archive_label = safe_rel(archive, input_root)
                        progress.log(f"[IMAGES] archive {i:,}/{len(archives):,}: {archive_label}")
                        archive_task = progress.add_task(
                            f"Archive bytes: {archive.name}",
                            total=safe_file_size(archive) or None,
                        )
                        index_tar_images(
                            archive,
                            input_root,
                            writer,
                            image_index,
                            stats,
                            args.limit_images,
                            progress,
                            archive_task,
                            indexed_task,
                            overall_task,
                            needed_image_keys,
                            needed_basenames,
                        )
                        progress.update(archive_task, completed=safe_file_size(archive))
                        progress.update(archive_files_task, advance=1)
                        image_index.commit()

                archive_work_allowed = not args.limit_images or stats["images_indexed"] < args.limit_images
                if not args.no_split_archives and archive_work_allowed:
                    progress.log(f"[IMAGES] split archive groups: {len(split_groups):,}")
                    split_groups_task = progress.add_task("Split archive groups", total=len(split_groups))
                    for i, parts in enumerate(split_groups, 1):
                        if args.limit_images and stats["images_indexed"] >= args.limit_images:
                            break
                        label = "+".join(safe_rel(p, input_root) for p in parts)
                        progress.log(f"[IMAGES] split archive {i:,}/{len(split_groups):,}: {label}")
                        split_task = progress.add_task(
                            f"Split archive bytes: {parts[0].name}",
                            total=sum_file_sizes(parts) or None,
                        )
                        index_split_tar_images(
                            parts,
                            input_root,
                            writer,
                            image_index,
                            stats,
                            args.limit_images,
                            progress,
                            split_task,
                            indexed_task,
                            overall_task,
                            needed_image_keys,
                            needed_basenames,
                        )
                        progress.update(split_task, completed=sum_file_sizes(parts))
                        progress.update(split_groups_task, advance=1)
                        image_index.commit()

            writer.flush_images()
            progress.log(f"[IMAGES] total indexed: {stats['images_indexed']:,}")
            write_records(
                json_files,
                input_root,
                writer,
                image_index,
                stats,
                args,
                progress,
                json_files_task,
                records_task,
                overall_task,
            )
            writer.flush_records()

        finally:
            image_index.close()
            if not args.keep_index:
                try:
                    index_path.unlink(missing_ok=True)
                    index_path.with_suffix(index_path.suffix + "-wal").unlink(missing_ok=True)
                    index_path.with_suffix(index_path.suffix + "-shm").unlink(missing_ok=True)
                except Exception:
                    pass

    writer.flush_images()
    writer.flush_records()
    elapsed = time.time() - start
    stats["final_images"] = writer.image_count
    stats["final_records"] = writer.record_count
    stats["final_primitives"] = writer.primitive_count
    stats["final_constraints"] = writer.constraint_count
    stats["final_dimensions"] = writer.dimension_count
    stats["final_dimension_refs"] = writer.dimension_ref_count
    write_schema_attrs(writer.root, args, stats, elapsed)

    print("[DONE]")
    print(json.dumps({"elapsed_seconds": round(elapsed, 3), "stats": dict(stats)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
