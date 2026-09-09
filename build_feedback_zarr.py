#!/usr/bin/env python3
"""Build a compact Primitive-DETR-compatible Zarr dataset from reviewed labels.

Input rows are produced by ``drawing_feedback.py export-labels``. This builder
is intentionally separate from ParaCAD's original archive so engineer-reviewed
local data remains auditable and can be fine-tuned without modifying the source
dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cli_progress import RichProgress, add_progress_argument

try:
    import zarr
except ImportError as exc:  # pragma: no cover - dependency guard
    print(f"[ERROR] {exc}. Install requirements-primitive-detr.txt.", file=sys.stderr)
    raise SystemExit(2)


TYPE_IDS = {"line": 1, "circle": 2, "arc": 3}


def read_rows(paths: list[Path], progress: RichProgress | None = None, task_id: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Reviewed-label JSONL not found: {path}")
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    rows.append(json.loads(line))
        if progress is not None:
            progress.advance(task_id)
    if len(rows) < 2:
        raise ValueError("At least two reviewed drawing rows are required for separate train and validation splits.")
    return rows


def image_path_for(row: dict[str, Any], image_root: Path) -> Path:
    candidate = Path(str(row.get("image_path", "")))
    path = candidate if candidate.is_absolute() else image_root / candidate
    if not path.is_file():
        raise FileNotFoundError(f"Image for reviewed label was not found: {path}")
    return path


def primitive_row(primitive: dict[str, Any]) -> tuple[int, list[float]]:
    kind = str(primitive.get("type", ""))
    geometry = primitive.get("geometry") or {}
    if kind not in TYPE_IDS:
        raise ValueError(f"Unsupported reviewed primitive type: {kind}")
    try:
        if kind == "line":
            values = [float(geometry["x1"]), float(geometry["y1"]), float(geometry["x2"]), float(geometry["y2"]), math.nan, math.nan, math.nan]
        elif kind == "circle":
            values = [float(geometry["cx"]), float(geometry["cy"]), math.nan, math.nan, float(geometry["radius"]), math.nan, math.nan]
        else:
            values = [
                float(geometry["cx"]),
                float(geometry["cy"]),
                math.nan,
                math.nan,
                float(geometry["radius"]),
                float(geometry["start_param"]),
                float(geometry["end_param"]),
            ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed reviewed {kind} geometry: {geometry}") from exc
    coordinate_indexes = (0, 1, 2, 3) if kind == "line" else (0, 1, 4)
    for index in coordinate_indexes:
        if index < len(values) and not math.isnan(values[index]) and not 0.0 <= values[index] <= 1.0:
            raise ValueError(f"Reviewed geometry must be normalized to [0, 1], got {values[index]} for {kind}.")
    return TYPE_IDS[kind], values


def create_array(group: Any, name: str, data: np.ndarray) -> None:
    chunks = tuple(max(1, min(100_000, size)) for size in data.shape) if data.ndim else ()
    if hasattr(group, "create_array"):
        array = group.create_array(name, shape=data.shape, chunks=chunks, dtype=data.dtype)
    else:  # pragma: no cover - Zarr 2 compatibility
        array = group.create_dataset(name, shape=data.shape, chunks=chunks, dtype=data.dtype)
    array[...] = data


def build(
    rows: list[dict[str, Any]],
    image_root: Path,
    output: Path,
    val_ratio: float,
    seed: int,
    progress: RichProgress | None = None,
    task_id: int | None = None,
) -> dict[str, int]:
    image_bytes: list[int] = []
    image_offsets = [0]
    image_ids: list[int] = []
    primitive_offsets: list[int] = []
    primitive_counts: list[int] = []
    primitive_types: list[int] = []
    primitive_geometry: list[list[float]] = []
    for record_index, row in enumerate(rows):
        encoded = image_path_for(row, image_root).read_bytes()
        image_bytes.extend(encoded)
        image_offsets.append(len(image_bytes))
        image_ids.append(record_index)
        primitive_offsets.append(len(primitive_types))
        for primitive in row.get("primitives", []):
            type_id, geometry = primitive_row(primitive)
            primitive_types.append(type_id)
            primitive_geometry.append(geometry)
        primitive_counts.append(len(primitive_types) - primitive_offsets[-1])
        if progress is not None:
            progress.advance(task_id)
    generator = np.random.default_rng(seed)
    indices = generator.permutation(len(rows)).astype(np.int64)
    validation_count = max(1, min(len(rows) - 1, round(len(rows) * val_ratio)))
    val_indices = np.sort(indices[:validation_count])
    train_indices = np.sort(indices[validation_count:])
    root = zarr.open_group(str(output), mode="w")
    images = root.require_group("images")
    records = root.require_group("records")
    primitives = root.require_group("primitives")
    splits = root.require_group("splits")
    create_array(images, "data_bytes", np.asarray(image_bytes, dtype=np.uint8))
    create_array(images, "data_offsets", np.asarray(image_offsets, dtype=np.uint64))
    create_array(records, "image_id", np.asarray(image_ids, dtype=np.int64))
    create_array(records, "primitive_offset", np.asarray(primitive_offsets, dtype=np.int64))
    create_array(records, "primitive_count", np.asarray(primitive_counts, dtype=np.int32))
    create_array(primitives, "type", np.asarray(primitive_types, dtype=np.uint8))
    geometry = np.asarray(primitive_geometry, dtype=np.float32).reshape((-1, 7)) if primitive_geometry else np.empty((0, 7), dtype=np.float32)
    create_array(primitives, "geometry_norm", geometry)
    create_array(splits, "train", train_indices)
    create_array(splits, "val", val_indices)
    create_array(splits, "test", np.empty(0, dtype=np.int64))
    root.attrs.update({"format": "feedback_primitive_detr/v1", "coordinate_space": "normalized_image_coordinates", "record_count": len(rows)})
    return {"records": len(rows), "train": len(train_indices), "val": len(val_indices), "primitives": len(primitive_types)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a compact Zarr fine-tuning dataset from engineer-reviewed label JSONL.")
    parser.add_argument("--labels", type=Path, action="append", required=True, help="Reviewed-label JSONL. Repeat for multiple files/drawings.")
    parser.add_argument("--image-root", type=Path, default=Path("."), help="Root used to resolve relative image_path values in the label rows.")
    parser.add_argument("--output", type=Path, default=Path("reviewed_feedback.zarr"))
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of the explicitly named output Zarr directory.")
    add_progress_argument(parser)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be between 0 and 1.")
    if args.output.exists():
        if not args.overwrite:
            print(f"[ERROR] Output already exists: {args.output}. Use --overwrite only if replacement is intended.", file=sys.stderr)
            return 2
        if not args.output.is_dir():
            print(f"[ERROR] --output exists but is not a directory: {args.output}", file=sys.stderr)
            return 2
        shutil.rmtree(args.output)
    progress = RichProgress(enabled=not args.no_progress).start()
    try:
        read_task = progress.add_task("Reading reviewed label files", total=len(args.labels))
        rows = read_rows(args.labels, progress, read_task)
        progress.complete(read_task, f"Read {len(rows):,} reviewed drawings")
        build_task = progress.add_task("Building reviewed feedback dataset", total=len(rows))
        summary = build(rows, args.image_root, args.output, args.val_ratio, args.seed, progress, build_task)
        progress.complete(build_task, "Reviewed feedback dataset complete")
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as exc:
        progress.stop()
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    progress.stop()
    print(f"[DONE] Wrote {args.output}: {summary['records']} records, {summary['primitives']} primitives, train={summary['train']}, val={summary['val']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
